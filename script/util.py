import json
import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bio_runs(tags):
    """BIO -> list (start, end) inclusive."""
    blks, i, n = [], 0, len(tags)
    while i < n:
        if tags[i] == "B":
            j = i
            while j + 1 < n and tags[j + 1] == "I":
                j += 1
            blks.append((i, j)); i = j + 1
        else:
            i += 1
    return blks


def bio_runs_ids(tags, b_tag, i_tag):
    """numeric BIO -> list (start, end)."""
    blks, i, n = [], 0, len(tags)
    while i < n:
        if tags[i] == b_tag:
            j = i
            while j + 1 < n and tags[j + 1] == i_tag:
                j += 1
            blks.append((i, j)); i = j + 1
        else:
            i += 1
    return blks


def bio_runs_ids_conf(tags, probs, b_tag, i_tag, conf_thr=0.0):
    """numeric BIO -> list (start, end); loai run co conf(B-tag) < conf_thr."""
    blks, i, n = [], 0, len(tags)
    while i < n:
        if tags[i] == b_tag:
            j = i
            while j + 1 < n and tags[j + 1] == i_tag:
                j += 1
            if float(probs[i, b_tag]) >= conf_thr:
                blks.append((i, j))
            i = j + 1
        else:
            i += 1
    return blks


def load_records(path):
    """absa_dependency JSON -> [{tokens, head, deprel, blocks:[...]}]."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for r in raw:
        sbs = bio_runs(r["span"])
        abs_ = bio_runs(r["aspect"])
        obs = bio_runs(r["opinion"])
        cat_field = r.get("category") or r.get("topic")
        blocks = []
        for (s, e) in sbs:
            cat = cat_field[s]
            sent = r["sentiment"][s]
            aspects = [a for a in abs_ if s <= a[0] and a[1] <= e]
            opinions = [o for o in obs if s <= o[0] and o[1] <= e]
            blocks.append({"span": (s, e), "category": cat, "sentiment": sent,
                           "aspects": aspects, "opinions": opinions})
        out.append({"tokens": r["tokens"], "head": r.get("head", []),
                    "deprel": r.get("deprel", []), "span_tags": r.get("span", []),
                    "aspect_tags": r.get("aspect", []), "opinion_tags": r.get("opinion", []),
                    "postag": r.get("postag", []),
                    "blocks": blocks})
    return out


def load_splits(base):
    return (load_records(os.path.join(base, "train.json")),
            load_records(os.path.join(base, "dev.json")),
            load_records(os.path.join(base, "test.json")))


# ---------------------------------------------------------------------------
# LABEL MAPS + DEPREL MAPS
# ---------------------------------------------------------------------------
def norm_cat(c):
    return c.replace(" ", "_")


def build_label_maps(records, mode):
    """(rel2id, id2rel, cat2id, id2cat, sent2id, id2sent). """
    cats = set()
    sents = set()
    rels = set(["O"])
    for r in records:
        for b in r["blocks"]:
            cats.add(norm_cat(b["category"]))
            sents.add(b["sentiment"])
    for c in cats:
        rels.add(f"C-{c}")
    for s in sents:
        rels.add(f"SENT-{s}")
    rels.update(["B-A", "I-A", "B-O", "I-O"])
    rel2id = {"O": 0}
    for rel in sorted(rels - {"O"}):
        rel2id[rel] = len(rel2id)
    id2rel = {i: rel for rel, i in rel2id.items()}
    cat2id = {c: i for i, c in enumerate(sorted(cats))}
    id2cat = {i: c for c, i in cat2id.items()}
    sent2id = {s: i for i, s in enumerate(sorted(sents))}
    id2sent = {i: s for s, i in sent2id.items()}
    return rel2id, id2rel, cat2id, id2cat, sent2id, id2sent


# ---------------------------------------------------------------------------
# TOKENIZER HELPERS
# ---------------------------------------------------------------------------
def _safe_word_ids(tokenizer, enc, tokens):
    try:
        wids = enc.word_ids()
        if wids is not None:
            return wids
    except (AttributeError, NotImplementedError, ValueError):
        pass
    wids = [None]
    for i, w in enumerate(tokens):
        n = len(tokenizer.tokenize(w))
        wids.extend([i] * n)
    wids.append(None)
    L = enc["input_ids"][0].size(0)
    if len(wids) > L:
        wids = wids[:L]
    return wids


def tokenize_first_n_eff(tokenizer, tokens, max_pieces):
    budget = max_pieces - 2
    n_eff = 0
    used = 0
    for w in tokens:
        n = len(tokenizer.tokenize(w))
        if n <= 0:
            continue
        if used + n > budget:
            break
        used += n
        n_eff += 1
    return n_eff


# ---------------------------------------------------------------------------
# SPAN-LEVEL DEPENDENCY RELATION FEATURES (deprel version)
# ---------------------------------------------------------------------------
DEP_ROLE_G = 7


def rel_group(rel):
    """Deprel -> coarse role group (0=root ... 6=other)."""
    if rel == "root":
        return 0
    if rel in ("nsubj", "csubj", "nsubj:pass"):
        return 1
    if rel in ("obj", "dobj", "iobj", "obl", "obl:arg", "obl:agent"):
        return 2
    if rel in ("nmod", "amod", "det", "nummod", "appos"):
        return 3
    if rel in ("advmod", "advcl", "xcomp", "ccomp", "acl", "acl:relcl"):
        return 4
    if rel in ("conj", "cc", "case", "mark", "aux", "aux:pass", "cop",
               "compound", "flat", "list", "fixed"):
        return 5
    return 6


def compute_dep_mat(N, head_s, deprel_s):
    """Dependency feature matrices: edge/dir/path/lca (NxN) + role (N x DEP_ROLE_G).
    head_s: parent index (1-based, 0 = none); deprel_s: relation per word.
    Dung lam span-level aspect-opinion relation feature (deprel version)."""
    from collections import deque
    parent = [-1] * N
    for w in range(N):
        p = head_s[w]
        if p is not None and 1 <= p <= N:
            parent[w] = p - 1
    adj = [[] for _ in range(N)]
    for i in range(N):
        if parent[i] >= 0:
            adj[i].append(parent[i]); adj[parent[i]].append(i)
    edge = np.zeros((N, N), dtype=np.float32)
    ddir = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        p = parent[i]
        if p >= 0:
            edge[i, p] = 1.0; edge[p, i] = 1.0
            ddir[i, p] = 1.0; ddir[p, i] = -1.0
    path = np.full((N, N), float(N), dtype=np.float32)
    for src in range(N):
        dist = [-1] * N; dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1; q.append(v)
        for j in range(N):
            if dist[j] >= 0:
                path[src, j] = dist[j]
    depth = [-1] * N
    def dfs(u, dd):
        depth[u] = dd
        for v in adj[u]:
            if depth[v] < 0:
                dfs(v, dd + 1)
    for i in range(N):
        if depth[i] < 0:
            dfs(i, 0)
    LOG = max(1, N.bit_length())
    up = [[-1] * N for _ in range(LOG)]
    up[0] = parent[:]
    for k in range(1, LOG):
        for i in range(N):
            up[k][i] = up[k - 1][up[k - 1][i]] if up[k - 1][i] >= 0 else -1
    def lca(u, v):
        if depth[u] < depth[v]:
            u, v = v, u
        dd = depth[u] - depth[v]
        k = 0
        while dd:
            if dd & 1:
                u = up[k][u]
            dd >>= 1; k += 1
        if u == v:
            return u
        for k in range(LOG - 1, -1, -1):
            if up[k][u] != up[k][v]:
                u = up[k][u]; v = up[k][v]
        return up[0][u]
    lca_d = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            l = lca(i, j)
            lca_d[i, j] = depth[l] if l >= 0 else 0.0
    role = np.zeros((N, DEP_ROLE_G), dtype=np.float32)
    for i in range(N):
        r = deprel_s[i] if i < len(deprel_s) else "root"
        role[i, rel_group(r)] = 1.0
    return {"edge": edge, "dir": ddir, "path": path, "lca": lca_d, "role": role}


