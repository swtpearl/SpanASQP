import random

import numpy as np

from .const import TYPE_EE, TYPE_IE, TYPE_EI, TYPE_II
from .util import norm_cat


# ---------------------------------------------------------------------------
# SPAN CANDIDATES (train: TOAN BO gold + negatives; infer: start/end layer)
# ---------------------------------------------------------------------------
def sample_negatives(spans, N, n_neg, max_span_len, rng,
                     neg_types=("rand", "merged"), span2blk=None):
    """Negatives: rand/merged.
    merged: merge 2 gold span khac cat/sent."""
    negs = []
    tries = 0
    while len(negs) < n_neg and tries < 150:
        tries += 1
        typ = rng.choice(neg_types) if neg_types else "rand"
        if typ == "merged" and len(spans) >= 2:
            a, b = rng.sample(list(spans), 2)
            s, e = min(a[0], b[0]), max(a[1], b[1])
            if e - s + 1 > max_span_len or e >= N:
                continue
            sp = (s, e)
            if span2blk is not None:
                ba, bb = span2blk.get(a), span2blk.get(b)
                if (ba is not None and bb is not None
                        and norm_cat(ba["category"]) == norm_cat(bb["category"])
                        and ba["sentiment"] == bb["sentiment"]):
                    continue
        else:
            s = rng.randint(0, max(N - 1, 0))
            l = rng.randint(1, min(max_span_len, N - s))
            sp = (s, s + l - 1)
            if any(not (sp[1] < g[0] or sp[0] > g[1]) for g in spans):
                continue
            if sp in negs:
                continue
            negs.append(sp)
            continue
        if sp in spans or sp in negs:
            continue
        negs.append(sp)
    return negs


def build_candidates(blocks, N, max_span_len, n_neg, rng,
                     neg_types=("rand", "merged")):
    spans, cand_map = [], []
    for b in blocks:
        s, e = b["span"]
        if e >= N:
            continue
        if e - s + 1 > max_span_len:
            continue
        spans.append((s, e))
        cand_map.append(b)
    if n_neg > 0:
        span2blk = dict(zip(spans, cand_map))
        negs = sample_negatives(spans, N, n_neg, max_span_len, rng,
                                neg_types=neg_types, span2blk=span2blk)
        spans = spans + negs
        cand_map = cand_map + [None] * len(negs)
    return spans, cand_map


def decode_spans(p_start, p_end, max_span_len, max_spans=None,
                 s_thr=0.40, e_thr=0.30):
    nms_iou = 0.50
    N = len(p_start)
    cands = []
    for s in range(N):
        if p_start[s] < s_thr:
            continue
        for e in range(s, min(s + max_span_len, N)):
            if p_end[e] >= e_thr:
                cands.append((s, e, float(p_start[s] * p_end[e])))
                break
    cands.sort(key=lambda x: -x[2]) # rank
    # check overlap 
    kept = []
    for (s, e, sc) in cands:
        if max_spans is not None and len(kept) >= max_spans:
            break
        suppressed = False
        for (ks, ke, _) in kept:
            inter = max(0, min(e, ke) - max(s, ks) + 1)
            union = (e - s + 1) + (ke - ks + 1) - inter
            iou = inter / union if union > 0 else 0.0
            if ks == s or ke == e or iou > nms_iou:
                suppressed = True
                break
        if not suppressed:
            kept.append((s, e, sc))
    return [(s, e) for (s, e, _) in kept]


# ---------------------------------------------------------------------------
# CLASSIFY TARGET 
# ---------------------------------------------------------------------------
def block_type(blk):
    has_a = len(blk["aspects"]) > 0
    has_o = len(blk["opinions"]) > 0
    if has_a and has_o:
        return TYPE_EE
    if (not has_a) and has_o:
        return TYPE_IE
    if has_a and (not has_o):
        return TYPE_EI
    return TYPE_II


def build_targets(cand_map, spans, cat2id, sent2id):
    """Per-span: type, cat, sent, mask.
    mask: 1.0 = gold span | 0 = negative."""
    M = len(cand_map)
    typ = np.zeros(M, dtype=np.int64)
    cat = np.zeros(M, dtype=np.int64)
    sent = np.zeros(M, dtype=np.int64)
    msk = np.zeros(M, dtype=np.float32)
    for i, blk in enumerate(cand_map):
        if blk is not None:
            typ[i] = block_type(blk)
            cat[i] = cat2id.get(norm_cat(blk["category"]), 0)
            sent[i] = sent2id.get(blk["sentiment"], 0)
            msk[i] = 1.0
    return typ, cat, sent, msk
