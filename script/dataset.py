import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .util import _safe_word_ids, tokenize_first_n_eff
from .util import compute_dep_mat, DEP_ROLE_G
from .candidates import build_candidates, build_targets


class SpanClsDataset(Dataset):
    def __init__(self, records, tokenizer, cat2id, sent2id,
                 max_pieces=256, max_words=128, max_span_len=12,
                 n_neg=4, seed=0, neg_types=("rand", "merged")):
        self.records = records
        self.tokenizer = tokenizer
        self.cat2id = cat2id
        self.sent2id = sent2id
        self.max_pieces = max_pieces
        self.max_words = max_words
        self.max_span_len = max_span_len
        self.n_neg = n_neg
        self.neg_types = neg_types
        self.rng = random.Random(seed)
        self._cache = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        rec = self.records[idx]
        tokens_all = rec["tokens"][: self.max_words]
        n_eff = tokenize_first_n_eff(self.tokenizer, tokens_all, self.max_pieces)
        n_eff = max(n_eff, 1)
        tokens = tokens_all[:n_eff]
        N = n_eff
        trunc_rec = {"tokens": tokens,
                     "blocks": [b for b in rec["blocks"] if b["span"][1] < N]}
        head_s = list(rec.get("head", []))[:N]
        deprel_s = list(rec.get("deprel", []))[:N]
        while len(head_s) < N:
            head_s.append(0)
        while len(deprel_s) < N:
            deprel_s.append("root")
        for i in range(N):
            if head_s[i] >= N:
                head_s[i] = 0
                deprel_s[i] = "root"
        enc = self.tokenizer(tokens, is_split_into_words=True, truncation=True,
                             max_length=self.max_pieces, return_tensors="pt")
        input_ids = enc["input_ids"][0]
        attn = enc["attention_mask"][0]
        word_ids = _safe_word_ids(self.tokenizer, enc, tokens)
        L = input_ids.size(0)
        G_word = N + 2
        p2w = np.zeros((G_word, L), dtype=np.float32)
        p2w[0, 0] = 1.0
        sep_idx = int(attn.sum().item()) - 1
        p2w[G_word - 1, sep_idx] = 1.0
        for p, w in enumerate(word_ids):
            if w is not None and w < N:
                p2w[w + 1, p] = 1.0
        dm = compute_dep_mat(N, head_s, deprel_s)
        dep_mat = np.stack([dm["edge"], dm["dir"], dm["path"], dm["lca"]], axis=-1)
        dep_role = dm["role"]

        # ---- gold word BIO + start/end indicators ----
        bio_aspect = np.zeros(N, dtype=np.int64)
        bio_opinion = np.zeros(N, dtype=np.int64)
        start_ind = np.zeros(N, dtype=np.float32)
        end_ind = np.zeros(N, dtype=np.float32)
        for b in trunc_rec["blocks"]:
            s, e = b["span"]
            start_ind[s] = 1.0
            end_ind[e] = 1.0
            for a in b["aspects"]:
                bio_aspect[a[0]] = 1
                bio_aspect[a[0] + 1:a[1] + 1] = 2
            for o in b["opinions"]:
                bio_opinion[o[0]] = 1
                bio_opinion[o[0] + 1:o[1] + 1] = 2

        # ---- candidate spans ----
        spans, cand_map = build_candidates(
            trunc_rec["blocks"], N, self.max_span_len,
            self.n_neg, self.rng, self.neg_types)
        sp_type, sp_cat, sp_sent, sp_mask = build_targets(
            cand_map, spans, self.cat2id, self.sent2id)

        item = {
            "input_ids": input_ids,
            "attention_mask": attn,
            "pieces2word": torch.from_numpy(p2w),
            "dep_mat": torch.from_numpy(dep_mat),
            "dep_role": torch.from_numpy(dep_role),
            "node_n": G_word,                    # word-node dim = N+2
            "n_words_word": N,                   # word count for BIO/span heads
            "word_mask": torch.ones(N, dtype=torch.float32),
            "start_ind": torch.from_numpy(start_ind),
            "end_ind": torch.from_numpy(end_ind),
            "bio_aspect": torch.from_numpy(bio_aspect),
            "bio_opinion": torch.from_numpy(bio_opinion),
            "sp_type": torch.from_numpy(sp_type),
            "sp_cat": torch.from_numpy(sp_cat),
            "sp_sent": torch.from_numpy(sp_sent),
            "sp_mask": torch.from_numpy(sp_mask),
            "cand_spans": spans,                 # list[(s,e)]
            "cand_map": cand_map,                # list[block|None]
            "blocks": trunc_rec["blocks"],
            "tokens": tokens,
        }
        self._cache[idx] = item
        return item


def collate_fn(batch):
    max_node = max(b["node_n"] for b in batch)
    max_l = max(b["input_ids"].size(0) for b in batch)
    max_w = max(b["n_words_word"] for b in batch)
    max_m = max(len(b["cand_spans"]) for b in batch)
    B = len(batch)
    input_ids = torch.zeros((B, max_l), dtype=torch.long)
    attn = torch.zeros((B, max_l), dtype=torch.long)
    p2w = torch.zeros((B, max_node, max_l))
    dep_mat = torch.zeros((B, max_w, max_w, 4))
    dep_role = torch.zeros((B, max_w, DEP_ROLE_G))
    word_mask = torch.zeros((B, max_w))
    start_ind = torch.zeros((B, max_w))
    end_ind = torch.zeros((B, max_w))
    bio_aspect = torch.zeros((B, max_w), dtype=torch.long)
    bio_opinion = torch.zeros((B, max_w), dtype=torch.long)
    sp_type = torch.zeros((B, max_m), dtype=torch.long)
    sp_cat = torch.zeros((B, max_m), dtype=torch.long)
    sp_sent = torch.zeros((B, max_m), dtype=torch.long)
    sp_mask = torch.zeros((B, max_m), dtype=torch.float32)
    for i, b in enumerate(batch):
        L = b["input_ids"].size(0)
        G = b["node_n"]
        Nw = b["n_words_word"]
        Mi = len(b["cand_spans"])
        input_ids[i, :L] = b["input_ids"]
        attn[i, :L] = b["attention_mask"]
        p2w[i, :G, :L] = b["pieces2word"]
        dep_mat[i, :Nw, :Nw] = b["dep_mat"]
        dep_role[i, :Nw] = b["dep_role"]
        word_mask[i, :Nw] = b["word_mask"]
        start_ind[i, :Nw] = b["start_ind"]
        end_ind[i, :Nw] = b["end_ind"]
        bio_aspect[i, :Nw] = b["bio_aspect"]
        bio_opinion[i, :Nw] = b["bio_opinion"]
        sp_type[i, :Mi] = b["sp_type"]
        sp_cat[i, :Mi] = b["sp_cat"]
        sp_sent[i, :Mi] = b["sp_sent"]
        sp_mask[i, :Mi] = b["sp_mask"]
    return {
        "input_ids": input_ids, "attention_mask": attn, "pieces2word": p2w,
        "dep_mat": dep_mat, "dep_role": dep_role,
        "word_mask": word_mask, "start_ind": start_ind, "end_ind": end_ind,
        "bio_aspect": bio_aspect, "bio_opinion": bio_opinion,
        "sp_type": sp_type, "sp_cat": sp_cat, "sp_sent": sp_sent, "sp_mask": sp_mask,
        "cand_spans": [b["cand_spans"] for b in batch],
        "cand_map": [b["cand_map"] for b in batch],
        "node_n": max_node, "n_words_word": max_w,
        "word_n": [b["n_words_word"] for b in batch],
        "blocks": [b["blocks"] for b in batch], "tokens": [b["tokens"] for b in batch],
    }


