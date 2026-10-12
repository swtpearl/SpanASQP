import numpy as np
import torch

from .const import TYPE_NONE, TYPE_EE, TYPE_IE, TYPE_EI, TYPE_II
from .util import bio_runs_ids_conf


def decode_mentions(model, out, bi, N, conf_thr=0.0):
    pa = torch.softmax(out["bio_a"][bi][:N], dim=-1).cpu().numpy()
    po = torch.softmax(out["bio_o"][bi][:N], dim=-1).cpu().numpy()
    ta = np.argmax(pa, axis=-1); to = np.argmax(po, axis=-1)
    return (bio_runs_ids_conf(ta, pa, 1, 2, conf_thr),
            bio_runs_ids_conf(to, po, 1, 2, conf_thr))


def classify_spans(model, out, bi, batch, type_mode="full", type_conf_thr=0.9):
    cand = batch["cand_spans"][bi]
    N = batch["n_words_word"]
    asp_w, opn_w = decode_mentions(model, out, bi, N, 0.0)
    type_probs = torch.softmax(out["head_type"][bi], dim=-1)
    type_confs, type_preds = type_probs.max(dim=-1)
    cat_preds = torch.argmax(out["head_cat"][bi], dim=-1)
    sent_preds = torch.argmax(out["head_sent"][bi], dim=-1)
    p_s = out["p_start"][bi].cpu().numpy()
    p_e = out["p_end"][bi].cpu().numpy()

    raw_blocks = []
    for idx in range(len(cand)):
        t = type_preds[idx].item()
        if t == TYPE_NONE:
            continue
        sp = cand[idx]
        score = float(p_s[sp[0]]) * float(p_e[sp[1]])
        blk = build_block(model, t, type_confs[idx].item(), sp, asp_w, opn_w,
                          cat_preds, sent_preds, idx, score,
                          type_mode, type_conf_thr)
        if blk is not None:
            raw_blocks.append(blk)
    return raw_blocks


def build_block(model, t, t_conf, cand_sp, asp_w, opn_w, cat_preds, sent_preds,
                 idx, score, type_mode="full", type_conf_thr=0.9):
    """Dung 1 block tu mentions trong span theo type_mode."""
    asps = [tuple(a) for a in asp_w if cand_sp[0] <= a[0] and a[1] <= cand_sp[1]]
    opns = [tuple(o) for o in opn_w if cand_sp[0] <= o[0] and o[1] <= cand_sp[1]]

    type_driven = (type_mode == "full") or \
                  (type_mode == "hybrid" and t_conf >= type_conf_thr)

    if type_mode == "full":
        if t in (TYPE_EE, TYPE_EI) and len(asps) == 0:
            return None
        if t in (TYPE_EE, TYPE_IE) and len(opns) == 0:
            return None

    if type_driven:
        if t == TYPE_II:
            asps, opns = [], []
        elif t == TYPE_IE:
            asps = []
        elif t == TYPE_EI:
            opns = []
    # else (bio-driven): giu nguyen asps/opns tu BIO

    return {"span": cand_sp,
            "category": model.id2cat[cat_preds[idx].item()],
            "sentiment": model.id2sent[sent_preds[idx].item()],
            "aspects": asps, "opinions": opns, "score": score}