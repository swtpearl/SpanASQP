import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

from .util import set_seed, norm_cat, bio_runs_ids
from .candidates import decode_spans
from .decode import classify_spans
from .loss import span_loss
from .metrics import compute_metrics, span_f1, quad_metrics, aope_f1


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def make_batch(item, device, cand_spans):
    """Batch size 1 tu 1 dataset item, voi cand_spans tu y (inference path)."""
    M = len(cand_spans)
    return {
        "input_ids": item["input_ids"].unsqueeze(0).to(device),
        "attention_mask": item["attention_mask"].unsqueeze(0).to(device),
        "pieces2word": item["pieces2word"].unsqueeze(0).to(device),
        "dep_mat": item["dep_mat"].unsqueeze(0).to(device),
        "dep_role": item["dep_role"].unsqueeze(0).to(device),
        "word_mask": item["word_mask"].unsqueeze(0).to(device),
        "start_ind": item["start_ind"].unsqueeze(0).to(device),
        "end_ind": item["end_ind"].unsqueeze(0).to(device),
        "bio_aspect": item["bio_aspect"].unsqueeze(0).to(device),
        "bio_opinion": item["bio_opinion"].unsqueeze(0).to(device),
        "sp_type": item["sp_type"][:M].unsqueeze(0).to(device),
        "sp_cat": item["sp_cat"][:M].unsqueeze(0).to(device),
        "sp_sent": item["sp_sent"][:M].unsqueeze(0).to(device),
        "sp_mask": item["sp_mask"][:M].unsqueeze(0).to(device),
        "node_n": item["node_n"], "n_words_word": item["n_words_word"],
        "cand_spans": [cand_spans], "cand_map": [None],
        "blocks": [item["blocks"]], "tokens": [item["tokens"]],
    }


def normalize_blocks(model, blocks):
    for b in blocks:
        if b["sentiment"] is None:
            b["sentiment"] = 0
        if isinstance(b["sentiment"], str):
            b["sentiment"] = model.sent2id.get(b["sentiment"], -1)
        if isinstance(b["category"], str):
            b["category"] = model.cat2id.get(b["category"], -1)
    return blocks


def gold_blocks(model, blks):
    return [{"span": b["span"], "category": model.cat2id.get(norm_cat(b["category"]), -1),
             "sentiment": model.sent2id.get(b["sentiment"], -1),
             "aspects": b["aspects"], "opinions": b["opinions"]} for b in blks]


# ---------------------------------------------------------------------------
# PREDICT (span probs -> decode spans -> classify)
# ---------------------------------------------------------------------------
def predict(model, item, device, max_spans=None, max_span_len=12, s_thr=0.40, e_thr=0.30,
            type_mode="full", type_conf_thr=0.9):

    model.eval()
    b1 = make_batch(item, device, [])
    with torch.no_grad():
        out1 = model(b1)
        ps = out1["p_start"][0].cpu().numpy()
        pe = out1["p_end"][0].cpu().numpy()
    pred_spans = decode_spans(ps, pe, max_span_len, max_spans, s_thr, e_thr)
    b2 = make_batch(item, device, pred_spans)
    with torch.no_grad():
        out2 = model(b2)
    blocks = classify_spans(model, out2, 0, b2, type_mode=type_mode, type_conf_thr=type_conf_thr)
    return blocks, pred_spans, ps, pe


# ---------------------------------------------------------------------------
# EVALUATE (test): report span / ATE / OTE / AOPE / ASQP / ASTE 
# ---------------------------------------------------------------------------
def evaluate(model, dataset, device, task="asqp", max_spans=None, max_span_len=12,
             s_thr=0.40, e_thr=0.30, type_mode="full", type_conf_thr=0.9):
    """End-to-end eval (bao gom span detection). dataset = SpanClsDataset.
    Report (dict):
      - span : start/end exact F1
      - ate / ote / aope
      - asqp : per-quad quad F1 (chi khi task=="asqp")
      - aste : per-quad triplet F1 (luon tinh duoc)
      - perquad_main : asqp (task asqp) | aste (task aste)
    Block-F1 (block_quad / block_subgroup) da chuyen sang script/analysis_utils.py."""
    pred_all, gold_all, span_all, records = [], [], [], []
    gq, pq = [], []
    model.eval()
    for i in range(len(dataset)):
        item = dataset[i]
        blocks, pred_spans, ps, pe = predict(model, item, device, max_spans, max_span_len,
                                             s_thr, e_thr, type_mode, type_conf_thr)
        pred_all.append(normalize_blocks(model, copy.deepcopy(blocks)))
        gold_all.append(gold_blocks(model, item["blocks"]))
        span_all.append([tuple(s) for s in pred_spans])
        gq.append((item["tokens"], item["blocks"]))
        pq.append((item["tokens"], blocks))
        rec = {"gold": gold_all[-1], "pred": pred_all[-1], "pred_spans": pred_spans,
               "p_start": ps, "p_end": pe}
        records.append(rec)

    res = compute_metrics(pred_all, gold_all, model.cat2id, model.sent2id, model.use_cat)
    aope = aope_f1(gq, pq, task)
    report = {
        "task": task,
        "span": span_f1(span_all, gold_all),
        "ate": res["ATE"], "ote": res["OTE"],
        "aope": {"f1": aope[0], "precision": aope[1], "recall": aope[2],
                 "tp": aope[3], "fp": aope[4], "fn": aope[5]},
        "asqp": None, "aste": None,
    }
    if task == "asqp":
        q_overall, _ = quad_metrics(gq, pq, task="asqp")
        report["asqp"] = q_overall
        report["perquad_main"] = q_overall
    t_overall, _ = quad_metrics(gq, pq, task="aste")
    report["aste"] = t_overall
    if task != "asqp":
        report["perquad_main"] = t_overall
    return report


# ---------------------------------------------------------------------------
# VALIDATE (val, batch, perquad main metric) + BIO monitor
# ---------------------------------------------------------------------------
def decode_batch(model, out, batch, bi, type_mode="full", type_conf_thr=0.9):
    return classify_spans(model, out, bi, batch, type_mode=type_mode, type_conf_thr=type_conf_thr)


def validate(model, dataloader, device, task="asqp", type_mode="full", type_conf_thr=0.9):
    """Per-quad (METRIC CHINH) tren val -> dung chon checkpoint."""
    model.eval()
    gq, pq = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = to_device(batch, device)
            out = model(batch)
            for bi in range(len(batch["tokens"])):
                blocks = decode_batch(model, out, batch, bi, type_mode, type_conf_thr)
                gq.append((batch["tokens"][bi], batch["blocks"][bi]))
                pq.append((batch["tokens"][bi], blocks))
    if not gq:
        return 0.0
    overall, _ = quad_metrics(gq, pq, task=task)
    return overall["f1"]


def runs_f1(pred_lists, gold_lists):
    """Micro exact-match P/R/F1 cho list (start,end) tung mau (one-to-one)."""
    tp = fp = fn = 0
    for prs, grs in zip(pred_lists, gold_lists):
        used = [False] * len(grs)
        for pr in prs:
            k = tuple(pr)
            m = False
            for i, gk in enumerate(grs):
                if not used[i] and tuple(gk) == k:
                    used[i] = True; m = True; break
            tp += m
            if not m:
                fp += 1
        fn += sum(1 for u in used if not u)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}


def bio_f1(model, dataloader, device):
    """Micro exact F1 cua mention runs (aspect/opinion, decode conf=0) tren val."""
    model.eval()
    a_pred, a_gold, o_pred, o_gold = [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = to_device(batch, device)
            out = model(batch)
            for bi in range(len(batch["tokens"])):
                n_i = int(batch["word_n"][bi]) if "word_n" in batch else batch["n_words_word"]
                for head_key, gold_key, plist, glist in (
                        ("bio_a", "bio_aspect", a_pred, a_gold),
                        ("bio_o", "bio_opinion", o_pred, o_gold)):
                    t = out[head_key][bi][:n_i].argmax(dim=-1).cpu().tolist()
                    g = batch[gold_key][bi][:n_i].cpu().tolist()
                    plist.append(bio_runs_ids(t, 1, 2))
                    glist.append(bio_runs_ids(g, 1, 2))
    return {"aspect": runs_f1(a_pred, a_gold),
            "opinion": runs_f1(o_pred, o_gold)}


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------
def train(model, train_loader, val_loader, device, task="asqp", epochs=2, lr=1e-5,
          weight_decay=0.01, patience=2, out_path=None, seed=25,
          w_span=0.5, w_bio=1.0, w_type=1.0, w_cat=1.0, w_sent=1.0,
          s_thr=0.40, e_thr=0.30, w_boundary=15.0, w_bg=0.1,
          history_out=None, type_mode="full", type_conf_thr=0.9):
    """Train span-classifier (deprel). Epoch report: loss | val_quad |
    val_bioA | val_bioO. Checkpoint select bang val per-quad F1 (metric chinh)."""
    set_seed(seed)
    no_decay = ["bias", "LayerNorm.weight", "ln_1.weight", "ln_2.weight", "gamma", "beta"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped, lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    best_score = -1.0
    best_state, counter = None, 0
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for batch in train_loader:
            batch = to_device(batch, device)
            model.zero_grad()
            out = model(batch)
            loss = span_loss(model, out, batch, w_span=w_span, w_bio=w_bio,
                             w_type=w_type, w_cat=w_cat, w_sent=w_sent,
                             w_boundary=w_boundary, w_bg=w_bg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            tot += loss.item()
        avg_loss = tot / len(train_loader)
        val_quad = validate(model, val_loader, device, task=task,
                            type_mode=type_mode, type_conf_thr=type_conf_thr)
        bio_met = bio_f1(model, val_loader, device)
        msg = ("  epoch %d: loss=%.4f | val_quad=%.4f | val_bioA=%.4f | val_bioO=%.4f"
               % (ep + 1, avg_loss, val_quad, bio_met["aspect"]["f1"], bio_met["opinion"]["f1"]))
        print(msg)
        if history_out is not None:
            history_out.append({"epoch": ep + 1, "train_loss": round(avg_loss, 4),
                                "val_quad_f1": round(val_quad, 4),
                                "dev_bio_aspect_f1": round(bio_met["aspect"]["f1"], 4),
                                "dev_bio_opinion_f1": round(bio_met["opinion"]["f1"], 4)})
        if val_quad > best_score:
            best_score = val_quad
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(best_state if best_state is not None else model.state_dict(), out_path)
    return best_score


# ---------------------------------------------------------------------------
# SAVE INFERENCE (inference_output.jsonl + metrics_final.json)
# ---------------------------------------------------------------------------
def save_inference(model, dataset, device, path, tokenizer, task="asqp",
                   max_spans=None, max_span_len=12, s_thr=0.40, e_thr=0.30,
                   type_mode="full", type_conf_thr=0.9, write_metrics=True):
    """Ghi inference_output.jsonl (p_start/p_end -> re-sim decode offline 0-cost).
    Neu write_metrics: tinh report = evaluate() va ghi metrics_final.json canh do."""
    model.eval()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n_lines = 0
    records = []
    with open(path, "w", encoding="utf-8") as f:
        for i in range(len(dataset)):
            item = dataset[i]
            blocks, pred_spans, ps, pe = predict(model, item, device, max_spans, max_span_len,
                                                 s_thr, e_thr, type_mode, type_conf_thr)
            pred = normalize_blocks(model, blocks)
            gold = gold_blocks(model, item["blocks"])
            toks = item["tokens"]
            text = tokenizer.decode(item["input_ids"], skip_special_tokens=True)
            line = {"tokens": toks, "text": text, "gold": gold, "pred": pred,
                    "pred_spans": pred_spans,
                    "p_start": [round(float(x), 4) for x in ps],
                    "p_end": [round(float(x), 4) for x in pe]}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            records.append(line)
            n_lines += 1
    print(f"saved inference_output.jsonl -> {path} ({n_lines} lines)")
    if write_metrics:
        report = evaluate(model, dataset, device, task=task, max_spans=max_spans,
                          max_span_len=max_span_len, s_thr=s_thr, e_thr=e_thr,
                          type_mode=type_mode, type_conf_thr=type_conf_thr)
        mpath = os.path.join(os.path.dirname(path), "metrics_final.json")
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("saved metrics_final.json ->", mpath)
    return records