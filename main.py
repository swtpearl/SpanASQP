import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from script.dataset import SpanClsDataset, collate_fn
from script.model import SpanClsModel
from script.train import train, evaluate, save_inference, predict
from script.util import (build_label_maps, get_device, load_splits, set_seed)


def parse_dep_feats(feats):
    if isinstance(feats, str):
        feats = [f.strip() for f in feats.split(",") if f.strip()]
    return tuple(feats)


def build_parser():
    p = argparse.ArgumentParser(description="ABSA Quad span-classifier (deprel)")
    p.add_argument("--config", type=str, default=None,
                   help="JSON config file (override mac dinh)")
    p.add_argument("--mode", type=str, default="train", choices=["train", "eval", "predict"])
    p.add_argument("--model_name", type=str, default="vinai/phobert-base")
    p.add_argument("--data_base", type=str, default="data/seg")
    p.add_argument("--out_dir", type=str, default="runs/main")
    p.add_argument("--task", type=str, default="asqp", choices=["asqp", "aste"])
    p.add_argument("--dep_feats", type=str, default="edge,dir,path,lca")
    p.add_argument("--ckpt", type=str, default=None, help=".pth checkpoint")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=25)
    p.add_argument("--max_pieces", type=int, default=256)
    p.add_argument("--max_words", type=int, default=128)
    p.add_argument("--max_span_len", type=int, default=12)
    p.add_argument("--n_neg", type=int, default=8)
    p.add_argument("--neg_types", type=str, default="rand,merged")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--node_hid", type=int, default=384)
    p.add_argument("--no_bert_last_4", action="store_true")
    p.add_argument("--w_span", type=float, default=0.5)
    p.add_argument("--w_boundary", type=float, default=15.0)
    p.add_argument("--w_bg", type=float, default=0.1)
    p.add_argument("--s_thr", type=float, default=0.90)
    p.add_argument("--e_thr", type=float, default=0.99)
    p.add_argument("--type_mode", type=str, default="full", choices=["full", "hybrid"])
    p.add_argument("--type_conf_thr", type=float, default=0.9)
    # p.add_argument("--limit", type=int, default=None,
    #                help="test nhanh")
    return p


def build_model(cfg, cat2id, id2cat, sent2id, id2sent):
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = SpanClsModel(
        cfg.model_name, cat2id, id2cat, sent2id, id2sent,
        node_hid=cfg.node_hid, dropout=cfg.dropout,
        use_bert_last_4=not cfg.no_bert_last_4,
        max_span_len=cfg.max_span_len, use_cat=(cfg.task == "asqp"),
        dep_feats=parse_dep_feats(cfg.dep_feats))
    return model, tok


def load_data(cfg, cat2id, sent2id, train_data, dev_data, test_data):
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    nt = tuple(x.strip() for x in cfg.neg_types.split(",") if x.strip())
    tr_ds = SpanClsDataset(train_data, tok, cat2id, sent2id, cfg.max_pieces, cfg.max_words,
                           cfg.max_span_len, cfg.n_neg, seed=0, neg_types=nt)
    va_ds = SpanClsDataset(dev_data, tok, cat2id, sent2id, cfg.max_pieces, cfg.max_words,
                           cfg.max_span_len, cfg.n_neg, seed=1, neg_types=nt)
    te_ds = SpanClsDataset(test_data, tok, cat2id, sent2id, cfg.max_pieces, cfg.max_words,
                           cfg.max_span_len, cfg.n_neg, seed=2, neg_types=nt)
    return tr_ds, va_ds, te_ds, tok


def run_train(cfg, model, tok, tr_ds, va_ds, te_ds, device):
    os.makedirs(cfg.out_dir, exist_ok=True)
    ckpt_path = cfg.ckpt or os.path.join(cfg.out_dir, "best_model.pt")
    tr_ld = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    va_ld = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    best = train(model, tr_ld, va_ld, device, task=cfg.task, epochs=cfg.epochs,
                 lr=cfg.lr, weight_decay=cfg.weight_decay, patience=cfg.patience,
                 out_path=ckpt_path, seed=cfg.seed, w_span=cfg.w_span,
                 s_thr=cfg.s_thr, e_thr=cfg.e_thr, w_boundary=cfg.w_boundary,
                 w_bg=cfg.w_bg, history_out=[],
                 type_mode=cfg.type_mode, type_conf_thr=cfg.type_conf_thr)
    print("[train] best val_quad=%.4f  ckpt -> %s" % (best, ckpt_path))

    report = evaluate(model, te_ds, device, task=cfg.task, max_spans=None,
                      max_span_len=cfg.max_span_len, s_thr=cfg.s_thr,
                      e_thr=cfg.e_thr, type_mode=cfg.type_mode,
                      type_conf_thr=cfg.type_conf_thr)
    print_report(cfg.task, report)
    if not cfg.no_save:
        out_jsonl = os.path.join(cfg.out_dir, "inference_output.jsonl")
        save_inference(model, te_ds, device, out_jsonl, tok, task=cfg.task,
                       max_spans=None, max_span_len=cfg.max_span_len,
                       s_thr=cfg.s_thr, e_thr=cfg.e_thr,
                       type_mode=cfg.type_mode, type_conf_thr=cfg.type_conf_thr,
                       write_metrics=True)
    return report


def run_eval(cfg, model, te_ds, device):
    if cfg.ckpt:
        model.load_state_dict(torch.load(cfg.ckpt, map_location=device))
        print("[eval] loaded ckpt:", cfg.ckpt)
    report = evaluate(model, te_ds, device, task=cfg.task, max_spans=None,
                      max_span_len=cfg.max_span_len, s_thr=cfg.s_thr,
                      e_thr=cfg.e_thr, type_mode=cfg.type_mode,
                      type_conf_thr=cfg.type_conf_thr)
    print_report(cfg.task, report)
    return report


def run_predict(cfg, model, te_ds, device):
    if cfg.ckpt:
        model.load_state_dict(torch.load(cfg.ckpt, map_location=device))
    model.eval()
    n = min(len(te_ds), cfg.limit or len(te_ds))
    for i in range(n):
        item = te_ds[i]
        blocks, spans, _, _ = predict(model, item, device, max_spans=None,
                                      max_span_len=cfg.max_span_len,
                                      s_thr=cfg.s_thr, e_thr=cfg.e_thr,
                                      type_mode=cfg.type_mode,
                                      type_conf_thr=cfg.type_conf_thr)
        print("=" * 60)
        print("[%d] %s" % (i, " ".join(item["tokens"])))
        print("   spans:", spans)
        for b in blocks:
            print("   quad:", b["span"], b["category"], b["sentiment"],
                  "aspects=", b["aspects"], "opinions=", b["opinions"])


def print_report(task, report):
    main = report.get("perquad_main") or {}
    print("[report %s] perquad(F1=%.4f P=%.3f R=%.3f) | span=%.4f ATE=%.4f OTE=%.4f AOPE=%.4f"
          % (task, main.get("f1", 0), main.get("precision", 0), main.get("recall", 0),
             report["span"]["f1"], report["ate"]["f1"], report["ote"]["f1"],
             report["aope"]["f1"]))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    cfg = build_parser().parse_args()
    set_seed(cfg.seed)
    device = get_device()
    print("[main] device=%s task=%s dep_feats=%s data_base=%s"
          % (device, cfg.task, parse_dep_feats(cfg.dep_feats), cfg.data_base))

    train_data, dev_data, test_data = load_splits(cfg.data_base)
    if cfg.limit:
        train_data = train_data[:cfg.limit]
        dev_data = dev_data[:cfg.limit]
        test_data = test_data[:cfg.limit]
    rel2id, id2rel, cat2id, id2cat, sent2id, id2sent = build_label_maps(train_data, "pair_grid")
    tr_ds, va_ds, te_ds, tok = load_data(cfg, cat2id, sent2id,
                                         train_data, dev_data, test_data)
    model, tok = build_model(cfg, cat2id, id2cat, sent2id, id2sent)
    model.to(device)
    print("[main] train/dev/test =", len(train_data), len(dev_data), len(test_data))

    t0 = time.time()
    if cfg.mode == "train":
        run_train(cfg, model, tok, tr_ds, va_ds, te_ds, device)
    elif cfg.mode == "eval":
        run_eval(cfg, model, te_ds, device)
    else:
        run_predict(cfg, model, te_ds, device)
    print("[main] done in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()