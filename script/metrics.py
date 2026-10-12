def block_quad_key(blk, cat2id, sent2id, use_cat=True):
    if use_cat:
        return (blk["category"], blk["sentiment"],
                frozenset(map(tuple, blk["aspects"])), frozenset(map(tuple, blk["opinions"])))
    return (blk["sentiment"],
            frozenset(map(tuple, blk["aspects"])), frozenset(map(tuple, blk["opinions"])))


def quad_f1(pred_blocks, gold_blocks, cat2id, sent2id, use_cat=True):
    tp = fp = fn = 0
    for pblks, gblks in zip(pred_blocks, gold_blocks):
        gset = [block_quad_key(g, cat2id, sent2id, use_cat) for g in gblks]
        used = [False] * len(gset)
        for pb in pblks:
            k = block_quad_key(pb, cat2id, sent2id, use_cat)
            matched = False
            for i, gk in enumerate(gset):
                if not used[i] and gk == k:
                    used[i] = True; matched = True; break
            tp += matched
            if not matched:
                fp += 1
        fn += sum(1 for u in used if not u)
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    f = 2 * p * r / (p + r) if p + r > 0 else 0
    return f, p, r, tp, fp, fn


def label_f1(gold_items, pred_items):
    tp = fp = fn = 0
    for g, p in zip(gold_items, pred_items):
        gs, ps = set(g), set(p)
        tp += len(gs & ps); fp += len(ps - gs); fn += len(gs - ps)
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    f = 2 * p * r / (p + r) if p + r > 0 else 0
    return f, p, r, tp, fp, fn


def span_f1(pred_spans_all, gold_blocks_all):
    """Span-layer exact F1 (start/end) vs gold block spans."""
    tp = fp = fn = 0
    for pspans, gblks in zip(pred_spans_all, gold_blocks_all):
        gset = [g["span"] for g in gblks]
        used = [False] * len(gset)
        for ps in pspans:
            k = tuple(ps)
            m = False
            for i, gk in enumerate(gset):
                if not used[i] and gk == k:
                    used[i] = True; m = True; break
            tp += m
            if not m:
                fp += 1
        fn += sum(1 for u in used if not u)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def compute_metrics(pred_all, gold_all, cat2id, sent2id, use_cat=True):
    """Block-level: Span / Category / Sentiment / ATE / OTE / Quad(4-tuple)."""
    span_g, span_p, cat_g, cat_p, sent_g, sent_p, ate_g, ate_p, ote_g, ote_p = (
        [], [], [], [], [], [], [], [], [], [])
    for pblks, gblks in zip(pred_all, gold_all):
        span_g.append([g["span"] for g in gblks]); span_p.append([p["span"] for p in pblks])
        cat_g.append([(g["span"][0], g["span"][1], g["category"]) for g in gblks])
        cat_p.append([(p["span"][0], p["span"][1], p["category"]) for p in pblks])
        sent_g.append([(g["span"][0], g["span"][1], g["sentiment"]) for g in gblks])
        sent_p.append([(p["span"][0], p["span"][1], p["sentiment"]) for p in pblks])
        ate_g.append([a for g in gblks for a in g["aspects"]])
        ate_p.append([a for p in pblks for a in p["aspects"]])
        ote_g.append([o for g in gblks for o in g["opinions"]])
        ote_p.append([o for p in pblks for o in p["opinions"]])
    res = {}
    for name, g, p in [("Span", span_g, span_p), ("Category", cat_g, cat_p),
                       ("Sentiment", sent_g, sent_p), ("ATE", ate_g, ate_p), ("OTE", ote_g, ote_p)]:
        f1, pr, re, tp, fp, fn = label_f1(g, p)
        res[name] = {"precision": pr, "recall": re, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    f1q, pq, rq, tpq, fpq, fnq = quad_f1(pred_all, gold_all, cat2id, sent2id, use_cat)
    res["Quad(4-tuple)"] = {"precision": pq, "recall": rq, "f1": f1q, "tp": tpq, "fp": fpq, "fn": fnq}
    return res


# ---------------------------------------------------------------------------
# METRIC CHINH 
# ---------------------------------------------------------------------------
_NULL = {"NULL"}


def _is_null(x):
    return x is None or str(x).strip() in _NULL


def _block_to_quads(blk, tokens, task="asqp"):
    """1 block -> list quad/triplet dang STRING (category/sentiment da la chuoi)."""
    a = [" ".join(tokens[s:e + 1]) for (s, e) in blk["aspects"]] or ["NULL"]
    o = [" ".join(tokens[s:e + 1]) for (s, e) in blk["opinions"]] or ["NULL"]
    cat = blk["category"]; sent = blk["sentiment"]
    if task == "aste":
        return [[x, y, sent] for x in a for y in o]           # triplet [a,o,s]
    return [[x, cat, sent, y] for x in a for y in o]           # quad [a,c,s,o]


def _tclass(q):
    if len(q) == 3:
        a, o, sent = q
    else:
        a, cat, sent, o = q
    return "EE" if (not _is_null(a)) and (not _is_null(o)) else \
           "IE" if _is_null(a) and (not _is_null(o)) else \
           "EI" if (not _is_null(a)) and _is_null(o) else "II"


def _qkey(q):
    return tuple((None if _is_null(v) else str(v)) for v in q)


def _micro_f1(gt, pt):
    tp = fp = fn = 0
    for g, p in zip(gt, pt):
        gs = {_qkey(x) for x in g}; ps = {_qkey(x) for x in p}
        tp += len(gs & ps); fp += len(ps - gs); fn += len(gs - ps)
    P = tp / (tp + fp) if tp + fp else 0
    R = tp / (tp + fn) if tp + fn else 0
    F = 2 * P * R / (P + R) if P + R else 0
    return {"precision": P, "recall": R, "f1": F, "tp": tp, "fp": fp, "fn": fn}


def quad_metrics(gold, pred, task="asqp"):
    """Per-quad metric (METRIC CHINH). gold/pred = list (tokens, blocks) voi
    category/sentiment la STRING. task="asqp" -> quad [a,c,s,o];
    task="aste" -> triplet [a,o,s]. Tra (overall, subgroup {EE,IE,EI,II,implicit})."""
    g = [[q for b in gl for q in _block_to_quads(b, tok_, task)] for tok_, gl in gold]
    p = [[q for b in pl for q in _block_to_quads(b, tok_, task)] for tok_, pl in pred]
    overall = _micro_f1(g, p)
    sub = {}
    for grp in ["EE", "IE", "EI", "II"]:
        gg = [[q for q in gq if _tclass(q) == grp] for gq in g]
        pp = [[q for q in pq if _tclass(q) == grp] for pq in p]
        m = _micro_f1(gg, pp)
        m["n_gold"] = sum(len(x) for x in gg); m["n_pred"] = sum(len(x) for x in pp)
        sub[grp] = m
    gg = [[q for q in gq if _tclass(q) != "EE"] for gq in g]
    pp = [[q for q in pq if _tclass(q) != "EE"] for pq in p]
    m = _micro_f1(gg, pp)
    m["n_gold"] = sum(len(x) for x in gg); m["n_pred"] = sum(len(x) for x in pp)
    sub["implicit"] = m
    return overall, sub


def aope_f1(gold, pred, task="asqp"):
    """Aspect-opinion PAIR extraction F1: pairs (a, o) tu moi block (cross-product),
    doc lap sentiment/category. gold/pred = list (tokens, blocks) string."""
    def pairs(pairs_in):
        out = []
        for tok_, blocks in pairs_in:
            row = set()
            for b in blocks:
                a = [" ".join(tok_[s:e + 1]) for (s, e) in b["aspects"]]
                o = [" ".join(tok_[s:e + 1]) for (s, e) in b["opinions"]]
                for x in a:
                    for y in o:
                        row.add((x, y))
            out.append(row)
        return out
    return label_f1(pairs(gold), pairs(pred))