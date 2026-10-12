import torch
import torch.nn as nn
import torch.nn.functional as F


def class_weights(sp_types, num_classes, device):
    """Inverse-frequency (clamp [0.2, 5]) cho head_type."""
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for t in sp_types:
        if t.numel() > 0:
            counts += torch.bincount(t, minlength=num_classes).float().cpu()
    total = counts.sum().clamp(min=1.0)
    w = total / (num_classes * counts.clamp(min=1.0))
    return w.clamp(0.2, 5.0).to(device)


def span_loss(model, out, batch, w_span=0.5, w_bio=1.0, w_type=1.0,
                  w_cat=1.0, w_sent=1.0, w_boundary=15.0, w_bg=0.1,
                  return_parts=False):
    """Loss: type (weighted CE) + cat/sent (CE masked gold spans)
    + weighted boundary BCE + BIO CE."""
    ce = nn.CrossEntropyLoss()
    tw = class_weights(batch["sp_type"], 5, model.node_proj.weight.device)
    ce_w = nn.CrossEntropyLoss(weight=tw)
    type_l, cat_l, sent_l = [], [], []
    for bi in range(len(batch["cand_spans"])):
        ht = out["head_type"][bi]
        Mi = ht.size(0)
        if Mi == 0:
            continue
        sp_t = batch["sp_type"][bi][:Mi]
        sp_m = batch["sp_mask"][bi][:Mi]
        sp_c = batch["sp_cat"][bi][:Mi]
        sp_s = batch["sp_sent"][bi][:Mi]
        type_l.append(ce_w(ht, sp_t.to(ht.device)))
        ids = torch.nonzero(sp_m > 0).squeeze(1)
        if ids.numel() > 0:
            sent_l.append(ce(out["head_sent"][bi][ids], sp_s[ids].to(ht.device)))
            if model.use_cat:   # ASTE (use_cat=False): head_cat la dummy 1-lop -> bo cat loss
                cat_l.append(ce(out["head_cat"][bi][ids], sp_c[ids].to(ht.device)))

    sw = batch["word_mask"]
    denom = sw.sum().clamp(min=1)
    is_b = (batch["start_ind"] + batch["end_ind"]).clamp(max=1).float()
    w_pos = w_bg + (w_boundary - w_bg) * is_b
    sp = (F.binary_cross_entropy(out["p_start"], batch["start_ind"].float(),
                                 reduction="none") * sw * w_pos).sum() / denom
    ep = (F.binary_cross_entropy(out["p_end"], batch["end_ind"].float(),
                                 reduction="none") * sw * w_pos).sum() / denom
    span_l = sp + ep
    ba = F.cross_entropy(out["bio_a"].transpose(1, 2), batch["bio_aspect"],
                         reduction="none") * sw
    bo = F.cross_entropy(out["bio_o"].transpose(1, 2), batch["bio_opinion"],
                         reduction="none") * sw
    bio_l = (ba.sum() + bo.sum()) / denom

    tl = torch.stack(type_l).mean() if type_l else torch.tensor(0.0, device=model.node_proj.weight.device)
    cl = torch.stack(cat_l).mean() if cat_l else torch.tensor(0.0, device=model.node_proj.weight.device)
    sl = torch.stack(sent_l).mean() if sent_l else torch.tensor(0.0, device=model.node_proj.weight.device)
    total = (w_type * tl + w_cat * cl + w_sent * sl + w_span * span_l +
             w_bio * bio_l)
    if return_parts:
        parts = {"type": tl, "cat": cl, "sent": sl, "span": span_l, "bio": bio_l}
        return total, {k: float(v.detach()) for k, v in parts.items()}
    return total