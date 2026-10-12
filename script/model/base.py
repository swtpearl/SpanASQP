import torch
import torch.nn as nn
from transformers import AutoModel

from ..util import DEP_ROLE_G

_DEP_IDX = {"edge": 0, "dir": 1, "path": 2, "lca": 3}


class BaseSpanClsModel(nn.Module):
    def __init__(self, model_name, cat2id, id2cat, sent2id, id2sent,
                 node_hid=384, dropout=0.33, use_bert_last_4=True,
                 max_span_len=12, use_cat=True, dep_feats=()):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.bert_dim = self.encoder.config.hidden_size
        self.cat2id = cat2id
        self.id2cat = id2cat
        self.sent2id = sent2id
        self.id2sent = id2sent
        self.use_bert_last_4 = use_bert_last_4
        self.node_hid = node_hid
        self.max_span_len = max_span_len
        self.use_cat = use_cat
        self.dep_feats = tuple(dep_feats)
        self.cat_dim = len(cat2id) if use_cat else 1   # ASTE: dummy 1-lop (triplet)

        self.node_proj = nn.Linear(self.bert_dim, node_hid)
        self.dropout = nn.Dropout(dropout)

        # ---- span start/end layer ----
        self.span_start = nn.Linear(node_hid, 1)
        self.span_end = nn.Linear(node_hid, 1)
        # ---- span representation [start; end; mean; pool_aspect; pool_opinion] ----
        _nf = sum(1 for f in dep_feats if f in _DEP_IDX)
        if "role" in dep_feats:
            _nf += 2 * DEP_ROLE_G
        span_in = node_hid * 5 + _nf
        self.span_mlp = nn.Linear(span_in, node_hid)

        # ---- fine BIO heads (aspect/opinion, 3-class each) ----
        self.bio_a_head = nn.Linear(node_hid, 3)
        self.bio_o_head = nn.Linear(node_hid, 3)

        # ---- per-span classifier heads ----
        self.head_type = nn.Linear(node_hid, 5)   # none/EE/IE/EI/II
        if use_cat:
            self.head_cat = nn.Linear(node_hid, len(cat2id))
        self.head_sent = nn.Linear(node_hid, len(sent2id))

    # ---- per-span heads ----
    def _span_heads(self, batch, bi, W, bio_a, bio_o, S, E, m, mean, st, en, Mi, N):
        # mention-aware span pooling: weight = B-tag + I-tag prob
        ba = torch.softmax(bio_a[bi][:N], dim=-1)          # [N,3]
        bo_ = torch.softmax(bio_o[bi][:N], dim=-1)
        wa = ba[:, 1] + ba[:, 2]                        # aspect weight
        wo = bo_[:, 1] + bo_[:, 2]                      # opinion weight
        num_a = (m * wa.unsqueeze(0)).sum(1)
        num_o = (m * wo.unsqueeze(0)).sum(1)
        pool_a = ((m.unsqueeze(-1) * wa.unsqueeze(0).unsqueeze(-1) * W[bi].unsqueeze(0)).sum(1)
                  / num_a.clamp(min=1e-6).unsqueeze(-1))
        pool_o = ((m.unsqueeze(-1) * wo.unsqueeze(0).unsqueeze(-1) * W[bi].unsqueeze(0)).sum(1)
                  / num_o.clamp(min=1e-6).unsqueeze(-1))
        feat = [st, en, mean, pool_a, pool_o]
        if self.dep_feats:
            dep_mat = batch["dep_mat"][bi][:N, :N]            # [N,N,4]
            mw_a = (m * wa.unsqueeze(0))                     # [Mi,N]
            mw_o = (m * wo.unsqueeze(0))
            da = mw_a / mw_a.sum(1).clamp(min=1e-6).unsqueeze(1)
            do_ = mw_o / mw_o.sum(1).clamp(min=1e-6).unsqueeze(1)
            for f in self.dep_feats:
                if f in _DEP_IDX:
                    D = dep_mat[:, :, _DEP_IDX[f]]           # [N,N]
                    feat.append(((da @ D) * do_).sum(-1).unsqueeze(-1))   # [Mi,1]
                elif f == "role":
                    role = batch["dep_role"][bi][:N]          # [N,G]
                    feat.append(da @ role)                    # [Mi,G]
                    feat.append(do_ @ role)
        v = torch.tanh(self.span_mlp(torch.cat(feat, dim=-1)))
        hc = self.head_cat(v) if self.use_cat else torch.zeros(Mi, 1, device=v.device)
        return self.head_type(v), hc, self.head_sent(v)

    def _get_bert_emb(self, input_ids, attention_mask):
        outs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.use_bert_last_4 and outs.hidden_states is not None:
            return torch.stack(outs.hidden_states[-4:], dim=-1).mean(-1)
        return outs.last_hidden_state

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pieces2word = batch["pieces2word"]
        node_n = batch["node_n"]
        N = batch["n_words_word"]

        bert_embs = self._get_bert_emb(input_ids, attention_mask)
        B, L, D = bert_embs.size()
        min_val = torch.min(bert_embs).item()
        _e = bert_embs.unsqueeze(1).expand(-1, node_n, -1, -1)
        _e = torch.masked_fill(_e, pieces2word.eq(0).unsqueeze(-1), min_val)
        word_reps, _ = torch.max(_e, dim=2)
        word_reps = self.dropout(self.node_proj(word_reps))
        node_mask = pieces2word.sum(-1).gt(0).float()
        H_rep = word_reps * node_mask.unsqueeze(-1)   # mask node padding
        W = H_rep[:, 1:1 + N]                             # [B,N,D] word nodes

        # ---- span start/end probabilities ----
        p_start = torch.sigmoid(self.span_start(W)).squeeze(-1)   # [B,N]
        p_end = torch.sigmoid(self.span_end(W)).squeeze(-1)

        # ---- BIO heads ----
        bio_a = self.bio_a_head(W)
        bio_o = self.bio_o_head(W)

        # ---- span representations + per-span heads (variable M per sample) ----
        spans = batch["cand_spans"]
        dev = input_ids.device
        head_type, head_cat, head_sent = [], [], []
        for bi in range(B):
            sp = spans[bi]
            if not sp:
                head_type.append(torch.zeros(0, 5, device=dev))
                head_cat.append(torch.zeros(0, self.cat_dim, device=dev))
                head_sent.append(torch.zeros(0, len(self.sent2id), device=dev))
                continue
            S = torch.tensor([s for (s, e) in sp], device=dev)
            E = torch.tensor([e for (s, e) in sp], device=dev)
            Mi = len(sp)
            ar = torch.arange(N, device=dev)
            m = ((ar.unsqueeze(0) >= S.unsqueeze(1)) &
                 (ar.unsqueeze(0) <= E.unsqueeze(1))).float()      # [Mi,N]
            mean = (m.unsqueeze(-1) * W[bi]).sum(1) / m.sum(1).clamp(min=1).unsqueeze(-1)
            st = W[bi][S]
            en = W[bi][E]
            ht, hc, hs = self._span_heads(batch, bi, W, bio_a, bio_o, S, E, m,
                                          mean, st, en, Mi, N)
            head_type.append(ht)
            head_cat.append(hc)
            head_sent.append(hs)

        out = {"p_start": p_start, "p_end": p_end,
               "bio_a": bio_a, "bio_o": bio_o,
               "head_type": head_type, "head_cat": head_cat, "head_sent": head_sent}
        return out


SpanClsModel = BaseSpanClsModel