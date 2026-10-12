# ============================================================================
# script/model - model chinh deprel: BaseSpanClsModel (alias SpanClsModel).
#   base.py : span layer + BIO + mention-aware span pooling + span-level
#             dependency relation features (dep_feats) + per-span type/cat/sent.
#   Da bo: e25d (POS-BIO), nospan (NoSpanModel), graph (AGDC) - 2026-09-19.
# ============================================================================
from .base import BaseSpanClsModel, SpanClsModel   # noqa: F401

__all__ = ["BaseSpanClsModel", "SpanClsModel"]