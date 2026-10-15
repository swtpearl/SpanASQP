# ECoVi: A Unified Vietnamese E-commerce ABSA Dataset for Comprehensive Benchmarking

This repository includes the annotated data and code implementations of the approaches used in our paper *ECoVi: A Unified Vietnamese E-commerce ABSA Dataset for Comprehensive Benchmarking*.

## 📌 Introduction

We introduce ECoVi, a benchmark dataset for Vietnamese e-commerce Aspect-Based Sentiment Analysis, annotated with five core elements: spans, aspect terms, aspect categories, opinion terms, and sentiment polarities. With these annotations, the dataset supports various ABSA subtasks, scaling up to sentiment quadruplet extraction. Building on this resource, we also propose SpanASQP, a span-based architecture that incorporates syntactic dependency representations for the aspect sentiment quadruple prediction task.

## 📊 Data

The data is located in the folders `data/` with 2 versions:

- `data/seg/` – text segmented with VnCoreNLP (default for training SpanASQP).
- `data/unseg/` – raw, unsegmented text (contains label annotations without segmentation and dependency fields).

The structure of the data is as follows:

| Columns | Description |
|---|---|
| `id` | Unique record id (e.g. `train-000001`). |
| `text` | Raw review text. |
| `text_segmented` | Review text after word segmentation. |
| `tokens` | List of tokens split by whitespace. |
| `postag` | POS tag per token. |
| `head` | Dependency head index per token (1-based; `0` = root). |
| `deprel` | Dependency relation label per token. |
| `span` | BIO tags marking the anchor span of each quad. |
| `aspect` | BIO tags marking aspect terms inside spans. |
| `topic` | Aspect category label per token (`NULL` for tokens outside a span). |
| `sentiment` | Sentiment polarity per token (`NULL` for tokens outside a span). |
| `opinion` | BIO tags marking opinion terms inside spans. |

## 🚀 Quick start

### Setup
```bash
conda create -n spanasqp python=3.12
conda activate spanasqp
pip install -r requirements.txt
```

### Training

- Train on segmented data (default):


```bash
python main.py --mode train \
    --model_name vinai/phobert-base \
    --data_base data/seg \
    --task asqp \
    --out_dir runs/main \
    --max_pieces 256 \
    --max_words 128 \
    --max_span_len 12 \
    --batch_size 16 \
    --lr 3e-5 \
    --epochs 100 \
    --patience 10
```

- Eval from a checkpoint:

```bash
python main.py --mode eval \
    --model_name vinai/phobert-base \
    --data_base data/seg \
    --task asqp \
    --ckpt runs/main/best_model.pt \
    --max_pieces 256 \
    --max_words 128 \
    --max_span_len 12
```

- Run inference / prediction:

```bash
python main.py --mode predict \
    --model_name vinai/phobert-base \
    --data_base data/seg \
    --task asqp \
    --ckpt runs/main/best_model.pt \
    --max_pieces 256 \
    --max_words 128 \
    --max_span_len 12
```

## Citation
> Paper under review.
