# IMDb Movie Review Sentiment Analysis using DistilBERT

## Project Overview

This project builds a binary sentiment classification system that predicts whether an IMDb movie review expresses **Positive** or **Negative** sentiment.

The project combines data engineering and deep learning:

- **Databricks** for data preparation using Medallion Architecture.
- **PyTorch and Hugging Face Transformers** for DistilBERT fine-tuning.
- **Streamlit** for interactive model deployment.

The fine-tuned model achieved **91.69% accuracy** on the test dataset.

## Project Architecture

```text
IMDb Dataset (Hugging Face)
          |
          v
Databricks - Medallion Architecture
          |
          v
     Bronze Layer
          |
          v
     Silver Layer
  (Text Cleaning)
          |
          v
      Gold Layer
 (Deduplication and Validation)
          |
          v
      CSV Export
          |
          v
     Google Colab
          |
          v
   DistilBERT Tokenization
          |
          v
  Model Fine-tuning (PyTorch)
          |
          v
    Model Evaluation
          |
          v
  Saved DistilBERT Model
          |
          v
   Streamlit Application
          |
          v
 Sentiment Prediction
```

## Dataset

**Source:** [IMDb Movie Reviews – Stanford NLP](https://huggingface.co/datasets/stanfordnlp/imdb)

The dataset contains movie reviews labelled as:

| Label | Sentiment |
|---|---|
| 0 | Negative |
| 1 | Positive |

### Dataset Preparation

| Dataset | Records |
|---|---:|
| Original training data | 25,000 |
| Original test data | 25,000 |
| Gold training data after deduplication | 24,877 |
| Gold test data | 25,000 |
| Removed overlapping training records | 123 |

The original test dataset was retained for final evaluation.

## Data Engineering – Medallion Architecture

The data preparation pipeline was implemented in Databricks.

### Bronze Layer
- Ingested original IMDb training and test datasets.
- Preserved raw review text and sentiment labels.
- Added ingestion timestamps.

### Silver Layer
- Replaced HTML line-break tags.
- Trimmed unnecessary whitespace.
- Created cleaned text columns.
- Generated review-length features.
- Validated missing values and labels.

### Gold Layer
- Identified exact duplicate reviews across training and test datasets.
- Removed overlapping reviews from the training dataset.
- Preserved the original test dataset.
- Validated that no exact text overlap remained.

## Deep
