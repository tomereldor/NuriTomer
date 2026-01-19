# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the **CTO (Clinical Trial Outcome)** benchmark - a large-scale clinical trial outcome dataset with weakly supervised labels. The main codebase is in the `CTOD/` directory, which implements multiple labeling pipelines that combine various data sources to predict clinical trial outcomes.

Paper: "Automatically Labeling $200B Life-Saving Datasets: A Large Clinical Trial Outcome Benchmark" (arXiv:2406.10292)

## Setup

```bash
pip install -r requirements.txt
```

Key dependencies: torch, transformers, sentence_transformers, snorkel, scikit-learn, llama_index, openai

For HINT model, create a conda environment:
```bash
conda create -n predict_drug_clinical_trial python==3.7
conda activate predict_drug_clinical_trial
conda install -c rdkit rdkit
pip install tqdm scikit-learn torch seaborn icd10-cm
```

## Architecture

The pipeline generates weak labels from four independent sources, then combines them:

### 1. LLM Predictions on PubMed (`CTOD/llm_prediction_on_pubmed/`)
Extracts trial outcomes from linked PubMed abstracts using GPT-3.5.
```bash
cd llm_prediction_on_pubmed
python extract_pubmed_abstracts.py --data_path <CTTI_PATH> --save_path <SAVE_PATH>
python retrieve_top2_abstracts.py --data_path <CTTI_PATH> --save_path <SAVE_PATH>
python get_llm_predictions.py --save_path <SAVE_PATH>
python clean_and_extract_final_outcomes.py --save_path <SAVE_PATH>
```

### 2. Clinical Trial Linkage (`CTOD/clinical_trial_linkage/`)
Links trials across phases and matches Phase 3 trials with FDA approvals.
```bash
cd clinical_trial_linkage
python extract_trial_info.py --data_path <CTTI_PATH>
python get_embedding_for_trial_linkage.py --root_folder <SAVE_PATH> --num_workers 2 --gpu_ids 0,1
python create_trial_linkage.py --root_folder <SAVE_PATH> --target_phase 'Phase 3'
python extract_outcome_from_trial_linkage.py --trial_linkage_path <PATH>
python match_fda_approvals.py --trial_linkage_path <PATH>
```

### 3. News Headlines (`CTOD/news_headlines/`)
Scrapes Google News for sponsor headlines and uses FinBERT sentiment analysis.
```bash
cd news_headlines
python get_news.py --mode=get_news              # Scraping (weeks)
python get_news.py --mode=process_news          # Sentiment embeddings
python get_news.py --mode=correspond_news_and_studies
```

### 4. Stock Prices (`CTOD/stock_price/`)
Calculates stock price SMA slopes around trial completion dates using yfinance.

### 5. Label Aggregation (`CTOD/labeling/`)
Combines weak labels using data programming (Snorkel) or supervised models (RF/LR/SVM).
```bash
python labeling/lfs.py --label_mode RF --CTTI_PATH <PATH> --HINT_PATH <PATH> ...
```
See `update_labels.sh` for full parameter examples.

### 6. HINT Model (`CTOD/TOP/HINT/`)
Hierarchical Interaction Network for trial outcome prediction:
```bash
python HINT/learn_phaseI.py
python HINT/learn_phaseII.py
python HINT/learn_phaseIII.py
```

### 7. Baselines (`CTOD/baselines/`)
```bash
python baselines.py          # SVM, XGBoost, MLP, RF, LR
python run_spot.py           # SPOT model
python biobert_trial_outcome.py  # BioBERT
```

## Data Sources

- **CTTI**: Clinical Trials Transformation Initiative data (pipe-delimited files from https://aact.ctti-clinicaltrials.org/download)
- **TOP**: Trial Outcome Prediction benchmark dataset (https://github.com/futianfan/clinical-trial-outcome-prediction)
- **FDA Orange Book**: Drug approval data
- **DrugBank**: Drug mapping database

## Main Entry Points

- `CTOD/pipeline.sh` - Full data processing pipeline
- `CTOD/update_labels.sh` - Label generation with multiple models
- Root notebooks for exploration: `getting_started_cto_vs_top.ipynb`, `exploring_company_examples.ipynb`
