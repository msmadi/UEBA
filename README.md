# UEBA
Multi-Axis Trust Modeling for Interpretable Account Hijacking Detection

# Hadith-Inspired Trust Modeling for UEBA

This repository contains the reference implementation and experimental
artifacts for the paper:

**“Multi-Axis Trust Modeling for Interpretable Account Hijacking Detection”**

The code implements a structured, interpretable feature framework for
detecting account hijacking and insider threats in user activity logs,
together with temporal extensions for sequence-aware detection.

---

## Overview

User and Entity Behavior Analytics (UEBA) systems often rely on opaque models
and low-level count features. This project introduces a **Hadith-inspired
multi-axis trust model**, translating classical trust criteria into
behavioral features grouped along five axes:

- **ʿAdālah (Integrity / Long-Term Stability)**
- **Ḍabṭ (Precision / Hygiene)**
- **Isnād (Contextual / Network Continuity)**
- **Reputation (Jarḥ wa-Taʿdīl)**
- **Anomaly Evidence (Shudhūdh / ʿIllah)**

We further extend this representation with **temporal sequence features**
that capture short-horizon behavioral drift across consecutive windows.

The framework is evaluated on:
- **CLUE-LDS** (cloud activity logs with injected hijack scenarios)
- **CERT Insider Threat Dataset r6.2** (realistic insider threat benchmark)

---


---

## Datasets

### CLUE-LDS
CLUE-LDS is publicly available but subject to redistribution restrictions.
This repository **does not include raw CLUE-LDS logs**.

To reproduce CLUE-LDS experiments:
1. Obtain the official CLUE-LDS dataset.
2. Place the CSV logs in `datasets/clue/data/`.
3. Run:
   ```bash
   python evaluate.py --csv clue_lds_logs.csv --max-events 500000 --n-hijacks 30 --save-plots

### CERT Insider Threat Dataset r6.2
  provider: Carnegie Mellon University CERT Division
  license: Research-only; redistribution prohibited
  raw_data_included: false
  justification: >
    Raw CERT logs cannot be redistributed due to licensing restrictions.
    The repository provides full preprocessing, feature extraction, and
    evaluation code so reviewers can reproduce results after obtaining
    the dataset from the official source.

#### data_summary:
  configurations:
    - name: CERT-500
      users: 500
      malicious_users: 29
      total_events: 185178
      windows: 6666
      positive_rate: 3.62%
      test_windows: 2000
      test_positives: 72
    - name: CERT-4000
      users: 4000
      malicious_users: 29
      total_events: 700335
      windows: 22622
      positive_rate: 1.07%
      test_windows: 6787
      test_positives: 72

#### features:
  hadith_axes:
    adalah:
      description: Long-term behavioral stability and consistency
      features: 5
    dabt:
      description: Precision, hygiene, and authentication regularity
      features: 7
    isnad:
      description: Contextual and session continuity
      features: 6
    reputation:
      description: Historical trust accumulation
      features: 4
    anomaly:
      description: Distributional deviation from historical behavior
      features: 4
  temporal_features:
    description: >
      Short-horizon sequence features capturing drift, transition entropy,
      and behavioral acceleration across consecutive windows.
    features: 16
  combined_features:
    total_dimensions: 42

#### models:
  supervised:
    - RandomForest
    - GradientBoosting
    - LSTM
  baselines:
    - RawCounts
    

#### experiments:
  scripts:
   ```bash
      python evaluate.py --csv data.csv

  
#### evaluation_metrics:
      - ROC-AUC
      - PR-AUC
      - F1
      - Precision
      - Recall

#### key_results:
  cert_500:
    best_model: Combined (Hadith + Temporal) + RandomForest
    roc_auc: 0.8439
    pr_auc: 0.4989
    f1: 0.5246
    improvement_over_hadith_only:
      roc_auc: "+8.80%"
      pr_auc: "+43.38%"
  cert_4000:
    best_model: Combined (Hadith + Temporal) + RandomForest
    roc_auc: 0.7151
    pr_auc: 0.2638
    f1: 0.3529
    improvement_over_hadith_only:
      roc_auc: "+14.12%"
      pr_auc: "+266.48%"

#### ablation_analysis:
  finding: >
    Temporal features become increasingly critical as dataset size grows
    and low-level signals weaken. Removing temporal dabt features causes
    the largest performance drop.
  temporal_importance:
    cert_500: 38.1%
    cert_4000: 43.6%

#### reproducibility_instructions:
  steps:
    - Obtain CERT r6.2 dataset from official source
    - Place CSV files in datasets/cert/data/
    - Run feature extraction scripts
    - Execute experiment scripts for desired configuration
  expected_runtime:
    cert_500: "~15 minutes on CPU"
    cert_4000: "~1–2 hours on CPU"

limitations:
  - Extremely sparse malicious behavior by design
  - Strong class imbalance (1–4%)
  - No redistribution of raw logs
  - Results depend on windowing and scenario labeling assumptions


#### contact:
  review_mode: Double-blind
  notes: >
    All artifacts listed here are sufficient to reproduce the CERT results
    reported in the paper after obtaining the dataset under the official license.


