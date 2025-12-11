# Evolution of Fairness in Large Langauge Models

## Overview

This framework evaluates how different language models perform binary classification tasks across 6 real-world datasets, measuring both accuracy and fairness metrics across sensitive demographic groups (race, gender, age).

**Supported Models:**
- **LLaMA family:** 8B, 70B, 405B (via Together AI)
- **Mistral:** 7B, Small-24B (via Together AI)
- **Gemini:** 2.5 Flash, 2.5 Pro (via Google)
- **GPT:** 4o-mini, 4o (via OpenAI)

**Datasets & Domains:**
1. **German Credit** (Finance) – Credit risk assessment  
2. **Adult Income** (Socioeconomic) – Income prediction (>\$50K)  
3. **COMPAS** (Criminal Justice) – Recidivism risk  
4. **Bank Marketing** (Finance) – Term deposit subscription  
5. **Folktables/ACS** (Socioeconomic) – Census income prediction  
6. **Diabetes Readmission** (Healthcare) – 30-day hospital readmission  

---

## Key Concepts

### Experiment Structure

Each experiment consists of:

1. **Data Loading**  
   Test split extracted from a real-world dataset (configurable test size).

2. **Prompting**  
   Standardized, domain-specific prompts instruct the model to output a JSON-like prediction with optional reasoning. Prompts are aligned across models for each dataset to support fair comparison.

3. **Classification**  
   The model outputs a predicted class label (e.g., `good`/`bad`, `>50K`/`<=50K`) and (optionally) reasoning. Outputs are normalized so that label variants such as `"Good"`, `"GOOD"`, `"good credit"` all map to the canonical class.

4. **Fairness Analysis**  
   The framework computes group-wise metrics and fairness statistics using protected attributes such as race, gender, or age group.

---

### Output Schema

Every prediction row follows a standardized schema:

```csv
record_id, ground_truth, prediction, sensitive_feature,
experiment_id, reasoning, api_success, attempts_made,
timestamp, correct
````

* `record_id` – Unique identifier for the sample
* `ground_truth` – True label from the dataset
* `prediction` – Normalized model prediction
* `sensitive_feature` – Group membership (e.g., race/sex/age group)
* `experiment_id` – Unique ID for this (dataset, model, config) run
* `reasoning` – Free-text explanation (if the model supplies one)
* `api_success` – Boolean, whether the API call succeeded
* `attempts_made` – How many retries were needed
* `timestamp` – ISO timestamp when prediction was generated
* `correct` – Boolean, prediction matches `ground_truth`

---

### Fairness Metrics

Core metrics computed per experiment:

* **Accuracy:** Overall accuracy and max accuracy gap between demographic groups
* **Demographic parity:** Differences in positive prediction rates between groups
* **Equalized odds:** Differences in true positive and false positive rates across groups
* **Per-group performance:** Accuracy, precision, recall, F1 for each protected group

Fairness metrics are reported both as **differences** (max gap between groups) and **ratios** (min/max across groups) to match common fairness guidelines (e.g., 80% rule).

---

## Installation

```bash
# Clone and enter the repository
git clone <your-repo>
cd fairness_experiments

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API keys in .env file
echo "TOGETHER_API_KEY=your_key_here" >> .env
echo "GEMINI_API_KEY=your_key_here"   >> .env
echo "OPENAI_API_KEY=your_key_here"   >> .env
```

> You only need keys for the providers you actually plan to run.

---

## Core Files

High-level description of the main modules:

* `config.py`

  * Central configuration file.
  * Registers datasets, model families, provider settings, and default hyperparameters (temperature, max tokens, test split, etc.).

* `data_loaders.py`

  * Implements dataset-specific loaders (German Credit, Adult Income, COMPAS, Bank Marketing, Folktables/ACS, Diabetes).
  * Returns standardized objects with `X_test`, `y_test`, `sf_test` (sensitive features), IDs, and a decoder for prompt generation.

* `prompts.py`

  * Builds domain-specific prompts for each dataset.
  * Incorporates dataset description, feature values, and fairness-aware instructions.

* `api_clients.py`

  * Wraps Together, Gemini, and OpenAI APIs with a unified interface.
  * Handles exponential backoff, retry logic, and response parsing/normalization.

* `run_experiment.py`

  * Core experiment engine for a single (dataset, model, provider) configuration.
  * Iterates over test samples, calls the model, logs predictions, writes checkpoints, and triggers fairness analysis.

* `run_mistral_experiments.py`, `run_gemini_experiments.py`, `run_gpt_experiments.py`, `run_sequential_by_dataset.py`, `run_all_experiments.py`

  * High-level runners that combine `config.py`, `data_loaders.py`, and `run_experiment.py` to execute multiple experiments in sequence.

* `fairness_analysis.py`

  * Consumes prediction CSVs and computes overall metrics, group-wise metrics, demographic parity, and equalized odds.
  * Writes per-group analysis CSVs and experiment-level fairness summaries as JSON.

* `cli.py`

  * Command-line interface for running experiments and listing/inspecting configurations without editing Python files.

---

## Quick Start

### 1. Run a Single Experiment (Direct)

This uses the default dataset/model configured inside `run_experiment.py` (or via `config.py`):

```bash
# Runs a single (dataset, model) experiment as defined in run_experiment.py
python run_experiment.py
```

Use this for quick sanity checks or when modifying core logic.

---

### 2. Run Full Model Families

#### LLaMA Models (Together AI)

```bash
python run_sequential_by_dataset.py
```

* Runs all 6 datasets × 3 LLaMA models (8B, 70B, 405B) sequentially.
* Good entry point for “one family vs datasets” analysis.

#### Mistral Models (Together AI)

```bash
python run_mistral_experiments.py
```

* Runs all 6 datasets × 2 Mistral models (7B, Small-24B).

#### Gemini Models (Google)

```bash
python run_gemini_experiments.py
```

* Runs all 6 datasets × 2 Gemini models (2.5 Flash, 2.5 Pro).

#### GPT Models (OpenAI)

```bash
python run_gpt_experiments.py
```

* Runs all 6 datasets × 2 GPT models (4o-mini, 4o).

---

### 3. Single Dataset Testing (Family-Specific Scripts)

Most family-specific runners accept an optional dataset argument:

```bash
# Run Mistral models only on German Credit
python run_mistral_experiments.py german_credit

# Run Gemini models only on COMPAS
python run_gemini_experiments.py compas
```

If no dataset is supplied, the script usually iterates over all registered datasets.

---

## Command-Line Interface (`cli.py`)

The CLI provides a more flexible and explicit way to run experiments without modifying code.

Common patterns:

### List Available Datasets

```bash
python cli.py --list-datasets
```

Expected output (example):

```text
Available datasets:
- german_credit
- adult_income
- compas
- bank_marketing
- folktables_acs
- diabetes_readmission
```

---

### Run Specific Dataset and Models

Run a single dataset across multiple model sizes:

```bash
# Run German Credit on LLaMA 8B and 70B
python cli.py --dataset german_credit --models 8b 70b
```

Typical behavior:

* Resolves `german_credit` via `config.py`.
* Maps `8b` and `70b` to concrete model names (e.g., LLaMA-3.1-8B, LLaMA-3.1-70B via Together).
* Runs all requested experiments sequentially with checkpoints and summaries.

---

### Run All Datasets for a Single Model

```bash
# Run all datasets on LLaMA 8B
python cli.py --dataset all --models 8b
```

This is useful for “fixed model, multi-domain fairness” analysis.

---

### Compare Existing Experiments

If you have stored experiment IDs (from filenames or logs), you can compare them:

```bash
python cli.py --compare-exps exp_id_1 exp_id_2 exp_id_3
```

Typical behavior:

* Loads `*_fairness.json` for each experiment ID.
* Produces a model comparison CSV summarizing accuracy and fairness metrics per experiment.

---

### Optional Parallel Execution

If enabled in your CLI implementation:

```bash
# Run German Credit on multiple models with simple parallelism
python cli.py --dataset german_credit --models 8b 70b 405b --parallel --workers 2
```

> Note: Parallel runs increase the risk of hitting API rate limits. For large experiments, sequential execution is safer.

---

## Execution Strategy

All execution scripts (including `cli.py`) follow the same conceptual pattern:

1. **Select dataset(s)** (via `config.py` and CLI arguments)
2. **Select model(s)** (LLaMA/Mistral/Gemini/GPT)
3. **For each (dataset, model)**:

   * Build experiment configuration
   * Load data and create prompts
   * Call the API with backoff & retries
   * Save predictions + checkpoint
   * Run fairness analysis
4. **Aggregate results** into per-experiment summaries and optional cross-model comparison tables

**Sequential execution** is the default to avoid hitting rate limits and to ensure comparability (same environment, same conditions).

---

## Output Structure

All experiment artifacts are written under:

```text
fairness_experiments/
├── configs/              # Experiment configurations (JSON)
├── predictions/          # Detailed predictions with reasoning (CSV)
├── analysis/             # Per-group analysis tables (CSV)
├── summaries/            # Fairness metrics and model comparisons (JSON/CSV)
├── checkpoints/          # Auto-save progress during runs (CSV)
└── logs/                 # Execution logs for debugging & auditing
```

Additionally, some runners may generate aggregate files such as:

```text
fairness_experiments/
└── master_summary_*.json   # High-level aggregate report across experiments
```

---

## Checkpoint & Resume

The framework **automatically saves progress** every N predictions (typically every 10). Each experiment has a corresponding checkpoint file in:

```text
fairness_experiments/checkpoints/
```

To resume an interrupted run:

```bash
# Simply rerun the same command
python run_mistral_experiments.py german_credit
# or
python cli.py --dataset german_credit --models 8b 70b
```

The experiment engine will:

* Detect the existing checkpoint
* Skip already-processed records
* Continue from where it left off

To start completely fresh:

```bash
rm fairness_experiments/checkpoints/*
```

---

## Understanding Results

### Individual Experiment Output

Each experiment has a fairness summary file (e.g., `*_fairness.json`):

```json
{
  "overall_metrics": {
    "accuracy": 0.7234,
    "precision": 0.7105,
    "recall": 0.6892,
    "f1_score": 0.6997
  },
  "fairness_differences": {
    "accuracy": 0.1245
  },
  "group_summary": {
    "accuracy": {
      "best_group": "Male_White",
      "best_value": 0.7891,
      "worst_group": "Female_Black",
      "worst_value": 0.6646,
      "gap": 0.1245
    }
  },
  "binary_metrics": {
    "demographic_parity_difference": 0.0823,
    "equalized_odds_difference": 0.1156
  }
}
```

Key interpretations:

* `overall_metrics` – aggregate performance across all groups
* `fairness_differences` – how much performance differs across groups
* `group_summary` – which group is most/least favored
* `binary_metrics` – fairness measures (DPD, EOD), where smaller magnitude is better

---

### Cross-Model Comparison

After a dataset completes, a comparison CSV can be produced, e.g.:

```csv
model,               accuracy, accuracy_difference, demographic_parity_diff, equalized_odds_diff
Mistral-7B,          0.7234,   0.1245,              0.0823,                 0.1156
Mistral-Small-24B,   0.7456,   0.0987,              0.0634,                 0.0941
```

* Higher accuracy + lower `accuracy_difference` + lower `demographic_parity_diff` and `equalized_odds_diff` indicates a **stronger and fairer** model on that dataset.

---

## Customization

### Add New Dataset

1. **Implement loader** in `data_loaders.py`:

```python
def load_your_dataset(test_size=0.20, random_state=42):
    # Load your data
    # Preprocess, split into X_test, y_test, sf_test (sensitive features), ids_test
    # Define feature_columns and a decoder that turns a row into text
    return {
        "X_test": X_test,
        "y_test": y_test,
        "sf_test": sf_test,
        "ids_test": ids_test,
        "feature_columns": feature_columns,
        "decoder": decode_your_dataset_features
    }
```

2. **Register** in `config.py`:

```python
DATASET_REGISTRY["your_dataset"] = DatasetConfig(
    name="Your Dataset Name",
    dataset_type=DatasetType.FINANCE,  # or SOCIOECONOMIC, CRIMINAL_JUSTICE, HEALTHCARE, ...
    sensitive_features=["race", "gender"],
    target_column="outcome",
    positive_class="approved",
    negative_class="denied",
    task_description="predict loan approval",
    loader_function="load_your_dataset"
)
```

3. **Run via CLI or scripts**:

```bash
python cli.py --dataset your_dataset --models 8b 70b
# or
python run_mistral_experiments.py your_dataset
```

---

### Modify Test Size or Hyperparameters

In the experiment configuration (e.g., in `run_experiment.py` or a family runner):

```python
exp_config = create_experiment_config(
    dataset_key=dataset_name,
    model_size=model_size,
    test_size=0.30,    # change from default 0.20
    random_state=42,
    temperature=0.0,
    max_tokens=2000
)
```

---

## Troubleshooting

**1. Rate Limits**

* The framework already uses exponential backoff in `api_clients.py`.
* If you still hit limits:

  * Reduce test size
  * Avoid parallel runs
  * Increase time between requests if needed

**2. API Key Errors**

```bash
# Check that keys exist
cat .env

# Reactivate environment to reload .env
source venv/bin/activate
```

**3. Missing Dependencies**

```bash
pip install -r requirements.txt --upgrade
```

**4. Dataset Loading Errors**

* Some datasets (especially Folktables/ACS) may download or process large files on first use.
* Ensure stable internet and enough memory/disk space.
* Verify any required raw files (e.g., ACS CSVs) exist in the `data/` directory.

---

## Citation

If you use this framework in academic work:

```bibtex
@software{fairness_experiments,
  title  = {Evolution of Fairness in Large Language Models},
  author = {Khaleeqa Aasiyah Garrett},
  year   = {2025},
  url    = {https://github.com/Khaleeks/EvolutionFairnessLLMS}
}
```

**For detailed methodological background and results, see the accompanying capstone report.**

```
```
