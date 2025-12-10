"""
Main Configuration Module
fairness_experiments/config.py
"""

from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import hashlib
import json
from datetime import datetime
from pathlib import Path

# ============================================================================
# ENUMS
# ============================================================================

class ModelSize(Enum):
    """Model names for different providers"""
    # Together AI models
    LLAMA_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    LLAMA_70B = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    LLAMA_405B = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"

    # Mistral models (via Together AI)
    MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"
    MISTRAL_SMALL_24B = "mistralai/Mistral-Small-24B-Instruct-2501"
    
    # Google Gemini models
    GEMINI_25_FLASH = "gemini-2.5-flash"
    GEMINI_25_PRO = "gemini-2.5-pro"
    
    # OpenAI GPT models
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"

class DatasetType(Enum):
    FINANCE = "finance"
    CRIMINAL_JUSTICE = "criminal_justice"
    SOCIOECONOMIC = "socioeconomic"
    HEALTHCARE = "healthcare"

class APIProvider(Enum):
    TOGETHER = "together"
    GEMINI = "gemini"
    OPENAI = "openai"

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class DatasetConfig:
    """Configuration for each dataset"""
    name: str
    dataset_type: DatasetType
    sensitive_features: List[str]
    target_column: str
    positive_class: str
    negative_class: str
    task_description: str
    loader_function: str
    feature_columns: Optional[List[str]] = None
    
@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run"""
    dataset_name: str
    model_name: str
    model_size: ModelSize
    api_provider: APIProvider
    dataset_type: DatasetType
    temperature: float
    max_tokens: int
    test_size: float
    random_state: int
    timestamp: str
    experiment_id: str

# ============================================================================
# DATASET REGISTRY
# ============================================================================

DATASET_REGISTRY = {
    "adult_income": DatasetConfig(
        name="Adult Income",
        dataset_type=DatasetType.SOCIOECONOMIC,
        sensitive_features=["race", "sex"],
        target_column="income",
        positive_class=">50K",
        negative_class="<=50K",
        task_description="predict whether annual income exceeds $50K",
        loader_function="load_adult_income"
    ),
    
    "compas": DatasetConfig(
        name="COMPAS Recidivism",
        dataset_type=DatasetType.CRIMINAL_JUSTICE,
        sensitive_features=["race", "sex"],
        target_column="two_year_recid",
        positive_class="low_risk",
        negative_class="high_risk",
        task_description="predict two-year recidivism risk",
        loader_function="load_compas"
    ),
    
    "german_credit": DatasetConfig(
        name="German Credit",
        dataset_type=DatasetType.FINANCE,
        sensitive_features=["Attribute9"],
        target_column="class",
        positive_class="good",
        negative_class="bad",
        task_description="assess credit risk for loan approval",
        loader_function="load_german_credit"
    ),
    
    "folktables": DatasetConfig(
        name="Folktables (ACS Income)",
        dataset_type=DatasetType.SOCIOECONOMIC,
        sensitive_features=["RAC1P", "SEX"],
        target_column="PINCP",
        positive_class="high_income",
        negative_class="low_income",
        task_description="predict high income from census data",
        loader_function="load_folktables"
    ),
    
    "bank_marketing": DatasetConfig(
        name="Bank Marketing",
        dataset_type=DatasetType.FINANCE,
        sensitive_features=["age", "job", "marital"],
        target_column="y",
        positive_class="yes",
        negative_class="no",
        task_description="predict if client will subscribe to term deposit",
        loader_function="load_bank_marketing"
    ),
    
    "diabetes_readmission": DatasetConfig(
        name="Diabetes Hospital Readmission",
        dataset_type=DatasetType.HEALTHCARE,
        sensitive_features=["age_group"],
        target_column="readmitted",
        positive_class="readmitted",
        negative_class="not_readmitted",
        task_description="predict 30-day hospital readmission for diabetic patients",
        loader_function="load_diabetes_readmission"
    ),
    
    # Backwards compatibility alias
    "heritage_health": DatasetConfig(
        name="Diabetes Hospital Readmission",
        dataset_type=DatasetType.HEALTHCARE,
        sensitive_features=["age_group"],
        target_column="readmitted",
        positive_class="readmitted",
        negative_class="not_readmitted",
        task_description="predict 30-day hospital readmission for diabetic patients",
        loader_function="load_heritage_health"
    ),
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_experiment_id(dataset_name: str, model_name: str, timestamp: str) -> str:
    """Generate short, unique experiment ID"""
    config_str = f"{dataset_name}_{model_name}_{timestamp}"
    return hashlib.md5(config_str.encode()).hexdigest()[:8]

def create_experiment_config(
    dataset_key: str,
    model_size: ModelSize,
    api_provider: APIProvider = APIProvider.TOGETHER,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    test_size: float = 0.20,
    random_state: int = 42
) -> ExperimentConfig:
    """Create a complete experiment configuration"""
    
    dataset_cfg = DATASET_REGISTRY[dataset_key]
    timestamp = datetime.now().isoformat()
    exp_id = generate_experiment_id(dataset_key, model_size.value, timestamp)
    
    return ExperimentConfig(
        dataset_name=dataset_key,
        model_name=model_size.value,
        model_size=model_size,
        api_provider=api_provider,
        dataset_type=dataset_cfg.dataset_type,
        temperature=temperature,
        max_tokens=max_tokens,
        test_size=test_size,
        random_state=random_state,
        timestamp=timestamp,
        experiment_id=exp_id
    )

def create_output_structure(base_dir: str = "fairness_experiments") -> Path:
    """Create organized output directory structure"""
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)
    
    (base_path / "configs").mkdir(exist_ok=True)
    (base_path / "predictions").mkdir(exist_ok=True)
    (base_path / "analysis").mkdir(exist_ok=True)
    (base_path / "summaries").mkdir(exist_ok=True)
    (base_path / "logs").mkdir(exist_ok=True)
    (base_path / "checkpoints").mkdir(exist_ok=True)  # NEW: For saving progress
    
    return base_path

def get_output_paths(exp_config: ExperimentConfig, base_dir: str = "fairness_experiments") -> Dict[str, Path]:
    """Generate standardized output file paths"""
    base_path = Path(base_dir)
    
    # Filename: dataset_model_expid
    prefix = f"{exp_config.dataset_name}_{exp_config.model_size.name}_{exp_config.experiment_id}"
    
    return {
        'config': base_path / "configs" / f"{prefix}_config.json",
        'predictions': base_path / "predictions" / f"{prefix}_predictions.csv",
        'analysis': base_path / "analysis" / f"{prefix}_analysis.csv",
        'fairness_summary': base_path / "summaries" / f"{prefix}_fairness.json",
        'log': base_path / "logs" / f"{prefix}.log",
        'checkpoint': base_path / "checkpoints" / f"{prefix}_checkpoint.csv"  # NEW
    }

def create_experiment_matrix(
    datasets: List[str] = None,
    models: List[ModelSize] = None,
    api_provider: APIProvider = APIProvider.TOGETHER
) -> List[ExperimentConfig]:
    """Create matrix of all experiment combinations"""
    
    if datasets is None:
        # Exclude the backwards compatibility alias
        datasets = [k for k in DATASET_REGISTRY.keys() if k != 'heritage_health']
    
    if models is None:
        models = [ModelSize.LLAMA_8B, ModelSize.LLAMA_70B, ModelSize.LLAMA_405B]
    
    experiments = []
    for dataset_key in datasets:
        for model_size in models:
            exp_config = create_experiment_config(
                dataset_key=dataset_key,
                model_size=model_size,
                api_provider=api_provider
            )
            experiments.append(exp_config)
    
    return experiments