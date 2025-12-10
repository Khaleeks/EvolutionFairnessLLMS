"""
Complete Dataset Loaders Module - All Real Data
fairness_experiments/data_loaders.py
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

# ============================================================================
# GERMAN CREDIT
# ============================================================================

GERMAN_CREDIT_MAPPINGS = {
    'Attribute1': {
        'A11': 'checking account < 0 DM', 'A12': 'checking account 0-200 DM',
        'A13': 'checking account >= 200 DM', 'A14': 'no checking account'
    },
    'Attribute3': {
        'A30': 'no credits taken/all credits paid back duly',
        'A31': 'all credits at this bank paid back duly',
        'A32': 'existing credits paid back duly till now',
        'A33': 'delay in paying off in the past',
        'A34': 'critical account/other credits existing'
    },
    'Attribute4': {
        'A40': 'car (new)', 'A41': 'car (used)', 'A42': 'furniture/equipment',
        'A43': 'radio/television', 'A44': 'domestic appliances', 'A45': 'repairs',
        'A46': 'education', 'A48': 'retraining', 'A49': 'business', 'A410': 'others'
    },
    'Attribute6': {
        'A61': 'savings < 100 DM', 'A62': 'savings 100-500 DM',
        'A63': 'savings 500-1000 DM', 'A64': 'savings >= 1000 DM',
        'A65': 'unknown/no savings account'
    },
    'Attribute7': {
        'A71': 'unemployed', 'A72': 'employed < 1 year',
        'A73': 'employed 1-4 years', 'A74': 'employed 4-7 years',
        'A75': 'employed >= 7 years'
    },
    'Attribute9': {
        'A91': 'male: divorced/separated', 'A92': 'female: divorced/separated/married',
        'A93': 'male: single', 'A94': 'male: married/widowed', 'A95': 'female: single'
    },
    'Attribute10': {'A101': 'none', 'A102': 'co-applicant', 'A103': 'guarantor'},
    'Attribute12': {
        'A121': 'real estate', 'A122': 'building society savings/life insurance',
        'A123': 'car or other', 'A124': 'unknown/no property'
    },
    'Attribute14': {'A141': 'bank', 'A142': 'stores', 'A143': 'none'},
    'Attribute15': {'A151': 'rent', 'A152': 'own', 'A153': 'for free'},
    'Attribute17': {
        'A171': 'unemployed/unskilled - non-resident', 'A172': 'unskilled - resident',
        'A173': 'skilled employee/official', 'A174': 'management/self-employed/highly qualified'
    },
    'Attribute19': {'A191': 'none', 'A192': 'yes, registered under customer name'},
    'Attribute20': {'A201': 'yes', 'A202': 'no'}
}

GERMAN_CREDIT_NAMES = {
    'Attribute1': 'Checking Account Status', 'Attribute2': 'Duration (months)',
    'Attribute3': 'Credit History', 'Attribute4': 'Purpose', 'Attribute5': 'Credit Amount (DM)',
    'Attribute6': 'Savings Account', 'Attribute7': 'Employment Duration',
    'Attribute8': 'Installment Rate (%)', 'Attribute9': 'Personal Status',
    'Attribute10': 'Other Debtors/Guarantors', 'Attribute11': 'Present Residence Since (years)',
    'Attribute12': 'Property', 'Attribute13': 'Age (years)', 'Attribute14': 'Other Installment Plans',
    'Attribute15': 'Housing', 'Attribute16': 'Number of Existing Credits',
    'Attribute17': 'Job', 'Attribute18': 'Number of Dependents',
    'Attribute19': 'Telephone', 'Attribute20': 'Foreign Worker'
}

def decode_german_credit_features(row: pd.Series, feature_columns: list) -> str:
    """Convert coded features to human-readable descriptions"""
    descriptions = []
    for col in feature_columns:
        if col not in row.index:
            continue
        value = row[col]
        feature_name = GERMAN_CREDIT_NAMES.get(col, col)
        if col in GERMAN_CREDIT_MAPPINGS and str(value) in GERMAN_CREDIT_MAPPINGS[col]:
            decoded_value = GERMAN_CREDIT_MAPPINGS[col][str(value)]
            descriptions.append(f"{feature_name}: {decoded_value}")
        else:
            descriptions.append(f"{feature_name}: {value}")
    return "\n".join(descriptions)

def load_german_credit(test_size: float = 0.20, random_state: int = 42):
    """Load German Credit dataset"""
    print("Loading German Credit dataset...")
    
    statlog_german_credit_data = fetch_ucirepo(id=144)
    X = statlog_german_credit_data.data.features
    y = statlog_german_credit_data.data.targets
    
    df = X.copy()
    df['class'] = y.iloc[:, 0].values
    df['record_id'] = df.index.map(lambda x: f"rec_{x:06d}")
    
    # Map labels: 1 = good, 2 = bad
    label_map = {1: 'good', 2: 'bad', '1': 'good', '2': 'bad'}
    df['class'] = df['class'].map(label_map)
    
    target_column = 'class'
    sensitive_feature = 'Attribute9'
    feature_columns = [col for col in df.columns if col not in [target_column, 'record_id', sensitive_feature]]
    
    X = df[feature_columns]
    y = df[target_column]
    sensitive_features = df[sensitive_feature].astype(str).str.strip()
    record_ids = df['record_id']
    
    X_train, X_test, y_train, y_test, sf_train, sf_test, ids_train, ids_test = train_test_split(
        X, y, sensitive_features, record_ids,
        test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    return {
        'X_test': X_test,
        'y_test': y_test,
        'sf_test': sf_test,
        'ids_test': ids_test,
        'feature_columns': feature_columns,
        'decoder': decode_german_credit_features
    }

# ============================================================================
# ADULT INCOME
# ============================================================================

def decode_adult_income_features(row: pd.Series, feature_columns: list) -> str:
    """Format adult income features"""
    descriptions = []
    for col in feature_columns:
        if col in row.index:
            descriptions.append(f"{col}: {row[col]}")
    return "\n".join(descriptions)

def load_adult_income(test_size: float = 0.20, random_state: int = 42):
    """Load Adult Income dataset"""
    print("Loading Adult Income dataset...")
    
    from datasets import load_dataset
    
    dataset = load_dataset("scikit-learn/adult-census-income")
    df = pd.DataFrame(dataset['train'])
    
    df['record_id'] = df.index.map(lambda x: f"rec_{x:06d}")
    
    # Standardize column names
    target_column = 'income' if 'income' in df.columns else 'class'
    df[target_column] = df[target_column].astype(str).str.strip().str.replace('.', '')
    df[target_column] = df[target_column].map({'>50K': '>50K', '<=50K': '<=50K', '1': '>50K', '0': '<=50K'})
    
    sensitive_features_cols = ['race', 'sex']
    feature_columns = [col for col in df.columns if col not in [target_column, 'record_id'] + sensitive_features_cols]
    
    X = df[feature_columns]
    y = df[target_column]
    sf = df['sex'].astype(str) + "_" + df['race'].astype(str)
    record_ids = df['record_id']
    
    X_train, X_test, y_train, y_test, sf_train, sf_test, ids_train, ids_test = train_test_split(
        X, y, sf, record_ids, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    return {
        'X_test': X_test, 'y_test': y_test, 'sf_test': sf_test, 'ids_test': ids_test,
        'feature_columns': feature_columns, 'decoder': decode_adult_income_features
    }

# ============================================================================
# COMPAS
# ============================================================================

def decode_compas_features(row: pd.Series, feature_columns: list) -> str:
    """Format COMPAS features"""
    descriptions = []
    for col in feature_columns:
        if col in row.index:
            descriptions.append(f"{col}: {row[col]}")
    return "\n".join(descriptions)

def load_compas(test_size: float = 0.20, random_state: int = 42):
    """Load COMPAS dataset"""
    print("Loading COMPAS dataset...")
    
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    df = pd.read_csv(url)
    
    df['record_id'] = df.index.map(lambda x: f"rec_{x:06d}")
    
    # Filter to African-American and Caucasian
    df = df[df['race'].isin(['African-American', 'Caucasian'])]
    
    target_column = 'two_year_recid'
    df[target_column] = df[target_column].map({0: 'low_risk', 1: 'high_risk'})
    
    sensitive_features_cols = ['race', 'sex']
    keep_features = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                     'priors_count', 'c_charge_degree', 'is_recid', 'is_violent_recid']
    
    feature_columns = [col for col in keep_features if col in df.columns and col not in [target_column]]
    
    X = df[feature_columns]
    y = df[target_column]
    sf = df['sex'].astype(str) + "_" + df['race'].astype(str)
    record_ids = df['record_id']
    
    X_train, X_test, y_train, y_test, sf_train, sf_test, ids_train, ids_test = train_test_split(
        X, y, sf, record_ids, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    return {
        'X_test': X_test, 'y_test': y_test, 'sf_test': sf_test, 'ids_test': ids_test,
        'feature_columns': feature_columns, 'decoder': decode_compas_features
    }

# ============================================================================
# BANK MARKETING
# ============================================================================

def decode_bank_marketing_features(row: pd.Series, feature_columns: list) -> str:
    """Format bank marketing features"""
    descriptions = []
    for col in feature_columns:
        if col in row.index:
            descriptions.append(f"{col}: {row[col]}")
    return "\n".join(descriptions)

def load_bank_marketing(test_size: float = 0.20, random_state: int = 42):
    """Load Bank Marketing dataset"""
    print("Loading Bank Marketing dataset...")
    
    bank = fetch_ucirepo(id=222)
    df = pd.concat([bank.data.features, bank.data.targets], axis=1)
    df['record_id'] = df.index.map(lambda x: f"rec_{x:06d}")
    
    target_column = 'y'
    df[target_column] = df[target_column].astype(str).str.strip()
    
    sensitive_features_cols = ['age', 'job', 'marital']
    feature_columns = [col for col in df.columns if col not in [target_column, 'record_id'] + sensitive_features_cols]
    
    X = df[feature_columns]
    y = df[target_column]
    sf = df['age'].astype(str) + "_" + df['job'].astype(str)
    record_ids = df['record_id']
    
    X_train, X_test, y_train, y_test, sf_train, sf_test, ids_train, ids_test = train_test_split(
        X, y, sf, record_ids, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    return {
        'X_test': X_test, 'y_test': y_test, 'sf_test': sf_test, 'ids_test': ids_test,
        'feature_columns': feature_columns, 'decoder': decode_bank_marketing_features
    }

# ============================================================================
# FOLKTABLES
# ============================================================================

def decode_folktables_features(row: pd.Series, feature_columns: list) -> str:
    """Format folktables features"""
    descriptions = []
    for col in feature_columns:
        if col in row.index:
            descriptions.append(f"{col}: {row[col]}")
    return "\n".join(descriptions)

def load_folktables(test_size: float = 0.20, random_state: int = 42):
    """Load Folktables ACS Income dataset"""
    print("Loading Folktables dataset...")
    
    from folktables import ACSDataSource, ACSIncome
    
    # Download 2018 ACS data for California
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA"], download=True)
    
    # Convert to features and target
    X, y, _ = ACSIncome.df_to_numpy(acs_data)
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=ACSIncome.features)
    df['PINCP'] = y
    df['PINCP'] = (df['PINCP'] > 50000).map({True: 'high_income', False: 'low_income'})
    
    # IMPORTANT: Limit to manageable size for API efficiency (195K+ is too large)
    MAX_SAMPLES = 5000
    if len(df) > MAX_SAMPLES:
        print(f"  Sampling {MAX_SAMPLES} from {len(df)} records for API efficiency...")
        df = df.sample(n=MAX_SAMPLES, random_state=random_state)
    
    df['record_id'] = df.index.map(lambda x: f"rec_{x:06d}")
    
    target_column = 'PINCP'
    sensitive_features_cols = ['RAC1P', 'SEX']
    feature_columns = [col for col in df.columns if col not in [target_column, 'record_id'] + sensitive_features_cols]
    
    X = df[feature_columns]
    y = df[target_column]
    sf = df['SEX'].astype(str) + "_" + df['RAC1P'].astype(str)
    record_ids = df['record_id']
    
    X_train, X_test, y_train, y_test, sf_train, sf_test, ids_train, ids_test = train_test_split(
        X, y, sf, record_ids, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    return {
        'X_test': X_test, 'y_test': y_test, 'sf_test': sf_test, 'ids_test': ids_test,
        'feature_columns': feature_columns, 'decoder': decode_folktables_features
    }

# ============================================================================
# DIABETES READMISSION (replacing Heritage Health)
# ============================================================================

def decode_diabetes_features(row: pd.Series, feature_columns: list) -> str:
    """Format diabetes readmission features"""
    descriptions = []
    for col in feature_columns:
        if col in row.index:
            descriptions.append(f"{col}: {row[col]}")
    return "\n".join(descriptions)

def load_diabetes_readmission(test_size: float = 0.20, random_state: int = 42):
    """
    Load Diabetes 130-US Hospitals dataset from UCI
    Predicts hospital readmission for diabetic patients
    """
    print("Loading Diabetes Readmission dataset...")
    
    # Load from UCI - dataset ID 296
    diabetes = fetch_ucirepo(id=296)
    df = pd.concat([diabetes.data.features, diabetes.data.targets], axis=1)
    
    df['record_id'] = df.index.map(lambda x: f"rec_{x:06d}")
    
    # Target: readmission status
    # Original has: NO, <30, >30
    # We'll make binary: readmitted (<30 days) vs not readmitted (NO or >30)
    target_column = 'readmitted'
    
    if target_column in df.columns:
        df[target_column] = df[target_column].astype(str).str.strip()
        # Binary classification: early readmission (<30) vs no early readmission
        df[target_column] = df[target_column].map({
            '<30': 'readmitted',
            '>30': 'not_readmitted',
            'NO': 'not_readmitted'
        })
    
    # Age is given in ranges like "[0-10)", "[10-20)", etc.
    # Create binary age group: <65 vs >=65
    if 'age' in df.columns:
        df['age_str'] = df['age'].astype(str)
        # Ages 70-80, 80-90, 90-100 are >=65
        df['age_group'] = df['age_str'].apply(
            lambda x: '>=65' if any(age in str(x) for age in ['70', '80', '90']) else '<65'
        )
    else:
        df['age_group'] = '<65'  # Default if age not available
    
    # Select relevant features for prediction
    potential_features = [
        'race', 'gender', 'time_in_hospital', 'num_lab_procedures',
        'num_procedures', 'num_medications', 'number_outpatient',
        'number_emergency', 'number_inpatient', 'number_diagnoses',
        'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed'
    ]
    
    # Keep only features that exist in the dataframe
    feature_columns = [col for col in potential_features if col in df.columns and col != 'age']
    
    # Remove race and gender from features (they're sensitive)
    sensitive_features_cols = ['age_group']
    if 'race' in feature_columns:
        sensitive_features_cols.append('race')
        feature_columns.remove('race')
    if 'gender' in feature_columns:
        feature_columns.remove('gender')
    
    # Clean data: remove rows with missing target
    df = df[df[target_column].notna()].copy()
    
    X = df[feature_columns]
    y = df[target_column]
    sf = df['age_group']
    record_ids = df['record_id']
    
    X_train, X_test, y_train, y_test, sf_train, sf_test, ids_train, ids_test = train_test_split(
        X, y, sf, record_ids, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    return {
        'X_test': X_test, 'y_test': y_test, 'sf_test': sf_test, 'ids_test': ids_test,
        'feature_columns': feature_columns, 'decoder': decode_diabetes_features
    }

# Alias for backwards compatibility
load_heritage_health = load_diabetes_readmission
decode_heritage_health_features = decode_diabetes_features

# ============================================================================
# LOADER REGISTRY
# ============================================================================

LOADER_REGISTRY = {
    'load_german_credit': load_german_credit,
    'load_adult_income': load_adult_income,
    'load_compas': load_compas,
    'load_bank_marketing': load_bank_marketing,
    'load_folktables': load_folktables,
    'load_diabetes_readmission': load_diabetes_readmission,
    'load_heritage_health': load_heritage_health,  # Alias for backwards compatibility

}

def load_dataset(dataset_name: str, test_size: float = 0.20, random_state: int = 42):
    """Load any dataset by name"""
    from config import DATASET_REGISTRY
    
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    loader_name = DATASET_REGISTRY[dataset_name].loader_function
    
    if loader_name not in LOADER_REGISTRY:
        raise ValueError(f"Loader not implemented: {loader_name}")
    
    return LOADER_REGISTRY[loader_name](test_size, random_state)