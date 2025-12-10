"""
Main Experiment Runner - With Checkpoint Support
fairness_experiments/run_experiment.py

NEW FEATURES:
- Automatic checkpointing every 10 predictions
- Resume from checkpoint if experiment interrupted
- No data loss even if process crashes
"""

import pandas as pd
import json
from typing import Dict, Any
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path

from config import ExperimentConfig, DATASET_REGISTRY, get_output_paths
from data_loaders import load_dataset
from api_clients import create_client
from prompts import generate_prompt
from fairness_analysis import perform_fairness_analysis

load_dotenv()

# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def load_checkpoint(checkpoint_path: Path) -> pd.DataFrame:
    """Load existing checkpoint if it exists"""
    if checkpoint_path.exists():
        try:
            df = pd.read_csv(checkpoint_path)
            print(f" Loaded checkpoint: {len(df)} predictions already completed")
            return df
        except Exception as e:
            print(f"  Could not load checkpoint: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def save_checkpoint(predictions_df: pd.DataFrame, checkpoint_path: Path):
    """Save checkpoint to disk"""
    try:
        predictions_df.to_csv(checkpoint_path, index=False)
    except Exception as e:
        print(f"  Could not save checkpoint: {e}")

# ============================================================================
# LABEL NORMALIZATION
# ============================================================================

def normalize_label(label: str, dataset_config) -> str:
    """
    Normalize prediction label to match expected format
    
    Handles common variations:
    - Case differences: ">50k" -> ">50K"
    - Spacing: "low risk" -> "low_risk"
    - Capitalization: "Good" -> "good"
    """
    if not label:
        return None
    
    label = str(label).strip().lower()
    
    # Get expected labels (also lowercase for comparison)
    pos_class = dataset_config.positive_class.lower()
    neg_class = dataset_config.negative_class.lower()
    
    # Direct match
    if label == pos_class:
        return dataset_config.positive_class
    if label == neg_class:
        return dataset_config.negative_class
    
    # Remove spaces/underscores for fuzzy matching
    label_clean = label.replace(" ", "").replace("_", "").replace("-", "")
    pos_clean = pos_class.replace(" ", "").replace("_", "").replace("-", "")
    neg_clean = neg_class.replace(" ", "").replace("_", "").replace("-", "")
    
    if label_clean == pos_clean:
        return dataset_config.positive_class
    if label_clean == neg_clean:
        return dataset_config.negative_class
    
    # Check if label contains the expected label
    if pos_clean in label_clean:
        return dataset_config.positive_class
    if neg_clean in label_clean:
        return dataset_config.negative_class
    
    # Return original if no match (will be marked incorrect)
    return label

# ============================================================================
# SINGLE EXPERIMENT EXECUTION
# ============================================================================

def run_single_experiment(exp_config: ExperimentConfig, resume: bool = True) -> Dict[str, Any]:
    """
    Run complete experiment: load data, generate predictions, analyze fairness
    
    Args:
        exp_config: Experiment configuration
        resume: If True, resume from checkpoint if it exists
    
    Returns:
        Dictionary with experiment results and output file paths
    """
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_config.experiment_id}")
    print(f"Dataset: {exp_config.dataset_name}")
    print(f"Model: {exp_config.model_size.name} ({exp_config.model_name})")
    print(f"API Provider: {exp_config.api_provider.value}")
    print(f"{'='*70}\n")
    
    # Get output paths
    output_paths = get_output_paths(exp_config)
    
    # Create output directories
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_paths['config'], 'w') as f:
        config_dict = {
            'experiment_id': exp_config.experiment_id,
            'dataset_name': exp_config.dataset_name,
            'model_name': exp_config.model_name,
            'model_size': exp_config.model_size.value,
            'api_provider': exp_config.api_provider.value,
            'dataset_type': exp_config.dataset_type.value,
            'temperature': exp_config.temperature,
            'max_tokens': exp_config.max_tokens,
            'test_size': exp_config.test_size,
            'random_state': exp_config.random_state,
            'timestamp': exp_config.timestamp
        }
        json.dump(config_dict, f, indent=2)
    
    print(f"   Configuration saved: {output_paths['config']}\n")
    
    # Step 1: Load dataset
    print("STEP 1: Loading dataset...")
    try:
        data = load_dataset(
            exp_config.dataset_name,
            test_size=exp_config.test_size,
            random_state=exp_config.random_state
        )
        
        X_test = data['X_test']
        y_test = data['y_test']
        sf_test = data['sf_test']
        ids_test = data['ids_test']
        feature_columns = data['feature_columns']
        decoder = data['decoder']
        
        print(f"   Loaded {len(X_test)} test samples\n")
        
    except Exception as e:
        print(f"    Failed to load dataset: {e}")
        return {
            'experiment_id': exp_config.experiment_id,
            'status': 'failed',
            'error': f"Dataset loading error: {str(e)}",
            'output_paths': {k: str(v) for k, v in output_paths.items()}
        }
    
    # Step 2: Check for existing checkpoint
    checkpoint_df = pd.DataFrame()
    completed_ids = set()
    
    if resume and output_paths['checkpoint'].exists():
        checkpoint_df = load_checkpoint(output_paths['checkpoint'])
        if len(checkpoint_df) > 0:
            completed_ids = set(checkpoint_df['record_id'].values)
            print(f" Resuming from checkpoint - skipping {len(completed_ids)} completed predictions\n")
    
    # Step 3: Initialize API client
    print("STEP 2: Initializing API client...")
    try:
        client = create_client(exp_config)
        print(f"   Client initialized: {exp_config.api_provider.value}\n")
    except Exception as e:
        print(f"    Failed to initialize client: {e}")
        return {
            'experiment_id': exp_config.experiment_id,
            'status': 'failed',
            'error': f"Client initialization error: {str(e)}",
            'output_paths': {k: str(v) for k, v in output_paths.items()}
        }
    
    # Get dataset config for label normalization
    dataset_cfg = DATASET_REGISTRY[exp_config.dataset_name]
    
    # Step 4: Generate predictions
    print("STEP 3: Generating predictions...")
    remaining_samples = [
        (idx, row, record_id, ground_truth, sensitive_feature)
        for (idx, row), record_id, ground_truth, sensitive_feature 
        in zip(X_test.iterrows(), ids_test, y_test, sf_test)
        if record_id not in completed_ids
    ]
    
    print(f"Processing {len(remaining_samples)} remaining samples (skipped {len(completed_ids)})...\n")
    
    prediction_records = checkpoint_df.to_dict('records') if len(checkpoint_df) > 0 else []
    checkpoint_interval = 10  # Save every 10 predictions
    
    for i, (idx, row, record_id, ground_truth, sensitive_feature) in enumerate(
        tqdm(remaining_samples, desc="Classifying"), 1
    ):
        # Decode features to human-readable format
        data_description = decoder(row, feature_columns)
        
        # Generate prompts
        system_prompt, user_prompt = generate_prompt(
            exp_config=exp_config,
            data_description=data_description,
            record_id=record_id
        )
        
        # Get prediction from API
        pred_record = client.classify(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            record_id=record_id
        )
        
        # NORMALIZE THE PREDICTION LABEL
        if pred_record['prediction_label']:
            pred_record['prediction_label'] = normalize_label(
                pred_record['prediction_label'],
                dataset_cfg
            )
        
        # Add ground truth and sensitive feature
        pred_record['ground_truth'] = ground_truth
        pred_record['sensitive_feature'] = sensitive_feature
        
        # Calculate correctness
        if pred_record['api_success'] and pred_record['prediction_label']:
            pred_record['correct'] = (pred_record['prediction_label'] == ground_truth)
        else:
            pred_record['correct'] = False
        
        prediction_records.append(pred_record)
        
        # Save checkpoint periodically
        if i % checkpoint_interval == 0:
            checkpoint_temp_df = pd.DataFrame(prediction_records)
            checkpoint_temp_df = checkpoint_temp_df.rename(columns={'prediction_label': 'prediction'})
            save_checkpoint(checkpoint_temp_df, output_paths['checkpoint'])
    
    # Convert to DataFrame
    predictions_df = pd.DataFrame(prediction_records)
    
    # Rename prediction_label to prediction
    predictions_df = predictions_df.rename(columns={'prediction_label': 'prediction'})
    
    # Define exact column order for output
    column_order = [
        'record_id',
        'ground_truth',
        'prediction',
        'sensitive_feature',
        'experiment_id',
        'reasoning',
        'api_success',
        'attempts_made',
        'timestamp',
        'correct'
    ]
    
    # Ensure all columns exist
    for col in column_order:
        if col not in predictions_df.columns:
            predictions_df[col] = None
    
    predictions_df = predictions_df[column_order]
    
    # Save final predictions
    predictions_df.to_csv(output_paths['predictions'], index=False)
    
    # Save final checkpoint
    save_checkpoint(predictions_df, output_paths['checkpoint'])
    
    # Print summary
    successful_count = predictions_df['api_success'].sum()
    failed_count = len(predictions_df) - successful_count
    
    print(f"\n{'='*70}")
    print(f"PREDICTION SUMMARY")
    print(f"{'='*70}")
    print(f"Total predictions: {len(predictions_df)}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {failed_count}")
    
    if successful_count > 0:
        accuracy = predictions_df['correct'].sum() / successful_count
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"\nPrediction distribution:")
        print(predictions_df['prediction'].value_counts())
        print(f"\nGround truth distribution:")
        print(predictions_df['ground_truth'].value_counts())
        print(f"\nAverage attempts: {predictions_df['attempts_made'].mean():.2f}")
    
    print(f"\n   Predictions saved: {output_paths['predictions']}")
    print(f"   Checkpoint saved: {output_paths['checkpoint']}")
    print(f"{'='*70}\n")
    
    # Step 5: Fairness Analysis
    if successful_count > 0:
        print("STEP 4: Performing fairness analysis...")
        try:
            analysis_results = perform_fairness_analysis(
                predictions_df=predictions_df,
                exp_config=exp_config,
                output_paths=output_paths
            )
            print(f"   Analysis saved: {output_paths['analysis']}")
            print(f"   Fairness summary saved: {output_paths['fairness_summary']}\n")
        except Exception as e:
            print(f"  Fairness analysis failed: {e}\n")
            analysis_results = {'error': str(e)}
    else:
        print("  Skipping fairness analysis (no successful predictions)\n")
        analysis_results = {'error': 'No successful predictions'}
    
    print(f"{'='*70}")
    print(f"EXPERIMENT COMPLETE: {exp_config.experiment_id}")
    print(f"{'='*70}")
    print(f"\nGenerated files:")
    print(f"  1. Config: {output_paths['config']}")
    print(f"  2. Predictions: {output_paths['predictions']}")
    print(f"  3. Analysis: {output_paths['analysis']}")
    print(f"  4. Fairness Summary: {output_paths['fairness_summary']}")
    print(f"  5. Checkpoint: {output_paths['checkpoint']}")
    print(f"{'='*70}\n")
    
    return {
        'experiment_id': exp_config.experiment_id,
        'status': 'completed',
        'successful_predictions': successful_count,
        'failed_predictions': failed_count,
        'accuracy': accuracy if successful_count > 0 else 0.0,
        'output_paths': {k: str(v) for k, v in output_paths.items()},
        'analysis_results': analysis_results
    }

# ============================================================================
# BATCH EXPERIMENT EXECUTION
# ============================================================================

def run_experiments_sequential(experiments: list) -> list:
    """Run experiments sequentially (safer for rate limits)"""
    results = []
    
    print(f"\n{'#'*70}")
    print(f"STARTING BATCH EXECUTION")
    print(f"Total experiments: {len(experiments)}")
    print(f"{'#'*70}\n")
    
    for i, exp_config in enumerate(experiments, 1):
        print(f"\n{'*'*70}")
        print(f"EXPERIMENT {i}/{len(experiments)}")
        print(f"{'*'*70}")
        
        try:
            result = run_single_experiment(exp_config)
            results.append(result)
        except Exception as e:
            print(f"    Experiment failed: {e}")
            results.append({
                'experiment_id': exp_config.experiment_id,
                'status': 'failed',
                'error': str(e)
            })
    
    print(f"\n{'#'*70}")
    print(f"BATCH EXECUTION COMPLETE")
    print(f"{'#'*70}\n")
    
    # Summary
    successful = sum(1 for r in results if r.get('status') == 'completed')
    failed = len(results) - successful
    
    print(f"Summary:")
    print(f"  Completed: {successful}")
    print(f"  Failed: {failed}")
    
    return results

def run_experiments_parallel(experiments: list, max_workers: int = 3) -> list:
    """Run experiments in parallel (use with caution for API rate limits)"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    print(f"\n{'#'*70}")
    print(f"STARTING PARALLEL EXECUTION")
    print(f"Total experiments: {len(experiments)}")
    print(f"Max workers: {max_workers}")
    print(f"{'#'*70}\n")
    
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_exp = {
            executor.submit(run_single_experiment, exp): exp
            for exp in experiments
        }
        
        for future in as_completed(future_to_exp):
            exp = future_to_exp[future]
            try:
                result = future.result()
                results.append(result)
                print(f"   Completed: {exp.experiment_id}")
            except Exception as e:
                print(f"    Failed: {exp.experiment_id} - {str(e)}")
                results.append({
                    'experiment_id': exp.experiment_id,
                    'status': 'failed',
                    'error': str(e)
                })
    
    print(f"\n{'#'*70}")
    print(f"PARALLEL EXECUTION COMPLETE")
    print(f"{'#'*70}\n")
    
    return results

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Example usage"""
    from config import create_experiment_config, ModelSize, APIProvider
    
    # Example: Run German Credit on Mistral models
    experiments = [
        create_experiment_config("german_credit", ModelSize.MISTRAL_7B, APIProvider.TOGETHER),
        create_experiment_config("german_credit", ModelSize.MISTRAL_SMALL_24B, APIProvider.TOGETHER),
    ]
    
    # Run sequentially (recommended for API rate limits)
    results = run_experiments_sequential(experiments)
    
    # Save batch summary
    summary_path = Path("fairness_experiments") / "batch_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n   Batch summary saved: {summary_path}")

if __name__ == "__main__":
    main()