"""
Master Execution Script
fairness_experiments/run_all_experiments.py

Run all 9 datasets across all 3 models (27 total experiments)
"""

import json
from pathlib import Path
from datetime import datetime
from config import (
    create_experiment_matrix, 
    create_output_structure, 
    ModelSize, 
    APIProvider,
    DATASET_REGISTRY
)
from run_experiment import run_experiments_sequential, run_experiments_parallel
from fairness_analysis import compare_models

# ============================================================================
# EXECUTION STRATEGIES
# ============================================================================

def run_all_datasets_all_models(parallel=False, max_workers=2):
    """
    Run ALL 9 datasets on ALL 3 models (27 experiments total)
    Using Together AI for everything
    
    Args:
        parallel: If True, run in parallel (use with caution for rate limits)
        max_workers: Number of parallel workers if parallel=True
    """
    
    print(f"\n{'#'*70}")
    print("MASTER EXECUTION: ALL DATASETS × ALL MODELS")
    print(f"API Provider: Together AI")
    print(f"Mode: {'Parallel' if parallel else 'Sequential'}")
    print(f"{'#'*70}\n")
    
    # Create output structure
    base_dir = create_output_structure()
    
    # Create full experiment matrix (Together AI for all)
    experiments = create_experiment_matrix()
    
    print(f"Total experiments to run: {len(experiments)}")
    print(f"Datasets: {len(DATASET_REGISTRY)}")
    print(f"Models: 3 (8B, 70B, 405B)")
    print(f"\nEstimated time (sequential): ~{len(experiments) * 5} minutes")
    print(f"  (assuming ~5 min per experiment with 200 test samples)\n")
    
    # Run experiments
    if parallel:
        results = run_experiments_parallel(experiments, max_workers=max_workers)
    else:
        results = run_experiments_sequential(experiments)
    
    # Save master summary
    timestamp = datetime.now().isoformat()
    master_summary = {
        'execution_timestamp': timestamp,
        'total_experiments': len(experiments),
        'api_provider': 'together',
        'execution_mode': 'parallel' if parallel else 'sequential',
        'results': results
    }
    
    summary_path = base_dir / f"master_summary_{timestamp.replace(':', '-')}.json"
    with open(summary_path, 'w') as f:
        json.dump(master_summary, f, indent=2, default=str)
    
    print(f"\n✓ Master summary saved: {summary_path}")
    
    return results

def run_single_dataset_all_models(dataset_name: str):
    """
    Run a single dataset across all 3 models (3 experiments)
    Using Together AI
    
    Args:
        dataset_name: Name of dataset (e.g., 'german_credit')
    """
    
    print(f"\n{'#'*70}")
    print(f"RUNNING DATASET: {dataset_name}")
    print(f"Models: LLAMA 8B, 70B, 405B")
    print(f"API: Together AI")
    print(f"{'#'*70}\n")
    
    if dataset_name not in DATASET_REGISTRY:
        print(f"✗ Unknown dataset: {dataset_name}")
        print(f"Available datasets: {list(DATASET_REGISTRY.keys())}")
        return
    
    # Create output structure
    create_output_structure()
    
    # Create experiments for this dataset only (Together AI)
    experiments = create_experiment_matrix(datasets=[dataset_name])
    
    print(f"Running {len(experiments)} experiments for {dataset_name}\n")
    
    # Run sequentially
    results = run_experiments_sequential(experiments)
    
    # Compare models
    exp_ids = [r['experiment_id'] for r in results if r.get('status') == 'completed']
    if len(exp_ids) > 1:
        print("\nComparing models...")
        compare_models(exp_ids)
    
    return results

def run_dataset_subset(dataset_names: list, models: list = None):
    """
    Run a custom subset of datasets and models
    Using Together AI
    
    Args:
        dataset_names: List of dataset names
        models: List of ModelSize enums (default: all 3)
    """
    
    print(f"\n{'#'*70}")
    print("RUNNING CUSTOM SUBSET")
    print(f"Datasets: {dataset_names}")
    print(f"Models: {[m.name for m in models] if models else 'All'}")
    print(f"API: Together AI")
    print(f"{'#'*70}\n")
    
    # Validate datasets
    for dataset_name in dataset_names:
        if dataset_name not in DATASET_REGISTRY:
            print(f"✗ Unknown dataset: {dataset_name}")
            return
    
    # Create output structure
    create_output_structure()
    
    # Create experiment matrix (Together AI)
    experiments = create_experiment_matrix(
        datasets=dataset_names,
        models=models
    )
    
    print(f"Running {len(experiments)} experiments\n")
    
    # Run sequentially
    results = run_experiments_sequential(experiments)
    
    return results

# ============================================================================
# EXAMPLES
# ============================================================================

def example_1_single_dataset():
    """Example: Run German Credit on all 3 models"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Dataset (German Credit) × All Models")
    print("="*70)
    
    results = run_single_dataset_all_models(
        dataset_name="german_credit",
        api_provider=APIProvider.GROQ
    )
    
    return results

def example_2_finance_datasets():
    """Example: Run all finance datasets on all models"""
    print("\n" + "="*70)
    print("EXAMPLE 2: All Finance Datasets × All Models")
    print("="*70)
    
    finance_datasets = [
        "german_credit",
        "bank_marketing"
    ]
    
    results = run_dataset_subset(dataset_names=finance_datasets)
    
    return results

def example_3_one_model_all_datasets():
    """Example: Run all datasets on just LLAMA 8B"""
    print("\n" + "="*70)
    print("EXAMPLE 3: All Datasets × LLAMA 8B Only")
    print("="*70)
    
    results = run_dataset_subset(
        dataset_names=list(DATASET_REGISTRY.keys()),
        models=[ModelSize.LLAMA_8B]
    )
    
    return results

def example_4_full_matrix():
    """Example: Run everything (27 experiments)"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Full Matrix (9 Datasets × 3 Models = 27 Experiments)")
    print("="*70)
    print("\n⚠️  WARNING: This will take a long time!")
    print("Estimated time: ~2-3 hours (sequential)")
    print("Consider running overnight or using parallel execution\n")
    
    # Uncomment to run
    # results = run_all_datasets_all_models(parallel=False)
    
    print("Uncomment the code in example_4_full_matrix() to run")
    return None

def example_5_compare_models_on_german_credit():
    """Example: Compare model performance on German Credit"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Compare Models on German Credit")
    print("="*70)
    
    # First run the experiments
    results = run_single_dataset_all_models(dataset_name="german_credit")
    
    # Extract experiment IDs
    exp_ids = [r['experiment_id'] for r in results if r.get('status') == 'completed']
    
    if len(exp_ids) > 0:
        print("\nGenerating cross-model comparison...")
        comparison_df = compare_models(exp_ids)
        return comparison_df
    else:
        print("No completed experiments to compare")
        return None

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point - customize this for your needs
    """
    
    print("\n" + "#"*70)
    print("# FAIRNESS EVALUATION FRAMEWORK")
    print("# Multi-Model, Multi-Dataset Analysis")
    print("# Using Together AI for all models")
    print("#"*70)
    
    print("\nAvailable datasets:")
    for i, (key, config) in enumerate(DATASET_REGISTRY.items(), 1):
        print(f"  {i}. {key:20s} - {config.name}")
    
    print("\nAvailable models:")
    print("  1. LLAMA 8B   (llama-3.1-8b-instant)")
    print("  2. LLAMA 70B  (llama-3.1-70b-versatile)")
    print("  3. LLAMA 405B (llama-3.1-405b-reasoning)")
    
    print("\n" + "="*70)
    print("EXECUTION OPTIONS")
    print("="*70)
    print("\n1. Run a single dataset on all 3 models")
    print("2. Run a subset of datasets")
    print("3. Run all datasets on all models (27 experiments)")
    print("4. Run examples")
    
    # Default: Run German Credit on all 3 models
    print("\n" + "="*70)
    print("DEFAULT: Running German Credit on all 3 models")
    print("="*70 + "\n")
    
    results = example_1_single_dataset()
    
    print("\n" + "#"*70)
    print("# EXECUTION COMPLETE")
    print("#"*70)
    
    # Print summary
    if results:
        successful = sum(1 for r in results if r.get('status') == 'completed')
        failed = len(results) - successful
        
        print(f"\nSummary:")
        print(f"  Total experiments: {len(results)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        
        if successful > 0:
            print(f"\nOutput files location: fairness_experiments/")
            print(f"  - configs/       (experiment configurations)")
            print(f"  - predictions/   (detailed predictions with reasoning)")
            print(f"  - analysis/      (analysis data)")
            print(f"  - summaries/     (fairness metrics)")

if __name__ == "__main__":
    main()