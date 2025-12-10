"""
Run Mistral Experiments - All 6 Datasets on Mistral-7B and Mistral-Small-24B
fairness_experiments/run_mistral_experiments.py

Runs experiments in this order:
1. Adult Income: Mistral-7B -> Mistral-Small-24B
2. German Credit: Mistral-7B -> Mistral-Small-24B
3. COMPAS: Mistral-7B -> Mistral-Small-24B
4. Bank Marketing: Mistral-7B -> Mistral-Small-24B
5. Folktables: Mistral-7B -> Mistral-Small-24B
6. Diabetes Readmission: Mistral-7B -> Mistral-Small-24B
"""

import json
from pathlib import Path
from datetime import datetime
from config import (
    create_experiment_config,
    ModelSize,
    APIProvider,
    DATASET_REGISTRY,
    create_output_structure
)
from run_experiment import run_single_experiment
from fairness_analysis import compare_models

def run_all_datasets_mistral():
    """
    Run all 6 datasets on both Mistral models (12 experiments total)
    
    For each dataset:
      1. Run Mistral-7B
      2. Run Mistral-Small-24B
      3. Compare the 2 models
      4. Move to next dataset
    """
    
    print("\n" + "="*70)
    print("MISTRAL MODELS EVALUATION: ALL DATASETS")
    print("="*70)
    print("\nExecution Order:")
    print("  For each dataset:")
    print("    1. Run Mistral-7B")
    print("    2. Run Mistral-Small-24B")
    print("    3. Compare results")
    print("    4. Move to next dataset")
    print("\n" + "="*70 + "\n")
    
    # Create output structure
    base_dir = create_output_structure()
    
    # All 6 datasets (excluding backwards compatibility alias)
    datasets = [
        'adult_income',
        'german_credit',
        'compas',
        'bank_marketing',
        'folktables',
        'diabetes_readmission'
    ]
    
    # Mistral models
    models = [ModelSize.MISTRAL_7B, ModelSize.MISTRAL_SMALL_24B]
    
    print(f"Total: {len(datasets)} datasets × {len(models)} models = {len(datasets) * len(models)} experiments\n")
    
    all_results = []
    dataset_summaries = {}
    
    # Process each dataset
    for dataset_idx, dataset_name in enumerate(datasets, 1):
        print("\n" + "#"*70)
        print(f"# DATASET {dataset_idx}/{len(datasets)}: {dataset_name.upper()}")
        print("#"*70 + "\n")
        
        dataset_config = DATASET_REGISTRY[dataset_name]
        print(f"Name: {dataset_config.name}")
        print(f"Type: {dataset_config.dataset_type.value}")
        print(f"Task: {dataset_config.task_description}")
        print(f"Sensitive Features: {', '.join(dataset_config.sensitive_features)}")
        print("\n" + "-"*70 + "\n")
        
        dataset_results = []
        dataset_exp_ids = []
        
        # Run both Mistral models for this dataset
        for model_idx, model_size in enumerate(models, 1):
            print(f"\n{'*'*70}")
            print(f"* MODEL {model_idx}/{len(models)}: {model_size.name}")
            print(f"* Dataset: {dataset_name} ({dataset_idx}/{len(datasets)})")
            print(f"{'*'*70}\n")
            
            # Create experiment config
            exp_config = create_experiment_config(
                dataset_key=dataset_name,
                model_size=model_size,
                api_provider=APIProvider.TOGETHER
            )
            
            # Run experiment
            try:
                result = run_single_experiment(exp_config)
                dataset_results.append(result)
                all_results.append(result)
                
                if result.get('status') == 'completed':
                    dataset_exp_ids.append(result['experiment_id'])
                    print(f"\nCompleted: {exp_config.experiment_id}")
                    print(f"  Accuracy: {result.get('accuracy', 0):.4f}")
                    print(f"  Successful predictions: {result.get('successful_predictions', 0)}")
                else:
                    print(f"\n Failed: {exp_config.experiment_id}")
                    print(f"  Error: {result.get('error', 'Unknown')}")
                    
            except Exception as e:
                print(f"\n Exception during experiment: {e}")
                result = {
                    'experiment_id': exp_config.experiment_id,
                    'status': 'failed',
                    'error': str(e),
                    'dataset': dataset_name,
                    'model': model_size.name
                }
                dataset_results.append(result)
                all_results.append(result)
        
        # Compare models for this dataset
        print("\n" + "="*70)
        print(f"DATASET COMPLETE: {dataset_name}")
        print("="*70)
        
        successful_count = sum(1 for r in dataset_results if r.get('status') == 'completed')
        failed_count = len(dataset_results) - successful_count
        
        print(f"\nResults for {dataset_name}:")
        print(f"  Models completed: {successful_count}/{len(models)}")
        print(f"  Models failed: {failed_count}/{len(models)}")
        
        if successful_count > 1:
            print(f"\n Comparing {successful_count} models...\n")
            try:
                comparison_df = compare_models(dataset_exp_ids)
                print(f"\n✓ Comparison complete for {dataset_name}")
            except Exception as e:
                print(f"\n Comparison failed: {e}")
        elif successful_count == 1:
            print("\n Only 1 model completed - skipping comparison")
        else:
            print("\n No models completed - skipping comparison")
        
        # Save dataset summary
        dataset_summaries[dataset_name] = {
            'completed_models': successful_count,
            'failed_models': failed_count,
            'experiment_ids': dataset_exp_ids,
            'results': dataset_results
        }
        
        print(f"\n{'='*70}")
        print(f"Progress: {dataset_idx}/{len(datasets)} datasets completed")
        print(f"{'='*70}\n")
        
        # Save intermediate results after each dataset
        intermediate_path = base_dir / f"progress_mistral_after_{dataset_name}.json"
        with open(intermediate_path, 'w') as f:
            json.dump({
                'completed_datasets': dataset_idx,
                'total_datasets': len(datasets),
                'current_dataset': dataset_name,
                'timestamp': datetime.now().isoformat(),
                'dataset_summaries': dataset_summaries
            }, f, indent=2, default=str)
        
        print(f" Progress saved: {intermediate_path}\n")
    
    # Final summary
    print("\n" + "#"*70)
    print("# ALL MISTRAL EXPERIMENTS COMPLETE!")
    print("#"*70 + "\n")
    
    total_completed = sum(1 for r in all_results if r.get('status') == 'completed')
    total_failed = len(all_results) - total_completed
    
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Total experiments: {len(all_results)}")
    print(f"Completed: {total_completed}")
    print(f"Failed: {total_failed}")
    print(f"Success rate: {total_completed/len(all_results)*100:.1f}%")
    print("\n" + "="*70 + "\n")
    
    # Print per-dataset summary
    print("PER-DATASET SUMMARY")
    print("="*70)
    for dataset_name, summary in dataset_summaries.items():
        status = f"{summary['completed_models']}/{len(models)} models"
        print(f"{dataset_name:25s} - {status}")
    print("="*70 + "\n")
    
    # Save final master summary
    timestamp = datetime.now().isoformat()
    master_summary = {
        'execution_timestamp': timestamp,
        'execution_strategy': 'sequential_by_dataset_mistral',
        'api_provider': 'together',
        'models': [m.value for m in models],
        'total_experiments': len(all_results),
        'completed': total_completed,
        'failed': total_failed,
        'dataset_summaries': dataset_summaries,
        'all_results': all_results
    }
    
    summary_path = base_dir / f"master_summary_mistral_{timestamp.replace(':', '-').replace('.', '_')}.json"
    with open(summary_path, 'w') as f:
        json.dump(master_summary, f, indent=2, default=str)
    
    print(f"Master summary saved: {summary_path}\n")
    
    return all_results, dataset_summaries

def run_single_dataset_mistral(dataset_name: str):
    """
    Run a single dataset on both Mistral models (for testing)
    
    Usage:
        python run_mistral_experiments.py german_credit
    """
    
    print(f"\n{'='*70}")
    print(f"MISTRAL COMPARISON - {dataset_name.upper()}")
    print(f"{'='*70}\n")
    
    if dataset_name not in DATASET_REGISTRY:
        print(f"Error: Unknown dataset '{dataset_name}'")
        print(f"Available datasets: {', '.join(DATASET_REGISTRY.keys())}")
        return
    
    models = [ModelSize.MISTRAL_7B, ModelSize.MISTRAL_SMALL_24B]
    results = []
    exp_ids = []
    
    for model_size in models:
        print(f"\n{'*'*70}")
        print(f"* Running {model_size.name} on {dataset_name}")
        print(f"{'*'*70}\n")
        
        exp_config = create_experiment_config(
            dataset_key=dataset_name,
            model_size=model_size,
            api_provider=APIProvider.TOGETHER
        )
        
        try:
            result = run_single_experiment(exp_config)
            results.append(result)
            
            if result.get('status') == 'completed':
                exp_ids.append(result['experiment_id'])
                print(f"\nCompleted: {exp_config.experiment_id}")
                print(f"  Accuracy: {result.get('accuracy', 0):.4f}")
        except Exception as e:
            print(f"\nFailed: {e}")
            results.append({
                'experiment_id': exp_config.experiment_id,
                'status': 'failed',
                'error': str(e)
            })
    
    # Compare if both succeeded
    if len(exp_ids) == 2:
        print("\nComparing Mistral-7B vs Mistral-Small-24B...\n")
        compare_models(exp_ids)
    
    return results

def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == '--help':
            print("\nUsage:")
            print("  python run_mistral_experiments.py                    # Run all 12 Mistral experiments")
            print("  python run_mistral_experiments.py german_credit      # Test single dataset")
            print("  python run_mistral_experiments.py --help             # Show this help")
            return
        else:
            # Single dataset test
            run_single_dataset_mistral(command)
            return
    
    print("\n" + "#"*70)
    print("# MISTRAL MODELS FAIRNESS EVALUATION")
    print("# Running all 6 datasets on Mistral-7B and Mistral-Small-24B")
    print("#"*70)
    
    # Verify API key is set
    import os
    if not os.getenv("TOGETHER_API_KEY"):
        print("\nTOGETHER_API_KEY environment variable not set!")
        print("Please set it in your .env file or environment")
        return
    
    # Run all experiments
    results, summaries = run_all_datasets_mistral()
    
    print("\n" + "#"*70)
    print("# EXECUTION COMPLETE")
    print("#"*70)
    print("\nAll results saved in: fairness_experiments/")
    print("  - configs/           (experiment configurations)")
    print("  - predictions/       (detailed predictions)")
    print("  - analysis/          (fairness analysis)")
    print("  - summaries/         (fairness metrics & comparisons)")
    print("\nMaster summary saved with timestamp")
    print("#"*70 + "\n")

if __name__ == "__main__":
    main()