"""
Run GPT Experiments - All 6 Datasets on GPT-4o and GPT-4o-mini
fairness_experiments/run_gpt_experiments.py

Runs experiments in this order:
1. Adult Income: GPT-4o-mini -> GPT-4o
2. German Credit: GPT-4o-mini -> GPT-4o
3. COMPAS: GPT-4o-mini -> GPT-4o
4. Bank Marketing: GPT-4o-mini -> GPT-4o
5. Folktables: GPT-4o-mini -> GPT-4o
6. Diabetes Readmission: GPT-4o-mini -> GPT-4o
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

def run_all_datasets_gpt():
    """
    Run all 6 datasets on both GPT models (12 experiments total)
    
    For each dataset:
      1. Run GPT-4o-mini
      2. Run GPT-4o
      3. Compare the 2 models
      4. Move to next dataset
    """
    
    print("\n" + "="*70)
    print("GPT MODELS EVALUATION: ALL DATASETS")
    print("="*70)
    print("\nExecution Order:")
    print("  For each dataset:")
    print("    1. Run GPT-4o-mini")
    print("    2. Run GPT-4o")
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
    
    # GPT models
    models = [ModelSize.GPT_4O_MINI, ModelSize.GPT_4O]
    
    print(f"Total: {len(datasets)} datasets x {len(models)} models = {len(datasets) * len(models)} experiments\n")
    
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
        
        # Run both GPT models for this dataset
        for model_idx, model_size in enumerate(models, 1):
            print(f"\n{'*'*70}")
            print(f"* MODEL {model_idx}/{len(models)}: {model_size.name}")
            print(f"* Dataset: {dataset_name} ({dataset_idx}/{len(datasets)})")
            print(f"{'*'*70}\n")
            
            # Create experiment config
            exp_config = create_experiment_config(
                dataset_key=dataset_name,
                model_size=model_size,
                api_provider=APIProvider.OPENAI
            )
            
            # Run experiment
            try:
                result = run_single_experiment(exp_config)
                dataset_results.append(result)
                all_results.append(result)
                
                if result.get('status') == 'completed':
                    dataset_exp_ids.append(result['experiment_id'])
                    print(f"\n[SUCCESS] Completed: {exp_config.experiment_id}")
                    print(f"  Accuracy: {result.get('accuracy', 0):.4f}")
                    print(f"  Successful predictions: {result.get('successful_predictions', 0)}")
                else:
                    print(f"\n[FAILED] Failed: {exp_config.experiment_id}")
                    print(f"  Error: {result.get('error', 'Unknown')}")
                    
            except Exception as e:
                print(f"\n[ERROR] Exception during experiment: {e}")
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
            print(f"\n[COMPARE] Comparing {successful_count} models...\n")
            try:
                comparison_df = compare_models(dataset_exp_ids)
                print(f"\n[SUCCESS] Comparison complete for {dataset_name}")
            except Exception as e:
                print(f"\n[WARNING] Comparison failed: {e}")
        elif successful_count == 1:
            print("\n[WARNING] Only 1 model completed - skipping comparison")
        else:
            print("\n[WARNING] No models completed - skipping comparison")
        
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
        intermediate_path = base_dir / f"progress_gpt_after_{dataset_name}.json"
        with open(intermediate_path, 'w') as f:
            json.dump({
                'completed_datasets': dataset_idx,
                'total_datasets': len(datasets),
                'current_dataset': dataset_name,
                'timestamp': datetime.now().isoformat(),
                'dataset_summaries': dataset_summaries
            }, f, indent=2, default=str)
        
        print(f"[SAVED] Progress saved: {intermediate_path}\n")
    
    # Final summary
    print("\n" + "#"*70)
    print("# ALL GPT EXPERIMENTS COMPLETE!")
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
        'execution_strategy': 'sequential_by_dataset_gpt',
        'api_provider': 'openai',
        'models': [m.value for m in models],
        'total_experiments': len(all_results),
        'completed': total_completed,
        'failed': total_failed,
        'dataset_summaries': dataset_summaries,
        'all_results': all_results
    }
    
    summary_path = base_dir / f"master_summary_gpt_{timestamp.replace(':', '-').replace('.', '_')}.json"
    with open(summary_path, 'w') as f:
        json.dump(master_summary, f, indent=2, default=str)
    
    print(f"[SUCCESS] Master summary saved: {summary_path}\n")
    
    return all_results, dataset_summaries

def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("\nUsage:")
        print("  python run_gpt_experiments.py              # Run all 12 GPT experiments")
        print("  python run_gpt_experiments.py --help       # Show this help")
        return
    
    print("\n" + "#"*70)
    print("# GPT MODELS FAIRNESS EVALUATION")
    print("# Running all 6 datasets on GPT-4o-mini and GPT-4o")
    print("#"*70)
    
    # Verify API key is set
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[ERROR] OPENAI_API_KEY environment variable not set!")
        print("Please set it in your .env file or environment")
        return
    
    # Run all experiments
    results, summaries = run_all_datasets_gpt()
    
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