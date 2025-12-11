"""
Run All Experiments - One Dataset at a Time
fairness_experiments/run_sequential_by_dataset.py

Runs experiments in this order:
1. Adult Income: 8B -> 70B -> 405B
2. German Credit: 8B -> 70B -> 405B
3. COMPAS: 8B -> 70B -> 405B
... and so on for all 9 datasets
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

def run_all_datasets_sequentially():
    """
    Run all 27 experiments, one dataset at a time
    
    For each dataset:
      1. Run 8B model
      2. Run 70B model  
      3. Run 405B model
      4. Compare the 3 models
      5. Move to next dataset
    """
    
    print("\n" + "="*70)
    print("SEQUENTIAL EXECUTION: ONE DATASET AT A TIME")
    print("="*70)
    print("\nExecution Order:")
    print("  For each dataset:")
    print("    1. Run 8B model")
    print("    2. Run 70B model")
    print("    3. Run 405B model")
    print("    4. Compare results")
    print("    5. Move to next dataset")
    print("\n" + "="*70 + "\n")
    
    # Create output structure
    base_dir = create_output_structure()
    
    # Get all datasets (excluding the alias)
    #datasets = [k for k in DATASET_REGISTRY.keys() if k != 'heritage_health']

    datasets = ['diabetes_readmission', 'folktables']
    
    # All models in order
    models = [ModelSize.LLAMA_405B]
    
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
        
        # Run all 3 models for this dataset
        for model_idx, model_size in enumerate(models, 1):
            print(f"\n{'*'*70}")
            print(f"* MODEL {model_idx}/3: {model_size.name}")
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
                    print(f"\n✓ Completed: {exp_config.experiment_id}")
                    print(f"  Accuracy: {result.get('accuracy', 0):.4f}")
                    print(f"  Successful predictions: {result.get('successful_predictions', 0)}")
                else:
                    print(f"\n✗ Failed: {exp_config.experiment_id}")
                    print(f"  Error: {result.get('error', 'Unknown')}")
                    
            except Exception as e:
                print(f"\n✗ Exception during experiment: {e}")
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
        print(f"  Models completed: {successful_count}/3")
        print(f"  Models failed: {failed_count}/3")
        
        if successful_count > 1:
            print(f"\n🔍 Comparing {successful_count} models...\n")
            try:
                comparison_df = compare_models(dataset_exp_ids)
                print(f"\n✓ Comparison complete for {dataset_name}")
            except Exception as e:
                print(f"\n⚠️  Comparison failed: {e}")
        elif successful_count == 1:
            print("\n⚠️  Only 1 model completed - skipping comparison")
        else:
            print("\n⚠️  No models completed - skipping comparison")
        
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
        intermediate_path = base_dir / f"progress_after_{dataset_name}.json"
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
    print("# ALL EXPERIMENTS COMPLETE!")
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
        status = f"{summary['completed_models']}/3 models"
        print(f"{dataset_name:25s} - {status}")
    print("="*70 + "\n")
    
    # Save final master summary
    timestamp = datetime.now().isoformat()
    master_summary = {
        'execution_timestamp': timestamp,
        'execution_strategy': 'sequential_by_dataset',
        'total_experiments': len(all_results),
        'completed': total_completed,
        'failed': total_failed,
        'dataset_summaries': dataset_summaries,
        'all_results': all_results
    }
    
    summary_path = base_dir / f"master_summary_{timestamp.replace(':', '-').replace('.', '_')}.json"
    with open(summary_path, 'w') as f:
        json.dump(master_summary, f, indent=2, default=str)
    
    print(f"✓ Master summary saved: {summary_path}\n")
    
    return all_results, dataset_summaries

def resume_from_dataset(start_dataset: str):
    """
    Resume execution from a specific dataset
    
    Usage:
        resume_from_dataset('compas')  # Start from COMPAS, skip completed ones
    """
    datasets = [k for k in DATASET_REGISTRY.keys() if k != 'heritage_health']
    
    if start_dataset not in datasets:
        print(f"Error: Unknown dataset '{start_dataset}'")
        print(f"Available datasets: {', '.join(datasets)}")
        return
    
    start_idx = datasets.index(start_dataset)
    remaining_datasets = datasets[start_idx:]
    
    print(f"\n{'='*70}")
    print(f"RESUMING FROM: {start_dataset}")
    print(f"{'='*70}")
    print(f"\nSkipping first {start_idx} datasets")
    print(f"Running {len(remaining_datasets)} remaining datasets:")
    for d in remaining_datasets:
        print(f"  - {d}")
    print(f"\n{'='*70}\n")
    
    # Temporarily replace registry
    from config import DATASET_REGISTRY as original_registry
    
    # Run only remaining datasets
    # (Implementation would modify the main function to accept dataset list)
    # For now, just inform user
    print("To resume, edit run_sequential_by_dataset.py:")
    print(f"  Change: datasets = [k for k in DATASET_REGISTRY.keys() ...]")
    print(f"  To:     datasets = {remaining_datasets}")

def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == '--resume' and len(sys.argv) > 2:
            resume_from_dataset(sys.argv[2])
            return
        elif command == '--help':
            print("\nUsage:")
            print("  python run_sequential_by_dataset.py              # Run all 27 experiments")
            print("  python run_sequential_by_dataset.py --resume compas   # Resume from COMPAS")
            print("  python run_sequential_by_dataset.py --help             # Show this help")
            return
    
    # Run all experiments
    results, summaries = run_all_datasets_sequentially()
    
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