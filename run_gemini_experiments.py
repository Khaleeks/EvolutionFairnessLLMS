"""
Run Gemini 2.5 Pro vs Flash Comparison
fairness_experiments/run_gemini_experiments.py

Runs all datasets on both Gemini models:
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

def run_gemini_comparison():
    """
    Run all datasets on both Gemini models
    
    For each dataset:
      1. Run Gemini Flash
      2. Run Gemini Pro
      3. Compare the 2 models
      4. Move to next dataset
    """
    
    print("\n" + "="*70)
    print("GEMINI MODEL COMPARISON")
    print("="*70)
    print("\nComparing:")
    print("  1. Gemini 2.5 Flash (gemini-2.5-flash)")
    print("  2. Gemini 2.5 Pro (gemini-2.5-pro)")
    print("\nAcross all datasets")
    print("\n" + "="*70 + "\n")
    
    # Create output structure
    base_dir = create_output_structure()
    
    # Get all datasets (excluding alias)
    datasets = [k for k in DATASET_REGISTRY.keys() if k != 'heritage_health']
    
    # Both Gemini models
    models = [ModelSize.GEMINI_25_FLASH, ModelSize.GEMINI_25_PRO]
    
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
        
        # Run both Gemini models for this dataset
        for model_idx, model_size in enumerate(models, 1):
            print(f"\n{'*'*70}")
            print(f"* MODEL {model_idx}/2: {model_size.name}")
            print(f"* Dataset: {dataset_name} ({dataset_idx}/{len(datasets)})")
            print(f"{'*'*70}\n")
            
            # Create experiment config
            exp_config = create_experiment_config(
                dataset_key=dataset_name,
                model_size=model_size,
                api_provider=APIProvider.GEMINI
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
        print(f"  Models completed: {successful_count}/2")
        print(f"  Models failed: {failed_count}/2")
        
        if successful_count == 2:
            print(f"\n📊 Comparing Gemini Flash vs Pro...\n")
            try:
                comparison_df = compare_models(dataset_exp_ids)
                print(f"\n✓ Comparison complete for {dataset_name}")
            except Exception as e:
                print(f"\n  Comparison failed: {e}")
        elif successful_count == 1:
            print("\n Only 1 model completed - skipping comparison")
        else:
            print("\n  No models completed - skipping comparison")
        
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
        intermediate_path = base_dir / f"gemini_progress_after_{dataset_name}.json"
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
    print("# ALL GEMINI EXPERIMENTS COMPLETE!")
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
        status = f"{summary['completed_models']}/2 models"
        print(f"{dataset_name:25s} - {status}")
    print("="*70 + "\n")
    
    # Save final master summary
    timestamp = datetime.now().isoformat()
    master_summary = {
        'execution_timestamp': timestamp,
        'execution_strategy': 'gemini_flash_vs_pro',
        'models': ['Gemini 2.5 Flash', 'Gemini 2.5 Pro'],
        'total_experiments': len(all_results),
        'completed': total_completed,
        'failed': total_failed,
        'dataset_summaries': dataset_summaries,
        'all_results': all_results
    }
    
    summary_path = base_dir / f"gemini_comparison_summary_{timestamp.replace(':', '-').replace('.', '_')}.json"
    with open(summary_path, 'w') as f:
        json.dump(master_summary, f, indent=2, default=str)
    
    print(f"✓ Master summary saved: {summary_path}\n")
    
    return all_results, dataset_summaries

def run_single_dataset_gemini(dataset_name: str):
    """
    Run a single dataset on both Gemini models (for testing)
    
    Usage:
        python run_gemini_experiments.py german_credit
    """
    
    print(f"\n{'='*70}")
    print(f"GEMINI COMPARISON - {dataset_name.upper()}")
    print(f"{'='*70}\n")
    
    if dataset_name not in DATASET_REGISTRY:
        print(f"Error: Unknown dataset '{dataset_name}'")
        print(f"Available datasets: {', '.join(DATASET_REGISTRY.keys())}")
        return
    
    models = [ModelSize.GEMINI_25_FLASH, ModelSize.GEMINI_25_PRO]
    results = []
    exp_ids = []
    
    for model_size in models:
        print(f"\n{'*'*70}")
        print(f"* Running {model_size.name} on {dataset_name}")
        print(f"{'*'*70}\n")
        
        exp_config = create_experiment_config(
            dataset_key=dataset_name,
            model_size=model_size,
            api_provider=APIProvider.GEMINI
        )
        
        try:
            result = run_single_experiment(exp_config)
            results.append(result)
            
            if result.get('status') == 'completed':
                exp_ids.append(result['experiment_id'])
                print(f"\n✓ Completed: {exp_config.experiment_id}")
                print(f"  Accuracy: {result.get('accuracy', 0):.4f}")
        except Exception as e:
            print(f"\n✗ Failed: {e}")
            results.append({
                'experiment_id': exp_config.experiment_id,
                'status': 'failed',
                'error': str(e)
            })
    
    # Compare if both succeeded
    if len(exp_ids) == 2:
        print("\n Comparing Flash vs Pro...\n")
        compare_models(exp_ids)
    
    return results

def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == '--help':
            print("\nUsage:")
            print("  python run_gemini_experiments.py                    # Run all datasets on both models")
            print("  python run_gemini_experiments.py german_credit      # Test single dataset")
            print("  python run_gemini_experiments.py --help             # Show this help")
            return
        else:
            # Single dataset test
            run_single_dataset_gemini(command)
            return
    
    # Run full comparison
    results, summaries = run_gemini_comparison()
    
    print("\n" + "#"*70)
    print("# GEMINI COMPARISON COMPLETE")
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