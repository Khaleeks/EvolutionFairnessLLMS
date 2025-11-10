"""
Command-Line Interface for Fairness Experiments
fairness_experiments/cli.py

Usage:
  python cli.py --dataset german_credit --models 8b 70b
  python cli.py --dataset all --models 8b
  python cli.py --list-datasets
  python cli.py --compare exp_id1 exp_id2 exp_id3
  
All experiments use Together AI
"""

import argparse
import sys
from config import APIProvider, ModelSize, DATASET_REGISTRY
from run_all_experiments import (
    run_single_dataset_all_models,
    run_dataset_subset,
    run_all_datasets_all_models
)
from fairness_analysis import compare_models

MODEL_MAP = {
    '8b': ModelSize.LLAMA_8B,
    '70b': ModelSize.LLAMA_70B,
    '405b': ModelSize.LLAMA_405B,
    'all': 'all'
}

API_MAP = {
    'together': APIProvider.TOGETHER
}

def list_datasets():
    """List all available datasets"""
    print("\n" + "="*70)
    print("AVAILABLE DATASETS")
    print("="*70 + "\n")
    
    for key, config in DATASET_REGISTRY.items():
        print(f"  {key:20s} - {config.name}")
        print(f"    Domain: {config.dataset_type.value}")
        print(f"    Sensitive: {', '.join(config.sensitive_features)}")
        print()

def run_experiments(args):
    """Run experiments based on CLI arguments"""
    
    # Determine models
    if args.models == ['all']:
        models = None  # Will use all models
    else:
        models = [MODEL_MAP[m] for m in args.models if m in MODEL_MAP]
    
    # Determine datasets
    if args.dataset == 'all':
        # Run all datasets
        if models is None:
            # Full matrix
            print("\n Running FULL MATRIX (27 experiments)")
            print(" This will take 2-3 hours!")
            print("Using Together AI for all models\n")
            
            confirm = input("Continue? (yes/no): ")
            if confirm.lower() != 'yes':
                print("Aborted.")
                return
            
            results = run_all_datasets_all_models(
                parallel=args.parallel,
                max_workers=args.workers
            )
        else:
            # All datasets, subset of models
            print(f"\n Running all datasets on {len(models)} model(s)")
            print("Using Together AI\n")
            results = run_dataset_subset(
                dataset_names=list(DATASET_REGISTRY.keys()),
                models=models
            )
    else:
        # Single dataset
        if args.dataset not in DATASET_REGISTRY:
            print(f"Unknown dataset: {args.dataset}")
            print("Use --list-datasets to see available datasets")
            return
        
        if models is None:
            # Single dataset, all models
            print(f"\n Running {args.dataset} on all 3 models")
            print("Using Together AI\n")
            results = run_single_dataset_all_models(dataset_name=args.dataset)
        else:
            # Single dataset, subset of models
            print(f"\n Running {args.dataset} on {len(models)} model(s)")
            print("Using Together AI\n")
            results = run_dataset_subset(
                dataset_names=[args.dataset],
                models=models
            )
    
    # Print summary
    if results:
        successful = sum(1 for r in results if r.get('status') == 'completed')
        failed = len(results) - successful
        
        print("\n" + "="*70)
        print("EXECUTION SUMMARY")
        print("="*70)
        print(f"Total: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print("="*70 + "\n")
        
        # Auto-compare if multiple experiments completed
        if successful > 1 and args.compare:
            exp_ids = [r['experiment_id'] for r in results if r.get('status') == 'completed']
            print("\n Generating comparison...\n")
            compare_models(exp_ids)

def compare_experiments(args):
    """Compare multiple experiments"""
    print(f"\n Comparing {len(args.exp_ids)} experiments\n")
    compare_models(args.exp_ids)

def main():
    parser = argparse.ArgumentParser(
        description='Multi-Model Fairness Evaluation Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all datasets
  python cli.py --list-datasets
  
  # Run German Credit on all 3 models
  python cli.py --dataset german_credit
  
  # Run German Credit on 8B and 70B onlyda
  python cli.py --dataset german_credit --models 8b 70b
  
  # Run all datasets on 8B only
  python cli.py --dataset all --models 8b
  
  # Run full matrix (27 experiments) in parallel
  python cli.py --dataset all --parallel --workers 3
  
  # Compare experiments
  python cli.py --compare exp_id1 exp_id2 exp_id3
  
  # Use Together AI instead of Groq
  python cli.py --dataset german_credit --api together
        """
    )
    
    # Main command group
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List datasets
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List all available datasets'
    )
    
    # Dataset selection
    parser.add_argument(
        '--dataset',
        type=str,
        default='german_credit',
        help='Dataset name or "all" for all datasets (default: german_credit)'
    )
    
    # Model selection
    parser.add_argument(
        '--models',
        nargs='+',
        default=['all'],
        choices=['8b', '70b', '405b', 'all'],
        help='Model sizes to run (default: all)'
    )
    
    # API provider
    parser.add_argument(
        '--api',
        type=str,
        default='groq',
        choices=['groq', 'together'],
        help='API provider (default: groq)'
    )
    
    # Parallel execution
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run experiments in parallel (use with caution for rate limits)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=2,
        help='Number of parallel workers (default: 2)'
    )
    
    # Auto-compare
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Automatically compare models after running'
    )
    
    # Manual comparison
    parser.add_argument(
        '--compare-exps',
        dest='exp_ids',
        nargs='+',
        help='Compare specific experiment IDs'
    )
    
    args = parser.parse_args()
    
    # Handle commands
    if args.list_datasets:
        list_datasets()
    elif args.exp_ids:
        compare_experiments(args)
    else:
        run_experiments(args)

if __name__ == '__main__':
    main()