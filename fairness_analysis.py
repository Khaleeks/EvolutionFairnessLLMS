"""
Fairness Analysis Module
fairness_experiments/fairness_analysis.py
"""

import pandas as pd
import json
from typing import Dict, Any
from pathlib import Path
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import ExperimentConfig, DATASET_REGISTRY

# ============================================================================
# FAIRNESS ANALYSIS
# ============================================================================

def perform_fairness_analysis(
    predictions_df: pd.DataFrame,
    exp_config: ExperimentConfig,
    output_paths: Dict[str, Path]
) -> Dict[str, Any]:
    """
    Perform comprehensive fairness analysis on predictions
    
    Args:
        predictions_df: DataFrame with columns matching output schema
        exp_config: Experiment configuration
        output_paths: Dictionary of output file paths
    
    Returns:
        Dictionary with fairness metrics and analysis results
    """
    
    print(f"\n{'='*70}")
    print(f"FAIRNESS ANALYSIS - {exp_config.experiment_id}")
    print(f"{'='*70}\n")
    
    # Filter to successful predictions only
    analysis_df = predictions_df[predictions_df['api_success'] == True].copy()
    
    if len(analysis_df) == 0:
        print("⚠ No successful predictions to analyze")
        return {'error': 'No successful predictions'}
    
    # Clean data
    mask = (
        (analysis_df['sensitive_feature'].notna()) &
        (analysis_df['sensitive_feature'] != 'None') &
        (analysis_df['prediction'].notna()) &
        (analysis_df['ground_truth'].notna())
    )
    
    analysis_df = analysis_df[mask].copy()
    
    print(f"Analyzing {len(analysis_df)} samples across {analysis_df['sensitive_feature'].nunique()} groups\n")
    
    # Save detailed analysis data
    analysis_df.to_csv(output_paths['analysis'], index=False)
    
    # Extract data for metrics
    y_true = analysis_df['ground_truth']
    y_pred = analysis_df['prediction']
    sensitive_features = analysis_df['sensitive_feature']
    
    # Define metrics
    metrics = {
        'accuracy': accuracy_score,
        'precision': lambda y_t, y_p: precision_score(y_t, y_p, average='weighted', zero_division=0),
        'recall': lambda y_t, y_p: recall_score(y_t, y_p, average='weighted', zero_division=0),
        'f1_score': lambda y_t, y_p: f1_score(y_t, y_p, average='weighted', zero_division=0)
    }
    
    # Compute MetricFrame
    try:
        mf = MetricFrame(
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features
        )
    except Exception as e:
        print(f"⚠ MetricFrame computation failed: {e}")
        return {'error': f'MetricFrame error: {str(e)}'}
    
    # Print overall metrics
    print("OVERALL METRICS")
    print("-" * 70)
    for metric_name, value in mf.overall.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Print metrics by group
    print("\n\nMETRICS BY SENSITIVE FEATURE")
    print("-" * 70)
    print(mf.by_group.to_string())
    
    # Print fairness metrics - differences
    print("\n\nFAIRNESS METRICS - DIFFERENCES")
    print("-" * 70)
    for metric_name, value in mf.difference().items():
        print(f"{metric_name}: {value:+.4f}")
    
    # Print fairness metrics - ratios
    print("\n\nFAIRNESS METRICS - RATIOS")
    print("-" * 70)
    for metric_name, value in mf.ratio().items():
        print(f"{metric_name}: {value:.4f}")
    
    # Group performance summary
    print("\n\nGROUP PERFORMANCE SUMMARY")
    print("-" * 70)
    group_summary = {}
    
    for metric_name in metrics.keys():
        group_perf = mf.by_group[metric_name]
        worst_group = group_perf.idxmin()
        best_group = group_perf.idxmax()
        worst_val = group_perf.min()
        best_val = group_perf.max()
        gap = best_val - worst_val
        
        print(f"\n{metric_name}:")
        print(f"  Best group: {best_group} = {best_val:.4f}")
        print(f"  Worst group: {worst_group} = {worst_val:.4f}")
        print(f"  Gap: {gap:.4f}")
        
        group_summary[metric_name] = {
            'best_group': str(best_group),
            'best_value': float(best_val),
            'worst_group': str(worst_group),
            'worst_value': float(worst_val),
            'gap': float(gap)
        }
    
    # Binary fairness metrics (if applicable)
    print("\n\nBINARY FAIRNESS METRICS")
    print("-" * 70)
    
    dataset_cfg = DATASET_REGISTRY[exp_config.dataset_name]
    positive_class = dataset_cfg.positive_class
    
    unique_labels = sorted(y_true.unique())
    binary_metrics = {}
    
    if len(unique_labels) == 2 and positive_class in unique_labels:
        y_true_binary = (y_true == positive_class).astype(int)
        y_pred_binary = (y_pred == positive_class).astype(int)
        
        try:
            dp_diff = demographic_parity_difference(
                y_true_binary, y_pred_binary,
                sensitive_features=sensitive_features
            )
            eo_diff = equalized_odds_difference(
                y_true_binary, y_pred_binary,
                sensitive_features=sensitive_features
            )
            
            print(f"Demographic Parity Difference: {dp_diff:+.4f}")
            print(f"Equalized Odds Difference: {eo_diff:+.4f}")
            
            binary_metrics = {
                'demographic_parity_difference': float(dp_diff),
                'equalized_odds_difference': float(eo_diff)
            }
        except Exception as e:
            print(f"Could not compute binary fairness metrics: {e}")
    else:
        print("Skipped (not applicable for this task)")
    
    # Group sizes
    print("\n\nGROUP SIZES")
    print("-" * 70)
    group_sizes = {}
    
    for group, size in sensitive_features.value_counts().items():
        pct = size / len(sensitive_features) * 100
        print(f"{group}: {size} samples ({pct:.1f}%)")
        group_sizes[str(group)] = {
            'count': int(size),
            'percentage': float(pct)
        }
    
    # Detailed class distributions
    print("\n\nDETAILED CLASS DISTRIBUTIONS")
    print("-" * 70)
    
    class_distributions = {}
    
    for group in sorted(sensitive_features.unique()):
        group_mask = sensitive_features == group
        group_true = y_true[group_mask]
        group_pred = y_pred[group_mask]
        
        print(f"\nGroup: {group} (n={sum(group_mask)})")
        
        # Ground truth distribution
        print("  Ground Truth:")
        gt_dist = {}
        for label in sorted(group_true.unique()):
            count = sum(group_true == label)
            pct = count / len(group_true) * 100 if len(group_true) > 0 else 0
            print(f"    {label}: {count} ({pct:.1f}%)")
            gt_dist[str(label)] = {'count': int(count), 'percentage': float(pct)}
        
        # Prediction distribution
        print("  Predictions:")
        pred_dist = {}
        for label in sorted(group_pred.unique()):
            count = sum(group_pred == label)
            pct = count / len(group_pred) * 100 if len(group_pred) > 0 else 0
            print(f"    {label}: {count} ({pct:.1f}%)")
            pred_dist[str(label)] = {'count': int(count), 'percentage': float(pct)}
        
        # Group accuracy
        group_accuracy = accuracy_score(group_true, group_pred)
        print(f"  Accuracy: {group_accuracy:.4f}")
        
        class_distributions[str(group)] = {
            'ground_truth': gt_dist,
            'predictions': pred_dist,
            'accuracy': float(group_accuracy)
        }
    
    # Create fairness summary
    fairness_summary = {
        'experiment_id': exp_config.experiment_id,
        'timestamp': exp_config.timestamp,
        'dataset': exp_config.dataset_name,
        'model': exp_config.model_name,
        'model_size': exp_config.model_size.value,
        'api_provider': exp_config.api_provider.value,
        'total_samples': len(analysis_df),
        'num_groups': int(analysis_df['sensitive_feature'].nunique()),
        'overall_metrics': {k: float(v) for k, v in mf.overall.items()},
        'fairness_differences': {k: float(v) for k, v in mf.difference().items()},
        'fairness_ratios': {k: float(v) for k, v in mf.ratio().items()},
        'group_summary': group_summary,
        'binary_metrics': binary_metrics,
        'group_sizes': group_sizes,
        'class_distributions': class_distributions,
        'metrics_by_group': mf.by_group.to_dict()
    }
    
    # Save fairness summary
    with open(output_paths['fairness_summary'], 'w') as f:
        json.dump(fairness_summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print("FAIRNESS ANALYSIS COMPLETE")
    print(f"{'='*70}\n")
    
    return fairness_summary

# ============================================================================
# CROSS-MODEL COMPARISON
# ============================================================================

def compare_models(experiment_ids: list, base_dir: str = "fairness_experiments"):
    """
    Compare fairness metrics across multiple models for the same dataset
    
    Args:
        experiment_ids: List of experiment IDs to compare
        base_dir: Base directory for experiment outputs
    """
    
    base_path = Path(base_dir)
    summaries = []
    
    print(f"\n{'='*70}")
    print("CROSS-MODEL FAIRNESS COMPARISON")
    print(f"{'='*70}\n")
    
    # Load all fairness summaries
    for exp_id in experiment_ids:
        summary_files = list(base_path.glob(f"summaries/*{exp_id}*fairness.json"))
        
        if summary_files:
            with open(summary_files[0], 'r') as f:
                summary = json.load(f)
                summaries.append(summary)
    
    if not summaries:
        print("No summaries found for comparison")
        return
    
    # Create comparison DataFrame
    comparison_data = []
    
    for summary in summaries:
        row = {
            'experiment_id': summary['experiment_id'],
            'model': summary['model_size'],
            'dataset': summary['dataset'],
            'accuracy': summary['overall_metrics']['accuracy'],
            'accuracy_difference': summary['fairness_differences']['accuracy'],
            'accuracy_ratio': summary['fairness_ratios']['accuracy'],
        }
        
        if 'binary_metrics' in summary and summary['binary_metrics']:
            row['demographic_parity_diff'] = summary['binary_metrics'].get('demographic_parity_difference', None)
            row['equalized_odds_diff'] = summary['binary_metrics'].get('equalized_odds_difference', None)
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("ACCURACY COMPARISON")
    print("-" * 70)
    print(comparison_df[['model', 'accuracy', 'accuracy_difference', 'accuracy_ratio']].to_string(index=False))
    
    if 'demographic_parity_diff' in comparison_df.columns:
        print("\n\nFAIRNESS COMPARISON")
        print("-" * 70)
        print(comparison_df[['model', 'demographic_parity_diff', 'equalized_odds_diff']].to_string(index=False))
    
    # Save comparison
    comparison_path = base_path / "summaries" / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    
    print(f"\n✓ Comparison saved: {comparison_path}")
    print(f"{'='*70}\n")
    
    return comparison_df