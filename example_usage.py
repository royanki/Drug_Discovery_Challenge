#!/usr/bin/env python3
"""
Example usage of the stacking pipeline.
"""

from stacking_pipeline import run_complete_pipeline, compare_base_learners
from stacking_pipeline import get_base_learner_config, create_default_config

# Example 1: Quick run with XGBoost
def example_quick_run():
    """Quick example with default settings."""
    
    data_path = "Data/WDR91.parquet"
    fingerprint_cols = ['ECFP4', 'ECFP6', 'FCFP4', 'FCFP6', 'TOPTOR', 'MACCS', 'RDK', 'AVALON']
    
    results = run_complete_pipeline(
        data_path=data_path,
        fingerprint_cols=fingerprint_cols,
        base_learner='xgboost'
    )
    
    return results

# Example 2: Custom base learner configuration
def example_custom_base_learner():
    """Example with custom base learner configuration."""
    
    # Custom XGBoost configuration
    custom_xgb_config = {
        'type': 'xgboost',
        'params': {
            'n_estimators': 200,  # More trees
            'max_depth': 8,       # Deeper trees
            'learning_rate': 0.05, # Slower learning
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
    }
    
    data_path = "Data/WDR91.parquet"
    fingerprint_cols = ['ECFP4', 'ECFP6', 'FCFP4', 'FCFP6']  # Subset of fingerprints
    
    results = run_complete_pipeline(
        data_path=data_path,
        fingerprint_cols=fingerprint_cols,
        base_learner=custom_xgb_config,
        ensemble_strategies=['weighted', 'calibrated'],  # Only these strategies
        cv_folds=3,  # Faster with fewer folds
        random_state=123
    )
    
    return results

# Example 3: Compare multiple base learners
def example_compare_base_learners():
    """Compare performance of different base learners."""
    
    results = compare_base_learners(['xgboost', 'random_forest', 'neural_net'])
    return results

# Example 4: Focus on specific fingerprints
def example_fingerprint_subset():
    """Example focusing on a subset of fingerprints."""
    
    data_path = "Data/WDR91.parquet"
    
    # Test different fingerprint combinations
    fingerprint_combinations = [
        ['ECFP4', 'ECFP6'],
        ['FCFP4', 'FCFP6'],
        ['ECFP4', 'FCFP4', 'MACCS'],
        ['ECFP4', 'ECFP6', 'FCFP4', 'FCFP6', 'TOPTOR', 'MACCS', 'RDK', 'AVALON']
    ]
    
    results_by_fp = {}
    
    for i, fp_combo in enumerate(fingerprint_combinations):
        print(f"\nTesting fingerprint combination {i+1}: {fp_combo}")
        
        results = run_complete_pipeline(
            data_path=data_path,
            fingerprint_cols=fp_combo,
            base_learner='xgboost',
            enable_visualizations=False  # Disable to speed up
        )
        
        # Extract best F1 score
        comparison_df = results['summary_report']['comparison_table']
        best_f1 = comparison_df['F1-Score'].max()
        best_approach = comparison_df['F1-Score'].idxmax()
        
        results_by_fp[f"Combo_{i+1}"] = {
            'fingerprints': fp_combo,
            'best_f1': best_f1,
            'best_approach': best_approach,
            'full_results': results
        }
    
    # Summary
    print("\n" + "="*60)
    print("FINGERPRINT COMBINATION COMPARISON")
    print("="*60)
    
    for combo_name, result in results_by_fp.items():
        fp_str = ", ".join(result['fingerprints'])
        print(f"{combo_name}: F1={result['best_f1']:.4f} ({result['best_approach']}) - {fp_str}")
    
    return results_by_fp

if __name__ == "__main__":
    # Run the quick example
    print("Running quick example...")
    results = example_quick_run()
    
    # Uncomment to run other examples
    # print("Running custom base learner example...")
    # custom_results = example_custom_base_learner()
    
    # print("Comparing base learners...")
    # comparison_results = example_compare_base_learners()
    
    # print("Testing fingerprint combinations...")
    # fp_results = example_fingerprint_subset()