#!/usr/bin/env python3
"""
Comprehensive Stacking Pipeline for Molecular Fingerprint Analysis

This script implements a complete stacking pipeline with:
- Plug-and-play base learners
- Multiple meta-learners (LR and RF focus)
- Various ensemble strategies
- Comprehensive analysis and comparison

Usage:
    python main.py --config config.json
    or modify the configuration in this script directly
"""

import argparse
import json
import time
from typing import Dict, List, Any

# Import all modules
from .preprocessing import load_and_preprocess_data
from .base_learners import get_base_learner_config, BASE_LEARNER_CONFIGS
from .meta_learners import MetaLearnerTrainer
from .ensemble_strategies import MetaLearnerEnsemble
from .analysis import StackingAnalyzer, visualize_performance_comparison
from .utils import print_metrics, create_comparison_table

class StackingPipeline:
    """Main stacking pipeline class."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}
        
    def run_complete_pipeline(self):
        """Run the complete stacking pipeline."""
        start_time = time.time()
        
        print("="*80)
        print("STARTING COMPREHENSIVE STACKING PIPELINE")
        print("="*80)
        
        # Step 1: Load and preprocess data
        print("\nStep 1: Loading and preprocessing data...")
        X_stacked, metadata_df, fp_ranges = load_and_preprocess_data(
            self.config['data_path'],
            self.config['fingerprint_cols']
        )
        
        # Import disynthon split function (user should provide this)
        from .disynthon_split import disynthon_split_real
        
        # Step 2: Train base learners and generate meta-features
        print("\nStep 2: Training base learners...")
        base_learner_config = self.config['base_learner_config']
        
        trainer = MetaLearnerTrainer(
            base_learner_config=base_learner_config,
            cv_folds=self.config.get('cv_folds', 5),
            random_state=self.config.get('random_state', 42)
        )
        
        meta_features, labels, fingerprint_cols = trainer.train_base_learners_cv(
            X_stacked, metadata_df, fp_ranges, disynthon_split_real
        )
        
        # Step 3: Train meta-learners
        print("\nStep 3: Training meta-learners...")
        meta_learner_results = trainer.train_meta_learners(meta_features, labels, fingerprint_cols)
        
        # Step 4: Create ensemble strategies
        print("\nStep 4: Creating ensemble strategies...")
        ensemble_creator = MetaLearnerEnsemble(
            meta_learner_results, meta_features, labels,
            random_state=self.config.get('random_state', 42)
        )
        
        # Apply different ensemble strategies
        ensemble_strategies = self.config.get('ensemble_strategies', ['weighted', 'threshold', 'calibrated'])
        
        if 'weighted' in ensemble_strategies:
            ensemble_creator.create_weighted_ensemble()
            
        if 'threshold' in ensemble_strategies:
            ensemble_creator.create_threshold_based_hybrid()
            
        if 'calibrated' in ensemble_strategies:
            ensemble_creator.create_calibrated_ensemble()
            
        if 'robust' in ensemble_strategies:
            ensemble_creator.create_robust_ensemble()
        
        ensemble_results = ensemble_creator.get_all_ensemble_results()
        
        # Step 5: Comprehensive analysis
        print("\nStep 5: Performing comprehensive analysis...")
        analyzer = StackingAnalyzer(
            meta_learner_results, ensemble_results, labels, fingerprint_cols
        )
        
        summary_report = analyzer.generate_summary_report()
        
        # Step 6: Visualizations (optional)
        if self.config.get('enable_visualizations', True):
            print("\nStep 6: Generating visualizations...")
            # Combine all results for visualization
            all_results = {**meta_learner_results}
            for name, result in ensemble_results.items():
                all_results[f"Ensemble_{name}"] = result
            
            comparison_df = create_comparison_table(all_results)
            visualize_performance_comparison(comparison_df)
        
        # Store results
        self.results = {
            'meta_learner_results': meta_learner_results,
            'ensemble_results': ensemble_results,
            'summary_report': summary_report,
            'meta_features': meta_features,
            'labels': labels,
            'fingerprint_cols': fingerprint_cols,
            'config': self.config
        }
        
        # Final summary
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n{'='*80}")
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"{'='*80}")
        
        return self.results
    
    def save_results(self, output_path: str):
        """Save results to file."""
        import pickle
        
        # Remove non-serializable objects (models) for saving
        serializable_results = self.results.copy()
        
        # Remove model objects
        for result in serializable_results['meta_learner_results'].values():
            if 'model' in result:
                del result['model']
        
        with open(output_path, 'wb') as f:
            pickle.dump(serializable_results, f)
        
        print(f"Results saved to {output_path}")

def create_default_config():
    """Create default configuration."""
    return {
        'data_path': "Data/WDR91.parquet",
        'fingerprint_cols': ['ECFP4', 'ECFP6', 'FCFP4', 'FCFP6', 'TOPTOR', 'MACCS', 'RDK', 'AVALON'],
        'base_learner_config': get_base_learner_config('xgboost'),
        'ensemble_strategies': ['weighted', 'threshold', 'calibrated', 'robust'],
        'cv_folds': 5,
        'random_state': 42,
        'enable_visualizations': True
    }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run stacking pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--base-learner', type=str, choices=list(BASE_LEARNER_CONFIGS.keys()),
                        default='xgboost', help='Base learner to use')
    parser.add_argument('--output', type=str, help='Output path for results')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
        
        # Override base learner if specified
        if args.base_learner:
            config['base_learner_config'] = get_base_learner_config(args.base_learner)
    
    # Run pipeline
    pipeline = StackingPipeline(config)
    results = pipeline.run_complete_pipeline()
    
    # Save results if output path specified
    if args.output:
        pipeline.save_results(args.output)
    
    # Print final recommendations
    print("\n" + "="*60)
    print("FINAL RECOMMENDATIONS")
    print("="*60)
    
    # Find best approach
    comparison_df = results['summary_report']['comparison_table']
    if not comparison_df.empty and 'F1-Score' in comparison_df.columns:
        best_f1 = comparison_df['F1-Score'].idxmax()
        best_f1_score = comparison_df.loc[best_f1, 'F1-Score']
        print(f"Best F1-Score: {best_f1} ({best_f1_score:.4f})")
        
        if 'AUC-ROC' in comparison_df.columns:
            best_auc = comparison_df['AUC-ROC'].idxmax()
            best_auc_score = comparison_df.loc[best_auc, 'AUC-ROC']
            print(f"Best AUC-ROC: {best_auc} ({best_auc_score:.4f})")
    
    return results

# Example usage functions
def run_with_xgboost():
    """Example: Run pipeline with XGBoost base learner."""
    config = create_default_config()
    config['base_learner_config'] = get_base_learner_config('xgboost')
    
    pipeline = StackingPipeline(config)
    return pipeline.run_complete_pipeline()

def run_with_random_forest():
    """Example: Run pipeline with Random Forest base learner."""
    config = create_default_config()
    config['base_learner_config'] = get_base_learner_config('random_forest')
    
    pipeline = StackingPipeline(config)
    return pipeline.run_complete_pipeline()

def run_with_neural_net():
    """Example: Run pipeline with Neural Network base learner."""
    config = create_default_config()
    config['base_learner_config'] = get_base_learner_config('neural_net')
    
    pipeline = StackingPipeline(config)
    return pipeline.run_complete_pipeline()

# Quick comparison function
def compare_base_learners(base_learners=['xgboost', 'random_forest', 'neural_net']):
    """Compare different base learners."""
    results_comparison = {}
    
    for base_learner in base_learners:
        print(f"\n{'='*60}")
        print(f"TESTING BASE LEARNER: {base_learner.upper()}")
        print(f"{'='*60}")
        
        config = create_default_config()
        config['base_learner_config'] = get_base_learner_config(base_learner)
        
        pipeline = StackingPipeline(config)
        results = pipeline.run_complete_pipeline()
        
        results_comparison[base_learner] = results['summary_report']['comparison_table']
    
    # Create final comparison
    print(f"\n{'='*80}")
    print("BASE LEARNER COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    for base_learner, comparison_df in results_comparison.items():
        if not comparison_df.empty and 'F1-Score' in comparison_df.columns:
            best_approach = comparison_df['F1-Score'].idxmax()
            best_score = comparison_df.loc[best_approach, 'F1-Score']
            print(f"{base_learner:<15}: Best F1 = {best_score:.4f} ({best_approach})")
    
    return results_comparison

if __name__ == "__main__":
    main()