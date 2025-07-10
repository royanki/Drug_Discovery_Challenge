#!/usr/bin/env python3
"""
Memory-Efficient Stacking Pipeline

This pipeline:
1. Trains base learners on individual fingerprints
2. Collects predictions to create meta-features
3. Trains meta-learners on these features
4. Combines meta-learners with different strategies
5. Compares all approaches
"""

import argparse
import json
import time
import numpy as np
import pandas as pd
import os
import gc
from typing import Dict, List, Any

from .preprocessing import preprocess_single_fingerprint_numpy
from .base_learners import get_base_learner_config, BaseLearnerFactory
from .meta_learners import MetaLearnerTrainer
from .ensemble_strategies import MetaLearnerEnsemble
from .analysis import StackingAnalyzer, visualize_performance_comparison
from .utils import print_metrics, create_comparison_table

class MemoryEfficientStacking:
    """Memory-efficient stacking implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}
        
    def generate_meta_features(self):
        """
        Generate meta-features by training individual base learners on each fingerprint.
        This is memory-efficient as it processes one fingerprint at a time.
        """
        data_path = self.config['data_path']
        fingerprint_cols = self.config['fingerprint_cols']
        base_learner_config = self.config['base_learner_config']
        cv_folds = self.config.get('cv_folds', 5)
        random_state = self.config.get('random_state', 42)
        
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        # Import disynthon split function (user should provide this)
        from .disynthon_split import disynthon_split_real
        
        # Storage for all meta-features
        all_meta_features_train = []
        all_meta_features_val = []
        all_meta_features_test = []
        
        # Track labels and indices
        labels = None
        fingerprint_meta_cols = []  # Keep track of what each meta-feature represents
        
        # Process one fingerprint at a time
        for fp_idx, fingerprint_col in enumerate(fingerprint_cols):
            print(f"\n{'-'*50}")
            print(f"Processing fingerprint {fp_idx+1}/{len(fingerprint_cols)}: {fingerprint_col}")
            print(f"{'-'*50}")
            
            # Load only necessary columns
            print(f"Loading {fingerprint_col}...")
            df = pd.read_parquet(data_path, columns=['BB1_ID', 'BB2_ID', fingerprint_col, 'LABEL'])
            
            # Preprocess
            fp_matrix, metadata_df, info = preprocess_single_fingerprint_numpy(df, fingerprint_col)
            
            # Apply split (only on first fingerprint)
            if fp_idx == 0:
                print("Splitting data...")
                train_meta, val_meta, test_meta = disynthon_split_real(metadata_df.reset_index())
                
                # Get indices
                train_indices = train_meta.index
                val_indices = val_meta.index
                test_indices = test_meta.index
                
                # Extract labels (same for all fingerprints)
                y_train = train_meta['LABEL'].values
                y_val = val_meta['LABEL'].values
                y_test = test_meta['LABEL'].values
                
                # Store labels
                labels = {
                    'train': y_train,
                    'val': y_val,
                    'test': y_test
                }
            
            # Split fingerprint data
            X_train_fp = fp_matrix[train_indices]
            X_val_fp = fp_matrix[val_indices]
            X_test_fp = fp_matrix[test_indices]
            
            print(f"Training base learner for {fingerprint_col}...")
            
            # For out-of-fold predictions
            oof_predictions = np.zeros(len(X_train_fp))
            val_predictions = []
            test_predictions = []
            
            for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(skf.split(X_train_fp, y_train)):
                print(f"  Fold {fold_idx+1}/{cv_folds}...")
                
                # Create model
                model = BaseLearnerFactory.create_base_learner(base_learner_config)
                
                # Train on fold
                model.fit(X_train_fp[fold_train_idx], y_train[fold_train_idx])
                
                # Get predictions
                oof_predictions[fold_val_idx] = model.predict_proba(X_train_fp[fold_val_idx])[:, 1]
                val_predictions.append(model.predict_proba(X_val_fp)[:, 1])
                test_predictions.append(model.predict_proba(X_test_fp)[:, 1])
            
            # Average validation and test predictions
            val_pred_avg = np.mean(val_predictions, axis=0)
            test_pred_avg = np.mean(test_predictions, axis=0)
            
            # Store meta-features for this fingerprint
            all_meta_features_train.append(oof_predictions)
            all_meta_features_val.append(val_pred_avg)
            all_meta_features_test.append(test_pred_avg)
            
            # Track meta-feature name
            fingerprint_meta_cols.append(f"{fingerprint_col}_{self.config['base_learner_config']['type']}")
            
            print(f"Completed {fingerprint_col} - adding meta-feature")
            
            # Explicitly delete large objects to free memory
            del fp_matrix, X_train_fp, X_val_fp, X_test_fp
            del df, metadata_df
            gc.collect()
        
        # Stack all meta-features
        X_meta_train = np.column_stack(all_meta_features_train)
        X_meta_val = np.column_stack(all_meta_features_val)
        X_meta_test = np.column_stack(all_meta_features_test)
        
        print(f"\nGenerated meta-features:")
        print(f"  Train: {X_meta_train.shape}")
        print(f"  Validation: {X_meta_val.shape}")
        print(f"  Test: {X_meta_test.shape}")
        
        meta_features = {
            'train': X_meta_train,
            'val': X_meta_val,
            'test': X_meta_test
        }
        
        return meta_features, labels, fingerprint_meta_cols
    
    def find_best_ensemble_strategy(self, meta_learner_results, ensemble_results, labels, 
                                   primary_metric='F1-Score', secondary_metric='AUC-ROC'):
        """
        Compare all strategies and determine the best performing one.
        
        Parameters:
        -----------
        meta_learner_results : dict
            Results from individual meta-learners
        ensemble_results : dict
            Results from ensemble strategies
        labels : dict
            True labels for evaluation
        primary_metric : str
            Primary metric for comparison (default: 'F1-Score')
        secondary_metric : str
            Secondary metric for tie-breaking (default: 'AUC-ROC')
        
        Returns:
        --------
        dict
            Information about the best strategy
        """
        print(f"Comparing strategies using {primary_metric} (primary) and {secondary_metric} (secondary)...")
        
        # Combine all results for comparison
        all_strategies = {}
        
        # Add individual meta-learners
        for name, result in meta_learner_results.items():
            all_strategies[f"MetaLearner_{name}"] = result
        
        # Add ensemble strategies
        for name, result in ensemble_results.items():
            all_strategies[f"Ensemble_{name}"] = result
        
        best_score = -1
        best_strategy = None
        best_info = {}
        
        strategy_comparison = {}
        
        for strategy_name, result in all_strategies.items():
            metrics = result['metrics']
            primary_score = metrics.get(primary_metric, 0)
            secondary_score = metrics.get(secondary_metric, 0)
            
            strategy_comparison[strategy_name] = {
                'primary_score': primary_score,
                'secondary_score': secondary_score,
                'all_metrics': metrics
            }
            
            # Check if this is the best strategy
            if (primary_score > best_score or 
                (primary_score == best_score and secondary_score > best_info.get('secondary_score', 0))):
                best_score = primary_score
                best_strategy = strategy_name
                best_info = {
                    'strategy_name': strategy_name,
                    'primary_score': primary_score,
                    'secondary_score': secondary_score,
                    'metrics': metrics,
                    'weights': self._extract_weights(strategy_name, result)
                }
        
        print(f"\nBest strategy: {best_strategy}")
        print(f"  {primary_metric}: {best_score:.4f}")
        print(f"  {secondary_metric}: {best_info['secondary_score']:.4f}")
        
        return {
            'best_strategy_name': best_strategy,
            'best_strategy_metrics': best_info,
            'all_strategy_comparison': strategy_comparison
        }

    def _extract_weights(self, strategy_name, result):
        """Extract weights from a strategy result."""
        if 'weights' in result:
            return result['weights']
        elif strategy_name.startswith('MetaLearner_'):
            learner_name = strategy_name.replace('MetaLearner_', '')
            return {learner_name: 1.0}
        else:
            # Dynamic fallback based on available meta-learners
            if hasattr(self, 'results') and 'meta_learner_results' in self.results:
                available_learners = list(self.results['meta_learner_results'].keys())
                n_learners = len(available_learners)
                equal_weight = 1.0 / n_learners
                return {learner: equal_weight for learner in available_learners}
            else:
                # Default set with new neural networks
                default_learners = [
                    'LogisticRegression', 'RandomForest',
                    'MLP_Compact', 'MLP_Deep', 'MLP_Wide', 'MLP_Balanced'
                ]
                equal_weight = 1.0 / len(default_learners)
                return {learner: equal_weight for learner in default_learners}
    
    def run_pipeline(self):
        """Run the complete pipeline."""
        start_time = time.time()
        
        print("="*80)
        print("MEMORY-EFFICIENT STACKING PIPELINE")
        print("="*80)
        
        print(f"\nConfiguration:")
        print(f"  Base learner: {self.config['base_learner_config']['type']}")
        print(f"  Fingerprints: {len(self.config['fingerprint_cols'])}")
        print(f"  CV folds: {self.config.get('cv_folds', 5)}")
        
        # Step 1: Generate meta-features
        print("\nStep 1: Generating meta-features from individual fingerprints...")
        meta_features, labels, fingerprint_meta_cols = self.generate_meta_features()
        
        # Step 2: Train meta-learners
        print("\nStep 2: Training meta-learners...")
        random_state = self.config.get('random_state', 42)
        
        trainer = MetaLearnerTrainer(
            self.config['base_learner_config'],
            self.config.get('cv_folds', 5),
            random_state
        )
        
        # Use the pre-generated meta-features directly
        meta_learner_results = trainer.train_meta_learners(meta_features, labels, fingerprint_meta_cols)
        
        # Step 3: Apply ensemble strategies
        print("\nStep 3: Applying meta-learner combination strategies...")
        ensemble_creator = MetaLearnerEnsemble(
            meta_learner_results, meta_features, labels, random_state
        )
        
        # Apply different ensemble strategies
        ensemble_strategies = self.config.get('ensemble_strategies', ['weighted', 'threshold', 'calibrated','neural_network', 'all_meta'])
        
        if 'weighted' in ensemble_strategies:
            ensemble_creator.create_weighted_ensemble()
            
        if 'threshold' in ensemble_strategies:
            ensemble_creator.create_threshold_based_hybrid()
            
        if 'calibrated' in ensemble_strategies:
            ensemble_creator.create_calibrated_ensemble()
            
        if 'robust' in ensemble_strategies:
            ensemble_creator.create_robust_ensemble()

        if 'neural_network' in ensemble_strategies:
            ensemble_creator.create_neural_network_ensemble()

        if 'all_meta' in ensemble_strategies:
            ensemble_creator.create_all_meta_ensemble()
        
        ensemble_results = ensemble_creator.get_all_ensemble_results()
        
        # Step 4: Analysis
        print("\nStep 4: Analyzing results...")
        analyzer = StackingAnalyzer(
            meta_learner_results, ensemble_results, labels, fingerprint_meta_cols
        )
        
        summary_report = analyzer.generate_summary_report()
        
        # NEW: Step 5: Determine best ensemble strategy
        print("\nStep 5: Determining best ensemble strategy...")
        
        # Compare all strategies and find the best one
        best_strategy_info = self.find_best_ensemble_strategy(
            meta_learner_results, ensemble_results, labels
        )
        
        # Step 6: Visualizations (optional)
        if self.config.get('enable_visualizations', True):
            print("\nStep 6: Generating visualizations...")
            # Combine all results for visualization
            all_results = {**meta_learner_results}
            for name, result in ensemble_results.items():
                all_results[f"Ensemble_{name}"] = result
            
            comparison_df = create_comparison_table(all_results)

            # Create a save path based on base learner type
            if self.config.get('save_visualizations', False):
                import os
                vis_dir = "results/visualizations"
                os.makedirs(vis_dir, exist_ok=True)

                base_learner_type = self.config['base_learner_config']['type']
                metrics_save_path = f"{vis_dir}/{base_learner_type}_metrics_comparison.png"
                feature_save_path = f"{vis_dir}/{base_learner_type}_feature_importance.png"

                # Visualize with save paths
                visualize_performance_comparison(comparison_df, metrics_save_path)

                # Call feature importance with save path - needs analyzer object
                analyzer.analyze_feature_importance(feature_save_path)
            else:
                visualize_performance_comparison(comparison_df)
        
        # Store results (UPDATED to include best strategy info)
        self.results = {
            'meta_learner_results': meta_learner_results,
            'ensemble_results': ensemble_results,
            'summary_report': summary_report,
            'meta_features': meta_features,
            'labels': labels,
            'fingerprint_meta_cols': fingerprint_meta_cols,
            'config': self.config,
            'best_strategy_info': best_strategy_info  # NEW: Add this
        }
        
        # Final timing
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
        
        # NEW: Add CV summary with best strategy information
        best_strategy_info = serializable_results.get('best_strategy_info', {})
        
        serializable_results['cv_summary'] = {
            'best_ensemble_strategy': best_strategy_info.get('best_strategy_name', 'Unknown'),
            'optimal_weights': best_strategy_info.get('best_strategy_metrics', {}).get('weights', {}),
            'fingerprint_meta_cols': serializable_results['fingerprint_meta_cols'],
            'config': self.config,
            'best_metrics': best_strategy_info.get('best_strategy_metrics', {}).get('metrics', {}),
            'strategy_comparison': best_strategy_info.get('all_strategy_comparison', {})
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(serializable_results, f)
        
        print(f"Results saved to {output_path}")
        print(f"Best strategy: {serializable_results['cv_summary']['best_ensemble_strategy']}")

def create_default_config():
    """Create default configuration."""
    return {
        'data_path': "Data/WDR91.parquet",
        'fingerprint_cols': ['ECFP4', 'ECFP6', 'FCFP4', 'FCFP6', 'TOPTOR', 'MACCS', 'RDK', 'AVALON','ATOMPAIR'],
        'base_learner_config': get_base_learner_config('xgboost'),
        'ensemble_strategies': ['weighted', 'threshold', 'calibrated', 'robust'],
        'cv_folds': 5,
        'random_state': 42,
        'enable_visualizations': True
    }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Memory-efficient stacking pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--base-learner', type=str, choices=['xgboost', 'random_forest', 'neural_net'],
                        default='xgboost', help='Base learner to use')
    parser.add_argument('--output', type=str, help='Output path for results')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualizations')
    parser.add_argument('--save-vis', action='store_true', help='Save visualizations to files')
    
    # NEW: Add final training arguments
    parser.add_argument('--final-train', action='store_true', 
                       help='Skip CV and run final training on complete data using saved CV results')
    parser.add_argument('--cv-results', type=str, 
                       help='Path to CV results pickle file (required for --final-train)')
    parser.add_argument('--test-data', type=str, 
                       help='Path to test dataset for final predictions')
    parser.add_argument('--submission-dir', type=str, default='results/submissions',
                       help='Directory to save submission files')
    
    args = parser.parse_args()
    
    # NEW: Handle final training workflow
    if args.final_train:
        return run_final_training_workflow(args)
    else:
        # Existing CV workflow
        return run_cv_workflow(args)

def run_final_training_workflow(args):
    """Run the final training workflow using CV results."""
    print("="*80)
    print("FINAL TRAINING AND PREDICTION WORKFLOW")
    print("="*80)
    
    # Validate required arguments
    if not args.cv_results:
        raise ValueError("--cv-results path is required when using --final-train")
    
    if not args.test_data:
        raise ValueError("--test-data path is required when using --final-train")
    
    if not os.path.exists(args.cv_results):
        raise FileNotFoundError(f"CV results file not found: {args.cv_results}")
    
    if not os.path.exists(args.test_data):
        raise FileNotFoundError(f"Test data file not found: {args.test_data}")
    
    # Import the final training class
    from .final_train_predict import FinalTrainPredict
    
    print(f"Loading CV results from: {args.cv_results}")
    print(f"Test data path: {args.test_data}")
    print(f"Submission directory: {args.submission_dir}")
    
    # Create final trainer
    final_trainer = FinalTrainPredict(
        cv_results_path=args.cv_results,
        config_overrides={
            'test_data_path': args.test_data,
            'submission_dir': args.submission_dir
        }
    )
    
    # Run complete pipeline
    results = final_trainer.run_complete_pipeline(
        test_data_path=args.test_data,
        output_dir=args.submission_dir
    )
    
    # Print final summary
    print_final_summary(results)
    
    # Save final results if output path specified
    if args.output:
        final_trainer.save_final_results(args.output)
    
    return results

def run_cv_workflow(args):
    """Run the existing CV workflow."""
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
        
    # Override base learner if specified
    if args.base_learner:
        config['base_learner_config'] = get_base_learner_config(args.base_learner)
    
    # Override folds if specified
    if args.folds:
        config['cv_folds'] = args.folds
    
    # Override visualizations if specified
    if args.no_vis:
        config['enable_visualizations'] = False
    
    # Override visualizations if specified
    if args.save_vis:
        config['save_visualizations'] = True

    # Run pipeline
    pipeline = MemoryEfficientStacking(config)
    results = pipeline.run_pipeline()
    
    # Save results if output path specified
    if args.output:
        pipeline.save_results(args.output)
    
    # Print final recommendations
    print_cv_recommendations(results)
    
    return results

def print_final_summary(results):
    """Print summary of final training results."""
    print("\n" + "="*60)
    print("FINAL TRAINING SUMMARY")
    print("="*60)
    
    # Model information
    base_models = results['models']['base_models']
    meta_models = results['models']['meta_models']
    
    print(f"Trained base models: {len(base_models)}")
    for model_name in base_models.keys():
        print(f"  - {model_name}")
    
    print(f"Trained meta models: {len(meta_models)}")
    for model_name in meta_models.keys():
        print(f"  - {model_name}")
    
    # Prediction information
    predictions = results['predictions']['predictions']
    print(f"\nGenerated predictions: {len(predictions)} molecules")
    print(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"Mean prediction: {predictions.mean():.4f}")
    print(f"Predictions > 0.5: {(predictions > 0.5).sum()}")
    
    # Submission files
    submissions = results['submissions']
    print(f"\nSubmission files created:")
    for file_type, path in submissions['file_paths'].items():
        if path:
            print(f"  - {file_type}: {path}")
    
    # Strategy used
    cv_summary = results['cv_summary']
    print(f"\nStrategy used: {cv_summary['best_ensemble_strategy']}")
    print(f"Optimal weights: {cv_summary['optimal_weights']}")

def print_cv_recommendations(results):
    """Print CV recommendations."""
    print("\n" + "="*60)
    print("FINAL RECOMMENDATIONS")
    print("="*60)
    
    # Find best approach
    comparison_df = results['summary_report']['comparison_table']
    if 'F1-Score' in comparison_df.columns:
        best_f1 = comparison_df['F1-Score'].idxmax()
        best_f1_score = comparison_df.loc[best_f1, 'F1-Score']
        print(f"Best F1-Score: {best_f1} ({best_f1_score:.4f})")
        
    if 'AUC-ROC' in comparison_df.columns:
        best_auc = comparison_df['AUC-ROC'].idxmax()
        best_auc_score = comparison_df.loc[best_auc, 'AUC-ROC']
        print(f"Best AUC-ROC: {best_auc} ({best_auc_score:.4f})")
    
    # Print best strategy from our comparison
    best_strategy_info = results.get('best_strategy_info', {})
    if best_strategy_info:
        print(f"Best Overall Strategy: {best_strategy_info['best_strategy_name']}")

# Example usage functions
def run_with_xgboost():
    """Run pipeline with XGBoost base learner."""
    config = create_default_config()
    config['base_learner_config'] = get_base_learner_config('xgboost')
    pipeline = MemoryEfficientStacking(config)
    return pipeline.run_pipeline()

def run_with_random_forest():
    """Run pipeline with Random Forest base learner."""
    config = create_default_config()
    config['base_learner_config'] = get_base_learner_config('random_forest')
    pipeline = MemoryEfficientStacking(config)
    return pipeline.run_pipeline()

def run_with_neural_net():
    """Run pipeline with Neural Network base learner."""
    config = create_default_config()
    config['base_learner_config'] = get_base_learner_config('neural_net')
    pipeline = MemoryEfficientStacking(config)
    return pipeline.run_pipeline()

def compare_base_learners(base_learners=['xgboost', 'random_forest', 'neural_net'], 
                          fingerprint_subset=None,
                          cv_folds=3):
    """Compare different base learners."""
    results_comparison = {}
    
    # Use subset of fingerprints if specified to speed up comparison
    config = create_default_config()
    if fingerprint_subset:
        config['fingerprint_cols'] = fingerprint_subset
    
    # Reduce folds for faster comparison
    config['cv_folds'] = cv_folds
    
    for base_learner in base_learners:
        print(f"\n{'='*60}")
        print(f"TESTING BASE LEARNER: {base_learner.upper()}")
        print(f"{'='*60}")
        
        config['base_learner_config'] = get_base_learner_config(base_learner)
        
        pipeline = MemoryEfficientStacking(config)
        results = pipeline.run_pipeline()
        
        # Store best meta-learner results
        comparison_df = results['summary_report']['comparison_table']
        results_comparison[base_learner] = {
            'comparison_table': comparison_df,
            'best_f1': comparison_df['F1-Score'].max(),
            'best_auc': comparison_df['AUC-ROC'].max() if 'AUC-ROC' in comparison_df.columns else None
        }
    
    # Create final comparison
    print(f"\n{'='*80}")
    print("BASE LEARNER COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Base Learner':<15} {'Best F1':<10} {'Best AUC':<10}")
    print("-" * 40)
    
    for base_learner, results in results_comparison.items():
        print(f"{base_learner:<15} {results['best_f1']:.4f}    {results['best_auc']:.4f if results['best_auc'] else 'N/A'}")
    
    return results_comparison

if __name__ == "__main__":
    main()