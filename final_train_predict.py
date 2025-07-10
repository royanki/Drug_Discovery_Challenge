#!/usr/bin/env python3
"""
Final Training and Test Prediction Module

This module:
1. Loads CV results from pickle files
2. Trains final models on complete training data
3. Generates predictions on test set
4. Creates submission files with proper chemical diversity selection
"""

import pickle
import numpy as np
import pandas as pd
import os
import gc
# Add these imports for RDKit-based diversity selection
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

class FinalTrainPredict:
    def __init__(self, cv_results_path, config_overrides=None):
        self.cv_results = self.load_cv_results(cv_results_path)
        self.config = self.cv_results['config']
        if config_overrides:
            self.config.update(config_overrides)
        
        # Store important info from CV
        self.fingerprint_cols = self.cv_results['fingerprint_meta_cols']
        self.best_weights = self.cv_results['cv_summary']['optimal_weights']
        self.best_strategy = self.cv_results['cv_summary']['best_ensemble_strategy']
    
    
    def load_cv_results(self, results_path):
        """Load CV results from pickle file."""
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        
        cv_summary = results['cv_summary']
        
        print(f"Loaded CV results:")
        print(f"  Best strategy: {cv_summary['best_ensemble_strategy']}")
        print(f"  Best F1-Score: {cv_summary['best_metrics'].get('F1-Score', 'N/A')}")
        print(f"  Optimal weights: {cv_summary['optimal_weights']}")
        
        return results
    
    def train_final_models(self):
        """
        Train final models on complete training data.
        Combines: train_final_base_learners() + train_final_meta_learners()
        """
        print("Training final models on complete training data...")
        
        # Step 1: Train base learners (from train_final_base_learners) and capture mask
        self.trained_base_models, self.feature_masks = self._train_final_base_learners()
        
        # Step 2: Generate complete meta-features for training
        complete_meta_features, complete_labels = self._generate_complete_meta_features()
        
        # Step 3: Train meta-learners (from train_final_meta_learners)
        self.trained_meta_models = self._train_final_meta_learners(
            complete_meta_features, complete_labels
        )
        
        return {
            'base_models': self.trained_base_models,
            'meta_models': self.trained_meta_models
        }
    
    def generate_final_predictions(self, test_data_path):
        
        """
        Generate predictions on test set.
        Combines: generate_test_meta_features() + apply_final_ensembles()
        """
        print("Generating final predictions on test set...")
        
        # Store test data path for diversity calculation
        self.config['test_data_path'] = test_data_path
        
        # Step 1: Generate meta-features for test set (from generate_test_meta_features)
        test_meta_features, test_metadata = self._generate_test_meta_features(test_data_path)
        
        # Step 2: Get predictions from meta-learners
        meta_predictions = self._get_meta_learner_test_predictions(test_meta_features)
        
        # Step 3: Apply ensemble strategy (from apply_final_ensembles)
        final_predictions = self._apply_final_ensemble(meta_predictions)
        
        # Store for later use in submission files
        self.test_predictions = final_predictions
        self.test_metadata = test_metadata
        
        return {
            'predictions': final_predictions,
            'metadata': test_metadata,
            'meta_features': test_meta_features,
            'individual_predictions': meta_predictions
        }
    
    def create_submission_files(self, output_dir="results/submissions"):
        """
        Create various submission files.
        Combines: get_top_confident_predictions() + get_diverse_predictions()
        
        Parameters:
        -----------
        output_dir : str
            Directory to save submission files
        
        Returns:
        --------
        dict
            Dictionary containing paths to created files and prediction summaries
        """
        print("Creating submission files...")
        
        # Ensure we have predictions
        if not hasattr(self, 'test_predictions') or not hasattr(self, 'test_metadata'):
            raise ValueError("Test predictions not found. Call generate_final_predictions() first.")
        
        predictions = self.test_predictions
        metadata = self.test_metadata
        
        # Step 1: Top confident predictions
        print("\nStep 1: Selecting top confident predictions...")
        top_200 = self._get_top_confident_predictions(predictions, metadata, n=200)
        top_500 = self._get_top_confident_predictions(predictions, metadata, n=500)
        
        # Step 2: Chemically diverse predictions using ALL PREPROCESSED fingerprints
        print("\nStep 2: Selecting chemically diverse predictions...")
        diverse_200 = self._get_diverse_predictions(
            predictions, 
            metadata, 
            n=200, 
            use_all_fingerprints=True  # Use all preprocessed fingerprints
        )
        
        # Step 3: Save files
        print("\nStep 3: Saving submission files...")
        file_paths = self._save_submission_files(top_200, top_500, diverse_200, output_dir)
        
        # Create summary
        summary = {
            'file_paths': file_paths,
            'statistics': {
                'total_predictions': len(predictions),
                'top_200': {
                    'count': top_200['count'],
                    'min_prob': top_200['probabilities'][-1],
                    'max_prob': top_200['probabilities'][0],
                    'mean_prob': top_200['probabilities'].mean()
                },
                'top_500': {
                    'count': top_500['count'],
                    'min_prob': top_500['probabilities'][-1],
                    'max_prob': top_500['probabilities'][0],
                    'mean_prob': top_500['probabilities'].mean()
                },
                'diverse_200': {
                    'count': diverse_200['count'],
                    'min_prob': diverse_200['probabilities'].min(),
                    'max_prob': diverse_200['probabilities'].max(),
                    'mean_prob': diverse_200['probabilities'].mean(),
                    'total_bits_used': diverse_200.get('total_bits_used', 'N/A'),
                    'method': diverse_200.get('method', 'N/A')
                }
            }
        }
        
        print("\nSubmission files created successfully!")
        print(f"Summary: {len(predictions)} total predictions processed")
        
        return summary

        # Private helper methods (these are the detailed implementations from earlier)
    def _train_final_base_learners(self):
        """
        Train final base learners on complete training data.
        """
        print("Training final base learners on complete training data...")
        
        # Access config and data paths from loaded CV results
        data_path = self.config['data_path']
        fingerprint_cols = self.config['fingerprint_cols']
        base_learner_config = self.config['base_learner_config']
        
        trained_models = {}
        feature_masks = {}
        
        # Rest of implementation is the same as original method
        for fp_idx, fingerprint_col in enumerate(fingerprint_cols):
            print("--"*60,"\n")
            print(f"Training final model for {fingerprint_col}...")
            
            # Load complete training data for this fingerprint
            df = pd.read_parquet(data_path, columns=['BB1_ID', 'BB2_ID', fingerprint_col, 'LABEL'])
            
            # Preprocess
            from .preprocessing import preprocess_single_fingerprint_numpy
            fp_matrix, metadata_df, info = preprocess_single_fingerprint_numpy(df, fingerprint_col)
            
            # Capture the training mask here for later reuse during prediction phase
            feature_masks[fingerprint_col] = info['feature_mask']
            print(f"Captured feature mask for {fingerprint_col}: {info['kept']} features kept")

            # Get all labels (no splitting)
            y_all = metadata_df['LABEL'].values
            
            # Create and train model on complete data
            from .base_learners import BaseLearnerFactory
            model = BaseLearnerFactory.create_base_learner(base_learner_config)
            model.fit(fp_matrix, y_all)
            
            # Store model with consistent naming
            model_name = f"{fingerprint_col}_{base_learner_config['type']}"
            trained_models[model_name] = model
            
            print(f"Completed training for {fingerprint_col}")
            
            # Clean up memory
            del fp_matrix, df, metadata_df
            gc.collect()
            
            self.feature_masks = feature_masks

        return trained_models, feature_masks
    
    def _train_final_meta_learners(self, meta_features, labels):
        """
        Train final meta-learners on complete meta-features.
        Adapted from the original train_final_meta_learners method.
        """
        print("Training final meta-learners on complete training data...")
        
        # Get fingerprint meta cols from CV results
        fingerprint_meta_cols = self.cv_results['fingerprint_meta_cols']
        
        # Import and get fresh meta-learners
        from .meta_learners import MetaLearnerTrainer

        trainer = MetaLearnerTrainer(
        base_learner_config=self.config.get('base_learner_config'),
        random_state=self.config.get('random_state', 42)
        )
        
        # Get fresh instances to avoid any training state
        meta_learners = trainer.get_fresh_meta_learners()

        print(f"Using {len(meta_learners)} meta-learners:")
        for name in meta_learners.keys():
            print(f"  - {name}")
        
        final_meta_results = {}
        
        for name, meta_learner in meta_learners.items():
            print(f"Training final {name} meta-learner...")
            
            # Train on complete meta-features
            meta_learner.fit(meta_features, labels)
            
            # Store results (we'll generate test predictions later)
            final_meta_results[name] = {
                'model': meta_learner,
                'feature_importances': self._get_feature_importances(meta_learner, fingerprint_meta_cols)
            }
        
        return final_meta_results

    def _generate_complete_meta_features(self):
        """
        Generate meta-features on complete training data using trained base models.
        Similar to generate_meta_features() but without CV splits.
        """
        print("Generating meta-features on complete training data...")
        
        data_path = self.config['data_path']
        fingerprint_cols = self.config['fingerprint_cols']
        
        all_meta_features = []
        labels = None

        if not hasattr(self, 'feature_masks'):
            raise ValueError("Feature masks not found! They should be captured during base model training.")
        
        # Process one fingerprint at a time
        for fp_idx, fingerprint_col in enumerate(fingerprint_cols):
            print("--"*60,"\n")
            print(f"Processing {fingerprint_col} for meta-feature generation...")
            
            # Get the feature_mask (to be used here)
            training_mask = self.feature_masks[fingerprint_col]

            # Load training data for this fingerprint
            df = pd.read_parquet(data_path, columns=['BB1_ID', 'BB2_ID', fingerprint_col, 'LABEL'])
            
            # Preprocess (apply feature_mask)
            from .preprocessing import preprocess_single_fingerprint_numpy
            fp_matrix, metadata_df, info = preprocess_single_fingerprint_numpy(df, fingerprint_col,feature_mask=training_mask)
            
            print(f"\nReused feature mask for {fingerprint_col}: {info['kept']} features (consistent with base training)")

            # Store labels from first fingerprint (same for all)
            if fp_idx == 0:
                labels = metadata_df['LABEL'].values
            
            # Get the trained base model for this fingerprint
            model_name = f"{fingerprint_col}_{self.config['base_learner_config']['type']}"
            if model_name not in self.trained_base_models:
                raise ValueError(f"No trained model found for {model_name}")
            
            model = self.trained_base_models[model_name]
            
            # Generate meta-features (prediction probabilities)
            meta_feature = model.predict_proba(fp_matrix)[:, 1]
            all_meta_features.append(meta_feature)
            
            print(f"\nGenerated meta-features for {fingerprint_col}: shape {meta_feature.shape}")
            print("--"*60,"\n")

            # Clean up memory
            del fp_matrix, df, metadata_df
            gc.collect()
        
        # Stack all meta-features
        complete_meta_features = np.column_stack(all_meta_features)
        
        print(f"Complete meta-features shape: {complete_meta_features.shape}")
        
        return complete_meta_features, labels
    
 
    def _get_parquet_columns(self, test_data_path):
        """Get column names from parquet file without loading data."""
        try:
            import pyarrow.parquet as pq
            
            # Read just the schema (no data loaded - very fast)
            parquet_file = pq.ParquetFile(test_data_path)
            columns = [field.name for field in parquet_file.schema]
            print(f"\nDetected {len(columns)} columns using PyArrow schema")
            return columns
            
        except (ImportError, Exception) as e:
            # Single fallback for both ImportError and any other exception
            if isinstance(e, ImportError):
                print("PyArrow not available, using pandas fallback...")
            else:
                print(f"PyArrow schema reading failed: {e}, using pandas fallback...")
            
            # Fallback: load with pandas (less efficient but works)
            df = pd.read_parquet(test_data_path)
            columns = df.columns.tolist()
            del df
            gc.collect()
            print(f"\nDetected {len(columns)} columns using pandas fallback")
            return columns

    def _generate_test_meta_features(self, test_data_path):
        """
        Generate meta-features for the test set using trained base learners.
        
        Parameters:
        -----------
        test_data_path : str
            Path to the test dataset
        
        Returns:
        --------
        tuple
            (test_meta_features, test_metadata)
        """
        print("\nGenerating meta-features for test set...")
        
        fingerprint_cols = self.config['fingerprint_cols']
        test_meta_features = []
        test_metadata = None
        
        # Step 1: Auto-detect available ID columns in test data
        try:
            # Read just the column names first (very fast)
            available_columns = self._get_parquet_columns(test_data_path)
            # Find ID columns using flexible detection
            id_columns = self._detect_id_columns(available_columns)
            print(f"Available columns in test data: {available_columns[:10]}...")  # Show first 10
            print(f"Detected ID columns: {id_columns}")
            
        except Exception as e:
            print(f"Warning: Could not detect columns in test data: {e}")
            id_columns = []
        
        if not hasattr(self, 'feature_masks'):
            raise ValueError("Feature masks not found! They should be captured during base model training.")
    
        print(f"Using feature masks captured during training for {len(self.feature_masks)} fingerprints")

        # Process each fingerprint and generate meta-features
        for fp_idx, fingerprint_col in enumerate(fingerprint_cols):
            print("--"*60,"\n")
            print(f"Processing test fingerprint {fp_idx+1}/{len(fingerprint_cols)}: {fingerprint_col}")
            
            # Load test data for this fingerprint
            load_columns = [fingerprint_col]
            if id_columns:
                load_columns.extend(id_columns)
            
            try:
                df_test = pd.read_parquet(test_data_path, columns=load_columns)
            except Exception as e:
                print(f"Warning: Could not load with ID columns {id_columns}, loading fingerprint only: {e}")
                df_test = pd.read_parquet(test_data_path, columns=[fingerprint_col])
                id_columns = []  # Reset since we couldn't load them
            
            # Add ID columns if none were found or loaded
            if not id_columns:
                df_test['ID'] = range(len(df_test))
                print(f"  Added dummy ID column (0 to {len(df_test)-1})")
            else:
                print(f"  Using existing ID columns: {id_columns}")
            
            training_mask = self.feature_masks[fingerprint_col]

            # Preprocess fingerprint
            from .preprocessing import preprocess_single_fingerprint_numpy
            fp_matrix_test, metadata_df_test, info = preprocess_single_fingerprint_numpy(df_test, fingerprint_col,feature_mask=training_mask)
            
            print(f"  Applied mask: {info['kept']} features kept (matches training)")

            # Store metadata from first fingerprint (should be same for all)
            if fp_idx == 0:
                test_metadata = metadata_df_test
            
            # Get the trained model for this fingerprint
            base_learner_type = self.config['base_learner_config']['type']
            model_name = f"{fingerprint_col}_{base_learner_type}"
            
            if model_name not in self.trained_base_models:
                raise ValueError(f"No trained model found for {model_name}. Available models: {list(self.trained_base_models.keys())}")
            
            model = self.trained_base_models[model_name]
            
            # Generate meta-features (prediction probabilities)
            test_meta_feature = model.predict_proba(fp_matrix_test)[:, 1]
            test_meta_features.append(test_meta_feature)
            
            print(f"Generated meta-features for {fingerprint_col}: shape {test_meta_feature.shape}")
            
            # Clean up memory
            del fp_matrix_test, df_test, metadata_df_test
            gc.collect()
        
        # Stack all meta-features
        X_meta_test = np.column_stack(test_meta_features)
        
        print(f"Final test meta-features shape: {X_meta_test.shape}")
        
        return X_meta_test, test_metadata

    def _detect_id_columns(self, available_columns):
        """
        Detect ID columns from available columns using flexible patterns.
        
        Parameters:
        -----------
        available_columns : list
            List of column names
        
        Returns:
        --------
        list
            List of detected ID column names
        """
        id_columns = []
        
        # Priority order for ID column detection
        id_patterns = [
            # Exact matches (highest priority)
            ['BB1_ID', 'BB2_ID'],  # Traditional building block IDs
            ['RandomID'],           # Your current test data format
            ['ID'],                 # Simple ID
            ['MOLECULE_ID'],        # Common alternative
            ['COMPOUND_ID'],        # Another common alternative
            ['SAMPLE_ID'],          # Another alternative
        ]
        
        # Check for exact pattern matches first
        for pattern in id_patterns:
            if all(col in available_columns for col in pattern):
                id_columns = pattern
                break
        
        # If no exact pattern found, look for any column containing 'ID'
        if not id_columns:
            potential_ids = [col for col in available_columns 
                            if 'ID' in col.upper() and not col.startswith('_')]
            if potential_ids:
                id_columns = potential_ids[:2]  # Take at most 2 ID columns
                print(f"  Found potential ID columns by pattern matching: {potential_ids}")
        
        # Special handling for index-like columns
        if not id_columns:
            index_like = [col for col in available_columns 
                         if col.lower() in ['index', 'idx', 'row', 'row_id', 'rowid']]
            if index_like:
                id_columns = index_like[:1]  # Take first index-like column
                print(f"  Found index-like columns: {index_like}")
        
        return id_columns
    
    def _apply_final_ensemble(self, meta_predictions):
        """
        Apply the best ensemble strategy from CV to test predictions.
        
        Parameters:
        -----------
        meta_predictions : dict
            Dictionary containing predictions from each meta-learner
            Format: {learner_name: prediction_probabilities}
        
        Returns:
        --------
        numpy.ndarray
            Final ensemble prediction probabilities
        """
        print("Applying final ensemble strategy to test predictions...")
        
        # Get the best strategy and weights from CV results
        best_strategy = self.best_strategy
        best_weights = self.best_weights
        
        print(f"Using best strategy from CV: {best_strategy}")
        print(f"Using weights: {best_weights}")
        
        # Apply ensemble based on the best strategy found during CV
        if best_strategy.startswith('Ensemble_'):
            # Extract the actual ensemble type (remove 'Ensemble_' prefix)
            ensemble_type = best_strategy.replace('Ensemble_', '')
            
            if ensemble_type == 'weighted':
                final_proba = self._apply_weighted_ensemble(meta_predictions, best_weights)
            elif ensemble_type == 'threshold':
                final_proba = self._apply_threshold_ensemble(meta_predictions)
            elif ensemble_type == 'calibrated':
                final_proba = self._apply_calibrated_ensemble(meta_predictions, best_weights)
            elif ensemble_type == 'robust':
                final_proba = self._apply_robust_ensemble(meta_predictions, best_weights)
            else:
                print(f"Warning: Unknown ensemble type {ensemble_type}, falling back to weighted ensemble")
                final_proba = self._apply_weighted_ensemble(meta_predictions, best_weights)
                
        elif best_strategy.startswith('MetaLearner_'):
            # Best strategy is a single meta-learner
            learner_name = best_strategy.replace('MetaLearner_', '')
            if learner_name in meta_predictions:
                print(f"Using single meta-learner: {learner_name}")
                final_proba = meta_predictions[learner_name]
            else:
                print(f"Warning: {learner_name} not found, falling back to weighted ensemble")
                final_proba = self._apply_weighted_ensemble(meta_predictions, best_weights)
        else:
            print(f"Warning: Unknown strategy type {best_strategy}, falling back to weighted ensemble")
            final_proba = self._apply_weighted_ensemble(meta_predictions, best_weights)
        
        print(f"Final ensemble predictions shape: {final_proba.shape}")
        return final_proba
    
    def _apply_weighted_ensemble(self, meta_predictions, weights):
        """Apply weighted ensemble strategy."""
        print("Applying weighted ensemble...")
        
        # Initialize ensemble prediction
        final_proba = np.zeros_like(list(meta_predictions.values())[0])
        total_weight = 0
        
        # Apply weights
        for learner_name, weight in weights.items():
            if learner_name in meta_predictions:
                final_proba += weight * meta_predictions[learner_name]
                total_weight += weight
                print(f"  {learner_name}: weight={weight:.3f}")
            else:
                print(f"  Warning: {learner_name} not found in meta_predictions")
        
        # Normalize if weights don't sum to 1
        if abs(total_weight - 1.0) > 1e-6:
            print(f"  Normalizing weights (sum was {total_weight:.3f})")
            final_proba = final_proba / total_weight
        
        return final_proba

    def _apply_threshold_ensemble(self, meta_predictions):
        """Apply threshold-based ensemble strategy."""
        print("Applying threshold-based ensemble...")
        
        # For simplicity, fall back to equal weighting for threshold ensemble on test set
        # The threshold logic was designed for validation set optimization
        print("  Note: Using equal weighting for threshold ensemble on test set")
        
        equal_weights = {name: 1.0/len(meta_predictions) for name in meta_predictions.keys()}
        return self._apply_weighted_ensemble(meta_predictions, equal_weights)

    def _apply_calibrated_ensemble(self, meta_predictions, weights):
        """Apply calibrated ensemble strategy."""
        print("Applying calibrated ensemble...")
        
        # For test set, we use the same weights as the weighted ensemble
        # since calibration was already applied during training
        return self._apply_weighted_ensemble(meta_predictions, weights)

    def _apply_robust_ensemble(self, meta_predictions, weights):
        """Apply robust ensemble strategy."""
        print("Applying robust ensemble...")
        
        # Use the optimized weights from the robust ensemble CV
        return self._apply_weighted_ensemble(meta_predictions, weights)

    def _get_meta_learner_test_predictions(self, test_meta_features):
        """
        Get predictions from all trained meta-learners on test set.
        
        Parameters:
        -----------
        test_meta_features : numpy.ndarray
            Meta-features for the test set
        
        Returns:
        --------
        dict
            Dictionary containing predictions from each meta-learner
        """
        print("Getting predictions from trained meta-learners...")
        
        meta_predictions = {}
        
        for name, meta_result in self.trained_meta_models.items():
            print(f"  Generating {name} predictions...")
            
            model = meta_result['model']
            test_proba = model.predict_proba(test_meta_features)[:, 1]
            meta_predictions[name] = test_proba
            
            print(f"    {name} predictions shape: {test_proba.shape}")
            print(f"    {name} prediction range: [{test_proba.min():.3f}, {test_proba.max():.3f}]")
        
        return meta_predictions

    def _get_top_confident_predictions(self, predictions, metadata, n):
        """
        Get the indices and values of the top N most confident predictions.
        
        Parameters:
        -----------
        predictions : numpy.ndarray
            Array of prediction probabilities
        metadata : pandas.DataFrame
            DataFrame containing metadata for each prediction
        n : int
            Number of top predictions to return
        
        Returns:
        --------
        dict
            Dictionary containing indices, probabilities, and metadata of top predictions
        """
        print(f"Selecting top {n} most confident predictions...")
        
        # Ensure we don't ask for more predictions than available
        n = min(n, len(predictions))
        
        # Get top N indices (highest probabilities)
        top_n_indices = np.argsort(predictions)[::-1][:n]
        
        # Extract top predictions
        top_probabilities = predictions[top_n_indices]
        top_metadata = metadata.iloc[top_n_indices].copy()
        
        # Add probability column to metadata
        top_metadata['Prediction_Probability'] = top_probabilities
        top_metadata['Confidence_Rank'] = range(1, n + 1)
        
        print(f"Top {n} predictions:")
        print(f"  Highest probability: {top_probabilities[0]:.4f}")
        print(f"  Lowest probability in top {n}: {top_probabilities[-1]:.4f}")
        print(f"  Mean probability: {top_probabilities.mean():.4f}")
        
        return {
            'indices': top_n_indices,
            'probabilities': top_probabilities,
            'metadata': top_metadata,
            'count': n
        }
    

    def _get_diverse_predictions(self, predictions, metadata, n=200, similarity_threshold=0.7, 
                           candidate_pool_size=1000, use_all_fingerprints=True):
        """
        Select chemically diverse molecules using preprocessed fingerprints (same as used for modeling).
        
        This applies the same zero-variance removal that was used during model building
        to ensure consistency between modeling and diversity calculation.
        """
        
        # Define fingerprint columns to use
        if use_all_fingerprints:
            fingerprint_cols = self.config.get('fingerprint_cols', 
                                              ['ECFP4', 'ECFP6', 'FCFP4', 'FCFP6', 'TOPTOR', 
                                               'MACCS', 'RDK', 'AVALON', 'ATOMPAIR'])
            print(f"Selecting {n} diverse predictions using ALL PREPROCESSED fingerprints: {fingerprint_cols}")
        else:
            fingerprint_cols = ['ECFP4']
            print(f"Selecting {n} diverse predictions using preprocessed ECFP4 only")
        
        print(f"Using Tanimoto similarity threshold: {similarity_threshold}")
        
        # Step 1: Select top candidates by probability
        candidate_pool_size = min(candidate_pool_size, len(predictions))
        sorted_indices = np.argsort(predictions)[::-1]
        candidate_indices = sorted_indices[:candidate_pool_size]
        
        print(f"Working with top {candidate_pool_size} predictions for diversity selection")
        
        # NEW: Determine what ID columns are available
        available_id_columns = self._get_available_id_columns(metadata)
        print(f"Available ID columns: {available_id_columns}")
        
        # Step 2: Load and preprocess each fingerprint (same as model building)
        test_data_path = self.config.get('test_data_path')
        if not test_data_path:
            print("Warning: No test_data_path found, falling back to simple diversity")
            return self._get_diverse_predictions_fallback(predictions, metadata, n)
        
        try:
            candidate_fingerprints = []
            valid_candidate_indices = []
            fingerprint_info = {}
            
            # Process each fingerprint type separately (same as during model building)
            for fp_col in fingerprint_cols:
                print(f"\nProcessing {fp_col} fingerprint...")
                
                # Load this specific fingerprint (only need the fingerprint column)
                df_fp = pd.read_parquet(test_data_path, columns=[fp_col])
                
                # Add index as identifier since we don't need actual IDs for fingerprint processing
                df_fp['INDEX_ID'] = range(len(df_fp))
                
                # Apply the same preprocessing as during model building
                from .preprocessing import preprocess_single_fingerprint_numpy
                fp_matrix, fp_metadata, info = preprocess_single_fingerprint_numpy(df_fp, fp_col)
                
                print(f"  {fp_col}: {info['kept']} bits kept, {info['removed']} bits removed")
                fingerprint_info[fp_col] = info
                
                # Store the processed fingerprint matrix for candidates
                if fp_col == fingerprint_cols[0]:  # First fingerprint
                    # Create mapping from candidate index to fingerprint row
                    for i, candidate_idx in enumerate(candidate_indices):
                        if candidate_idx < len(fp_matrix):  # Ensure index is valid
                            candidate_fingerprints.append([fp_matrix[candidate_idx]])
                            valid_candidate_indices.append(candidate_idx)
                else:  # Subsequent fingerprints
                    # Add to existing candidate fingerprints
                    candidates_to_remove = []
                    for i, candidate_idx in enumerate(valid_candidate_indices):
                        if candidate_idx < len(fp_matrix):
                            candidate_fingerprints[i].append(fp_matrix[candidate_idx])
                        else:
                            candidates_to_remove.append(i)
                    
                    # Remove candidates that don't have all fingerprints
                    for i in reversed(candidates_to_remove):
                        candidate_fingerprints.pop(i)
                        valid_candidate_indices.pop(i)
                
                # Clean up
                del df_fp, fp_matrix, fp_metadata
                gc.collect()
            
            # Step 3: Combine preprocessed fingerprints for each candidate
            print("Combining preprocessed fingerprints...")
            combined_fingerprints = []
            
            for candidate_fps in candidate_fingerprints:
                # Concatenate all fingerprint types for this candidate
                combined_vector = np.concatenate(candidate_fps)
                
                # Convert to RDKit-compatible fingerprint
                fp = DataStructs.ExplicitBitVect(len(combined_vector))
                for bit_idx, bit_val in enumerate(combined_vector):
                    if bit_val:
                        fp.SetBit(bit_idx)
                
                combined_fingerprints.append(fp)
            
            # Print fingerprint combination summary
            total_bits = sum(info['kept'] for info in fingerprint_info.values())
            print(f"Combined fingerprint summary:")
            for fp_col, info in fingerprint_info.items():
                print(f"  {fp_col}: {info['kept']} bits")
            print(f"  Total combined bits: {total_bits}")
            print(f"Successfully created combined fingerprints for {len(combined_fingerprints)} candidates")
            
            if len(combined_fingerprints) == 0:
                print("Warning: No valid combined fingerprints created, falling back to simple diversity")
                return self._get_diverse_predictions_fallback(predictions, metadata, n)
            
            candidate_fingerprints = combined_fingerprints  # Use for clustering
            
        except Exception as e:
            print(f"Error processing fingerprint data: {e}")
            print("Falling back to simple diversity selection")
            return self._get_diverse_predictions_fallback(predictions, metadata, n)
        
        # Step 4: Calculate Tanimoto distance matrix
        print("\nCalculating Tanimoto distance matrix on preprocessed fingerprints...")
        n_fps = len(candidate_fingerprints)
        distance_matrix = []
        
        for i in range(n_fps):
            distances = []
            for j in range(i):
                # Calculate Tanimoto similarity
                tanimoto_sim = DataStructs.TanimotoSimilarity(candidate_fingerprints[i], candidate_fingerprints[j])
                # Convert to distance (1 - similarity)
                tanimoto_dist = 1.0 - tanimoto_sim
                distances.append(tanimoto_dist)
            distance_matrix.append(distances)
        
        # Step 5: Cluster using Butina algorithm
        print(f"\nClustering (Butina) with threshold {similarity_threshold}...")
        try:
            clusters = Butina.ClusterData(
                distance_matrix, 
                n_fps, 
                1.0 - similarity_threshold,  # Convert similarity threshold to distance
                isDistData=True
            )
            print(f"Generated {len(clusters)} clusters")
        except Exception as e:
            print(f"Clustering failed: {e}, falling back to simple diversity")
            return self._get_diverse_predictions_fallback(predictions, metadata, n)
        
        # Step 6: Select representatives from each cluster
        diverse_indices = []
        cluster_info = []
        
        for cluster_idx, cluster in enumerate(clusters):
            if len(diverse_indices) >= n:
                break
            
            # Get the molecule with highest probability in this cluster
            cluster_predictions = [predictions[valid_candidate_indices[i]] for i in cluster]
            best_in_cluster_idx = np.argmax(cluster_predictions)
            selected_idx = valid_candidate_indices[cluster[best_in_cluster_idx]]
            
            diverse_indices.append(selected_idx)
            cluster_info.append({
                'cluster_id': cluster_idx,
                'cluster_size': len(cluster),
                'selected_probability': predictions[selected_idx]
            })
        
        # Step 7: Fill remaining slots with highest probability molecules not yet selected
        if len(diverse_indices) < n:
            print(f"Adding {n - len(diverse_indices)} more molecules by probability...")
            original_sorted_indices = np.argsort(predictions)[::-1]
            for idx in original_sorted_indices:
                if idx not in diverse_indices:
                    diverse_indices.append(idx)
                    if len(diverse_indices) >= n:
                        break
        
        # Limit to requested number
        diverse_indices = diverse_indices[:n]
        
        # Step 8: Prepare results
        diverse_indices = np.array(diverse_indices)
        diverse_probabilities = predictions[diverse_indices]
        diverse_metadata = metadata.iloc[diverse_indices].copy()
        
        # Add diversity information
        diverse_metadata['Prediction_Probability'] = diverse_probabilities
        diverse_metadata['Diversity_Rank'] = range(1, len(diverse_indices) + 1)
        diverse_metadata['Selection_Method'] = ['Cluster_Representative'] * min(len(cluster_info), len(diverse_indices))
        
        # Add remaining as 'High_Probability' if we filled extra slots
        if len(diverse_indices) > len(cluster_info):
            diverse_metadata.loc[diverse_metadata.index[len(cluster_info):], 'Selection_Method'] = 'High_Probability'
        
        method_name = f"preprocessed_combined_{len(fingerprint_cols)}_fingerprints" if use_all_fingerprints else "preprocessed_ECFP4"
        
        print(f"Selected {len(diverse_indices)} diverse predictions using preprocessed fingerprints:")
        print(f"  From {len(cluster_info)} clusters: {min(len(cluster_info), len(diverse_indices))} molecules")
        print(f"  Additional high-probability: {max(0, len(diverse_indices) - len(cluster_info))} molecules")
        print(f"  Highest probability: {diverse_probabilities.max():.4f}")
        print(f"  Lowest probability: {diverse_probabilities.min():.4f}")
        print(f"  Mean probability: {diverse_probabilities.mean():.4f}")
        
        # Updated statistics that don't assume specific ID columns
        id_stats = self._get_id_statistics(diverse_metadata, available_id_columns)
        for stat_name, stat_value in id_stats.items():
            print(f"  {stat_name}: {stat_value}")
        
        return {
            'indices': diverse_indices,
            'probabilities': diverse_probabilities,
            'metadata': diverse_metadata,
            'count': len(diverse_indices),
            'cluster_info': cluster_info,
            'method': method_name,
            'fingerprint_info': fingerprint_info,
            'total_bits_used': sum(info['kept'] for info in fingerprint_info.values())
        }


    def _get_available_id_columns(self, metadata):
        """
        Determine what ID columns are available in the metadata.
        Updated to use the same flexible detection logic.
        """
        available_columns = metadata.columns.tolist()
        return self._detect_id_columns(available_columns)

    def _get_id_statistics(self, metadata, available_id_columns):
        """Get statistics about ID columns in a flexible way."""
        stats = {}
        
        if 'BB1_ID' in available_id_columns and 'BB2_ID' in available_id_columns:
            stats['Unique BB1_IDs'] = len(metadata['BB1_ID'].unique())
            stats['Unique BB2_IDs'] = len(metadata['BB2_ID'].unique())
        elif 'RandomID' in available_id_columns:
            stats['Unique RandomIDs'] = len(metadata['RandomID'].unique())
        elif available_id_columns:
            for col in available_id_columns:
                stats[f'Unique {col}s'] = len(metadata[col].unique())
        else:
            stats['Unique molecules'] = len(metadata)
        
        return stats


    def _get_diverse_predictions_fallback(self, predictions, metadata, n):
        """
        Fallback diversity selection that works with any ID column structure.
        """
        print("Using fallback diversity selection...")
        
        # Determine what ID columns are available
        available_id_columns = self._get_available_id_columns(metadata)
        print(f"Using fallback diversity with columns: {available_id_columns}")
        
        # Sort by prediction probability
        sorted_indices = np.argsort(predictions)[::-1]
        
        diverse_indices = [sorted_indices[0]]  # Start with highest probability
        
        if not available_id_columns:
            # No ID columns available - just select by probability
            print("No ID columns found - selecting by probability only")
            diverse_indices = sorted_indices[:n].tolist()
        else:
            # Use available ID columns for diversity
            used_identifiers = set()
            
            # Get identifier for first molecule
            first_identifier = self._get_molecule_identifier(metadata.iloc[sorted_indices[0]], available_id_columns)
            used_identifiers.add(first_identifier)
            
            # Select diverse molecules
            for idx in sorted_indices[1:]:
                if len(diverse_indices) >= n:
                    break
                
                identifier = self._get_molecule_identifier(metadata.iloc[idx], available_id_columns)
                
                # Simple diversity criterion: different identifiers
                if identifier not in used_identifiers:
                    diverse_indices.append(idx)
                    used_identifiers.add(identifier)
            
            # Fill remaining slots with highest probabilities if needed
            if len(diverse_indices) < n:
                for idx in sorted_indices:
                    if idx not in diverse_indices:
                        diverse_indices.append(idx)
                        if len(diverse_indices) >= n:
                            break
        
        # Prepare results
        diverse_indices = np.array(diverse_indices[:n])
        diverse_probabilities = predictions[diverse_indices]
        diverse_metadata = metadata.iloc[diverse_indices].copy()
        
        diverse_metadata['Prediction_Probability'] = diverse_probabilities
        diverse_metadata['Diversity_Rank'] = range(1, len(diverse_indices) + 1)
        diverse_metadata['Selection_Method'] = 'Fallback_Diversity'
        
        return {
            'indices': diverse_indices,
            'probabilities': diverse_probabilities,
            'metadata': diverse_metadata,
            'count': len(diverse_indices),
            'method': 'fallback_diversity'
        }

    def _get_molecule_identifier(self, row, available_id_columns):
        """Get a unique identifier for a molecule from available ID columns."""
        if 'BB1_ID' in available_id_columns and 'BB2_ID' in available_id_columns:
            return (row['BB1_ID'], row['BB2_ID'])
        elif 'RandomID' in available_id_columns:
            return row['RandomID']
        elif available_id_columns:
            # Use the first available ID column
            return row[available_id_columns[0]]
        else:
            # Fallback to index
            return row.name

    def _get_feature_importances(self, model, fingerprint_cols):
        """Extract feature importances from meta-learner."""
        feature_importances = {}
        
        if hasattr(model, 'coef_'):
            # For LogisticRegression
            for i, fp in enumerate(fingerprint_cols):
                feature_importances[fp] = model.coef_[0][i]
        elif hasattr(model, 'feature_importances_'):
            # For RandomForest
            for i, fp in enumerate(fingerprint_cols):
                feature_importances[fp] = model.feature_importances_[i]
        else:
            # For models without direct feature importance
            for i, fp in enumerate(fingerprint_cols):
                feature_importances[fp] = 0.0
        
        return feature_importances

    def _save_submission_files(self, top_200, top_500, diverse_200, output_dir):
        """
        Save submission files to disk.
        
        Parameters:
        -----------
        top_200 : dict
            Top 200 confident predictions
        top_500 : dict
            Top 500 confident predictions
        diverse_200 : dict
            200 diverse predictions
        output_dir : str
            Directory to save files
        """
        print(f"Saving submission files to {output_dir}...")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base learner type for filename
        base_learner_type = self.config['base_learner_config']['type']
        
        # Save top 200
        top_200_path = os.path.join(output_dir, f"top_200_{base_learner_type}.csv")
        top_200['metadata'].to_csv(top_200_path, index=False)
        print(f"  Saved top 200 predictions to: {top_200_path}")
        
        # Save top 500
        top_500_path = os.path.join(output_dir, f"top_500_{base_learner_type}.csv")
        top_500['metadata'].to_csv(top_500_path, index=False)
        print(f"  Saved top 500 predictions to: {top_500_path}")
        
        # Save diverse 200
        if diverse_200 is not None:
            diverse_200_path = os.path.join(output_dir, f"diverse_200_{base_learner_type}.csv")
            diverse_200['metadata'].to_csv(diverse_200_path, index=False)
            print(f"  Saved diverse 200 predictions to: {diverse_200_path}")
        
        # Save all predictions with probabilities
        all_predictions_path = os.path.join(output_dir, f"all_predictions_{base_learner_type}.csv")
        all_metadata = self.test_metadata.copy()
        all_metadata['Prediction_Probability'] = self.test_predictions
        all_metadata = all_metadata.sort_values('Prediction_Probability', ascending=False)
        all_metadata.to_csv(all_predictions_path, index=False)
        print(f"  Saved all predictions to: {all_predictions_path}")
        
        return {
            'top_200_path': top_200_path,
            'top_500_path': top_500_path,
            'diverse_200_path': diverse_200_path if diverse_200 is not None else None,
            'all_predictions_path': all_predictions_path
        }

    def run_complete_pipeline(self, test_data_path, output_dir="results/submissions"):
        """
        Run the complete final training and prediction pipeline.
        
        Parameters:
        -----------
        test_data_path : str
            Path to test dataset
        output_dir : str
            Directory to save submission files
        
        Returns:
        --------
        dict
            Complete results including models, predictions, and file paths
        """
        print("="*80)
        print("RUNNING COMPLETE FINAL PIPELINE")
        print("="*80)
        
        # Step 1: Train final models
        print("\nStep 1: Training final models...")
        model_results = self.train_final_models()
        
        # Step 2: Generate predictions
        print("\nStep 2: Generating final predictions...")
        prediction_results = self.generate_final_predictions(test_data_path)
        
        # Step 3: Create submission files
        print("\nStep 3: Creating submission files...")
        submission_results = self.create_submission_files(output_dir)
        
        print("\n" + "="*80)
        print("COMPLETE PIPELINE FINISHED SUCCESSFULLY")
        print("="*80)
        
        return {
            'models': model_results,
            'predictions': prediction_results,
            'submissions': submission_results,
            'config': self.config,
            'cv_summary': self.cv_results['cv_summary']
        }

    def save_final_results(self, output_path):
        """Save final training results."""
        import pickle
        
        # Create serializable results
        final_results = {
            'config': self.config,
            'cv_summary': self.cv_results['cv_summary'],
            'test_predictions': self.test_predictions.tolist(),
            'test_metadata': self.test_metadata.to_dict(),
            'prediction_stats': {
                'total_predictions': len(self.test_predictions),
                'mean_prediction': float(self.test_predictions.mean()),
                'max_prediction': float(self.test_predictions.max()),
                'min_prediction': float(self.test_predictions.min()),
                'predictions_above_05': int((self.test_predictions > 0.5).sum())
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(final_results, f)
        
        print(f"Final results saved to {output_path}")