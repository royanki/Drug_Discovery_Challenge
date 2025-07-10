#!/usr/bin/env python3
"""
Final Training and Test Prediction Module

This module:
1. Loads CV results from pickle files
2. Trains final models on complete training data
3. Generates predictions on test set
4. Creates submission files
"""

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
        
        # Step 1: Train base learners (from train_final_base_learners)
        self.trained_base_models = self._train_final_base_learners()
        
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
        
        # Step 1: Generate meta-features for test set (from generate_test_meta_features)
        test_meta_features, test_metadata = self._generate_test_meta_features(test_data_path)
        
        # Step 2: Get predictions from meta-learners
        meta_predictions = self._get_meta_learner_predictions(test_meta_features)
        
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
        
        # Step 2: Chemically diverse predictions
        print("\nStep 2: Selecting chemically diverse predictions...")
        diverse_200 = self._get_diverse_predictions(predictions, metadata, n=200)
        
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
                    'mean_prob': diverse_200['probabilities'].mean()
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
        
        # Rest of implementation is the same as original method
        for fp_idx, fingerprint_col in enumerate(fingerprint_cols):
            print(f"Training final model for {fingerprint_col}...")
            
            # Load complete training data for this fingerprint
            df = pd.read_parquet(data_path, columns=['BB1_ID', 'BB2_ID', fingerprint_col, 'LABEL'])
            
            # Preprocess
            from .preprocessing import preprocess_single_fingerprint_numpy
            fp_matrix, metadata_df, info = preprocess_single_fingerprint_numpy(df, fingerprint_col)
            
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
        
        return trained_models
    
    def _train_final_meta_learners(self, meta_features, labels):
        """
        Train final meta-learners on complete meta-features.
        Adapted from the original train_final_meta_learners method.
        """
        print("Training final meta-learners on complete training data...")
        
        # Get fingerprint meta cols from CV results
        fingerprint_meta_cols = self.cv_results['fingerprint_meta_cols']
        
        # Create meta-learner instances (only LR and RF as per your requirements)
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        
        meta_learners = {
            'LogisticRegression': LogisticRegression(
                random_state=self.config.get('random_state', 42),
                max_iter=1000,
                class_weight='balanced'
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.get('random_state', 42),
                class_weight='balanced',
                n_jobs=-1
            )
        }
        
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
        
        # Process one fingerprint at a time
        for fp_idx, fingerprint_col in enumerate(fingerprint_cols):
            print(f"Processing {fingerprint_col} for meta-feature generation...")
            
            # Load training data for this fingerprint
            df = pd.read_parquet(data_path, columns=['BB1_ID', 'BB2_ID', fingerprint_col, 'LABEL'])
            
            # Preprocess
            from .preprocessing import preprocess_single_fingerprint_numpy
            fp_matrix, metadata_df, info = preprocess_single_fingerprint_numpy(df, fingerprint_col)
            
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
            
            print(f"Generated meta-features for {fingerprint_col}: shape {meta_feature.shape}")
            
            # Clean up memory
            del fp_matrix, df, metadata_df
            gc.collect()
        
        # Stack all meta-features
        complete_meta_features = np.column_stack(all_meta_features)
        
        print(f"Complete meta-features shape: {complete_meta_features.shape}")
        
        return complete_meta_features, labels
    
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
        
        # Process each fingerprint and generate meta-features
        for fp_idx, fingerprint_col in enumerate(fingerprint_cols):
            print(f"Processing test fingerprint {fp_idx+1}/{len(fingerprint_cols)}: {fingerprint_col}")
            
            # Load test data for this fingerprint (no LABEL column in test data)
            df_test = pd.read_parquet(test_data_path, columns=['BB1_ID', 'BB2_ID', fingerprint_col])
            
            # Preprocess fingerprint
            from .preprocessing import preprocess_single_fingerprint_numpy
            fp_matrix_test, metadata_df_test, info = preprocess_single_fingerprint_numpy(df_test, fingerprint_col)
            
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
    
    def _get_diverse_predictions(self, predictions, metadata, n=200, similarity_threshold=0.8):
        """
        Select chemically diverse molecules from top predictions.
        
        Note: This is a simplified implementation. For full chemical diversity,
        you would need molecular fingerprints and RDKit.
        
        Parameters:
        -----------
        predictions : numpy.ndarray
            Array of prediction probabilities
        metadata : pandas.DataFrame
            DataFrame containing metadata for each prediction
        n : int
            Number of diverse molecules to select
        similarity_threshold : float
            Similarity threshold for diversity selection
        
        Returns:
        --------
        dict
            Dictionary containing indices, probabilities, and metadata of diverse predictions
        """
        print(f"Selecting {n} chemically diverse predictions...")
        print("Note: Using simplified diversity selection based on BB1_ID and BB2_ID")
        
        # Sort by prediction probability
        sorted_indices = np.argsort(predictions)[::-1]
        
        # Start with highest probability prediction
        diverse_indices = [sorted_indices[0]]
        used_bb1_ids = {metadata.iloc[sorted_indices[0]]['BB1_ID']}
        used_bb2_ids = {metadata.iloc[sorted_indices[0]]['BB2_ID']}
        
        # Select diverse molecules
        for idx in sorted_indices[1:]:
            if len(diverse_indices) >= n:
                break
                
            bb1_id = metadata.iloc[idx]['BB1_ID']
            bb2_id = metadata.iloc[idx]['BB2_ID']
            
            # Simple diversity criterion: different building blocks
            if bb1_id not in used_bb1_ids or bb2_id not in used_bb2_ids:
                diverse_indices.append(idx)
                used_bb1_ids.add(bb1_id)
                used_bb2_ids.add(bb2_id)
        
        # If we don't have enough diverse molecules, fill with highest remaining probabilities
        if len(diverse_indices) < n:
            for idx in sorted_indices:
                if idx not in diverse_indices:
                    diverse_indices.append(idx)
                    if len(diverse_indices) >= n:
                        break
        
        # Convert to numpy array and limit to n
        diverse_indices = np.array(diverse_indices[:n])
        
        # Extract results
        diverse_probabilities = predictions[diverse_indices]
        diverse_metadata = metadata.iloc[diverse_indices].copy()
        
        # Add diversity information
        diverse_metadata['Prediction_Probability'] = diverse_probabilities
        diverse_metadata['Diversity_Rank'] = range(1, len(diverse_indices) + 1)
        
        print(f"Selected {len(diverse_indices)} diverse predictions:")
        print(f"  Highest probability: {diverse_probabilities.max():.4f}")
        print(f"  Lowest probability: {diverse_probabilities.min():.4f}")
        print(f"  Mean probability: {diverse_probabilities.mean():.4f}")
        print(f"  Unique BB1_IDs: {len(diverse_metadata['BB1_ID'].unique())}")
        print(f"  Unique BB2_IDs: {len(diverse_metadata['BB2_ID'].unique())}")
        
        return {
            'indices': diverse_indices,
            'probabilities': diverse_probabilities,
            'metadata': diverse_metadata,
            'count': len(diverse_indices)
        }

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

