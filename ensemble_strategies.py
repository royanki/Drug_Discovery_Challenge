import numpy as np
from sklearn.linear_model import Ridge
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize_scalar, minimize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
# from .pytorch_neural_net import PyTorchNeuralNetClassifier
from sklearn.model_selection import train_test_split
from .utils import calculate_comprehensive_metrics

class MetaLearnerEnsemble:
    """Class for combining meta-learners using different strategies."""
    
    def __init__(self, meta_learner_results, meta_features, labels, random_state=42):
        self.meta_learner_results = meta_learner_results
        self.meta_features = meta_features
        self.labels = labels
        self.random_state = random_state
        self.ensemble_results = {}

    def create_weighted_ensemble(self, optimization_metric='f1'):
        """Create weighted ensemble by optimizing weights for all meta-learners."""
        print("\nCreating weighted ensemble...")
        
        X_meta_val = self.meta_features['val']
        y_val = self.labels['val']
        
        # Get predictions from all meta-learners
        meta_proba = {}
        for name in self.meta_learner_results.keys():
            meta_proba[name] = self.meta_learner_results[name]['predictions']['proba']
        
        # Meta-learner names for optimization
        meta_learner_names = list(meta_proba.keys())
        n_learners = len(meta_learner_names)
        
        # Define objective function for multiple weights
        def objective(weights):
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)
            
            # Calculate weighted ensemble prediction
            ensemble_proba = np.zeros_like(meta_proba[meta_learner_names[0]])
            for i, name in enumerate(meta_learner_names):
                ensemble_proba += weights[i] * meta_proba[name]
            
            ensemble_pred = (ensemble_proba >= 0.5).astype(int)
            
            if optimization_metric == 'f1':
                from sklearn.metrics import f1_score
                return -f1_score(y_val, ensemble_pred)
            elif optimization_metric == 'auc':
                from sklearn.metrics import roc_auc_score
                return -roc_auc_score(y_val, ensemble_proba)
        
        # Initial weights (equal weighting)
        initial_weights = np.ones(n_learners) / n_learners
        
        # Constraints to ensure weights sum to 1 and are non-negative
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # weights sum to 1
        ]
        
        # Bounds to ensure weights are between 0 and 1
        bounds = [(0.0, 1.0) for _ in range(n_learners)]
        
        # Find optimal weights
        from scipy.optimize import minimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Get optimized weights
        optimal_weights = result.x
        
        # Print weights
        print("Optimal weights:")
        for i, name in enumerate(meta_learner_names):
            print(f"  {name}: {optimal_weights[i]:.3f}")
        
        # Create final ensemble
        ensemble_proba = np.zeros_like(meta_proba[meta_learner_names[0]])
        for i, name in enumerate(meta_learner_names):
            ensemble_proba += optimal_weights[i] * meta_proba[name]
        
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = calculate_comprehensive_metrics(y_val, ensemble_pred, ensemble_proba)
        
        # Store weights in a more accessible format
        weights_dict = {name: weight for name, weight in zip(meta_learner_names, optimal_weights)}
        
        self.ensemble_results['weighted'] = {
            'metrics': metrics,
            'weights': weights_dict,  # Store as dictionary for easier access
            'predictions': {'proba': ensemble_proba, 'binary': ensemble_pred}
        }
        
        return metrics, ensemble_proba, weights_dict
    
 

    def create_threshold_based_hybrid(self, confidence_thresholds=None, default_high_threshold=0.7, default_low_threshold=0.3):
        """
        Create hybrid ensemble using threshold-based selection with any number of meta-learners.
        
        Parameters:
        -----------
        confidence_thresholds : dict, optional
            Dictionary of thresholds for each meta-learner. Format:
            {learner_name: {'high': high_threshold, 'low': low_threshold}}
        default_high_threshold : float, default=0.7
            Default high confidence threshold for positive predictions
        default_low_threshold : float, default=0.3
            Default low confidence threshold for negative predictions
        """
        print("\nCreating threshold-based hybrid ensemble...")
        
        y_val = self.labels['val']
        
        # Get all meta-learner predictions
        meta_probas = {}
        for name in self.meta_learner_results.keys():
            meta_probas[name] = self.meta_learner_results[name]['predictions']['proba']
        
        # Set default thresholds if none provided
        if confidence_thresholds is None:
            confidence_thresholds = {}
            for name in meta_probas.keys():
                confidence_thresholds[name] = {
                    'high': default_high_threshold,
                    'low': default_low_threshold
                }
        
        # Initialize hybrid prediction array
        hybrid_proba = np.zeros_like(list(meta_probas.values())[0])
        
        # Determine where each learner is confident
        high_conf_regions = {}
        for name, proba in meta_probas.items():
            # High confidence positive predictions
            high_conf_regions[f"{name}_high"] = proba >= confidence_thresholds[name]['high']
            
            # High confidence negative predictions
            high_conf_regions[f"{name}_low"] = proba <= confidence_thresholds[name]['low']
        
        # Calculate confidence strength for each learner
        # This is used to prioritize learners when multiple are confident
        confidence_strength = {}
        for name, proba in meta_probas.items():
            high_regions = high_conf_regions[f"{name}_high"]
            low_regions = high_conf_regions[f"{name}_low"]
            
            # For high confidence regions, measure how far above threshold
            if np.any(high_regions):
                confidence_strength[f"{name}_high"] = np.mean(proba[high_regions] - confidence_thresholds[name]['high'])
            else:
                confidence_strength[f"{name}_high"] = 0
                
            # For low confidence regions, measure how far below threshold
            if np.any(low_regions):
                confidence_strength[f"{name}_low"] = np.mean(confidence_thresholds[name]['low'] - proba[low_regions])
            else:
                confidence_strength[f"{name}_low"] = 0
        
        # Sort regions by confidence strength
        sorted_regions = sorted(
            confidence_strength.keys(), 
            key=lambda x: confidence_strength[x], 
            reverse=True
        )
        
        # Create mask to track which samples are already assigned
        assigned = np.zeros_like(hybrid_proba, dtype=bool)
        sample_counts = {}
        
        # Assign predictions based on confidence, prioritizing highest confidence regions
        for region in sorted_regions:
            name, conf_type = region.rsplit('_', 1)
            mask = high_conf_regions[region] & ~assigned
            
            if np.any(mask):
                hybrid_proba[mask] = meta_probas[name][mask]
                assigned[mask] = True
                sample_counts[region] = np.sum(mask)
        
        # Use weighted average for uncertain cases
        uncertain = ~assigned
        if np.any(uncertain):
            # Equal weights for uncertain regions
            weights = {name: 1.0 / len(meta_probas) for name in meta_probas.keys()}
            
            # Apply weighted average
            for name, proba in meta_probas.items():
                hybrid_proba[uncertain] += weights[name] * proba[uncertain]
                
            sample_counts['uncertain'] = np.sum(uncertain)
        
        # Print assignment statistics
        print("\nSample assignment statistics:")
        for region, count in sample_counts.items():
            if region == 'uncertain':
                print(f"  Uncertain (using ensemble): {count} samples")
            else:
                name, conf_type = region.rsplit('_', 1)
                conf_label = "positive" if conf_type == "high" else "negative"
                print(f"  {name} high confidence {conf_label}: {count} samples")
        
        hybrid_pred = (hybrid_proba >= 0.5).astype(int)
        metrics = calculate_comprehensive_metrics(y_val, hybrid_pred, hybrid_proba)
        
        self.ensemble_results['threshold'] = {
            'metrics': metrics,
            'thresholds': confidence_thresholds,
            'predictions': {'proba': hybrid_proba, 'binary': hybrid_pred}
        }
        
        return metrics, hybrid_proba
    

    
    def create_calibrated_ensemble(self, weights=None):
        """
        Create ensemble with calibrated probabilities using all available meta-learners.
        
        Parameters:
        -----------
        weights : dict, optional
            Dictionary of weights for each meta-learner. If None, equal weights are used.
        """
        print("\nCreating calibrated ensemble...")
        
        X_meta_train = self.meta_features['train']
        X_meta_val = self.meta_features['val']
        y_train = self.labels['train']
        y_val = self.labels['val']
        
        # Get all meta-learner models
        calibrated_models = {}
        calibrated_probas = {}
        excluded_models = []

        # Process each meta-learner
        for name, result in self.meta_learner_results.items():
            # if name == "MLP":
            #     excluded_models.append(name)
            #     print(f"  Excluding {name} from calibration")
            #     continue

            print(f"  Calibrating {name}...")
            
            # Get base model
            base_model = result['model']
            
            # Create and fit calibration model
            calibrated = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
            calibrated.fit(X_meta_train, y_train)
            
            # Store calibrated model
            calibrated_models[name] = calibrated
            
            # Get calibrated probabilities
            calibrated_probas[name] = calibrated.predict_proba(X_meta_val)[:, 1]
        
        # If no weights provided, use equal weighting
        if weights is None:
            n_learners = len(calibrated_probas)
            weights = {name: 1.0 / n_learners for name in calibrated_probas.keys()}
        
        # Validate weights
        if set(weights.keys()) != set(calibrated_probas.keys()):
            raise ValueError("Weights must be provided for all meta-learners")
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Print weights being used
        print("\nUsing weights:")
        for name, weight in weights.items():
            print(f"  {name}: {weight:.3f}")
        
        # Combine calibrated probabilities using weights
        calibrated_ensemble_proba = np.zeros_like(list(calibrated_probas.values())[0])
        for name, proba in calibrated_probas.items():
            calibrated_ensemble_proba += weights[name] * proba
        
        calibrated_ensemble_pred = (calibrated_ensemble_proba >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = calculate_comprehensive_metrics(y_val, calibrated_ensemble_pred, calibrated_ensemble_proba)
        
        self.ensemble_results['calibrated'] = {
            'metrics': metrics,
            'calibrated_models': calibrated_models,
            'weights': weights,
            'predictions': {'proba': calibrated_ensemble_proba, 'binary': calibrated_ensemble_pred}
        }
        
        return metrics, calibrated_ensemble_proba


    
    def create_robust_ensemble(self, holdout_ratio=0.3, weight_step=0.1):
        """
        Create robust ensemble using holdout validation with any number of meta-learners.
        
        Parameters:
        -----------
        holdout_ratio : float, default=0.3
            Ratio of meta-training data to hold out for weight optimization
        weight_step : float, default=0.1
            Step size for weight grid search (smaller values give more precise weights
            but increase computation time)
        """
        print(f"\nCreating robust ensemble (holdout ratio: {holdout_ratio})...")
        
        X_meta_train = self.meta_features['train']
        X_meta_val = self.meta_features['val']
        y_train = self.labels['train']
        y_val = self.labels['val']
        
        # Further split meta-training data
        X_comb_train, X_comb_val, y_comb_train, y_comb_val = train_test_split(
            X_meta_train, y_train, test_size=holdout_ratio, 
            random_state=self.random_state, stratify=y_train
        )

        # IMPORT META-LEARNERS FROM MetaLearnerTrainer (no hardcoding!)
        try:
            from .meta_learners import MetaLearnerTrainer
            
            # Create trainer to get fresh meta-learner instances
            trainer = MetaLearnerTrainer(random_state=self.random_state)
            
            # Get fresh instances of all meta-learners
            available_meta_learners = trainer.get_fresh_meta_learners()
            
            print(f"Imported {len(available_meta_learners)} meta-learners from MetaLearnerTrainer")
            
        except Exception as e:
            print(f"Error importing meta-learners: {e}")
            print("Falling back to weighted ensemble")
            return self.create_weighted_ensemble()
        
        # Get all meta-learner classes
        meta_learner_models = {}
        meta_learner_probas = {}
        
        # Retrain all meta-learners on reduced training set
        for name, result in self.meta_learner_results.items():
            print(f"  Retraining {name} on holdout data...")
            
            # Create a new instance of the same model type
            if name in available_meta_learners:
                print(f"  Retraining {name} on holdout data...")
                
                try:
                    # Get fresh model instance from MetaLearnerTrainer
                    model = available_meta_learners[name]
                    
                    # Fit model on holdout training data
                    model.fit(X_comb_train, y_comb_train)
                    meta_learner_models[name] = model
                    
                    # Generate predictions on holdout validation data
                    meta_learner_probas[name] = model.predict_proba(X_comb_val)[:, 1]
                    
                    print(f"    ✓ {name} retrained successfully")
                    
                except Exception as e:
                    print(f"    ❌ Error retraining {name}: {e}")
                    print(f"       Excluding {name} from robust ensemble")
                    continue
            else:
                print(f"  Warning: {name} not found in MetaLearnerTrainer, skipping")

        
        # For more than 2 meta-learners, we need a more sophisticated weight optimization
        # approach than the simple grid search used for 2 meta-learners
        
        if len(meta_learner_models) == 2:
            # If we only have 2 meta-learners, use the original grid search approach
            meta_names = list(meta_learner_models.keys())
            first_name, second_name = meta_names[0], meta_names[1]
            
            best_f1 = 0
            best_weights = {}
            
            for first_weight in np.arange(0.0, 1.1, weight_step):
                second_weight = 1 - first_weight
                ensemble_proba = (first_weight * meta_learner_probas[first_name] + 
                                 second_weight * meta_learner_probas[second_name])
                ensemble_pred = (ensemble_proba >= 0.5).astype(int)
                
                from sklearn.metrics import f1_score
                f1 = f1_score(y_comb_val, ensemble_pred)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_weights = {first_name: first_weight, second_name: second_weight}
        else:
            # For more than 2 meta-learners, use optimization approach
            from scipy.optimize import minimize
            
            meta_names = list(meta_learner_models.keys())
            n_learners = len(meta_names)
            
            # Objective function to minimize (negative F1 score)
            def objective(weights):
                # Normalize weights to sum to 1
                weights = weights / np.sum(weights)
                
                # Calculate weighted ensemble prediction
                ensemble_proba = np.zeros_like(meta_learner_probas[meta_names[0]])
                for i, name in enumerate(meta_names):
                    ensemble_proba += weights[i] * meta_learner_probas[name]
                
                ensemble_pred = (ensemble_proba >= 0.5).astype(int)
                
                from sklearn.metrics import f1_score
                return -f1_score(y_comb_val, ensemble_pred)
            
            # Initial weights (equal weighting)
            initial_weights = np.ones(n_learners) / n_learners
            
            # Constraints to ensure weights sum to 1 and are non-negative
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # weights sum to 1
            ]
            
            # Bounds to ensure weights are between 0 and 1
            bounds = [(0.0, 1.0) for _ in range(n_learners)]
            
            # Find optimal weights
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            # Get optimized weights
            optimal_weights = result.x
            best_f1 = -result.fun  # Convert back from negative
            
            # Store as dictionary
            best_weights = {name: weight for name, weight in zip(meta_names, optimal_weights)}
        
        # Print weights and performance
        print("\nBest weights found:")
        for name, weight in best_weights.items():
            print(f"  {name}: {weight:.3f}")
        print(f"Validation F1 with best weights: {best_f1:.3f}")
        
        # Apply to final validation set
        final_probas = {}
        for name, model in meta_learner_models.items():
            final_probas[name] = model.predict_proba(X_meta_val)[:, 1]
        
        # Calculate final ensemble prediction
        final_ensemble_proba = np.zeros_like(final_probas[meta_names[0]])
        for name, weight in best_weights.items():
            final_ensemble_proba += weight * final_probas[name]
        
        final_ensemble_pred = (final_ensemble_proba >= 0.5).astype(int)
        
        metrics = calculate_comprehensive_metrics(y_val, final_ensemble_pred, final_ensemble_proba)
        
        self.ensemble_results['robust'] = {
            'metrics': metrics,
            'weights': best_weights,
            'models': meta_learner_models,
            'predictions': {'proba': final_ensemble_proba, 'binary': final_ensemble_pred}
        }
        
        return metrics, final_ensemble_proba, best_weights


    
    def get_all_ensemble_results(self):
        """Return all ensemble results."""
        return self.ensemble_results


    def create_neural_network_ensemble(self):
        """
        Create ensemble specifically for neural network meta-learners.
        Combines multiple NN predictions with optimized weights.
        """
        print("\nCreating Neural Network Ensemble...")
        
        y_val = self.labels['val']
        
        # Get predictions from all NN meta-learners
        nn_predictions = {}
        nn_names = ['MLP_Compact', 'MLP_Deep', 'MLP_Wide', 'MLP_Balanced']
        
        for name in nn_names:
            if name in self.meta_learner_results:
                nn_predictions[name] = self.meta_learner_results[name]['predictions']['proba']
                print(f"  Using {name} predictions")
            else:
                print(f"  Warning: {name} not found in results")
        
        if len(nn_predictions) < 2:
            print("  Not enough NN meta-learners for ensemble")
            return None
        
        # Optimize weights for NN ensemble
        from scipy.optimize import minimize
        
        nn_names_list = list(nn_predictions.keys())
        n_nns = len(nn_names_list)
        
        def objective(weights):
            weights = weights / np.sum(weights)  # Normalize
            
            ensemble_proba = np.zeros_like(nn_predictions[nn_names_list[0]])
            for i, name in enumerate(nn_names_list):
                ensemble_proba += weights[i] * nn_predictions[name]
            
            ensemble_pred = (ensemble_proba >= 0.5).astype(int)
            
            from sklearn.metrics import f1_score
            return -f1_score(y_val, ensemble_pred)
        
        # Optimize weights
        initial_weights = np.ones(n_nns) / n_nns
        bounds = [(0.0, 1.0) for _ in range(n_nns)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        
        print("Optimal NN ensemble weights:")
        for i, name in enumerate(nn_names_list):
            print(f"  {name}: {optimal_weights[i]:.3f}")
        
        # Create final ensemble
        nn_ensemble_proba = np.zeros_like(nn_predictions[nn_names_list[0]])
        for i, name in enumerate(nn_names_list):
            nn_ensemble_proba += optimal_weights[i] * nn_predictions[name]
        
        nn_ensemble_pred = (nn_ensemble_proba >= 0.5).astype(int)
        
        # Calculate metrics
        from .utils import calculate_comprehensive_metrics
        metrics = calculate_comprehensive_metrics(y_val, nn_ensemble_pred, nn_ensemble_proba)
        
        weights_dict = {name: weight for name, weight in zip(nn_names_list, optimal_weights)}
        
        self.ensemble_results['neural_network'] = {
            'metrics': metrics,
            'weights': weights_dict,
            'predictions': {'proba': nn_ensemble_proba, 'binary': nn_ensemble_pred}
        }
        
        return metrics, nn_ensemble_proba, weights_dict
 

    def create_all_meta_ensemble(self):
        """
        Create ensemble using ALL meta-learners (traditional + multiple NNs).
        This is the ultimate ensemble approach.
        """
        print("\nCreating All Meta-Learners Ensemble...")
        
        y_val = self.labels['val']
        
        # Get predictions from ALL meta-learners
        all_predictions = {}
        for name in self.meta_learner_results.keys():
            all_predictions[name] = self.meta_learner_results[name]['predictions']['proba']
        
        print(f"Combining {len(all_predictions)} meta-learners:")
        for name in all_predictions.keys():
            print(f"  - {name}")
        
        # Optimize weights for all meta-learners
        from scipy.optimize import minimize
        
        learner_names = list(all_predictions.keys())
        n_learners = len(learner_names)
        
        def objective(weights):
            weights = weights / np.sum(weights)
            
            ensemble_proba = np.zeros_like(all_predictions[learner_names[0]])
            for i, name in enumerate(learner_names):
                ensemble_proba += weights[i] * all_predictions[name]
            
            ensemble_pred = (ensemble_proba >= 0.5).astype(int)
            
            from sklearn.metrics import f1_score
            return -f1_score(y_val, ensemble_pred)
        
        # Optimize
        initial_weights = np.ones(n_learners) / n_learners
        bounds = [(0.0, 1.0) for _ in range(n_learners)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        
        print("Optimal all-meta-learners weights:")
        for i, name in enumerate(learner_names):
            print(f"  {name}: {optimal_weights[i]:.3f}")
        
        # Create final ensemble
        all_ensemble_proba = np.zeros_like(all_predictions[learner_names[0]])
        for i, name in enumerate(learner_names):
            all_ensemble_proba += optimal_weights[i] * all_predictions[name]
        
        all_ensemble_pred = (all_ensemble_proba >= 0.5).astype(int)
        
        # Calculate metrics
        from .utils import calculate_comprehensive_metrics
        metrics = calculate_comprehensive_metrics(y_val, all_ensemble_pred, all_ensemble_proba)
        
        weights_dict = {name: weight for name, weight in zip(learner_names, optimal_weights)}
        
        self.ensemble_results['all_meta'] = {
            'metrics': metrics,
            'weights': weights_dict,
            'predictions': {'proba': all_ensemble_proba, 'binary': all_ensemble_pred}
        }
        
        return metrics, all_ensemble_proba, weights_dict
 
