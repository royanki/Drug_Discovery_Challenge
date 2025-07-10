import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
# from .pytorch_neural_net import PyTorchNeuralNetClassifier
from .utils import calculate_comprehensive_metrics, print_metrics

class MetaLearnerTrainer:
    """Class for training meta-learners."""
    
    def __init__(self, base_learner_config=None, cv_folds=5, random_state=42):
        """Initialize with base learner config for training."""
        self.base_learner_config = base_learner_config
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Define meta-learners
        self.meta_learners = {
            'LogisticRegression': LogisticRegression(
                random_state=random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'MLP_Compact': MLPClassifier(
                hidden_layer_sizes=(64, 32),        # Smaller, faster
                activation='relu',
                solver='adam',
                alpha=0.001,                        # L2 regularization
                batch_size='auto',                  # Let sklearn decide
                learning_rate_init=0.001,
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=random_state
            ),
            'MLP_Deep': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),   # Deeper network
                activation='relu',
                solver='adam',
                alpha=0.005,                        # More regularization for deeper net
                batch_size='auto',
                learning_rate_init=0.0005,          # Lower LR for deeper net
                max_iter=250,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,                # More patience for deeper net
                random_state=random_state
            ),
            'MLP_Wide': MLPClassifier(
                hidden_layer_sizes=(256, 128),      # Wider network
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate_init=0.001,
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=random_state
            ),
            'MLP_Balanced': MLPClassifier(
                hidden_layer_sizes=(200, 100, 50),  # Balanced architecture
                activation='relu',
                solver='adam',
                alpha=0.003,                        # Moderate regularization
                batch_size='auto',
                learning_rate_init=0.0008,
                max_iter=220,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=12,
                random_state=random_state
            )
            # KNN and MLP are removed but could be added back in future
            # # Add KNN meta-learner
            # 'KNN': KNeighborsClassifier(
            #     n_neighbors=5,
            #     weights='uniform',
            #     algorithm='auto',
            #     leaf_size=30,
            #     p=2,  # Euclidean distance
            #     metric='minkowski',
            #     n_jobs=-1  # Use all available cores
            # ),
            # # Add MLP meta-learner
            # 'MLP': PyTorchNeuralNetClassifier(
            #     hidden_layer_sizes=(100, 32),  # Relatively small for meta-learning
            #     activation='relu',
            #     solver='adam',
            #     alpha=0.0001,
            #     batch_size=512,#256,
            #     learning_rate_init=0.001,
            #     max_iter=100,
            #     early_stopping=True,
            #     validation_fraction=0.1,
            #     random_state=random_state,
            #     verbose=0,
            #     dropout_rate=0.2
                
            # ),

        }
    
    def get_fresh_meta_learners(self):
        """
        Return fresh instances of all meta-learners.
        Useful for final training to avoid any state from CV training.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier

        fresh_meta_learners = {
            'LogisticRegression': LogisticRegression(
                random_state=random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'MLP_Compact': MLPClassifier(
                hidden_layer_sizes=(64, 32),        # Smaller, faster
                activation='relu',
                solver='adam',
                alpha=0.001,                        # L2 regularization
                batch_size='auto',                  # Let sklearn decide
                learning_rate_init=0.001,
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=random_state
            ),
            'MLP_Deep': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),   # Deeper network
                activation='relu',
                solver='adam',
                alpha=0.005,                        # More regularization for deeper net
                batch_size='auto',
                learning_rate_init=0.0005,          # Lower LR for deeper net
                max_iter=250,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,                # More patience for deeper net
                random_state=random_state
            ),
            'MLP_Wide': MLPClassifier(
                hidden_layer_sizes=(256, 128),      # Wider network
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate_init=0.001,
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=random_state
            ),
            'MLP_Balanced': MLPClassifier(
                hidden_layer_sizes=(200, 100, 50),  # Balanced architecture
                activation='relu',
                solver='adam',
                alpha=0.003,                        # Moderate regularization
                batch_size='auto',
                learning_rate_init=0.0008,
                max_iter=220,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=12,
                random_state=random_state
            )
        }

    def train_meta_learners(self, meta_features, labels, fingerprint_cols):
        """Train all meta-learners and evaluate performance."""
        X_meta_train = meta_features['train']
        X_meta_val = meta_features['val']
        y_train = labels['train']
        y_val = labels['val']
        
        # Check class imbalance
        pos_ratio = y_train.mean()
        print(f"\nClass distribution in meta-training: {pos_ratio:.4f} positive, {1-pos_ratio:.4f} negative")
        
        # Train and evaluate each meta-learner
        results = {}
        
        for name, meta_learner in self.meta_learners.items():
            print(f"\nTraining and evaluating {name}...")

            # Train meta-learner
            meta_learner.fit(X_meta_train, y_train)
            
            # Get predictions
            y_meta_pred_proba = meta_learner.predict_proba(X_meta_val)[:, 1]
            y_meta_pred = meta_learner.predict(X_meta_val)
            
            # Calculate metrics
            metrics = calculate_comprehensive_metrics(y_val, y_meta_pred, y_meta_pred_proba)
            
            # Get feature importances
            feature_importances = self._get_feature_importances(meta_learner, fingerprint_cols)
            
            # Store results
            results[name] = {
                'model': meta_learner,
                'metrics': metrics,
                'feature_importances': feature_importances,
                'predictions': {
                    'proba': y_meta_pred_proba,
                    'binary': y_meta_pred
                }
            }
            
            # Print metrics
            print_metrics(metrics, f"{name} Meta-learner Metrics")
        
        return results
    
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
            # For models without direct feature importance (KNN, MLP)
            # Either return None or compute permutation importance
            for i, fp in enumerate(fingerprint_cols):
                feature_importances[fp] = 0.0  # Placeholder
        
        return feature_importances