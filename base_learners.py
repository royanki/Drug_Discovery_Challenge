import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
# custom implementation in Pytorch is not responding on 8GB M1 Apple,so going back to 
# scikit-learn implementation
# from .pytorch_neural_net import PyTorchNeuralNetClassifier 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as SKLogisticRegression

class BaseLearnerFactory:
    """Factory for creating different base learners."""
    
    @staticmethod
    def create_base_learner(config):
        """Create a base learner based on configuration."""
        learner_type = config['type'].lower()
        params = config.get('params', {})
        
        if learner_type == 'xgboost':
            return xgb.XGBClassifier(**params)
        elif learner_type == 'random_forest':
            return RandomForestClassifier(**params)
        elif learner_type == 'neural_net':
            return MLPClassifier(**params)
            # return PyTorchNeuralNetClassifier(**params)
        elif learner_type == 'svm':
            return SVC(probability=True, **params)
        elif learner_type == 'logistic_regression':
            return SKLogisticRegression(**params)
        else:
            raise ValueError(f"Unknown base learner type: {learner_type}")

# Predefined configurations
BASE_LEARNER_CONFIGS = {
    'xgboost': {
        'type': 'xgboost',
        'params': {
            'n_estimators': 500,          # More trees
            'max_depth': 8,               # Deeper trees for complex patterns
            'learning_rate': 0.05,        # Lower LR, more trees
            'subsample': 0.85,            # More data per tree
            'colsample_bytree': 0.85,     # More features per tree
            'reg_alpha': 0.1,             # L1 regularization
            'reg_lambda': 1.0,            # L2 regularization
            'scale_pos_weight': 2.0,      # Handle class imbalance
            'early_stopping_rounds': 10,  # Early Stopping
        }
    },
    'random_forest': {
        'type': 'random_forest',
        'params': {
            'n_estimators': 100,
            'max_depth': None,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
    },
    'neural_net': {
        'type': 'neural_net',
        'params': {
            'hidden_layer_sizes': (256, 32),#(128, 32),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 512,
            'learning_rate_init': 0.001,
            'max_iter': 200,
            'early_stopping': True,
            'random_state': 42,
            'tol': 1e-4,
            'validation_fraction': 0.1,
            'beta_1': 0.9,
            'beta_2': 0.999
        }
    },
    'svm': {
        'type': 'svm',
        'params': {
            'C': 1.0,
            'kernel': 'rbf',
            'class_weight': 'balanced',
            'random_state': 42
        }
    }
}

def get_base_learner_config(name):
    """Get predefined base learner configuration."""
    if name in BASE_LEARNER_CONFIGS:
        return BASE_LEARNER_CONFIGS[name].copy()
    else:
        raise ValueError(f"Unknown base learner: {name}. Available: {list(BASE_LEARNER_CONFIGS.keys())}")