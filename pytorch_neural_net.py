import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class PyTorchNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[100, 100], output_dim=2, dropout_rate=0.2):
        super(PyTorchNeuralNetModel, self).__init__()
        
        # Build layers dynamically based on hidden_dims
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)


class PyTorchNeuralNetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 hidden_layer_sizes=(100, 100),
                 activation='relu', 
                 solver='adam',
                 alpha=0.0001,  # L2 regularization parameter
                 batch_size=64, 
                 learning_rate_init=0.001,
                 max_iter=200,
                 tol=1e-4,
                 early_stopping=False,
                 validation_fraction=0.1,
                 beta_1=0.9,  # For adam optimizer
                 beta_2=0.999,  # For adam optimizer
                 epsilon=1e-8,  # For adam optimizer
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 shuffle=True,
                 n_iter_no_change=10,
                 dropout_rate=0.2,
                 device=None):
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.shuffle = shuffle
        self.n_iter_no_change = n_iter_no_change
        self.dropout_rate = dropout_rate
        
        # Set device (GPU if available, otherwise CPU)
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')  # Apple Silicon GPU
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
            
        self.scaler = StandardScaler()
        self._model = None
    
    # Add this property right here, after __init__
    @property
    def _estimator_type(self):
        return "classifier"

    def _get_optimizer(self):
        if self.solver == 'adam':
            return optim.Adam(
                self._model.parameters(),
                lr=self.learning_rate_init,
                betas=(self.beta_1, self.beta_2),
                eps=self.epsilon,
                weight_decay=self.alpha
            )
        elif self.solver == 'sgd':
            return optim.SGD(
                self._model.parameters(),
                lr=self.learning_rate_init,
                momentum=self.beta_1,
                weight_decay=self.alpha
            )
        else:
            raise ValueError(f"Solver {self.solver} not supported. Use 'adam' or 'sgd'.")
    
    def _get_criterion(self):
        return nn.CrossEntropyLoss()
    
    def _initialize_model(self, input_dim, output_dim):
        # Create a new model instance
        model = PyTorchNeuralNetModel(
            input_dim=input_dim,
            hidden_dims=list(self.hidden_layer_sizes),
            output_dim=output_dim,
            dropout_rate=self.dropout_rate
        )
        return model.to(self.device)
    
    def fit(self, X, y):
        # Check inputs and set up internal variables
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.X_shape_ = X.shape
        
        # Scale the input features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Handle validation if early stopping is enabled
        if self.early_stopping:
            val_size = int(self.validation_fraction * len(dataset))
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size], 
                generator=torch.Generator().manual_seed(self.random_state or 42)
            )
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=self.shuffle
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )
        else:
            train_loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=self.shuffle
            )
            val_loader = None
        
        # Initialize model
        self._model = self._initialize_model(X.shape[1], self.n_classes_)
        
        # Set up optimizer and loss function
        optimizer = self._get_optimizer()
        criterion = self._get_criterion()
        
        # Training loop
        best_loss = float('inf')
        no_improvement_count = 0
        best_model_state = None
        
        for epoch in range(self.max_iter):
            # Training phase
            self._model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self._model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
                
            train_loss /= len(train_loader.dataset)
            
            # Validation phase if early stopping is enabled
            if self.early_stopping and val_loader is not None:
                self._model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self._model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item() * batch_X.size(0)
                        
                val_loss /= len(val_loader.dataset)
                
                # Check for improvement
                if val_loss + self.tol < best_loss:
                    best_loss = val_loss
                    no_improvement_count = 0
                    best_model_state = self._model.state_dict().copy()
                else:
                    no_improvement_count += 1
                    
                # Early stopping check
                if no_improvement_count >= self.n_iter_no_change:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
                    
                if self.verbose:
                    print(f"Epoch {epoch+1}/{self.max_iter}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                # Without validation
                if self.verbose:
                    print(f"Epoch {epoch+1}/{self.max_iter}, Train Loss: {train_loss:.4f}")
        
        # Load best model if early stopping was used
        if self.early_stopping and best_model_state is not None:
            self._model.load_state_dict(best_model_state)
        
        self.is_fitted_ = True
        return self
    
    def predict_proba(self, X):
        check_is_fitted(self, ["is_fitted_"])
        X = check_array(X)
        
        # Scale the input features
        X_scaled = self.scaler.transform(X)
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Set model to evaluation mode
        self._model.eval()
        
        # Get predictions
        with torch.no_grad():
            logits = self._model(X_tensor)
            probas = torch.softmax(logits, dim=1)
            
        return probas.cpu().numpy()
    
    def predict(self, X):
        check_is_fitted(self, ["is_fitted_"])
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

    def decision_function(self, X):
        """
        Return decision function values for samples in X.
        
        For binary classification, returns the raw decision function values.
        For multi-class classification, returns the one-vs-rest decision function.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        array-like of shape (n_samples,) for binary classification or 
        (n_samples, n_classes) for multi-class classification
            The decision function values.
        """
        check_is_fitted(self, ["is_fitted_"])
        X = check_array(X)
        
        # Scale the input features
        X_scaled = self.scaler.transform(X)
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Set model to evaluation mode
        self._model.eval()
        
        # Get raw logits (pre-softmax)
        with torch.no_grad():
            logits = self._model(X_tensor)
        
        # For binary classification (2 classes), return scores for positive class
        if self.n_classes_ == 2:
            return logits[:, 1].cpu().numpy()
        
        # For multi-class, return all logits
        return logits.cpu().numpy()

        
# Example of how to use it in your pipeline:
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Instead of using MLPClassifier from sklearn:
# from sklearn.neural_network import MLPClassifier

# Replace with:
# from pytorch_neural_net import PyTorchNeuralNetClassifier

# Your pipeline setup remains the same:
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', PyTorchNeuralNetClassifier(
        hidden_layer_sizes=(200, 200, 100),
        max_iter=100,
        early_stopping=True,
        batch_size=128,
        verbose=1
    ))
])

# And the rest of your code works as before:
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)
"""
