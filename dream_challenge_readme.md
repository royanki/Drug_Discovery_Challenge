# DREAM Challenge: Drug Discovery Target 2035 (Step 1)

A comprehensive, memory-efficient stacking ensemble pipeline for molecular fingerprint-based drug discovery, featuring **two-phase workflow**, **chemical diversity selection**, and **intelligent submission generation** for the DREAM Challenge.

## 🎯 Overview

This experimental pipeline addresses the DREAM Challenge Drug Discovery Target 2035 (Step 1) by implementing a sophisticated stacking ensemble approach that:

- **Trains base learners** on individual molecular fingerprints (ECFP4, ECFP6, FCFP4, FCFP6, TOPTOR, MACCS, RDK, AVALON, ATOMPAIR)
- **Generates meta-features** through cross-validation to avoid overfitting
- **Combines meta-learners** using multiple ensemble strategies (weighted, threshold-based, calibrated, robust)
- **Selects diverse compounds** using RDKit-based Tanimoto similarity clustering
- **Generates submission files** in the required DREAM Challenge format

### 🚀 Key Features

- **🧪 Two-Phase Workflow**: Cross-validation for model selection + Final training for submissions
- **⚡ Memory-Efficient Processing**: Processes one fingerprint at a time to handle large datasets on limited hardware
- **🔬 Chemical Diversity Selection**: RDKit-based Tanimoto similarity clustering with preprocessed fingerprints
- **📊 Multiple Base Learners**: XGBoost, Random Forest, Neural Networks (PyTorch-based)
- **🤖 Advanced Meta-Learning**: Logistic Regression, Random Forest, KNN, Multiple Neural Network architectures
- **🎯 Intelligent Submission Generation**: Top confident + chemically diverse predictions
- **🛡️ Robust Error Handling**: Comprehensive fallback mechanisms and validation

## 📋 Requirements

### Hardware Requirements
- **Minimum RAM**: 8GB (tested on Mac M1 with 8GB RAM)
- **Recommended RAM**: 16GB+ for larger datasets
- **GPU**: Optional (PyTorch will use GPU if available)
- **Storage**: 2-5GB free space for intermediate results and submissions

### Software Requirements
- **Python**: 3.8+ (recommended: 3.9 or 3.10)
- **Operating System**: macOS, Linux, Windows

## 🔧 Installation

### Option 1: Using Conda (Recommended)

```bash
# Create a new conda environment
conda create -n dream-challenge python=3.9
conda activate dream-challenge

# Install core dependencies
conda install -c conda-forge pandas numpy scikit-learn matplotlib seaborn
conda install -c conda-forge rdkit  # Critical for chemical diversity
conda install -c conda-forge xgboost

# Install PyTorch (CPU version)
conda install pytorch torchvision cpuonly -c pytorch

# Install remaining packages with pip
pip install pyarrow tqdm memory-profiler
```

### Option 2: Using pip

```bash
# Create virtual environment
python -m venv dream-challenge-env
source dream-challenge-env/bin/activate  # On Windows: dream-challenge-env\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### requirements.txt
```txt
# Core Data Science Libraries
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0

# Machine Learning Libraries
xgboost>=1.5.0
torch>=1.10.0
torchvision>=0.11.0

# RDKit for Chemical Diversity (CRITICAL)
rdkit-pypi>=2022.3.1

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Data Processing
scipy>=1.7.0
pyarrow>=6.0.0

# Utilities
tqdm>=4.62.0
memory-profiler>=0.60.0

# Development (Optional)
jupyter>=1.0.0
pytest>=6.0.0
```

## 🚀 Quick Start

### 1. Setup Project Structure

```bash
# Clone or download the pipeline
git clone <repository-url>
cd dream-challenge-pipeline

# Create required directories
mkdir -p Data results/cv_results results/submissions

# Place your data files
# Data/WDR91.parquet          # Training data
# Data/test_set.parquet       # Test data for predictions
```

### 2. Configure Data Split (Important!)

**⚠️ Critical Step**: You must provide your own `disynthon_split.py` file:

```python
# disynthon_split.py - User-provided file
def disynthon_split_real(metadata_df):
    """
    Split data into train/validation/test sets.
    Customize this function based on your splitting strategy.
    
    Parameters:
    -----------
    metadata_df : pandas.DataFrame
        DataFrame containing metadata (BB1_ID, BB2_ID, LABEL)
    
    Returns:
    --------
    tuple: (train_meta, val_meta, test_meta)
        Three DataFrames with the splits
    """
    from sklearn.model_selection import train_test_split
    
    # Example implementation - customize as needed
    train_val, test_meta = train_test_split(
        metadata_df, test_size=0.2, stratify=metadata_df['LABEL'], random_state=42
    )
    
    train_meta, val_meta = train_test_split(
        train_val, test_size=0.25, stratify=train_val['LABEL'], random_state=42
    )
    
    return train_meta, val_meta, test_meta
```

### 3. Run Complete Workflow

#### Phase 1: Cross-Validation (Model Selection)

```bash
# Test with XGBoost base learner
python main.py --base-learner xgboost \
               --output results/cv_results/xgb_cv_results.pkl \
               --folds 5

# Compare different base learners
python main.py --base-learner random_forest \
               --output results/cv_results/rf_cv_results.pkl

python main.py --base-learner neural_net \
               --output results/cv_results/nn_cv_results.pkl
```

#### Phase 2: Final Training & Submission Generation

```bash
# Use best CV results for final training
python main.py --final-train \
               --cv-results results/cv_results/xgb_cv_results.pkl \
               --test-data Data/test_set.parquet \
               --submission-dir results/submissions/xgb \
               --output results/final_xgb_results.pkl
```

#### Phase 3: Convert to DREAM Challenge Format

```bash
# Convert pipeline outputs to official submission format
python create_submission.py \
    --submission-dir results/submissions/xgb \
    --team-name YourTeamName \
    --output-dir results/final_submissions
```

## 📁 Generated Output Files

### Cross-Validation Results
```
results/cv_results/
├── xgb_cv_results.pkl        # CV results with best strategy & weights
├── rf_cv_results.pkl         # Random Forest CV results
└── nn_cv_results.pkl         # Neural Network CV results
```

### Submission Files
```
results/submissions/xgb/
├── top_200_xgboost.csv       # 🏆 Top 200 most confident predictions
├── top_500_xgboost.csv       # 🏆 Top 500 most confident predictions
├── diverse_200_xgboost.csv   # 🧬 200 chemically diverse predictions
└── all_predictions_xgboost.csv # 📋 All predictions ranked by confidence
```

### Final DREAM Challenge Submission
```
results/final_submissions/
└── TeamYourTeamName.csv      # 📤 Ready for DREAM Challenge submission
```

## 💻 Programmatic Usage

### Example 1: Quick CV Run

```python
from main import run_with_xgboost, create_default_config

# Run with default settings
results = run_with_xgboost()

# Print best strategy
print(f"Best strategy: {results['best_strategy_info']['best_strategy_name']}")
print(f"Best F1-Score: {results['best_strategy_info']['best_strategy_metrics']['metrics']['F1-Score']:.4f}")
```

### Example 2: Custom Configuration

```python
from main import MemoryEfficientStacking, create_default_config
from base_learners import get_base_learner_config

# Create custom configuration
config = create_default_config()
config['fingerprint_cols'] = ['ECFP4', 'ECFP6', 'MACCS']  # Subset for faster testing
config['cv_folds'] = 3
config['base_learner_config'] = get_base_learner_config('random_forest')

# Run pipeline
pipeline = MemoryEfficientStacking(config)
results = pipeline.run_pipeline()

# Save results
pipeline.save_results('results/custom_cv_results.pkl')
```

### Example 3: Compare Base Learners

```python
from main import compare_base_learners

# Quick comparison with subset of fingerprints
results = compare_base_learners(
    base_learners=['xgboost', 'random_forest'],
    fingerprint_subset=['ECFP4', 'ECFP6'],
    cv_folds=3
)

# Print comparison
for learner, metrics in results.items():
    print(f"{learner}: F1={metrics['best_f1']:.4f}, AUC={metrics['best_auc']:.4f}")
```

### Example 4: Final Training from Code

```python
from final_train_predict import FinalTrainPredict

# Load CV results and run final training
trainer = FinalTrainPredict('results/cv_results/xgb_cv_results.pkl')

# Run complete final pipeline
results = trainer.run_complete_pipeline(
    test_data_path='Data/test_set.parquet',
    output_dir='results/submissions/final'
)

# Access predictions
predictions = results['predictions']['predictions']
print(f"Generated {len(predictions)} predictions")
print(f"Mean confidence: {predictions.mean():.4f}")
```

## ⚙️ Configuration Options

### Base Learner Options
```python
# Available base learners
base_learners = {
    'xgboost': 'XGBoost Classifier (recommended)',
    'random_forest': 'Random Forest Classifier',
    'neural_net': 'PyTorch Neural Network'
}
```

### Meta-Learner Options
```python
# Automatically included meta-learners
meta_learners = {
    'LogisticRegression': 'L2-regularized Logistic Regression',
    'RandomForest': 'Random Forest meta-learner',
    'KNN': 'K-Nearest Neighbors',
    'MLP': 'Multi-layer Perceptron (PyTorch)'
}
```

### Ensemble Strategies
```python
# Available ensemble strategies (automatically compared)
ensemble_strategies = [
    'weighted',      # Optimized weighted combination
    'threshold',     # Threshold-based hybrid approach
    'calibrated',    # Calibrated probability ensemble
    'robust'         # Robust validation-based combination
]
```

## 🧬 Chemical Diversity Details

### Fingerprint Processing
- **Preprocessing Consistency**: Same zero-variance removal for modeling and diversity
- **Combined Fingerprints**: All 9 fingerprint types concatenated (~8,000-12,000 informative bits)
- **Memory Efficient**: Processes one fingerprint type at a time

### Diversity Algorithm
1. **Candidate Selection**: Top 1,000 predictions by confidence
2. **Fingerprint Combination**: Concatenate all preprocessed fingerprints
3. **Similarity Calculation**: RDKit Tanimoto similarity
4. **Clustering**: Butina algorithm (similarity threshold = 0.7)
5. **Representative Selection**: Highest confidence from each cluster
6. **Gap Filling**: Add high-confidence molecules to reach target count

### Fallback Mechanisms
- **RDKit Failure**: Falls back to building block diversity
- **Column Mismatch**: Flexible ID column detection
- **Memory Issues**: Automatic memory cleanup and garbage collection

## 📊 Performance Metrics

The pipeline reports comprehensive evaluation metrics:

### Standard Classification Metrics
- **Precision, Recall, F1-Score**: Standard binary classification metrics
- **AUC-ROC**: Area under ROC curve
- **PR AUC**: Area under Precision-Recall curve

### Drug Discovery Specific Metrics
- **PPV@128**: Positive Predictive Value at top 128 predictions
- **PPV@256**: Positive Predictive Value at top 256 predictions
- **Hits@128**: Number of actual positives in top 128 predictions
- **Hits@256**: Number of actual positives in top 256 predictions

### Diversity Metrics
- **Cluster Count**: Number of chemical clusters identified
- **Average Cluster Size**: Mean molecules per cluster
- **Tanimoto Statistics**: Similarity distribution analysis

## 🔧 Troubleshooting

### Common Issues

#### 1. **Import Errors**
```bash
# Error: ImportError: attempted relative import with no known parent package
# Solution: Run as module from project root
python main.py  # ✅ Correct

# Don't run from subdirectory
cd subdirectory && python ../main.py  # ❌ Wrong
```

#### 2. **RDKit Installation Issues**
```bash
# If pip install fails:
pip install rdkit-pypi

# If that fails, use conda:
conda install -c conda-forge rdkit
```

#### 3. **Memory Issues**
```python
# Reduce fingerprints for testing
config['fingerprint_cols'] = ['ECFP4', 'ECFP6']  # Instead of all 9

# Reduce CV folds
config['cv_folds'] = 3  # Instead of 5

# Reduce candidate pool for diversity
candidate_pool_size = 500  # Instead of 1000
```

#### 4. **Missing Test Data Path**
```bash
# Error: Test data file not found
# Solution: Always specify test data path
python main.py --final-train \
               --cv-results results.pkl \
               --test-data Data/YOUR_TEST_FILE.parquet  # ✅ Required
```

#### 5. **disynthon_split.py Missing**
```python
# Create your own splitting function
def disynthon_split_real(metadata_df):
    # Implement your data splitting strategy
    # Must return (train_meta, val_meta, test_meta)
    pass
```

### Performance Optimization

#### For Limited Memory (8GB)
```python
# Optimize configuration
config = {
    'fingerprint_cols': ['ECFP4', 'ECFP6', 'MACCS'],  # Fewer fingerprints
    'cv_folds': 3,                                     # Fewer folds
    'base_learner_config': {
        'type': 'random_forest',                       # Memory efficient
        'params': {'n_estimators': 50}                 # Smaller ensemble
    }
}
```

#### For Faster Development
```python
# Quick testing configuration
config['fingerprint_cols'] = ['ECFP4']        # Single fingerprint
config['cv_folds'] = 2                        # Minimal CV
config['ensemble_strategies'] = ['weighted']  # Single strategy
```

## 🔬 File Structure

```
dream-challenge-pipeline/
├── main.py                      # 🎯 Main pipeline orchestrator
├── final_train_predict.py       # 🆕 Final training & prediction
├── create_submission.py         # 📤 DREAM Challenge format converter
├── base_learners.py             # 🤖 Base learner configurations
├── meta_learners.py             # 🧠 Meta-learner training
├── ensemble_strategies.py       # 🎭 Ensemble combination strategies
├── analysis.py                  # 📊 Analysis and visualization
├── utils.py                     # 🛠️ Helper functions
├── preprocessing.py             # ⚙️ Data preprocessing
├── pytorch_neural_net.py        # 🧠 PyTorch neural network wrapper
├── disynthon_split.py           # 🎯 User-provided data splitting
├── requirements.txt             # 📋 Python dependencies
├── README.md                    # 📖 This file
└── Data/                        # 📁 Data directory
    ├── WDR91.parquet           # Training data
    └── test_set.parquet        # Test data
```

## 🎯 Expected Workflow Timeline

### Development Phase (1-2 hours)
1. **Setup**: Install dependencies, prepare data (15 min)
2. **Test Run**: Single fingerprint, few folds (15 min)
3. **Full CV**: All fingerprints, full cross-validation (30-60 min)
4. **Compare**: Multiple base learners (optional, +30 min)

### Production Phase (30-60 minutes)
1. **Final Training**: Train on complete data (15-30 min)
2. **Generate Predictions**: Test set predictions (5-10 min)
3. **Create Submissions**: Convert to DREAM format (2-5 min)
4. **Validate**: Check submission format (1-2 min)

## 🤝 Contributing

This is an experimental project for the DREAM Challenge. Key areas for improvement:

- **Base Learners**: Add support for additional algorithms (SVM, CatBoost, etc.)
- **Meta-Learners**: Experiment with deep learning meta-learners
- **Ensemble Strategies**: Develop novel combination approaches
- **Chemical Diversity**: Implement alternative diversity metrics
- **Memory Optimization**: Further reduce memory footprint
- **Visualization**: Enhanced analysis and plotting capabilities

## 📄 License

This project is provided for the DREAM Challenge: Drug Discovery Target 2035. Please refer to the DREAM Challenge terms and conditions for usage guidelines.

## 🙏 Acknowledgments

- **DREAM Challenge Organizers**: For providing the framework and data
- **RDKit Community**: For excellent cheminformatics tools
- **Scikit-learn Team**: For robust machine learning implementations
- **PyTorch Team**: For flexible deep learning capabilities

## 📞 Support

For questions related to:
- **Pipeline Usage**: Check troubleshooting section above
- **DREAM Challenge**: Refer to official DREAM Challenge documentation
- **Chemical Diversity**: Consult RDKit documentation
- **General ML**: Refer to scikit-learn documentation

---

**🎯 Ready to tackle the DREAM Challenge!** This pipeline provides a comprehensive, memory-efficient approach to molecular property prediction with advanced ensemble techniques and chemical diversity selection.
