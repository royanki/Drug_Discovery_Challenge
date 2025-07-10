# Comprehensive Stacking Pipeline

A modular, extensible pipeline for stacking ensemble learning with molecular fingerprints, featuring **final training on complete data** and **intelligent submission file generation** for drug discovery challenges.

## 🎯 New Features

- **🚀 Two-Phase Workflow**: Cross-validation for model selection + Final training for submissions
- **🧪 Chemical Diversity Selection**: RDKit-based Tanimoto similarity clustering with preprocessed fingerprints
- **📊 Intelligent Submission Generation**: Top confident + chemically diverse predictions
- **⚡ Memory-Efficient Processing**: One fingerprint at a time with zero-variance removal
- **🔄 Preprocessed Fingerprint Consistency**: Same preprocessing for modeling and diversity

## Installation

```bash
# Install required packages
pip install pandas numpy scikit-learn xgboost matplotlib seaborn rdkit-pypi

# ⚠️ **IMPORTANT**: RDKit is now required for chemical diversity selection
# If installation fails, try: conda install -c conda-forge rdkit

# Clone or download the pipeline
git clone <your-repo-url>
cd stacking_pipeline
```

## 🔥 Quick Start - Two-Phase Workflow

### **Phase 1: Cross-Validation (Model Selection)**

```bash
# Run CV with XGBoost to find best ensemble strategy
python -m stacking_pipeline.main --base-learner xgboost --output results/xgb_cv_results.pkl

# Run CV with Random Forest
python -m stacking_pipeline.main --base-learner random_forest --output results/rf_cv_results.pkl

# Run CV with Neural Network  
python -m stacking_pipeline.main --base-learner neural_net --output results/nn_cv_results.pkl
```

### **Phase 2: Final Training & Submission Generation**

```bash
# 🎯 Use XGBoost CV results for final training on complete data
python -m stacking_pipeline.main --final-train \
               --cv-results results/xgb_cv_results.pkl \
               --test-data Data/challenge_test_set.parquet \
               --submission-dir results/xgb_submissions

# 🎯 Use Random Forest CV results  
python -m stacking_pipeline.main --final-train \
               --cv-results results/rf_cv_results.pkl \
               --test-data Data/challenge_test_set.parquet \
               --submission-dir results/rf_submissions
```

> **🔴 CRITICAL**: <span style='color:red'>The pipeline uses relative imports and **MUST** be run as a module with `python -m stacking_pipeline.main`. Running `python main.py` directly will cause ImportError.</span>

## 📁 Generated Submission Files

The final training generates **4 submission files**:

```
results/xgb_submissions/
├── top_200_xgboost.csv        # 🏆 Top 200 most confident predictions
├── top_500_xgboost.csv        # 🏆 Top 500 most confident predictions  
├── diverse_200_xgboost.csv    # 🧬 200 chemically diverse predictions
└── all_predictions_xgboost.csv # 📋 All predictions ranked by confidence
```

### 🧬 Chemical Diversity Selection Features

- **🔬 Uses ALL 9 fingerprints**: ECFP4, ECFP6, FCFP4, FCFP6, TOPTOR, MACCS, RDK, AVALON, ATOMPAIR
- **⚙️ Consistent Preprocessing**: Same zero-variance removal as model training
- **🎯 RDKit Tanimoto Clustering**: Butina algorithm for proper chemical diversity
- **🛡️ Robust Fallback**: Building block diversity if RDKit fails
- **📊 Detailed Logging**: Complete transparency of selection process

## ⚠️ **CRITICAL**: Test Data Path Requirements

> **🔴 WARNING**: You **MUST** specify the test data path when running final training!

### **Option 1: Command Line (Recommended)**
```bash
python main.py --final-train \
               --cv-results results/cv_results.pkl \
               --test-data Data/YOUR_TEST_FILE.parquet  # 🔴 REQUIRED!
```

### **Option 2: Default Configuration**
```python
def create_default_config():
    return {
        'data_path': "Data/WDR91.parquet",
        'test_data_path': "Data/YOUR_TEST_FILE.parquet",  # Add this line
        # ... other config
    }
```

### **Option 3: JSON Configuration File**
```json
{
    "data_path": "Data/WDR91.parquet",
    "test_data_path": "Data/YOUR_TEST_FILE.parquet",
    "fingerprint_cols": ["ECFP4", "ECFP6", "FCFP4", "FCFP6", "TOPTOR", "MACCS", "RDK", "AVALON", "ATOMPAIR"]
}
```

## 🚀 Complete Workflow Example

```bash
# 1️⃣ Run cross-validation (find best strategy & weights)
python -m stacking_pipeline.main --base-learner xgboost \
               --output results/xgb_cv_results.pkl \
               --folds 5

# 2️⃣ Train final models on complete data & generate submissions
python -m stacking_pipeline.main --final-train \
               --cv-results results/xgb_cv_results.pkl \
               --test-data Data/challenge_test_set.parquet \
               --submission-dir results/final_submissions \
               --output results/final_results.pkl

# 3️⃣ Check your submission files!
ls results/final_submissions/
```

## ⚙️ Advanced Configuration

### **Meta-Learner Selection**
> **🟡 NOTE**: Currently uses Logistic Regression + Random Forest meta-learners only
> 
> KNN and MLP meta-learners are excluded for performance reasons

### **Ensemble Strategies**
Available strategies (automatically compared during CV):

- `weighted`: 🎯 Optimized weighted combination
- `threshold`: 🔄 Threshold-based hybrid approach  
- `calibrated`: 📊 Calibrated probability ensemble
- `robust`: 🛡️ Robust validation-based combination

### **Memory-Efficient Processing**
```python
# The pipeline processes one fingerprint at a time to save memory
# Each fingerprint undergoes zero-variance removal before modeling
# Same preprocessing is applied for diversity calculation
```

## 📊 Understanding CV Results

The CV phase saves critical information for final training:

```python
cv_results = {
    'cv_summary': {
        'best_ensemble_strategy': 'Ensemble_weighted',
        'optimal_weights': {'LogisticRegression': 0.3, 'RandomForest': 0.7},
        'fingerprint_meta_cols': ['ECFP4_xgboost', 'ECFP6_xgboost', ...],
        'best_metrics': {'F1-Score': 0.8234, 'AUC-ROC': 0.9156, ...}
    }
}
```

## 🔧 File Structure (Updated)

```
project_root/
├── stacking_pipeline/
│   ├── __init__.py               # Package initialization
│   ├── main.py                   # 🎯 Main pipeline with two-phase workflow
│   ├── final_train_predict.py    # 🆕 Final training & prediction module
│   ├── base_learners.py          # Base learner configurations
│   ├── meta_learners.py          # Meta-learner training (LR + RF only)
│   ├── ensemble_strategies.py    # Meta-learner combination strategies
│   ├── analysis.py               # Analysis and visualization
│   ├── utils.py                  # Helper functions
│   ├── preprocessing.py          # Data preprocessing with zero-variance removal
│   └── disynthon_split.py        # Custom split function (user-provided)
├── requirements.txt              # Dependencies list
├── Data/                         # Data directory
│   ├── WDR91.parquet            # Training data
│   └── test_set.parquet         # Test data
└── results/                     # Output directory
    ├── cv_results.pkl           # CV results
    └── submissions/             # Final submission files
```

> **🔴 CRITICAL**: The pipeline is a Python package with relative imports. Always run with `python -m stacking_pipeline.main` from the project root directory.

## 🎯 Performance Metrics

The pipeline reports comprehensive metrics:

- **📊 Standard Metrics**: Precision, Recall, F1-Score, AUC-ROC
- **🎯 Challenge-Specific**: PPV@128, PPV@256, Hits@128, Hits@256
- **📈 Advanced**: PR AUC, Calibration metrics
- **🧬 Diversity Metrics**: Chemical diversity statistics

## ⚠️ **WARNINGS & TROUBLESHOOTING**

### **🔴 Critical Issues**

1. **Missing RDKit**: 
   ```bash
   # If you get import errors for RDKit:
   conda install -c conda-forge rdkit
   # or
   pip install rdkit-pypi
   ```

2. **Import Errors with Relative Paths**:
   ```
   ImportError: attempted relative import with no known parent package
   ```
   **Solution**: The pipeline uses absolute imports. Run `python main.py` directly from the project root directory where all Python files are located.

3. **Test Data Path Not Found**:
   ```
   Error: Test data file not found: Data/test_set.parquet
   ```
   **Solution**: Verify your test file path is correct!

4. **CV Results Missing**:
   ```
   Error: --cv-results path is required when using --final-train
   ```
   **Solution**: Run CV phase first to generate .pkl file!

### **🟡 Performance Warnings**

1. **Memory Issues**: 
   - Reduce fingerprints: `--fingerprint-subset ['ECFP4', 'ECFP6']`
   - Reduce CV folds: `--folds 3`

2. **Slow Diversity Calculation**:
   - Uses top 1000 candidates by default
   - Falls back to building block diversity if RDKit fails

### **🟢 Best Practices**

1. **🎯 Start with CV**: Always run cross-validation first
2. **💾 Save Results**: Use `--output` to save CV results  
3. **🧪 Test Different Base Learners**: Compare XGBoost vs Random Forest vs Neural Net
4. **📊 Check Diversity**: Ensure diverse_200.csv has good chemical diversity
5. **🔍 Validate Submissions**: Check that submission files have expected formats
6. **📁 Project Structure**: Keep all Python files in the same directory and run from project root

## 🆕 New Command Line Options

```bash
# CV Phase Options
python main.py --base-learner xgboost --output results.pkl --folds 5 --save-vis

# Final Training Phase Options  
python main.py --final-train \
               --cv-results results.pkl \        # 🔴 REQUIRED
               --test-data test.parquet \        # 🔴 REQUIRED  
               --submission-dir submissions/ \   # Optional (default: results/submissions)
               --output final_results.pkl       # Optional

# Utility Options
--no-vis              # Disable visualizations
--save-vis            # Save visualizations to files
--folds N             # Number of CV folds (default: 5)
```

## 🧬 Chemical Diversity Technical Details

### **Preprocessing Consistency**
> **🟢 IMPORTANT**: Diversity calculation uses the **same preprocessed fingerprints** as model training
> 
> - Zero-variance bits removed per fingerprint
> - ECFP4: ~1847 bits (after removing ~201 zero-variance)
> - MACCS: ~167 bits (after removing many zero-variance)
> - Combined fingerprint: ~8000-12000 total informative bits

### **Diversity Algorithm**
1. **Candidate Selection**: Top 1000 predictions by confidence
2. **Fingerprint Combination**: Concatenate all 9 preprocessed fingerprints
3. **Similarity Calculation**: RDKit Tanimoto similarity
4. **Clustering**: Butina algorithm (threshold=0.7)
5. **Representative Selection**: Highest confidence from each cluster
6. **Gap Filling**: Add high-confidence molecules if needed

## 📋 Quick Reference

### **Typical Workflow**
```bash
# 1. CV Phase (once per base learner)
python -m stacking_pipeline.main --base-learner xgboost --output xgb_cv.pkl

# 2. Final Phase (generates submissions)  
python -m stacking_pipeline.main --final-train --cv-results xgb_cv.pkl --test-data test.parquet

# 3. Submit files from results/submissions/
```

### **File Formats**
All CSV files contain:
- `BB1_ID`, `BB2_ID`: Building block identifiers
- `Prediction_Probability`: Model confidence score
- `Confidence_Rank` or `Diversity_Rank`: Selection ranking
- `Selection_Method`: How molecule was selected

## 🎉 Ready for Drug Discovery Challenges!

This updated pipeline provides everything needed for modern drug discovery competitions:

✅ **Two-phase workflow** for optimal model selection and final training  
✅ **Chemical diversity selection** using state-of-the-art RDKit methods  
✅ **Memory-efficient processing** of large molecular datasets  
✅ **Consistent preprocessing** between modeling and diversity calculation  
✅ **Comprehensive submission formats** (confident + diverse predictions)  
✅ **Robust error handling** with meaningful warnings and fallbacks  

> **🎯 Success Tip**: Run CV on a subset of fingerprints first to find the best setup, then run final training with all fingerprints for maximum performance!