#!/usr/bin/env python3
"""
Test script to ensure the pipeline works correctly.
"""

import numpy as np
import pandas as pd
from stacking_pipeline import run_complete_pipeline, get_base_learner_config

def create_test_data():
    """Create synthetic test data for pipeline testing."""
    np.random.seed(42)
    
    # Create synthetic fingerprints
    n_samples = 1000
    n_features = 100  # Smaller for testing
    
    data = {
        'BB1_ID': [f'BB1_{i}' for i in range(n_samples)],
        'BB2_ID': [f'BB2_{i}' for i in range(n_samples)],
        'LABEL': np.random.binomial(1, 0.2, n_samples)  # 20% positive class
    }
    
    # Create synthetic fingerprints
    fingerprints = ['ECFP4', 'ECFP6', 'FCFP4']
    for fp in fingerprints:
        # Each fingerprint is a list of binary values
        data[fp] = [np.random.binomial(1, 0.1, n_features).tolist() for _ in range(n_samples)]
    
    df = pd.DataFrame(data)
    return df

def test_basic_pipeline():
    """Test basic pipeline functionality."""
    print("Testing basic pipeline functionality...")
    
    # Create test data
    test_df = create_test_data()
    test_path = "test_data.parquet"
    test_df.to_parquet(test_path)
    
    try:
        # Run pipeline
        results = run_complete_pipeline(
            data_path=test_path,
            fingerprint_cols=['ECFP4', 'ECFP6', 'FCFP4'],
            base_learner='xgboost',
            ensemble_strategies=['weighted'],
            cv_folds=3,  # Faster for testing
            enable_visualizations=False
        )
        
        # Basic checks
        assert 'meta_learner_results' in results
        assert 'ensemble_results' in results
        assert 'summary_report' in results
        
        print("✓ Basic pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Basic pipeline test failed: {e}")
        return False
    
    finally:
        # Clean up
        import os
        if os.path.exists(test_path):
            os.remove(test_path)

def test_different_base_learners():
    """Test different base learners."""
    print("Testing different base learners...")
    
    test_df = create_test_data()
    test_path = "test_data.parquet"
    test_df.to_parquet(test_path)
    
    base_learners = ['xgboost', 'random_forest']
    results = {}
    
    try:
        for bl in base_learners:
            print(f"  Testing {bl}...")
            result = run_complete_pipeline(
                data_path=test_path,
                fingerprint_cols=['ECFP4', 'ECFP6'],
                base_learner=bl,
                ensemble_strategies=['weighted'],
                cv_folds=2,
                enable_visualizations=False
            )
            results[bl] = result
            
        print("✓ Different base learners test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Different base learners test failed: {e}")
        return False
    
    finally:
        import os
        if os.path.exists(test_path):
            os.remove(test_path)

def test_custom_config():
    """Test custom configuration."""
    print("Testing custom configuration...")
    
    test_df = create_test_data()
    test_path = "test_data.parquet"
    test_df.to_parquet(test_path)
    
    # Custom XGBoost config
    custom_config = {
        'type': 'xgboost',
        'params': {
            'n_estimators': 20,  # Small for testing
            'max_depth': 3,
            'learning_rate': 0.3,
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
    }
    
    try:
        results = run_complete_pipeline(
            data_path=test_path,
            fingerprint_cols=['ECFP4'],
            base_learner=custom_config,
            ensemble_strategies=['weighted'],
            cv_folds=2,
            enable_visualizations=False
        )
        
        print("✓ Custom configuration test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Custom configuration test failed: {e}")
        return False
    
    finally:
        import os
        if os.path.exists(test_path):
            os.remove(test_path)

def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("RUNNING PIPELINE TESTS")
    print("="*60)
    
    tests = [
        test_basic_pipeline,
        test_different_base_learners,
        test_custom_config
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("="*60)
    print(f"TESTS COMPLETED: {passed}/{total} passed")
    print("="*60)
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)