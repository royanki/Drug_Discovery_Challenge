"""
Comprehensive Stacking Pipeline Package

A modular, extensible pipeline for stacking with multiple fingerprints,
supporting plug-and-play base learners and meta-learner combination strategies.
"""

# from .main import StackingPipeline, run_with_xgboost, run_with_random_forest, run_with_neural_net
from .main import MemoryEfficientStacking, run_with_xgboost, run_with_random_forest, run_with_neural_net
from .main import compare_base_learners, create_default_config
from .base_learners import get_base_learner_config, BASE_LEARNER_CONFIGS
from .utils import calculate_comprehensive_metrics, print_metrics, create_comparison_table

__version__ = "1.0.0"
__author__ = "Your Name"

# Main function for easy import
def run_complete_pipeline(data_path, fingerprint_cols, base_learner='xgboost', 
                         ensemble_strategies=None, **kwargs):
    """
    Convenience function to run the complete pipeline.
    
    Parameters:
    -----------
    data_path : str
        Path to the parquet file
    fingerprint_cols : list
        List of fingerprint column names
    base_learner : str or dict
        Either a string name of predefined learner or a config dict
    ensemble_strategies : list
        List of ensemble strategies to apply
    **kwargs : additional config parameters
    
    Returns:
    --------
    dict : Complete results from the pipeline
    """
    # Create configuration
    config = create_default_config()
    config['data_path'] = data_path
    config['fingerprint_cols'] = fingerprint_cols
    
    # Set base learner
    if isinstance(base_learner, str):
        config['base_learner_config'] = get_base_learner_config(base_learner)
    else:
        config['base_learner_config'] = base_learner
    
    # Set ensemble strategies
    if ensemble_strategies is not None:
        config['ensemble_strategies'] = ensemble_strategies
    
    # Update with any additional parameters
    config.update(kwargs)
    
    # Run pipeline
    # pipeline = StackingPipeline(config)
    pipeline = MemoryEfficientStacking(config)
    return pipeline.run_complete_pipeline()