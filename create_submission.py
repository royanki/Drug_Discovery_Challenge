#!/usr/bin/env python3
"""
Challenge Submission Format Converter

This script converts the pipeline output CSV files into the required challenge submission format:
- RandomID (str): Anonymized IDs from test set
- Sel_200 (int): Binary flag for 200 diverse compounds (0 or 1)
- Sel_500 (int): Binary flag for 500 diverse compounds (0 or 1) 
- Score (float): Probability/confidence score for all compounds

Usage:
    python create_submission.py --submission-dir results/xgb_submissions --team-name MyTeam
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path

def find_submission_files(submission_dir, base_learner_type=None):
    """
    Find the generated CSV files in the submission directory.
    
    Parameters:
    -----------
    submission_dir : str
        Directory containing the generated CSV files
    base_learner_type : str, optional
        Base learner type (xgboost, random_forest, neural_net)
        If None, will auto-detect from files
    
    Returns:
    --------
    dict: Dictionary with paths to each file type
    """
    submission_path = Path(submission_dir)
    
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission directory not found: {submission_dir}")
    
    # Auto-detect base learner type if not provided
    if base_learner_type is None:
        csv_files = list(submission_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {submission_dir}")
        
        # Extract base learner type from filename
        sample_file = csv_files[0].name
        if 'xgboost' in sample_file:
            base_learner_type = 'xgboost'
        elif 'random_forest' in sample_file:
            base_learner_type = 'random_forest'  
        elif 'neural_net' in sample_file:
            base_learner_type = 'neural_net'
        else:
            raise ValueError(f"Cannot auto-detect base learner type from {sample_file}")
    
    # Define expected file patterns
    file_patterns = {
        'all_predictions': f"all_predictions_{base_learner_type}.csv",
        'top_500': f"top_500_{base_learner_type}.csv",
        'diverse_200': f"diverse_200_{base_learner_type}.csv"
    }
    
    # Find actual files
    found_files = {}
    for file_type, pattern in file_patterns.items():
        file_path = submission_path / pattern
        if file_path.exists():
            found_files[file_type] = str(file_path)
        else:
            print(f"Warning: {pattern} not found in {submission_dir}")
    
    print(f"Found {len(found_files)} submission files:")
    for file_type, path in found_files.items():
        print(f"  {file_type}: {Path(path).name}")
    
    return found_files, base_learner_type

def load_and_validate_files(file_paths):
    """
    Load CSV files and validate they have required columns.
    
    Parameters:
    -----------
    file_paths : dict
        Dictionary with file paths for each type
    
    Returns:
    --------
    dict: Dictionary with loaded DataFrames
    """
    dataframes = {}
    
    # Required columns for each file type
    required_columns = {
        'all_predictions': ['RandomID', 'Prediction_Probability'],
        'top_500': ['RandomID', 'Prediction_Probability'],
        'diverse_200': ['RandomID', 'Prediction_Probability']
    }
    
    for file_type, file_path in file_paths.items():
        print(f"\nLoading {file_type} from {Path(file_path).name}...")
        
        try:
            df = pd.read_csv(file_path)
            print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Check for required columns
            required_cols = required_columns[file_type]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"  Available columns: {list(df.columns)}")
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Show sample data
            print(f"  Sample data:")
            print(f"    RandomID range: {df['RandomID'].iloc[0]} to {df['RandomID'].iloc[-1]}")
            print(f"    Score range: [{df['Prediction_Probability'].min():.4f}, {df['Prediction_Probability'].max():.4f}]")
            
            dataframes[file_type] = df
            
        except Exception as e:
            print(f"  ERROR loading {file_type}: {e}")
            raise
    
    return dataframes

def create_submission_format(dataframes):
    """
    Create the submission format DataFrame.
    
    Parameters:
    -----------
    dataframes : dict
        Dictionary with loaded DataFrames
    
    Returns:
    --------
    pandas.DataFrame: Submission format DataFrame
    """
    print("\nCreating submission format...")
    
    # Start with all predictions as the base (contains all RandomIDs and scores)
    if 'all_predictions' not in dataframes:
        raise ValueError("all_predictions file is required but not found")
    
    submission_df = dataframes['all_predictions'][['RandomID', 'Prediction_Probability']].copy()
    submission_df = submission_df.rename(columns={'Prediction_Probability': 'Score'})
    
    print(f"Base submission with {len(submission_df)} compounds")
    
    # Initialize selection columns with zeros
    submission_df['Sel_200'] = 0
    submission_df['Sel_500'] = 0
    
    # Mark Sel_500 (top 500 compounds)
    if 'top_500' in dataframes:
        top_500_ids = set(dataframes['top_500']['RandomID'])
        submission_df.loc[submission_df['RandomID'].isin(top_500_ids), 'Sel_500'] = 1
        sel_500_count = submission_df['Sel_500'].sum()
        print(f"Marked {sel_500_count} compounds for Sel_500")
        
        if sel_500_count != 500:
            print(f"WARNING: Expected exactly 500 compounds for Sel_500, got {sel_500_count}")
    else:
        print("WARNING: No top_500 file found, Sel_500 will be all zeros")
    
    # Mark Sel_200 (diverse 200 compounds)  
    if 'diverse_200' in dataframes:
        diverse_200_ids = set(dataframes['diverse_200']['RandomID'])
        submission_df.loc[submission_df['RandomID'].isin(diverse_200_ids), 'Sel_200'] = 1
        sel_200_count = submission_df['Sel_200'].sum()
        print(f"Marked {sel_200_count} compounds for Sel_200")
        
        if sel_200_count != 200:
            print(f"WARNING: Expected exactly 200 compounds for Sel_200, got {sel_200_count}")
    else:
        print("WARNING: No diverse_200 file found, Sel_200 will be all zeros")
    
    # Reorder columns to match submission format
    submission_df = submission_df[['RandomID', 'Sel_200', 'Sel_500', 'Score']]
    
    # Validate data types
    submission_df['RandomID'] = submission_df['RandomID'].astype(str)
    submission_df['Sel_200'] = submission_df['Sel_200'].astype(int)
    submission_df['Sel_500'] = submission_df['Sel_500'].astype(int)
    submission_df['Score'] = submission_df['Score'].astype(float)
    
    print(f"\nSubmission DataFrame created:")
    print(f"  Shape: {submission_df.shape}")
    print(f"  Columns: {list(submission_df.columns)}")
    print(f"  Data types: {submission_df.dtypes.to_dict()}")
    
    return submission_df

def validate_submission_format(submission_df):
    """
    Validate the submission format meets requirements.
    
    Parameters:
    -----------
    submission_df : pandas.DataFrame
        Submission DataFrame to validate
    
    Returns:
    --------
    bool: True if valid, raises exception if invalid
    """
    print("\nValidating submission format...")
    
    # Check required columns
    required_columns = ['RandomID', 'Sel_200', 'Sel_500', 'Score']
    missing_columns = [col for col in required_columns if col not in submission_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check data types
    if submission_df['RandomID'].dtype != 'object':
        raise ValueError("RandomID must be string type")
    
    if submission_df['Sel_200'].dtype != 'int64':
        raise ValueError("Sel_200 must be integer type")
        
    if submission_df['Sel_500'].dtype != 'int64':
        raise ValueError("Sel_500 must be integer type")
    
    if not np.issubdtype(submission_df['Score'].dtype, np.floating):
        raise ValueError("Score must be float type")
    
    # Check binary values
    sel_200_values = set(submission_df['Sel_200'].unique())
    if not sel_200_values.issubset({0, 1}):
        raise ValueError(f"Sel_200 must contain only 0 and 1, found: {sel_200_values}")
    
    sel_500_values = set(submission_df['Sel_500'].unique())
    if not sel_500_values.issubset({0, 1}):
        raise ValueError(f"Sel_500 must contain only 0 and 1, found: {sel_500_values}")
    
    # Check selection counts
    sel_200_count = submission_df['Sel_200'].sum()
    sel_500_count = submission_df['Sel_500'].sum()
    
    if sel_200_count != 200:
        print(f"WARNING: Sel_200 count is {sel_200_count}, expected exactly 200")
    
    if sel_500_count != 500:
        print(f"WARNING: Sel_500 count is {sel_500_count}, expected exactly 500")
    
    # Check for missing values
    missing_counts = submission_df.isnull().sum()
    if missing_counts.any():
        raise ValueError(f"Missing values found: {missing_counts.to_dict()}")
    
    # Check Score range
    score_min = submission_df['Score'].min()
    score_max = submission_df['Score'].max()
    if not (0.0 <= score_min <= 1.0) or not (0.0 <= score_max <= 1.0):
        print(f"WARNING: Scores outside [0,1] range: [{score_min:.4f}, {score_max:.4f}]")
    
    print("✅ Submission format validation passed!")
    
    # Print summary statistics
    print(f"\nSubmission Summary:")
    print(f"  Total compounds: {len(submission_df):,}")
    print(f"  Sel_200 marked: {sel_200_count}")
    print(f"  Sel_500 marked: {sel_500_count}")
    print(f"  Score range: [{score_min:.4f}, {score_max:.4f}]")
    print(f"  Mean score: {submission_df['Score'].mean():.4f}")
    
    return True

def main():
    """Main function to convert pipeline outputs to submission format."""
    parser = argparse.ArgumentParser(description='Convert pipeline outputs to challenge submission format')
    parser.add_argument('--submission-dir', type=str, required=True,
                       help='Directory containing pipeline output CSV files')
    parser.add_argument('--team-name', type=str, required=True,
                       help='Team name for output file (e.g., "MyTeam" -> "TeamMyTeam.csv")')
    parser.add_argument('--base-learner', type=str, choices=['xgboost', 'random_forest', 'neural_net'],
                       help='Base learner type (auto-detected if not specified)')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for submission file (default: current directory)')
    
    args = parser.parse_args()
    
    try:
        # Step 1: Find submission files
        print("="*60)
        print("CHALLENGE SUBMISSION FORMAT CONVERTER")
        print("="*60)
        
        file_paths, base_learner_type = find_submission_files(args.submission_dir, args.base_learner)
        
        if not file_paths:
            raise ValueError("No valid submission files found")
        
        # Step 2: Load and validate files
        dataframes = load_and_validate_files(file_paths)
        
        # Step 3: Create submission format
        submission_df = create_submission_format(dataframes)
        
        # Step 4: Validate submission format
        validate_submission_format(submission_df)
        
        # Step 5: Save submission file
        output_filename = f"Team{args.team_name}.csv"
        output_path = Path(args.output_dir) / output_filename
        
        print(f"\nSaving submission file...")
        submission_df.to_csv(output_path, index=False)
        
        print(f"✅ Submission file created successfully!")
        print(f"   File: {output_path}")
        print(f"   Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
        print(f"   Ready for submission to challenge!")
        
        return str(output_path)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise

if __name__ == "__main__":
    main()

# Example usage functions for direct import
def create_submission_from_directory(submission_dir, team_name, output_dir="."):
    """
    Convenience function to create submission file from directory.
    
    Parameters:
    -----------
    submission_dir : str
        Directory containing pipeline CSV outputs
    team_name : str  
        Team name for output file
    output_dir : str, optional
        Output directory (default: current directory)
    
    Returns:
    --------
    str: Path to created submission file
    """
    file_paths, base_learner_type = find_submission_files(submission_dir)
    dataframes = load_and_validate_files(file_paths)
    submission_df = create_submission_format(dataframes)
    validate_submission_format(submission_df)
    
    output_filename = f"Team{team_name}.csv"
    output_path = Path(output_dir) / output_filename
    submission_df.to_csv(output_path, index=False)
    
    return str(output_path)

# Quick test function
def test_submission_format():
    """Test function with dummy data."""
    print("Testing submission format creation...")
    
    # Create dummy data
    n_compounds = 1000
    dummy_data = {
        'RandomID': [f"ID_{i:06d}" for i in range(n_compounds)],
        'Score': np.random.random(n_compounds),
        'Sel_200': [1 if i < 200 else 0 for i in range(n_compounds)],
        'Sel_500': [1 if i < 500 else 0 for i in range(n_compounds)]
    }
    
    test_df = pd.DataFrame(dummy_data)
    validate_submission_format(test_df)
    print("✅ Test passed!")

if __name__ == "__main__":
    # Uncomment to run test
    # test_submission_format()
    main()