"""
Implementation of disynthon_split_real function for DEL data.
This function ensures molecules with the same disynthon (BB1+BB2) are grouped together.
"""

import pandas as pd
import random
from typing import Tuple

# Define constants
BB_COLS = ['BB1_ID', 'BB2_ID']
DISYNTHON_KEY = 'DISYNTHON'

def disynthon_split_real(df: pd.DataFrame, 
                         frac_train: float = 0.8, 
                         frac_val: float = 0.1, 
                         frac_test: float = 0.1,
                         seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs a disynthon-aware split on real DEL data using BB1_ID and BB2_ID.
    Ensures molecules with the same disynthon (BB1+BB2) are grouped together.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with BB1_ID, BB2_ID columns
    frac_train : float
        Fraction of data for training set
    frac_val : float
        Fraction of data for validation set
    frac_test : float
        Fraction of data for test set
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Training, validation, and test DataFrames
    """
    assert sum([frac_train, frac_val, frac_test]) == 1.0, "Fractions must sum to 1"

    df = df.copy()
    df[DISYNTHON_KEY] = df[BB_COLS].astype(str).agg('_'.join, axis=1)

    # Group by disynthon
    disynthon_groups = df.groupby(DISYNTHON_KEY)
    disynthon_list = list(disynthon_groups.groups.keys())
    random.Random(seed).shuffle(disynthon_list)

    n_total = len(df)
    n_train = int(n_total * frac_train)
    n_val = int(n_total * frac_val)

    train_idx, val_idx, test_idx = [], [], []
    counts = {'train': 0, 'val': 0, 'test': 0}

    for disynthon in disynthon_list:
        indices = list(disynthon_groups.groups[disynthon])
        if counts['train'] + len(indices) <= n_train:
            train_idx.extend(indices)
            counts['train'] += len(indices)
        elif counts['val'] + len(indices) <= n_val:
            val_idx.extend(indices)
            counts['val'] += len(indices)
        else:
            test_idx.extend(indices)
            counts['test'] += len(indices)

    print(f"Split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    return (
        df.loc[train_idx].drop(columns=DISYNTHON_KEY),
        df.loc[val_idx].drop(columns=DISYNTHON_KEY),
        df.loc[test_idx].drop(columns=DISYNTHON_KEY)
    )