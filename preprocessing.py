import numpy as np
import pandas as pd

def parse_fingerprint_strings(fingerprint_series):
    """
    Parse fingerprint strings into numpy arrays.
    Handles various string formats: comma-separated, space-separated, etc.
    """
    print("Parsing fingerprint strings into numpy arrays...")
    
    # Get first non-null value to determine format
    sample_fp = None
    for fp in fingerprint_series:
        if pd.notna(fp) and fp is not None:
            sample_fp = fp
            break
    
    if sample_fp is None:
        raise ValueError("All fingerprint values are null or empty")
    
    # Convert to string if not already
    sample_fp_str = str(sample_fp)
    
    # Determine separator and parsing method
    if ',' in sample_fp_str:
        print("  Detected comma-separated fingerprint format")
        separator = ','
    elif ' ' in sample_fp_str.strip():
        print("  Detected space-separated fingerprint format")
        separator = ' '
    else:
        # Might be a single long string of digits
        print("  Detected concatenated digit fingerprint format")
        separator = None
    
    parsed_fingerprints = []
    
    for i, fp in enumerate(fingerprint_series):
        if pd.isna(fp) or fp is None:
            # Handle missing values - create zero vector
            if parsed_fingerprints:
                # Use length of first parsed fingerprint
                fp_array = np.zeros(len(parsed_fingerprints[0]), dtype=np.float32)
            else:
                # Default length - will be adjusted later
                fp_array = np.zeros(2048, dtype=np.float32)
        else:
            fp_str = str(fp).strip()
            
            if separator:
                # Split by separator and convert to float
                try:
                    fp_values = [float(x.strip()) for x in fp_str.split(separator) if x.strip()]
                    fp_array = np.array(fp_values, dtype=np.float32)
                except ValueError as e:
                    print(f"  Warning: Could not parse fingerprint at index {i}: {e}")
                    # Create zero vector with same length as previous
                    if parsed_fingerprints:
                        fp_array = np.zeros(len(parsed_fingerprints[0]), dtype=np.float32)
                    else:
                        fp_array = np.zeros(2048, dtype=np.float32)
            else:
                # Try to parse as individual digits
                try:
                    fp_array = np.array([float(digit) for digit in fp_str], dtype=np.float32)
                except ValueError as e:
                    print(f"  Warning: Could not parse fingerprint at index {i}: {e}")
                    if parsed_fingerprints:
                        fp_array = np.zeros(len(parsed_fingerprints[0]), dtype=np.float32)
                    else:
                        fp_array = np.zeros(2048, dtype=np.float32)
        
        parsed_fingerprints.append(fp_array)
        
        # Progress indicator for large datasets
        if (i + 1) % 50000 == 0:
            print(f"  Parsed {i + 1}/{len(fingerprint_series)} fingerprints...")
    
    # Convert to 2D numpy array
    try:
        fingerprint_matrix = np.vstack(parsed_fingerprints)
        print(f"  Successfully parsed fingerprints into matrix shape: {fingerprint_matrix.shape}")
        return fingerprint_matrix
    except ValueError as e:
        print(f"  Error stacking fingerprints - inconsistent lengths: {e}")
        # Find the most common length
        lengths = [len(fp) for fp in parsed_fingerprints]
        most_common_length = max(set(lengths), key=lengths.count)
        print(f"  Standardizing all fingerprints to length: {most_common_length}")
        
        # Pad or truncate to common length
        standardized_fps = []
        for fp in parsed_fingerprints:
            if len(fp) < most_common_length:
                # Pad with zeros
                padded_fp = np.zeros(most_common_length, dtype=np.float32)
                padded_fp[:len(fp)] = fp
                standardized_fps.append(padded_fp)
            elif len(fp) > most_common_length:
                # Truncate
                standardized_fps.append(fp[:most_common_length])
            else:
                standardized_fps.append(fp)
        
        fingerprint_matrix = np.vstack(standardized_fps)
        print(f"  Standardized fingerprint matrix shape: {fingerprint_matrix.shape}")
        return fingerprint_matrix

def remove_zero_variance_numpy(fingerprint_matrix, threshold=1e-10, feature_mask=None):
    """
    Remove zero variance features directly from numpy array.
    Much more memory efficient than DataFrame expansion.
    
    Parameters:
    -----------
    fingerprint_matrix : numpy.ndarray
        Input fingerprint matrix of shape (n_samples, n_features)
    threshold : float, default=1e-10
        Variance threshold below which features are considered zero-variance
    feature_mask : numpy.ndarray, optional
        Boolean mask of shape (n_features,) indicating which features to keep
        If provided, skips variance computation and uses this mask directly
        If None, computes variance and creates mask (training mode)
    
    Returns:
    --------
    tuple: (filtered_matrix, info_dict)
        - filtered_matrix: numpy array with selected features only
        - info_dict: dictionary containing:
            - 'removed': number of features removed
            - 'kept': number of features kept  
            - 'feature_mask': boolean array indicating kept features
    """
    print(f"Original fingerprint shape: {fingerprint_matrix.shape}")
    
    # Convert to float32 for faster computation
    if fingerprint_matrix.dtype != np.float32:
        fp_data = fingerprint_matrix.astype(np.float32)
    else:
        fp_data = fingerprint_matrix
    
    if feature_mask is not None:
        # Use provided mask (test mode - apply same selection as training)
        print("Using provided feature mask for consistency with training data")
        keep_mask = feature_mask
        n_removed = (~keep_mask).sum()
        n_kept = keep_mask.sum()
        
        print(f"Applied training mask: removed {n_removed}, kept {n_kept} features")
    else:
        # Compute mask from data (training mode - analyze this data)
        print("Computing feature mask from data variance analysis")
        # Fast variance calculation
        means = fp_data.mean(axis=0)
        variances = ((fp_data - means) ** 2).mean(axis=0)
        
        # Boolean mask with small threshold instead of exactly 0
        keep_mask = variances > threshold
        n_removed = (~keep_mask).sum()
        n_kept = keep_mask.sum()
        
        print(f"Computed variance-based mask: removed {n_removed}, kept {n_kept} features")
    
    # Filter the matrix using the mask
    if n_removed > 0:
        filtered_matrix = fp_data[:, keep_mask]
    else:
        filtered_matrix = fp_data
    
    print(f"Final fingerprint shape: {filtered_matrix.shape}")
    
    return filtered_matrix, {
        'removed': n_removed, 
        'kept': n_kept,
        'feature_mask': keep_mask  # Always return the mask for future use
    }

def preprocess_single_fingerprint_numpy(df, fingerprint_col, feature_mask=None):
    """
    Extract fingerprint, remove zero variance, but keep as numpy array.
    Returns the fingerprint matrix and a dataframe with other columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing fingerprint data
    fingerprint_col : str  
        Name of the fingerprint column to process
    feature_mask : numpy.ndarray, optional
        Boolean mask indicating which features to keep (for test data consistency)
        If None, computes mask from data (training mode)
        If provided, uses the mask directly (test mode)
    
    Returns:
    --------
    tuple: (filtered_matrix, metadata_df, info)
        - filtered_matrix: numpy array with selected features
        - metadata_df: dataframe with non-fingerprint columns  
        - info: dict with 'removed', 'kept', and 'feature_mask' keys
    
    Updated to handle both numpy arrays and string representations.
    """
    print(f"Preprocessing fingerprint column: {fingerprint_col}")
    
    # Check the type of fingerprint data
    sample_fp = df[fingerprint_col].iloc[0]
    
    if isinstance(sample_fp, str):
        print("  Detected string fingerprint format - parsing...")
        # Parse string fingerprints
        fingerprint_matrix = parse_fingerprint_strings(df[fingerprint_col])
    elif isinstance(sample_fp, np.ndarray):
        print("  Detected numpy array fingerprint format")
        # Extract fingerprint data as numpy array (existing behavior)
        fingerprint_matrix = np.stack(df[fingerprint_col].values)
    elif hasattr(sample_fp, '__iter__') and not isinstance(sample_fp, str):
        print("  Detected iterable fingerprint format - converting to numpy array...")
        # Handle lists or other iterables
        try:
            fingerprint_matrix = np.array([np.array(fp, dtype=np.float32) for fp in df[fingerprint_col]])
        except Exception as e:
            print(f"  Error converting iterable fingerprints: {e}")
            print("  Falling back to string parsing...")
            fingerprint_matrix = parse_fingerprint_strings(df[fingerprint_col])
    else:
        raise ValueError(f"Unsupported fingerprint data type: {type(sample_fp)}")
    
    # Remove zero variance features (with optional mask for consistency)
    filtered_matrix, info = remove_zero_variance_numpy(fingerprint_matrix, feature_mask=feature_mask)
    
    # Create a dataframe with just the metadata columns
    metadata_cols = [col for col in df.columns if col != fingerprint_col]
    metadata_df = df[metadata_cols].copy()
    
    return filtered_matrix, metadata_df, info

def load_and_preprocess_data(data_path, fingerprint_cols):
    """
    Load and preprocess all fingerprints.
    Returns stacked features and metadata.
    """
    fingerprint_data = []
    fingerprint_ranges = {}
    current_idx = 0
    
    # Load and process each fingerprint
    for fp in fingerprint_cols:
        print(f"Loading and preprocessing {fp}...")
        df = pd.read_parquet(data_path, columns=['BB1_ID', 'BB2_ID', fp, 'LABEL'])
        
        # Extract and process fingerprint
        fp_matrix, metadata_df, info = preprocess_single_fingerprint_numpy(df, fp)
        
        # Store the range of indices for this fingerprint
        fingerprint_ranges[fp] = (current_idx, current_idx + fp_matrix.shape[1])
        current_idx += fp_matrix.shape[1]
        
        fingerprint_data.append(fp_matrix)
        
        # Use metadata from first fingerprint (they should all be the same)
        if fp == fingerprint_cols[0]:
            base_metadata = metadata_df
    
    # Stack all fingerprints horizontally
    X_stacked = np.hstack(fingerprint_data)
    print(f"Stacked feature matrix shape: {X_stacked.shape}")
    
    return X_stacked, base_metadata, fingerprint_ranges