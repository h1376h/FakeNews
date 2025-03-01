import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional, Union
import logging
import os
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define expected feature types and ranges
FEATURE_METADATA = {
    # Content features
    'content_polarity': {'type': 'float', 'range': [-1.0, 1.0], 'description': 'Average sentiment polarity'},
    'content_subjectivity': {'type': 'float', 'range': [0.0, 1.0], 'description': 'Average subjectivity score'},
    'content_disagreement': {'type': 'float', 'range': [0.0, 1.0], 'description': 'Amount of tweets expressing disagreement'},
    'content_num_question': {'type': 'int', 'range': [0, float('inf')], 'description': 'Number of tweets containing question marks'},
    'content_ratio_question': {'type': 'float', 'range': [0.0, 1.0], 'description': 'Ratio of tweets with question marks'},
    'content_num_exclamation': {'type': 'int', 'range': [0, float('inf')], 'description': 'Number of tweets containing exclamation marks'},
    'content_ratio_exclamation': {'type': 'float', 'range': [0.0, 1.0], 'description': 'Ratio of tweets with exclamation marks'},
    'content_num_first_person': {'type': 'int', 'range': [0, float('inf')], 'description': 'Number of tweets containing first-person pronouns'},
    'content_ratio_first_person': {'type': 'float', 'range': [0.0, 1.0], 'description': 'Ratio of tweets with first-person pronouns'},
    'content_num_second_person': {'type': 'int', 'range': [0, float('inf')], 'description': 'Number of tweets containing second-person pronouns'},
    'content_ratio_second_person': {'type': 'float', 'range': [0.0, 1.0], 'description': 'Ratio of tweets with second-person pronouns'},
    'content_num_third_person': {'type': 'int', 'range': [0, float('inf')], 'description': 'Number of tweets containing third-person pronouns'},
    'content_ratio_third_person': {'type': 'float', 'range': [0.0, 1.0], 'description': 'Ratio of tweets with third-person pronouns'},
    'content_num_smiley': {'type': 'int', 'range': [0, float('inf')], 'description': 'Number of tweets containing smileys'},
    'content_ratio_smiley': {'type': 'float', 'range': [0.0, 1.0], 'description': 'Ratio of tweets with smileys'},
    'content_num_info_request': {'type': 'int', 'range': [0, float('inf')], 'description': 'Number of tweets requesting information'},
    'content_ratio_info_request': {'type': 'float', 'range': [0.0, 1.0], 'description': 'Ratio of tweets requesting information'},
    'content_num_support': {'type': 'int', 'range': [0, float('inf')], 'description': 'Number of tweets supporting source tweet'},
    'content_ratio_support': {'type': 'float', 'range': [0.0, 1.0], 'description': 'Ratio of tweets supporting source tweet'},
    'content_num_disagreement': {'type': 'int', 'range': [0, float('inf')], 'description': 'Number of tweets expressing disagreement'},
    'content_ratio_disagreement': {'type': 'float', 'range': [0.0, 1.0], 'description': 'Ratio of tweets expressing disagreement'},
    'content_num_polarity': {'type': 'int', 'range': [0, float('inf')], 'description': 'Number of tweets containing polarity'},
    'content_num_subjectivity': {'type': 'int', 'range': [0, float('inf')], 'description': 'Number of tweets containing subjectivity'},
    
    # Structural features
    'structural_num_tweets_with_mentions': {'type': 'int', 'range': [0, float('inf')], 'description': 'Number of tweets containing mentions'},
    'structural_ratio_tweets_with_hashtags': {'type': 'float', 'range': [0.0, 1.0], 'description': 'Ratio of tweets containing hashtags'},
    'structural_conversation_depth': {'type': 'int', 'range': [0, float('inf')], 'description': 'Depth of conversation tree'},
    'structural_num_tweets_with_hashtags': {'type': 'int', 'range': [0, float('inf')], 'description': 'Number of tweets containing hashtags'},
    'structural_thread_lifetime_minutes': {'type': 'float', 'range': [0.0, float('inf')], 'description': 'Time between first and last tweet'},
    'structural_num_tweets_with_urls': {'type': 'int', 'range': [0, float('inf')], 'description': 'Number of tweets containing URLs'},
    'structural_num_tweets': {'type': 'int', 'range': [0, float('inf')], 'description': 'Total number of tweets in thread'},
    'structural_num_tweets_with_media': {'type': 'int', 'range': [0, float('inf')], 'description': 'Number of tweets with images/video'},
    'structural_ratio_tweets_with_urls': {'type': 'float', 'range': [0.0, 1.0], 'description': 'Ratio of tweets containing URLs'},
    'structural_avg_tweet_length': {'type': 'float', 'range': [0.0, float('inf')], 'description': 'Average length of tweets'},
    'structural_ratio_tweets_with_media': {'type': 'float', 'range': [0.0, 1.0], 'description': 'Ratio of tweets containing media'},
    'structural_ratio_tweets_with_mentions': {'type': 'float', 'range': [0.0, 1.0], 'description': 'Ratio of tweets containing mentions'},
    'structural_num_retweets': {'type': 'int', 'range': [0, float('inf')], 'description': 'Number of retweets'},
    'structural_ratio_retweets': {'type': 'float', 'range': [0.0, 1.0], 'description': 'Ratio of retweets to total tweets'},
    
    # User features
    'user_avg_account_age_days': {'type': 'float', 'range': [0.0, float('inf')], 'description': 'Average account age in days'},
    'user_avg_followers_count': {'type': 'float', 'range': [0.0, float('inf')], 'description': 'Average number of followers per user'},
    'user_avg_friends_count': {'type': 'float', 'range': [0.0, float('inf')], 'description': 'Average number of friends per user'},
    'user_avg_statuses_count': {'type': 'float', 'range': [0.0, float('inf')], 'description': 'Average number of statuses per user'},
    'user_num_verified': {'type': 'int', 'range': [0, float('inf')], 'description': 'Number of verified users'},
    'user_network_density': {'type': 'float', 'range': [0.0, 1.0], 'description': 'Density of user interaction network'},
    'user_avg_account_age_at_tweet': {'type': 'float', 'range': [0.0, float('inf')], 'description': 'Average time between account creation and tweet'},
    'user_source_verified': {'type': 'int', 'range': [0, 1], 'description': 'Whether source tweet author is verified (binary)'},
    'user_source_account_age_days': {'type': 'float', 'range': [0.0, float('inf')], 'description': 'Source author\'s account age in days'},
    'user_source_account_age_at_tweet': {'type': 'float', 'range': [0.0, float('inf')], 'description': 'Time between source author\'s account creation and tweet'},
    'user_verified_ratio': {'type': 'float', 'range': [0.0, 1.0], 'description': 'Ratio of verified users'},
    'user_followers_friends_ratio': {'type': 'float', 'range': [0.0, float('inf')], 'description': 'Ratio of followers to friends'},
    'user_interaction_count': {'type': 'int', 'range': [0, float('inf')], 'description': 'Number of user interactions (mentions/retweets)'},
    'user_unique_authors': {'type': 'int', 'range': [0, float('inf')], 'description': 'Number of unique authors in thread'},
    'user_avg_interactions_per_author': {'type': 'float', 'range': [0.0, float('inf')], 'description': 'Average interactions per author'},
    
    # Temporal features
    'temporal_account_age_slope': {'type': 'float', 'range': [-float('inf'), float('inf')], 'description': 'Trend in user account ages over time'},
    'temporal_followers_count_slope': {'type': 'float', 'range': [-float('inf'), float('inf')], 'description': 'Trend in follower counts over time'},
    'temporal_statuses_count_slope': {'type': 'float', 'range': [-float('inf'), float('inf')], 'description': 'Trend in status counts over time'},
    'temporal_tweets_per_minute_slope': {'type': 'float', 'range': [-float('inf'), float('inf')], 'description': 'Trend in tweet frequency over time'},
    'temporal_friends_count_slope': {'type': 'float', 'range': [-float('inf'), float('inf')], 'description': 'Trend in friend counts over time'},
    'temporal_interaction_slope': {'type': 'float', 'range': [-float('inf'), float('inf')], 'description': 'Trend in user interactions over time'},
    'temporal_age_tweet_diff_slope': {'type': 'float', 'range': [-float('inf'), float('inf')], 'description': 'Trend in time between tweets over time'},
    'temporal_network_density_slope': {'type': 'float', 'range': [-float('inf'), float('inf')], 'description': 'Trend in network density over time'}
}

def validate_feature_types(df: pd.DataFrame, feature_metadata: Dict = FEATURE_METADATA) -> Tuple[bool, Dict]:
    """Validate that features have the correct data types.
    
    Args:
        df: DataFrame containing features to validate
        feature_metadata: Dictionary mapping feature names to expected types and ranges
        
    Returns:
        Tuple of (is_valid, issues_dict) where issues_dict maps feature names to lists of issues
    """
    issues = defaultdict(list)
    is_valid = True
    
    for col in df.columns:
        # Skip non-feature columns
        if not any(col.startswith(prefix) for prefix in ['content_', 'structural_', 'user_', 'temporal_']):
            continue
            
        # Check if feature is in metadata
        if col not in feature_metadata:
            issues[col].append(f"Feature not defined in metadata")
            is_valid = False
            continue
            
        # Get expected type
        expected_type = feature_metadata[col]['type']
        
        # Check type
        if expected_type == 'int':
            if not pd.api.types.is_integer_dtype(df[col]):
                # Try to convert to int if it's float with no decimal part
                if pd.api.types.is_float_dtype(df[col]) and df[col].dropna().apply(lambda x: x.is_integer()).all():
                    logger.warning(f"Feature {col} is float but contains only integer values. Converting to int.")
                    df[col] = df[col].astype('Int64')  # Use nullable integer type
                else:
                    issues[col].append(f"Expected integer type, got {df[col].dtype}")
                    is_valid = False
        elif expected_type == 'float':
            if not pd.api.types.is_float_dtype(df[col]):
                issues[col].append(f"Expected float type, got {df[col].dtype}")
                is_valid = False
    
    return is_valid, dict(issues)

def validate_feature_ranges(df: pd.DataFrame, feature_metadata: Dict = FEATURE_METADATA) -> Tuple[bool, Dict]:
    """Validate that features have values within the expected ranges.
    
    Args:
        df: DataFrame containing features to validate
        feature_metadata: Dictionary mapping feature names to expected types and ranges
        
    Returns:
        Tuple of (is_valid, issues_dict) where issues_dict maps feature names to lists of issues
    """
    issues = defaultdict(list)
    is_valid = True
    
    for col in df.columns:
        # Skip non-feature columns
        if not any(col.startswith(prefix) for prefix in ['content_', 'structural_', 'user_', 'temporal_']):
            continue
            
        # Check if feature is in metadata
        if col not in feature_metadata:
            continue  # Already reported in validate_feature_types
            
        # Get expected range
        min_val, max_val = feature_metadata[col]['range']
        
        # Check range (ignoring NaN values)
        non_nan_values = df[col].dropna()
        if len(non_nan_values) > 0:
            actual_min = non_nan_values.min()
            actual_max = non_nan_values.max()
            
            if actual_min < min_val:
                issues[col].append(f"Values below minimum: min={actual_min}, expected min={min_val}")
                is_valid = False
                
            if actual_max > max_val and max_val != float('inf'):
                issues[col].append(f"Values above maximum: max={actual_max}, expected max={max_val}")
                is_valid = False
    
    return is_valid, dict(issues)

def fix_feature_types(df: pd.DataFrame, feature_metadata: Dict = FEATURE_METADATA) -> pd.DataFrame:
    """Fix feature types to match expected types.
    
    Args:
        df: DataFrame containing features to fix
        feature_metadata: Dictionary mapping feature names to expected types and ranges
        
    Returns:
        DataFrame with fixed feature types
    """
    df_fixed = df.copy()
    
    for col in df.columns:
        # Skip non-feature columns
        if not any(col.startswith(prefix) for prefix in ['content_', 'structural_', 'user_', 'temporal_']):
            continue
            
        # Check if feature is in metadata
        if col not in feature_metadata:
            logger.warning(f"Feature {col} not defined in metadata, skipping type conversion")
            continue
            
        # Get expected type
        expected_type = feature_metadata[col]['type']
        
        # Convert type
        try:
            if expected_type == 'int':
                df_fixed[col] = df[col].fillna(0).astype('int64')
            elif expected_type == 'float':
                df_fixed[col] = df[col].astype('float64')
        except Exception as e:
            logger.error(f"Error converting {col} to {expected_type}: {str(e)}")
    
    return df_fixed

def fix_feature_ranges(df: pd.DataFrame, feature_metadata: Dict = FEATURE_METADATA) -> pd.DataFrame:
    """Fix feature values to be within expected ranges.
    
    Args:
        df: DataFrame containing features to fix
        feature_metadata: Dictionary mapping feature names to expected types and ranges
        
    Returns:
        DataFrame with fixed feature ranges
    """
    df_fixed = df.copy()
    
    for col in df.columns:
        # Skip non-feature columns
        if not any(col.startswith(prefix) for prefix in ['content_', 'structural_', 'user_', 'temporal_']):
            continue
            
        # Check if feature is in metadata
        if col not in feature_metadata:
            logger.warning(f"Feature {col} not defined in metadata, skipping range fixing")
            continue
            
        # Get expected range
        min_val, max_val = feature_metadata[col]['range']
        
        # Fix range
        if min_val > -float('inf'):
            df_fixed[col] = df_fixed[col].clip(lower=min_val)
            
        if max_val < float('inf'):
            df_fixed[col] = df_fixed[col].clip(upper=max_val)
    
    return df_fixed

def ensure_feature_consistency(df: pd.DataFrame, fix_issues: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """Ensure features have consistent types and ranges.
    
    Args:
        df: DataFrame containing features to validate and fix
        fix_issues: Whether to fix issues automatically
        
    Returns:
        Tuple of (fixed_df, issues_dict) where issues_dict contains all validation issues
    """
    all_issues = {}
    
    # Validate feature types
    types_valid, type_issues = validate_feature_types(df)
    all_issues.update(type_issues)
    
    # Validate feature ranges
    ranges_valid, range_issues = validate_feature_ranges(df)
    
    # Merge range issues with type issues
    for feature, issues in range_issues.items():
        if feature in all_issues:
            all_issues[feature].extend(issues)
        else:
            all_issues[feature] = issues
    
    # Fix issues if requested
    if fix_issues:
        # Fix types first
        df_fixed = fix_feature_types(df)
        
        # Then fix ranges
        df_fixed = fix_feature_ranges(df_fixed)
        
        return df_fixed, all_issues
    else:
        return df, all_issues

def validate_datasets_consistency(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """Validate consistency of features across multiple datasets.
    
    Args:
        datasets: Dictionary mapping dataset names to DataFrames
        
    Returns:
        Dictionary mapping dataset names to dictionaries of issues
    """
    all_dataset_issues = {}
    
    for name, df in datasets.items():
        _, issues = ensure_feature_consistency(df, fix_issues=False)
        if issues:
            all_dataset_issues[name] = issues
    
    return all_dataset_issues

def align_datasets_features(datasets: Dict[str, pd.DataFrame], output_dir: str = None) -> Dict[str, pd.DataFrame]:
    """Align features across multiple datasets to ensure consistency.
    
    Args:
        datasets: Dictionary mapping dataset names to DataFrames
        output_dir: Directory to save aligned datasets (optional)
        
    Returns:
        Dictionary mapping dataset names to aligned DataFrames
    """
    aligned_datasets = {}
    
    # Fix each dataset
    for name, df in datasets.items():
        logger.info(f"Aligning features for dataset: {name}")
        df_fixed, issues = ensure_feature_consistency(df, fix_issues=True)
        
        if issues:
            logger.warning(f"Fixed {len(issues)} feature issues in {name}")
            for feature, feature_issues in issues.items():
                logger.warning(f"  {feature}: {', '.join(feature_issues)}")
        
        aligned_datasets[name] = df_fixed
        
        # Save if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{name}_aligned.csv")
            df_fixed.to_csv(output_path, index=False)
            logger.info(f"Saved aligned dataset to: {output_path}")
    
    return aligned_datasets 