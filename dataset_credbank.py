import json
import os
import pandas as pd
import sweetviz as sv
import numpy as np
from typing import Dict, List, Tuple, Any, Union
from collections import defaultdict
import warnings
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
from utils.features import (
    CredbankStructuralFeatureExtractor,
    CredbankUserFeatureExtractor,
    CredbankContentFeatureExtractor,
    CredbankTemporalFeatureExtractor
)
from utils.dataset_alignment import save_feature_sets

# CREDBANK dataset processor for credibility assessment using social media features
def process_list_column(column_data):
    """Process a string representation of a list into an actual list."""
    try:
        return eval(column_data)
    except:
        return []

# Loads raw CREDBANK dataset from multiple data files containing credibility ratings and tweets
def load_credbank_dataset_raw(base_path: str = 'data/credbank/CREDBANK', output_dir: str = None, save_csv: bool = False) -> pd.DataFrame:
    """Load the raw CREDBANK dataset files and combine them into a single DataFrame.
    
    Args:
        base_path: Path to the CREDBANK dataset directory
        output_dir: Directory to save output files (defaults to base_path if None)
        save_csv: Whether to save intermediate CSV files
        
    Returns:
        DataFrame containing the combined raw dataset
    """
    # Use base_path as output_dir if not specified
    output_dir = output_dir or base_path
    
    # Define file paths
    credibility_ratings_path = os.path.join(base_path, 'cred_event_TurkRatings.data')
    event_annotations_path = os.path.join(base_path, 'eventNonEvent_annotations.data')
    search_tweets_path = os.path.join(base_path, 'cred_event_SearchTweets.data')
    
    # Verify paths
    for path in [credibility_ratings_path, event_annotations_path, search_tweets_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    
    print("Loading CREDBANK datasets...")
    
    # Load credibility ratings
    cred_ratings_df = pd.read_csv(
        credibility_ratings_path, 
        sep='\t',
        names=['topic_key', 'topic_terms', 'Cred_Ratings', 'Reasons'],
        skiprows=1  # Skip header row
    )
    
    # Load event annotations
    event_annotations_df = pd.read_csv(
        event_annotations_path,
        sep='\t',
        names=['timespan_key', 'topic_terms', 'isEvent'],
        skiprows=1  # Skip header row
    )
    
    # Process tweets line by line
    print("Processing search tweets...")
    tweet_metrics = []
    
    with tqdm(total=sum(1 for _ in open(search_tweets_path)), desc="Processing tweets") as pbar:
        with open(search_tweets_path, 'r') as f:
            for line in f:
                try:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:  # Ensure we have all required columns
                        topic_key = parts[0]
                        tweet_list_str = parts[3]  # The tweet list is in the fourth column
                        tweet_list = process_list_column(tweet_list_str)
                        unique_authors = set(tuple_item[1].split('AUTHOR=')[1] for tuple_item in tweet_list)
                        
                        # Extract additional metrics
                        tweet_metrics.append({
                            'topic_key': topic_key,
                            'tweet_count': int(len(tweet_list)),
                            'unique_authors': int(len(unique_authors))
                        })
                except Exception as e:
                    print(f"\nError processing line: {str(e)}")
                finally:
                    pbar.update(1)
    
    search_tweets_df = pd.DataFrame(tweet_metrics)
    
    # Merge all DataFrames
    merged_df = pd.merge(
        cred_ratings_df,
        event_annotations_df,
        left_on='topic_key',
        right_on='timespan_key',
        how='left',
        suffixes=('', '_event')  # Keep original column names, add _event suffix for duplicates
    )
    
    # Drop duplicate topic_terms column
    if 'topic_terms_event' in merged_df.columns:
        merged_df.drop('topic_terms_event', axis=1, inplace=True)
    
    # Merge with tweets data
    merged_df = pd.merge(
        merged_df,
        search_tweets_df,
        on='topic_key',
        how='left'
    )
    
    # Save if requested
    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'credbank_raw_dataset.csv')
        merged_df.to_csv(output_path, index=False, header=True)
        print(f"Saved raw dataset to: {output_path}")
    
    return merged_df

# Processes raw CREDBANK data into a structured DataFrame with basic credibility features
def load_credbank_dataset_extended(raw_dataset: Union[Dict, str, None] = None, base_path: str = 'data/credbank/CREDBANK', output_dir: str = None, save_csv: bool = False) -> pd.DataFrame:
    """Load and process the CREDBANK dataset with basic features but without the feature extractors.
    Can either take a raw dataset dict or load from a CSV file.
    
    Args:
        raw_dataset: Either a raw dataset dict from load_credbank_dataset_raw or a path to a CSV file
        base_path: Path to the CREDBANK dataset directory (used if raw_dataset is None)
        output_dir: Directory to save output files (defaults to base_path if None)
        save_csv: Whether to save intermediate CSV files
        
    Returns:
        DataFrame containing the processed dataset with basic features
    """
    # Use base_path as output_dir if not specified
    output_dir = output_dir or base_path
    
    if isinstance(raw_dataset, str) and os.path.exists(raw_dataset):
        # Load from specified CSV path
        print(f"Loading extended dataset from CSV: {raw_dataset}")
        return pd.read_csv(raw_dataset)
    
    if raw_dataset is None:
        raw_dataset = load_credbank_dataset_raw(base_path)
    
    # Process list columns in ratings DataFrame
    list_columns = ['topic_terms', 'Cred_Ratings', 'Reasons']
    for col in list_columns:
        raw_dataset[col] = raw_dataset[col].apply(process_list_column)
    
    # Create a more robust mapping function that handles different formats
    def normalize_topic_terms(terms):
        """Normalize topic terms for consistent matching"""
        if isinstance(terms, str):
            # Handle string format from event annotations
            terms = terms.lower().replace('[', '').replace(']', '').replace("'", '').replace('"', '')
            return sorted(term.strip() for term in terms.split(','))
        elif isinstance(terms, list):
            # Handle list format from credibility ratings
            return sorted(str(term).lower().strip() for term in terms)
        return []

    # Create mapping with normalized terms
    topic_event_map = defaultdict(list)
    for _, row in raw_dataset.iterrows():
        normalized_terms = tuple(normalize_topic_terms(row['topic_terms']))
        if normalized_terms and pd.notna(row.get('isEvent')):  # Only add if we have valid terms and isEvent
            topic_event_map[normalized_terms].append({
                'time_key': row['timespan_key'],
                'isEvent': int(row['isEvent']) if pd.notna(row['isEvent']) else 0  # Default to 0 if NaN
            })

    # Modified matching function
    def get_event_info(topic_terms):
        normalized_terms = tuple(normalize_topic_terms(topic_terms))
        # Try exact match first
        if normalized_terms in topic_event_map:
            return topic_event_map[normalized_terms]
        
        # If no exact match, try partial matching
        normalized_terms_set = set(normalized_terms)
        for stored_terms, values in topic_event_map.items():
            stored_terms_set = set(stored_terms)
            if len(normalized_terms_set & stored_terms_set) >= 2:  # At least 2 terms match
                return values
        return []

    # Apply the matching to get event info
    raw_dataset['event_info'] = raw_dataset['topic_terms'].apply(get_event_info)
    
    # Extract time_keys and isEvent from event_info
    raw_dataset['time_keys'] = raw_dataset['event_info'].apply(
        lambda x: [item['time_key'] for item in x] if x else []
    )
    raw_dataset['isEvent'] = raw_dataset['event_info'].apply(
        lambda x: [item['isEvent'] for item in x] if x else [0]  # Default to [0] for no events
    )
    
    # Clean up temporary column
    raw_dataset.drop('event_info', axis=1, inplace=True)
    
    # Calculate average credibility rating
    raw_dataset['avg_credibility'] = raw_dataset['Cred_Ratings'].apply(
        lambda x: sum(map(float, x)) / len(x) if x else 0
    )
    
    # Add additional fields
    raw_dataset['num_credibility_ratings'] = raw_dataset['Cred_Ratings'].apply(len)
    raw_dataset['num_reasons'] = raw_dataset['Reasons'].apply(len)
    
    # Add credibility distribution columns
    for rating in ['certainly_inaccurate', 'probably_inaccurate', 'uncertain', 'probably_accurate', 'certainly_accurate']:
        raw_dataset[f'num_{rating}'] = raw_dataset['Cred_Ratings'].apply(
            lambda x: [int(r) for r in x].count({'certainly_inaccurate': -2, 'probably_inaccurate': -1, 
                                                'uncertain': 0, 'probably_accurate': 1, 
                                                'certainly_accurate': 2}[rating])
        )
    
    # Calculate agreement metrics
    def calculate_agreement(ratings):
        ratings = [int(r) for r in ratings]
        total = len(ratings)
        if total == 0:
            return 0
        mode_count = max(ratings.count(r) for r in set(ratings))
        return mode_count / total
    
    raw_dataset['rater_agreement'] = raw_dataset['Cred_Ratings'].apply(calculate_agreement)
    
    # Save if requested
    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'credbank_extended_dataset.csv')
        raw_dataset.to_csv(output_path, index=False, header=True)
        print(f"Saved extended dataset to: {output_path}")
    
    return raw_dataset

# Extracts advanced features from CREDBANK data using specialized feature extractors
def load_credbank_features_dataset(extended_dataset: Union[pd.DataFrame, str, None] = None, base_path: str = 'data/credbank', output_dir: str = None, save_csv: bool = False, include_additional_features: bool = False) -> pd.DataFrame:
    """Load or create the CREDBANK features dataset with all feature extractors applied.
    Can either take an extended dataset DataFrame or load from a CSV file.
    
    Args:
        extended_dataset: Either a DataFrame from load_credbank_dataset_extended or a path to a CSV file
        base_path: Path to the CREDBANK dataset directory (used if extended_dataset is None)
        output_dir: Directory to save output files (defaults to base_path if None)
        save_csv: Whether to save intermediate CSV files
        include_additional_features: Whether to include additional features
        
    Returns:
        DataFrame containing only the extracted features, source, and label
    """
    # Use base_path as output_dir if not specified
    output_dir = output_dir or base_path
    
    if isinstance(extended_dataset, str) and os.path.exists(extended_dataset):
        # Load from specified CSV path
        print(f"Loading features dataset from CSV: {extended_dataset}")
        return pd.read_csv(extended_dataset)
    
    if extended_dataset is None:
        extended_dataset = load_credbank_dataset_extended(base_path=base_path, output_dir=output_dir, save_csv=save_csv)
    
    # Add binary label (1 for low credibility/potentially fake, 0 for high credibility/real)
    extended_dataset['label'] = (extended_dataset['avg_credibility'] < 0).astype(int)
    
    # Extract all features
    df_with_features = extract_all_features(extended_dataset, include_additional_features)
    
    # Get feature columns
    feature_columns = [col for col in df_with_features.columns if any(
        col.startswith(prefix) for prefix in 
        ['structural_', 'user_', 'content_', 'temporal_']
    )]
    
    # Create features DataFrame
    features_df = df_with_features[feature_columns].copy()
    features_df.insert(0, 'source', 'credbank')
    features_df['label'] = df_with_features['label']
    
    # Save if requested
    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        save_feature_sets(features_df, output_dir, 'credbank')
    
    return features_df

# Applies all CREDBANK-specific feature extractors to generate the complete feature set
def extract_all_features(df: pd.DataFrame, include_additional_features: bool = False) -> pd.DataFrame:
    """Extract all CREDBANK-specific features from the dataset.
    
    Args:
        df: Input DataFrame
        include_additional_features: Whether to include additional features not in the paper
        
    Returns:
        DataFrame with all extracted features
    """
    # Make a copy of the input DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Initialize feature extractors
    extractors = [
        CredbankStructuralFeatureExtractor(result_df),
        CredbankUserFeatureExtractor(result_df, include_additional_features),
        CredbankContentFeatureExtractor(result_df, include_additional_features),
        CredbankTemporalFeatureExtractor(result_df, include_additional_features)
    ]
    
    # Apply each extractor and update the DataFrame
    for extractor in extractors:
        # Get features from the current extractor
        updated_df = extractor.extract_features()
        
        # Add any new columns from the updated DataFrame
        new_columns = set(updated_df.columns) - set(result_df.columns)
        for col in new_columns:
            result_df[col] = updated_df[col]
    
    return result_df

# Prepares DataFrame for Sweetviz analysis by converting complex data types to simple ones
def prepare_df_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for Sweetviz analysis by handling complex data types"""
    analysis_df = df.copy()
    for column in analysis_df.columns:
        # Convert list columns to their lengths
        if analysis_df[column].apply(lambda x: isinstance(x, list)).any():
            analysis_df[column] = analysis_df[column].apply(lambda x: len(x) if isinstance(x, list) else x)
        # Convert unhashable types to strings
        elif analysis_df[column].apply(lambda x: isinstance(x, (dict, set))).any():
            analysis_df[column] = analysis_df[column].apply(str)
    return analysis_df

# Main execution function to process CREDBANK dataset and generate analysis reports
def main():
    """Main execution function"""
    warnings.filterwarnings('ignore')
    
    # Set output directory
    output_dir = 'data/credbank'
    os.makedirs(output_dir, exist_ok=True)

    # Load raw dataset
    print("Loading raw dataset...")
    raw_df = load_credbank_dataset_raw(output_dir=output_dir, save_csv=True)

    # Load extended dataset
    print("\nCreating extended dataset...")
    extended_df = load_credbank_dataset_extended(raw_df, output_dir=output_dir, save_csv=True)
    
    # Extract all features once
    print("\nExtracting features...")
    all_features_df = load_credbank_features_dataset(extended_df, output_dir=output_dir, save_csv=False, include_additional_features=True)

    # Split into paper and all features using dataset_alignment.save_feature_sets
    paper_features_df, all_features_df = save_feature_sets(all_features_df, output_dir, 'credbank')

    # Generate analysis report
    try:
        analysis_df = prepare_df_for_analysis(paper_features_df)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            sweet_report = sv.analyze(analysis_df)
            output_path = os.path.join(output_dir, 'credbank_paper_features_analysis_report.html')
            sweet_report.show_html(output_path) # Could use sweet_report.show_notebook()
            print(f"Analysis report saved to: {output_path}")
    except Exception as e:
        print(f"Warning: Could not generate Sweetviz report. Error: {str(e)}")

    # Generate analysis report
    try:
        analysis_df = prepare_df_for_analysis(all_features_df)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            sweet_report = sv.analyze(analysis_df)
            output_path = os.path.join(output_dir, 'credbank_all_features_analysis_report.html')
            sweet_report.show_html(output_path) # Could use sweet_report.show_notebook()
            print(f"Analysis report saved to: {output_path}")
    except Exception as e:
        print(f"Warning: Could not generate Sweetviz report. Error: {str(e)}")
    
    print("\nDataset processing complete!")
    print(f"Raw dataset shape: {raw_df.shape}")
    print(f"Extended dataset shape: {extended_df.shape}")
    print(f"Paper features shape: {paper_features_df.shape}")
    print(f"All features shape: {all_features_df.shape}")

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main() 