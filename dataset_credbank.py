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
from datetime import datetime
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# CREDBANK dataset processor for credibility assessment using social media features
def process_list_column(column_data):
    """Process a string representation of a list into an actual list."""
    # If already a list or numpy array, return it
    if isinstance(column_data, (list, np.ndarray)):
        return column_data
        
    # If None or NaN, return empty list
    if column_data is None or pd.isna(column_data):
        return []
        
    # If it's a string, try to evaluate it as a list
    if isinstance(column_data, str):
        try:
            result = eval(column_data)
            if isinstance(result, (list, np.ndarray)):
                return result
            else:
                return [result]  # Convert single items to a list
        except:
            # If eval fails, try to split by comma (simple CSV format)
            try:
                if ',' in column_data:
                    return [item.strip() for item in column_data.split(',')]
                else:
                    return [column_data]  # Single item
            except:
                return []
    
    # For any other type, try to convert to list or return empty list
    try:
        return list(column_data)
    except:
        return []

# Parse Twitter date format to datetime object
def parse_twitter_date(date_str):
    """Parse Twitter's date format to datetime object."""
    if date_str is None or pd.isna(date_str):
        return None
        
    # If already a datetime object, return it
    if isinstance(date_str, datetime):
        return date_str
        
    date_str = str(date_str).strip()
    formats = [
        '%a %b %d %H:%M:%S +0000 %Y',  # Twitter API format
        '%Y-%m-%d %H:%M:%S',           # Standard format
        '%Y-%m-%dT%H:%M:%S.%fZ',       # ISO format
        '%Y-%m-%dT%H:%M:%SZ',          # ISO format without milliseconds
        '%Y-%m-%d',                    # Date only format
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    
    # Try to handle other potential formats
    try:
        # Try pandas to_datetime as a fallback with error handling
        dt = pd.to_datetime(date_str, errors='coerce')
        if pd.notna(dt):
            return dt.to_pydatetime()
    except:
        pass
        
    # Log the failure for debugging
    logging.debug(f"Failed to parse date string: {date_str}")
    return None

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
                        
                        # Try to parse the tweet list as JSON first
                        try:
                            tweet_list = json.loads(tweet_list_str)
                        except json.JSONDecodeError:
                            # If JSON parsing fails, try eval with safety checks
                            try:
                                tweet_list = eval(tweet_list_str)
                            except:
                                # If both methods fail, use the process_list_column function
                                tweet_list = process_list_column(tweet_list_str)
                        
                        # Extract temporal and structural data
                        timestamps = []
                        followers = []
                        friends = []
                        statuses = []
                        account_created_ats = []
                        texts = []
                        user_ids = []
                        tweet_ids = []
                        in_reply_to_status_ids = []
                        
                        for tweet_data in tweet_list:
                            try:
                                # Handle different tweet data formats
                                if isinstance(tweet_data, dict):
                                    # Direct dictionary format
                                    tweet_obj = tweet_data
                                elif isinstance(tweet_data, (list, tuple)) and len(tweet_data) >= 4:
                                    # Tuple format with tweet data in position 3
                                    tweet_obj = tweet_data[3] if isinstance(tweet_data[3], dict) else {}
                                else:
                                    # Skip invalid formats
                                    continue
                                
                                # Extract created_at time
                                if 'created_at' in tweet_obj:
                                    created_at = parse_twitter_date(tweet_obj['created_at'])
                                    if created_at:
                                        timestamps.append(created_at.strftime('%Y-%m-%d %H:%M:%S'))
                                
                                # Extract user metrics
                                user_data = tweet_obj.get('user', {})
                                
                                # Extract follower count
                                follower_count = user_data.get('followers_count')
                                if follower_count is not None and str(follower_count).isdigit():
                                    followers.append(int(follower_count))
                                
                                # Extract friends count
                                friend_count = user_data.get('friends_count')
                                if friend_count is not None and str(friend_count).isdigit():
                                    friends.append(int(friend_count))
                                
                                # Extract statuses count
                                status_count = user_data.get('statuses_count')
                                if status_count is not None and str(status_count).isdigit():
                                    statuses.append(int(status_count))
                                
                                # Extract account creation date
                                if 'created_at' in user_data:
                                    account_created = parse_twitter_date(user_data['created_at'])
                                    if account_created:
                                        account_created_ats.append(account_created.strftime('%Y-%m-%d %H:%M:%S'))
                                
                                # Extract text
                                tweet_text = tweet_obj.get('text', '')
                                if tweet_text:
                                    texts.append(tweet_text)
                                
                                # Extract user ID
                                user_id = str(user_data.get('id_str', ''))
                                if not user_id:
                                    user_id = str(user_data.get('id', ''))
                                if user_id:
                                    user_ids.append(user_id)
                                
                                # Extract tweet ID
                                tweet_id = str(tweet_obj.get('id_str', ''))
                                if not tweet_id:
                                    tweet_id = str(tweet_obj.get('id', ''))
                                if tweet_id:
                                    tweet_ids.append(tweet_id)
                                
                                # Extract in_reply_to_status_id
                                reply_id = str(tweet_obj.get('in_reply_to_status_id_str', ''))
                                if not reply_id:
                                    reply_id = str(tweet_obj.get('in_reply_to_status_id', ''))
                                if reply_id and reply_id != '0' and reply_id.lower() != 'none':
                                    in_reply_to_status_ids.append(reply_id)
                                
                            except Exception as e:
                                logging.error(f"Error processing tweet: {str(e)}")
                                continue
                        
                        # Store the extracted data
                        tweet_metrics.append({
                            'topic_key': topic_key,
                            'created_at_times': timestamps if timestamps else None,
                            'followers_counts': followers if followers else None,
                            'friends_counts': friends if friends else None,
                            'statuses_counts': statuses if statuses else None,
                            'account_created_ats': account_created_ats if account_created_ats else None,
                            'texts': texts if texts else None,
                            'user_ids': user_ids if user_ids else None,
                            'tweet_ids': tweet_ids if tweet_ids else None,
                            'in_reply_to_status_ids': in_reply_to_status_ids if in_reply_to_status_ids else None,
                            'tweet_count': len(timestamps)
                        })
                except Exception as e:
                    logging.error(f"Error processing line: {str(e)}")
                finally:
                    pbar.update(1)
    
    if not tweet_metrics:
        warnings.warn("No valid tweet data found")
        tweet_metrics_df = pd.DataFrame(columns=[
            'topic_key', 'created_at_times', 'followers_counts', 'friends_counts', 
            'statuses_counts', 'tweet_count'
        ])
    else:
        tweet_metrics_df = pd.DataFrame(tweet_metrics)
    
    # Merge datasets
    merged_df = cred_ratings_df.copy()
    
    # Add tweet metrics
    if not tweet_metrics_df.empty:
        merged_df = pd.merge(
            merged_df, 
            tweet_metrics_df,
            on='topic_key',
            how='left'
        )
    else:
        # Add empty columns if no tweet data
        for col in ['created_at_times', 'followers_counts', 'friends_counts', 
                   'statuses_counts', 'account_created_ats', 'texts', 'user_ids',
                   'tweet_ids', 'in_reply_to_status_ids', 'tweet_count']:
            merged_df[col] = None
    
    # Add event annotations
    merged_df = pd.merge(
        merged_df,
        event_annotations_df,
        on='topic_terms',
        how='left'
    )
    
    # Save raw dataset if requested
    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'credbank_raw.csv')
        merged_df.to_csv(output_path, index=False)
        print(f"Raw dataset saved to: {output_path}")
    
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
        df = pd.read_csv(raw_dataset)
        
        def safe_eval(x):
            if pd.isna(x):
                return []
            if not isinstance(x, str):
                if isinstance(x, (list, np.ndarray)):
                    return x
                return []
            try:
                val = eval(x)
                return val if isinstance(val, (list, np.ndarray)) else [val]
            except:
                # Try to split by comma if it looks like a CSV
                try:
                    if ',' in x:
                        return [item.strip() for item in x.split(',')]
                    else:
                        return [x]  # Single item
                except:
                    return []
        
        # Convert string representations of lists back to actual lists
        list_columns = ['topic_terms', 'Cred_Ratings', 'Reasons', 'created_at_times', 
                       'followers_counts', 'friends_counts', 'statuses_counts']
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(safe_eval)
        
        # Extract timestamps from topic_key if created_at_times is empty
        if 'created_at_times' not in df.columns or df['created_at_times'].isna().all():
            timestamp_pattern = re.compile(r'(\d{8}_\d{6})')
            
            # Extract timestamps from topic_key
            def extract_timestamps(topic_key):
                if pd.isna(topic_key):
                    return []
                
                # Try to find timestamps in the topic_key
                matches = timestamp_pattern.findall(str(topic_key))
                
                # Convert to datetime format
                timestamps = []
                for match in matches:
                    try:
                        # Format: YYYYMMDD_HHMMSS
                        year = match[:4]
                        month = match[4:6]
                        day = match[6:8]
                        hour = match[9:11]
                        minute = match[11:13]
                        second = match[13:15]
                        timestamp = f"{year}-{month}-{day} {hour}:{minute}:{second}"
                        timestamps.append(timestamp)
                    except Exception as e:
                        logging.warning(f"Error parsing timestamp {match}: {str(e)}")
                        continue
                
                # If no timestamps found, try to extract from timespan_key format
                if not timestamps and '-' in str(topic_key):
                    try:
                        # Try to extract from format like "term1_term2_term3-20141024_170629-20141024_181626"
                        parts = str(topic_key).split('-')
                        for part in parts:
                            if re.match(r'\d{8}_\d{6}', part):
                                year = part[:4]
                                month = part[4:6]
                                day = part[6:8]
                                hour = part[9:11]
                                minute = part[11:13]
                                second = part[13:15]
                                timestamp = f"{year}-{month}-{day} {hour}:{minute}:{second}"
                                timestamps.append(timestamp)
                    except Exception as e:
                        logging.warning(f"Error parsing timespan from topic_key {topic_key}: {str(e)}")
                
                return timestamps
            
            df['created_at_times'] = df['topic_key'].apply(extract_timestamps)
            print(f"Extracted timestamps from topic_key: {df['created_at_times'].str.len().sum()} timestamps found")
        
        return df
    
    if raw_dataset is None:
        raw_dataset = load_credbank_dataset_raw(base_path)
    
    # Process list columns in ratings DataFrame
    list_columns = ['topic_terms', 'Cred_Ratings', 'Reasons']
    for col in list_columns:
        raw_dataset[col] = raw_dataset[col].apply(process_list_column)
    
    # Extract event info from topic_terms and timespan_key
    raw_dataset['event_info'] = raw_dataset.apply(
        lambda row: [
            {
                'time_key': time_key,
                'isEvent': is_event
            }
            for time_key, is_event in zip(
                [row['timespan_key']] if not pd.isna(row['timespan_key']) else [],
                [row['isEvent']] if not pd.isna(row['isEvent']) else []
            )
        ],
        axis=1
    )
    
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
    
    # Extract timestamps from topic_key
    timestamp_pattern = re.compile(r'(\d{8}_\d{6})')
    
    # Extract timestamps from topic_key
    def extract_timestamps(topic_key):
        if pd.isna(topic_key):
            return []
        
        # Try to find timestamps in the topic_key
        matches = timestamp_pattern.findall(str(topic_key))
        
        # Convert to datetime format
        timestamps = []
        for match in matches:
            try:
                # Format: YYYYMMDD_HHMMSS
                year = match[:4]
                month = match[4:6]
                day = match[6:8]
                hour = match[9:11]
                minute = match[11:13]
                second = match[13:15]
                timestamp = f"{year}-{month}-{day} {hour}:{minute}:{second}"
                timestamps.append(timestamp)
            except Exception as e:
                logging.warning(f"Error parsing timestamp {match}: {str(e)}")
                continue
        
        # If no timestamps found, try to extract from timespan_key format
        if not timestamps and '-' in str(topic_key):
            try:
                # Try to extract from format like "term1_term2_term3-20141024_170629-20141024_181626"
                parts = str(topic_key).split('-')
                for part in parts:
                    if re.match(r'\d{8}_\d{6}', part):
                        year = part[:4]
                        month = part[4:6]
                        day = part[6:8]
                        hour = part[9:11]
                        minute = part[11:13]
                        second = part[13:15]
                        timestamp = f"{year}-{month}-{day} {hour}:{minute}:{second}"
                        timestamps.append(timestamp)
            except Exception as e:
                logging.warning(f"Error parsing timespan from topic_key {topic_key}: {str(e)}")
        
        return timestamps
    
    # Apply the function to extract timestamps
    raw_dataset['created_at_times'] = raw_dataset['topic_key'].apply(extract_timestamps)
    
    # Log the results
    timestamp_count = sum(len(times) for times in raw_dataset['created_at_times'])
    print(f"Extracted timestamps from topic_key: {timestamp_count} timestamps found in {len(raw_dataset)} rows")
    
    # If we have timespan_key column, also extract timestamps from there as a backup
    if 'timespan_key' in raw_dataset.columns:
        def extract_from_timespan(timespan_key, existing_times):
            # Skip if we already have timestamps
            if existing_times and len(existing_times) > 0:
                return existing_times
                
            if pd.isna(timespan_key):
                return existing_times
                
            # Try to extract timestamps from timespan_key
            matches = timestamp_pattern.findall(str(timespan_key))
            timestamps = []
            
            for match in matches:
                try:
                    year = match[:4]
                    month = match[4:6]
                    day = match[6:8]
                    hour = match[9:11]
                    minute = match[11:13]
                    second = match[13:15]
                    timestamp = f"{year}-{month}-{day} {hour}:{minute}:{second}"
                    timestamps.append(timestamp)
                except:
                    continue
                    
            return timestamps if timestamps else existing_times
        
        # Only use timespan_key if created_at_times is empty
        raw_dataset['created_at_times'] = raw_dataset.apply(
            lambda row: extract_from_timespan(row.get('timespan_key'), row['created_at_times']), 
            axis=1
        )
        
        # Log the updated results
        timestamp_count = sum(len(times) for times in raw_dataset['created_at_times'])
        print(f"After adding timespan_key timestamps: {timestamp_count} timestamps found in {len(raw_dataset)} rows")
    
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
    for i, extractor in enumerate(extractors):
        logging.info(f"Applying extractor {i+1}/{len(extractors)}: {extractor.__class__.__name__}")
        try:
            # Get features from the current extractor
            updated_df = extractor.extract_features()
            
            # Add any new columns from the updated DataFrame
            new_columns = set(updated_df.columns) - set(result_df.columns)
            for col in new_columns:
                result_df[col] = updated_df[col]
                
            # Log the number of features extracted
            feature_cols = [col for col in updated_df.columns if col not in df.columns]
            logging.info(f"Extracted {len(feature_cols)} features from {extractor.__class__.__name__}")
            
        except Exception as e:
            logging.error(f"Error in {extractor.__class__.__name__}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
    
    # Check if we have any temporal features
    temporal_features = [col for col in result_df.columns if col.startswith('temporal_')]
    if not temporal_features:
        logging.warning("No temporal features were extracted. This may indicate an issue with the dataset.")
    
    # Check if we have any structural features
    structural_features = [col for col in result_df.columns if col.startswith('structural_')]
    if not structural_features:
        logging.warning("No structural features were extracted. This may indicate an issue with the dataset.")
    
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