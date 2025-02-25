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
import xml.etree.ElementTree as ET
from utils.features import (
    BuzzFeedStructuralFeatureExtractor,
    BuzzFeedUserFeatureExtractor,
    BuzzFeedContentFeatureExtractor,
    BuzzFeedTemporalFeatureExtractor
)
from utils.dataset_alignment import save_feature_sets
from utils.thread_capture import ThreadCaptureTool

# Global constants
NUM_PROCESSES = mp.cpu_count()

# Parses a single XML article file and extracts its content, metadata, and social media reactions
def parse_article_xml(file_path: str) -> Dict:
    """Parse a single XML article file and extract relevant data"""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        article_data = {
            'article_id': os.path.splitext(os.path.basename(file_path))[0],
            'title': root.find('title').text if root.find('title') is not None else '',
            'author': root.find('author').text if root.find('author') is not None else '',
            'orientation': root.find('orientation').text if root.find('orientation') is not None else '',
            'veracity': root.find('veracity').text if root.find('veracity') is not None else '',
            'mainText': root.find('mainText').text if root.find('mainText') is not None else '',
            'uri': root.find('uri').text if root.find('uri') is not None else '',
            'tweet_count': 0,  # Initialize tweet-related fields
            'unique_authors': 0,
            'reaction_texts': [],  # Store tweet replies
            'reaction_authors': [],  # Store authors of replies
            'reaction_timestamps': []  # Store timestamps of replies
        }
        
        # Extract hyperlinks
        hyperlinks = root.findall('hyperlink')
        article_data['hyperlink_count'] = len(hyperlinks)
        
        # Extract paragraphs
        paragraphs = root.findall('paragraph')
        article_data['paragraph_count'] = len(paragraphs)
        
        # Extract Twitter thread data
        twitter_threads = root.findall('twitter_thread')
        if twitter_threads:
            for thread in twitter_threads:
                # Extract tweet replies
                replies = thread.findall('reply')
                for reply in replies:
                    if reply.find('text') is not None:
                        article_data['reaction_texts'].append(reply.find('text').text)
                    if reply.find('author') is not None:
                        article_data['reaction_authors'].append(reply.find('author').text)
                    if reply.find('timestamp') is not None:
                        article_data['reaction_timestamps'].append(reply.find('timestamp').text)
            
            article_data['tweet_count'] = len(article_data['reaction_texts'])
            article_data['unique_authors'] = len(set(article_data['reaction_authors']))
        
        return article_data
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

# BuzzFeed dataset processor for fake news detection using article and social media features
def load_buzzfeed_dataset_raw(base_path: str = 'data/buzzfeed', output_dir: str = None, save_csv: bool = False) -> pd.DataFrame:
    """Load and process the raw BuzzFeed dataset from XML files.
    Returns the raw dataset structure as a DataFrame.
    
    Args:
        base_path: Path to the BuzzFeed dataset directory
        output_dir: Directory to save output files (defaults to base_path if None)
        save_csv: Whether to save intermediate CSV files
        
    Returns:
        DataFrame containing raw article data
    """
    # Use base_path as output_dir if not specified
    output_dir = output_dir or base_path
    
    # Define paths
    articles_path = os.path.join(base_path, 'articles')
    
    # Verify paths
    if not os.path.exists(articles_path):
        raise FileNotFoundError(f"Articles directory not found at: {articles_path}")
    
    print("Loading BuzzFeed dataset from XML files...")
    
    # Process XML files in parallel
    articles_data = []
    xml_files = [f for f in os.listdir(articles_path) if f.endswith('.xml')]
    
    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        futures = []
        for xml_file in xml_files:
            file_path = os.path.join(articles_path, xml_file)
            futures.append(executor.submit(parse_article_xml, file_path))
        
        for future in tqdm(futures, desc="Processing articles"):
            article_data = future.result()
            if article_data:
                articles_data.append(article_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(articles_data)
    
    # Save if requested
    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'buzzfeed_raw_dataset.csv')
        df.to_csv(output_path, index=False)
        print(f"Saved raw dataset to: {output_path}")
    
    return df

# Processes raw BuzzFeed data into a structured DataFrame with basic article and reaction features
def load_buzzfeed_dataset_extended(raw_dataset: Union[List[Dict], str, None] = None, base_path: str = 'data/buzzfeed', output_dir: str = None, save_csv: bool = False) -> pd.DataFrame:
    """Load and process the BuzzFeed dataset with basic features but without the feature extractors.
    Can either take a raw dataset list or load from a CSV file.
    
    Args:
        raw_dataset: Either a raw dataset list from load_buzzfeed_dataset_raw or a path to a CSV file
        base_path: Path to the BuzzFeed dataset directory (used if raw_dataset is None)
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
        raw_dataset = load_buzzfeed_dataset_raw(base_path)
    
    # Create DataFrame from raw data
    df = pd.DataFrame(raw_dataset)
    
    # Map veracity values to binary labels
    veracity_map = {
        'mostly true': 'real',
        'mixture of true and false': 'fake',
        'mostly false': 'fake',
        'no factual content': 'fake'
    }
    df['rating'] = df['veracity'].map(veracity_map)
    
    # Save if requested
    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'buzzfeed_extended_dataset.csv')
        df.to_csv(output_path, index=False)
        print(f"Saved extended dataset to: {output_path}")
    
    return df

# Extracts advanced features from BuzzFeed articles using specialized feature extractors
def load_buzzfeed_features_dataset(extended_dataset: Union[pd.DataFrame, str, None] = None, base_path: str = 'data/buzzfeed', output_dir: str = None, save_csv: bool = False, include_additional_features: bool = False) -> pd.DataFrame:
    """Load or create the BuzzFeed features dataset with all feature extractors applied.
    Can either take an extended dataset DataFrame or load from a CSV file.
    
    Args:
        extended_dataset: Either a DataFrame from load_buzzfeed_dataset_extended or a path to a CSV file
        base_path: Path to the BuzzFeed dataset directory (used if extended_dataset is None)
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
        extended_dataset = load_buzzfeed_dataset_extended(base_path=base_path, output_dir=output_dir, save_csv=save_csv)
    
    # Add binary label (1 for fake, 0 for real)
    extended_dataset['label'] = (extended_dataset['rating'] == 'fake').astype(int)
    
    # Extract all features
    df_with_features = extract_all_features(extended_dataset, include_additional_features)
    
    # Get feature columns
    feature_columns = [col for col in df_with_features.columns if any(
        col.startswith(prefix) for prefix in 
        ['structural_', 'user_', 'content_', 'temporal_']
    )]
    
    # Create features DataFrame
    features_df = df_with_features[feature_columns].copy()
    features_df.insert(0, 'source', 'buzzfeed')
    features_df['label'] = df_with_features['label']
    
    # Save if requested
    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        save_feature_sets(features_df, output_dir, 'buzzfeed')
    
    return features_df

# Applies all BuzzFeed-specific feature extractors to generate the complete feature set
def extract_all_features(df: pd.DataFrame, include_additional_features: bool = False) -> pd.DataFrame:
    """Extract all BuzzFeed-specific features from the dataset.
    
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
        BuzzFeedStructuralFeatureExtractor(result_df),
        BuzzFeedUserFeatureExtractor(result_df, include_additional_features),
        BuzzFeedContentFeatureExtractor(result_df, include_additional_features),
        BuzzFeedTemporalFeatureExtractor(result_df, include_additional_features)
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

# Loads BuzzFeed dataset with threaded structure for better alignment with PHEME
def load_buzzfeed_threaded_dataset(extended_dataset: Union[pd.DataFrame, str, None] = None, 
                                 base_path: str = 'data/buzzfeed', 
                                 output_dir: str = None, 
                                 save_csv: bool = False) -> pd.DataFrame:
    """Load BuzzFeed dataset with threaded structure.
    
    This implements the extraction of Twitter threads from BuzzFeed's Facebook dataset:
    1. Extract the 10 most shared stories from left-wing pages
    2. Extract the 10 most shared stories from right-wing pages
    3. Search Twitter for these headlines
    4. Keep the top 3 most retweeted posts for each headline
    5. This results in approximately 35 topics with journalist-provided labels
       (15 "mostly true", 20 "mostly false")
       
    Args:
        extended_dataset: Extended dataset DataFrame or path to CSV file
        base_path: Path to BuzzFeed dataset directory
        output_dir: Directory to save output files (defaults to base_path if None)
        save_csv: Whether to save intermediate CSV files
        
    Returns:
        DataFrame containing the threaded dataset
    """
    # Use base_path as output_dir if not specified
    output_dir = output_dir or base_path
    
    # Load extended dataset if needed
    if extended_dataset is None:
        extended_path = os.path.join(base_path, 'buzzfeed_extended.csv')
        if os.path.exists(extended_path):
            extended_dataset = pd.read_csv(extended_path)
        else:
            extended_dataset = load_buzzfeed_dataset_extended(base_path=base_path, output_dir=output_dir, save_csv=True)
    elif isinstance(extended_dataset, str):
        extended_dataset = pd.read_csv(extended_dataset)
    
    print(f"Starting BuzzFeed thread capture with {len(extended_dataset)} articles")
    
    # Initialize thread capture tool
    thread_capture_tool = ThreadCaptureTool(base_path=os.path.dirname(base_path))
    
    # Process based on political orientation and veracity
    # Get the 10 most shared stories from left-wing pages
    left_wing = extended_dataset[extended_dataset['orientation'] == 'left']
    left_most_shared = left_wing.sort_values('hyperlink_count', ascending=False).head(10)
    
    # Get the 10 most shared stories from right-wing pages
    right_wing = extended_dataset[extended_dataset['orientation'] == 'right']
    right_most_shared = right_wing.sort_values('hyperlink_count', ascending=False).head(10)
    
    # Combine both sets
    most_shared = pd.concat([left_most_shared, right_most_shared], ignore_index=True)
    
    print(f"Selected {len(most_shared)} most shared stories for Twitter search")
    
    # Generate Twitter threads for each headline
    print("Generating Twitter threads for headlines...")
    
    if thread_capture_tool.twitter_api_available:
        all_threads = []
        for _, row in most_shared.iterrows():
            # Convert article data to dict for thread capture
            article_dict = row.to_dict()
            
            # Search for this headline on Twitter and capture threads
            # Keep the top 3 most retweeted posts per headline
            threads = thread_capture_tool._search_twitter_for_headline(article_dict.get('title', ''))
            
            # Sort by retweet count and take top 3
            top_threads = sorted(threads, key=lambda x: x.get('retweet_count', 0), reverse=True)[:3]
            
            for thread in top_threads:
                # Add article metadata to each thread
                thread['article_id'] = article_dict.get('article_id', '')
                thread['title'] = article_dict.get('title', '')
                thread['orientation'] = article_dict.get('orientation', '')
                thread['veracity'] = article_dict.get('veracity', '')
                all_threads.append(thread)
        
        # Convert threads to DataFrame
        threaded_df = thread_capture_tool._flatten_thread_data(all_threads)
    else:
        # If Twitter API is not available, generate mock threads
        print("Twitter API not available. Generating mock threads from existing data...")
        mock_threads = thread_capture_tool._generate_mock_twitter_threads(most_shared)
        threaded_df = thread_capture_tool._flatten_thread_data(mock_threads)
    
    # Add binary labels based on veracity
    threaded_df['label'] = threaded_df['rating'].apply(
        lambda x: 1 if x == 'fake' else 0
    )
    
    # Save if requested
    if save_csv:
        output_path = os.path.join(output_dir, 'buzzfeed_threaded.csv')
        threaded_df.to_csv(output_path, index=False)
        print(f"Saved threaded dataset to: {output_path}")
    
    print(f"BuzzFeed threaded dataset created with {len(threaded_df)} threads")
    return threaded_df

# Extracts features from BuzzFeed threaded dataset for better alignment with PHEME
def load_buzzfeed_threaded_features_dataset(threaded_dataset: Union[pd.DataFrame, str, None] = None, 
                                          base_path: str = 'data/buzzfeed', 
                                          output_dir: str = None, 
                                          save_csv: bool = False,
                                          include_additional_features: bool = False) -> pd.DataFrame:
    """Load or create the features dataset from BuzzFeed threaded data.
    
    Args:
        threaded_dataset: Either a DataFrame from load_buzzfeed_threaded_dataset or a path to a CSV file
        base_path: Path to the BuzzFeed dataset directory
        output_dir: Directory to save output files (defaults to base_path if None)
        save_csv: Whether to save intermediate CSV files
        include_additional_features: Whether to include additional features
        
    Returns:
        DataFrame containing only the extracted features, source, and label
    """
    # Use base_path as output_dir if not specified
    output_dir = output_dir or base_path
    
    # Load threaded dataset if needed
    if threaded_dataset is None:
        if isinstance(threaded_dataset, str) and os.path.exists(threaded_dataset):
            threaded_dataset = pd.read_csv(threaded_dataset)
        else:
            print("Loading threaded dataset first...")
            threaded_dataset = load_buzzfeed_threaded_dataset(base_path=base_path, output_dir=output_dir, save_csv=save_csv)
    
    # Extract all features
    print("Extracting features from threaded dataset...")
    df_with_features = extract_all_features(threaded_dataset, include_additional_features)
    
    # Get feature columns
    feature_columns = [col for col in df_with_features.columns if any(
        col.startswith(prefix) for prefix in 
        ['structural_', 'user_', 'content_', 'temporal_']
    )]
    
    # Create features DataFrame
    features_df = df_with_features[feature_columns].copy()
    features_df.insert(0, 'source', 'buzzfeed_threaded')
    features_df['label'] = df_with_features['label']
    
    # Save if requested
    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        # Use a different name to distinguish from regular features
        output_path = os.path.join(output_dir, 'buzzfeed_threaded_paper_features.csv')
        features_df.to_csv(output_path, index=False)
        print(f"Saved threaded features to: {output_path}")
        
        # Also save with all_features suffix for consistency
        if include_additional_features:
            all_features_path = os.path.join(output_dir, 'buzzfeed_threaded_all_features.csv')
            features_df.to_csv(all_features_path, index=False)
            print(f"Saved all threaded features to: {all_features_path}")
    
    return features_df

# Main execution function to process BuzzFeed dataset and generate analysis reports
def main():
    """Main execution function"""
    warnings.filterwarnings('ignore')
    
    # Set output directory
    output_dir = 'data/buzzfeed'
    os.makedirs(output_dir, exist_ok=True)

    # Load raw dataset
    print("Loading raw dataset...")
    raw_df = load_buzzfeed_dataset_raw(output_dir=output_dir, save_csv=True)
    
    # Load extended dataset
    print("\nCreating extended dataset...")
    extended_df = load_buzzfeed_dataset_extended(raw_df, output_dir=output_dir, save_csv=True)
    
    # Extract features
    print("\nExtracting features...")
    all_features_df = load_buzzfeed_features_dataset(extended_df, output_dir=output_dir, save_csv=True, include_additional_features=True)
    
    # Create threaded dataset
    print("\nCreating threaded dataset...")
    threaded_df = load_buzzfeed_threaded_dataset(extended_df, output_dir=output_dir, save_csv=True)
    
    # Extract features from threaded dataset
    print("\nExtracting features from threaded dataset...")
    threaded_features_df = load_buzzfeed_threaded_features_dataset(threaded_df, output_dir=output_dir, save_csv=True, include_additional_features=True)
    
    print("\nDataset processing complete!")
    print(f"Raw dataset shape: {raw_df.shape}")
    print(f"Extended dataset shape: {extended_df.shape}")
    print(f"Features dataset shape: {all_features_df.shape}")
    print(f"Threaded dataset shape: {threaded_df.shape}")
    print(f"Threaded features dataset shape: {threaded_features_df.shape}")

if __name__ == "__main__":
    main()