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
    
    # Extract all features once
    print("\nExtracting features...")
    all_features_df = load_buzzfeed_features_dataset(extended_df, output_dir=output_dir, save_csv=False, include_additional_features=True)

    # Split into paper and all features using dataset_alignment.save_feature_sets
    paper_features_df, all_features_df = save_feature_sets(all_features_df, output_dir, 'buzzfeed')

    # Generate analysis report
    try:
        analysis_df = prepare_df_for_analysis(paper_features_df)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            sweet_report = sv.analyze(analysis_df)
            output_path = os.path.join(output_dir, 'buzzfeed_paper_features_analysis_report.html')
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
            output_path = os.path.join(output_dir, 'buzzfeed_all_features_analysis_report.html')
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
    main()