#!/usr/bin/env python3
"""
Dataset Alignment Implementation

This script implements the dataset alignment process described in the paper:
1. Extracting Twitter threads from BuzzFeed's Facebook dataset
2. Aligning labels across datasets (particularly CREDBANK's Likert scale conversion)
3. Capturing Twitter's threaded structure for all datasets

The alignment process ensures consistent feature sets and labels across the
BuzzFeed, CREDBANK, and PHEME datasets.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union
import logging
import sys
from tqdm import tqdm
from utils.thread_capture import ThreadCaptureTool
from utils.dataset_alignment import (
    align_buzzfeed_threads,
    align_pheme_threads,
    align_credbank_threads,
    convert_credbank_scale,
    calculate_disagreement_score,
    save_feature_sets
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def extract_twitter_threads_from_buzzfeed(
    base_path: str = 'data/buzzfeed',
    output_dir: str = None,
    save_csv: bool = False
) -> pd.DataFrame:
    """
    Extract Twitter threads from BuzzFeed's Facebook dataset.
    
    As described in the paper:
    1. Extract the 10 most shared stories from left-wing pages
    2. Extract the 10 most shared stories from right-wing pages
    3. Search Twitter for these headlines
    4. Keep the top 3 most retweeted posts for each headline
    5. Results in ~35 topics with journalist-provided labels (15 "mostly true", 20 "mostly false")
    
    Args:
        base_path: Path to BuzzFeed dataset directory
        output_dir: Output directory for saving results (defaults to base_path if None)
        save_csv: Whether to save intermediate CSV files
        
    Returns:
        DataFrame containing extracted Twitter threads with labels
    """
    logger.info("Extracting Twitter threads from BuzzFeed Facebook dataset...")
    output_dir = output_dir or base_path
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize thread capture tool for Twitter API access
    thread_capture = ThreadCaptureTool(base_path=os.path.dirname(base_path))
    
    # Load BuzzFeed dataset
    try:
        from dataset_buzzfeed import load_buzzfeed_dataset_raw
        buzzfeed_df = load_buzzfeed_dataset_raw(base_path=base_path, output_dir=output_dir, save_csv=save_csv)
    except ImportError:
        logger.error("Could not import BuzzFeed dataset module. Make sure dataset_buzzfeed.py is available.")
        return pd.DataFrame()
    
    # Check if the dataset is empty
    if buzzfeed_df.empty:
        logger.error("BuzzFeed dataset is empty")
        return pd.DataFrame()
    
    # Log available columns for debugging
    logger.info(f"Available columns in BuzzFeed dataset: {', '.join(buzzfeed_df.columns)}")
    
    # Use tweet_count and hyperlink_count as engagement metrics
    # If tweet_count is available, use it as the primary sorting metric
    # Otherwise, fall back to hyperlink_count or other metrics
    if 'tweet_count' in buzzfeed_df.columns:
        engagement_metric = 'tweet_count'
    elif 'hyperlink_count' in buzzfeed_df.columns:
        engagement_metric = 'hyperlink_count'
    else:
        # If neither is available, create a composite engagement score
        logger.warning("No direct engagement metric found. Creating a composite score.")
        # Use a safe approach - check if columns exist before using them
        composite_columns = []
        if 'paragraph_count' in buzzfeed_df.columns:
            composite_columns.append('paragraph_count')
        if 'share_count' in buzzfeed_df.columns:
            composite_columns.append('share_count')
        if 'reaction_count' in buzzfeed_df.columns:
            composite_columns.append('reaction_count')
            
        # If we have no usable columns, add a dummy column
        if not composite_columns:
            logger.warning("No engagement metrics found. Using row index as fallback.")
            buzzfeed_df['engagement_score'] = buzzfeed_df.index
        else:
            # Sum all available metrics
            buzzfeed_df['engagement_score'] = 0
            for col in composite_columns:
                buzzfeed_df['engagement_score'] += buzzfeed_df[col].fillna(0)
                
        engagement_metric = 'engagement_score'
    
    logger.info(f"Using '{engagement_metric}' as engagement metric for sorting")
    
    # Select most shared stories from each political orientation
    left_wing = buzzfeed_df[buzzfeed_df['orientation'] == 'left'].sort_values(
        by=engagement_metric, ascending=False).head(10)
    right_wing = buzzfeed_df[buzzfeed_df['orientation'] == 'right'].sort_values(
        by=engagement_metric, ascending=False).head(10)
    
    # Combine into one balanced dataset
    selected_stories = pd.concat([left_wing, right_wing])
    
    # Extract Twitter threads for each headline
    threads = []
    for _, article in tqdm(selected_stories.iterrows(), total=len(selected_stories), 
                          desc="Searching Twitter for headlines"):
        article_data = article.to_dict()
        article_threads = align_buzzfeed_threads(article_data, thread_capture_tool=thread_capture)
        threads.extend(article_threads)
    
    # Convert to DataFrame
    buzzfeed_threads_df = pd.DataFrame(threads) if threads else pd.DataFrame()
    
    # Log statistics
    if not buzzfeed_threads_df.empty and 'label' in buzzfeed_threads_df.columns:
        true_count = len(buzzfeed_threads_df[buzzfeed_threads_df['label'] == 0])
        false_count = len(buzzfeed_threads_df[buzzfeed_threads_df['label'] == 1])
        logger.info(f"Extracted {len(buzzfeed_threads_df)} Twitter threads from BuzzFeed")
        logger.info(f"Label distribution: {true_count} mostly true, {false_count} mostly false")
    else:
        logger.warning("No Twitter threads extracted or label column not found")
    
    # Save to CSV if requested
    if save_csv and not buzzfeed_threads_df.empty:
        output_path = os.path.join(output_dir, "buzzfeed_twitter_threads.csv")
        buzzfeed_threads_df.to_csv(output_path, index=False)
        logger.info(f"Saved BuzzFeed Twitter threads to {output_path}")
    
    return buzzfeed_threads_df

def align_credbank_labels(
    base_path: str = 'data/credbank',
    output_dir: str = None,
    save_csv: bool = False,
    quantile_threshold: float = 0.15
) -> pd.DataFrame:
    """
    Align CREDBANK labels by converting Likert scale ratings to binary labels.
    
    As described in the paper:
    - The grand mean of CREDBANK's accuracy assessments is 1.7
    - The median is 1.767
    - The 25th and 75th quartiles are 1.6 and 1.867 respectively
    - Events below the bottom quantile_threshold% (mean rating < 1.467) are negative samples
    - Events above the top quantile_threshold% (mean rating > 1.9) are positive samples
    - Events between these thresholds are removed from the dataset
    
    Args:
        base_path: Path to CREDBANK dataset directory
        output_dir: Output directory for saving results (defaults to base_path if None)
        save_csv: Whether to save intermediate CSV files
        quantile_threshold: Quantile threshold for label assignment (default: 0.15 for 15%)
        
    Returns:
        DataFrame containing CREDBANK events with binary labels
    """
    logger.info(f"Aligning CREDBANK labels using {quantile_threshold*100}% quantile thresholds...")
    output_dir = output_dir or base_path
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CREDBANK dataset
    try:
        from dataset_credbank import load_credbank_dataset_raw
        credbank_df = load_credbank_dataset_raw(base_path=base_path, output_dir=output_dir, save_csv=save_csv)
    except ImportError:
        logger.error("Could not import CREDBANK dataset module. Make sure dataset_credbank.py is available.")
        return pd.DataFrame()
    
    # Check if the dataset is empty
    if credbank_df.empty:
        logger.error("CREDBANK dataset is empty")
        return pd.DataFrame()
    
    # Log available columns for debugging
    logger.info(f"Available columns in CREDBANK dataset: {', '.join(credbank_df.columns)}")
    
    # Identify the ratings column
    ratings_column = None
    possible_ratings_columns = ['ratings', 'accuracy_ratings', 'turkRatings', 'cred_ratings', 'Cred_Ratings']
    for col in possible_ratings_columns:
        if col in credbank_df.columns:
            ratings_column = col
            break
    
    if ratings_column is None:
        logger.error("No ratings column found in CREDBANK dataset")
        return pd.DataFrame()
    
    logger.info(f"Using '{ratings_column}' as ratings column")
    
    # Calculate mean accuracy rating for each event
    try:
        credbank_df['mean_accuracy'] = credbank_df[ratings_column].apply(
            lambda x: np.mean([int(r) for r in eval(x)]) if isinstance(x, str) else 
            np.mean(x) if isinstance(x, list) else np.nan
        )
    except Exception as e:
        logger.error(f"Error calculating mean accuracy: {str(e)}")
        # Try an alternative approach
        try:
            credbank_df['mean_accuracy'] = credbank_df[ratings_column].apply(
                lambda x: np.mean([int(r) for r in x.strip('[]').split(',')]) 
                if isinstance(x, str) and '[' in x else np.nan
            )
        except Exception as e2:
            logger.error(f"Alternative approach also failed: {str(e2)}")
            return pd.DataFrame()
    
    # Drop NaN values
    credbank_df = credbank_df.dropna(subset=['mean_accuracy'])
    
    if len(credbank_df) == 0:
        logger.error("No valid mean accuracy ratings calculated")
        return pd.DataFrame()
    
    # Calculate quantiles
    low_quantile = credbank_df['mean_accuracy'].quantile(quantile_threshold)
    high_quantile = credbank_df['mean_accuracy'].quantile(1 - quantile_threshold)
    
    # For reference, paper values:
    # - Low threshold (15% quantile): 1.467
    # - High threshold (85% quantile): 1.9
    logger.info(f"Calculated quantiles: Low ({quantile_threshold*100}%): {low_quantile}, "
                f"High ({(1-quantile_threshold)*100}%): {high_quantile}")
    
    # Apply binary labeling using the convert_credbank_scale function
    try:
        credbank_df['label_info'] = credbank_df[ratings_column].apply(
            lambda x: convert_credbank_scale(eval(x) if isinstance(x, str) else 
                                           x if isinstance(x, list) else [])
        )
    except Exception as e:
        logger.error(f"Error converting ratings scale: {str(e)}")
        # Directly apply the quantile thresholds
        credbank_df['label'] = np.nan
        credbank_df.loc[credbank_df['mean_accuracy'] <= low_quantile, 'label'] = 1  # negative samples
        credbank_df.loc[credbank_df['mean_accuracy'] >= high_quantile, 'label'] = 0  # positive samples
        credbank_df['is_valid'] = ~credbank_df['label'].isna()
    else:
        # Extract the components
        credbank_df['label'] = credbank_df['label_info'].apply(lambda x: x[0])
        credbank_df['confidence'] = credbank_df['label_info'].apply(lambda x: x[1])
        credbank_df['is_valid'] = credbank_df['label_info'].apply(lambda x: x[2])
    
    # Filter out invalid entries (those falling in the middle range)
    credbank_df_labeled = credbank_df[credbank_df['is_valid']].copy()
    
    # Log statistics
    original_count = len(credbank_df)
    labeled_count = len(credbank_df_labeled)
    positive_count = len(credbank_df_labeled[credbank_df_labeled['label'] == 0])
    negative_count = len(credbank_df_labeled[credbank_df_labeled['label'] == 1])
    
    logger.info(f"Original CREDBANK dataset: {original_count} events")
    logger.info(f"After quantile filtering: {labeled_count} events ({labeled_count/original_count*100:.2f}%)")
    logger.info(f"Label distribution: {positive_count} positive (accurate), {negative_count} negative (inaccurate)")
    
    # Save to CSV if requested
    if save_csv and not credbank_df_labeled.empty:
        output_path = os.path.join(output_dir, "credbank_labeled.csv")
        credbank_df_labeled.to_csv(output_path, index=False)
        logger.info(f"Saved labeled CREDBANK dataset to {output_path}")
    
    return credbank_df_labeled

def capture_threaded_structure(
    credbank_df: pd.DataFrame = None,
    buzzfeed_df: pd.DataFrame = None,
    base_path: str = 'data',
    output_dir: str = None,
    save_csv: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Capture the threaded structure of Twitter conversations for CREDBANK and BuzzFeed.
    
    Args:
        credbank_df: CREDBANK DataFrame with labels
        buzzfeed_df: BuzzFeed DataFrame with Twitter threads
        base_path: Base path for datasets
        output_dir: Output directory for saving results (defaults to base_path if None)
        save_csv: Whether to save intermediate CSV files
        
    Returns:
        Tuple (threaded_credbank_df, threaded_buzzfeed_df)
    """
    output_dir = output_dir or base_path
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize empty DataFrames for results
    threaded_credbank_df = pd.DataFrame()
    threaded_buzzfeed_df = pd.DataFrame()
    
    # Initialize thread capture tool
    try:
        from utils.thread_capture import ThreadCaptureTool
        thread_tool = ThreadCaptureTool(base_path=base_path)
        logger.info("Successfully initialized ThreadCaptureTool")
    except Exception as e:
        logger.error(f"Error initializing ThreadCaptureTool: {str(e)}")
        logger.warning("Using fallback implementation without Twitter API access")
        # Create a minimal implementation for testing
        class FallbackThreadCaptureTool:
            def __init__(self, base_path=None):
                self.base_path = base_path
                
            def capture_credbank_threads(self, credbank_df=None):
                if credbank_df is None or credbank_df.empty:
                    return pd.DataFrame()
                # Basic thread structure with just the source tweet
                thread_data = []
                for _, row in credbank_df.iterrows():
                    if 'text' in row:
                        thread_data.append({
                            'id': row.get('topic_key', f"topic_{_}"),
                            'thread_root': row.get('text', ''),
                            'thread_depth': 0,
                            'thread_size': 1,
                            'label': row.get('label', 0)
                        })
                return pd.DataFrame(thread_data) if thread_data else pd.DataFrame()
                
            def capture_buzzfeed_threads(self, buzzfeed_df=None):
                if buzzfeed_df is None or buzzfeed_df.empty:
                    return pd.DataFrame()
                # Basic thread structure with just the source tweet
                thread_data = []
                for idx, row in buzzfeed_df.iterrows():
                    if 'title' in row:
                        thread_data.append({
                            'id': row.get('article_id', f"article_{idx}"),
                            'thread_root': row.get('title', ''),
                            'thread_depth': 0,
                            'thread_size': 1,
                            'label': row.get('label', 0),
                            'source': 'buzzfeed'  # Add source field to ensure proper alignment
                        })
                return pd.DataFrame(thread_data) if thread_data else pd.DataFrame({'source': ['buzzfeed'], 'label': [0]})
                
            def save_threaded_datasets(self, credbank_threads=None, buzzfeed_threads=None, output_dir=None):
                return None, None
        
        thread_tool = FallbackThreadCaptureTool(base_path=base_path)
    
    # Process CREDBANK threads if available
    if credbank_df is not None and not credbank_df.empty:
        logger.info("Processing CREDBANK threads...")
        try:
            threaded_credbank_df = thread_tool.capture_credbank_threads(credbank_df)
            if threaded_credbank_df.empty:
                logger.warning("ThreadCaptureTool returned empty CREDBANK threads")
            else:
                logger.info(f"Generated {len(threaded_credbank_df)} CREDBANK threads")
        except Exception as e:
            logger.error(f"Error capturing CREDBANK threads: {str(e)}")
    
    # Process BuzzFeed threads if available
    if buzzfeed_df is not None and not buzzfeed_df.empty:
        logger.info("Processing BuzzFeed threads...")
        try:
            threaded_buzzfeed_df = thread_tool.capture_buzzfeed_threads(buzzfeed_df)
            if threaded_buzzfeed_df.empty:
                logger.warning("ThreadCaptureTool returned empty BuzzFeed threads")
            else:
                logger.info(f"Generated {len(threaded_buzzfeed_df)} BuzzFeed threads")
        except Exception as e:
            logger.error(f"Error capturing BuzzFeed threads: {str(e)}")
    
    # Ensure we're returning DataFrames even if processing failed
    threaded_credbank_df = threaded_credbank_df if threaded_credbank_df is not None else pd.DataFrame()
    threaded_buzzfeed_df = threaded_buzzfeed_df if threaded_buzzfeed_df is not None else pd.DataFrame()
    
    # Save results if requested
    if save_csv and (not threaded_credbank_df.empty or not threaded_buzzfeed_df.empty):
        try:
            # Check if the thread_capture tool has a save method
            if hasattr(thread_tool, 'save_threaded_datasets'):
                thread_tool.save_threaded_datasets(
                    credbank_threads=threaded_credbank_df if not threaded_credbank_df.empty else None,
                    buzzfeed_threads=threaded_buzzfeed_df if not threaded_buzzfeed_df.empty else None,
                    output_dir=output_dir
                )
            else:
                # Manual save if method doesn't exist
                if not threaded_credbank_df.empty:
                    threaded_credbank_df.to_csv(os.path.join(output_dir, "credbank_threaded.csv"), index=False)
                    logger.info(f"Saved threaded CREDBANK dataset to {os.path.join(output_dir, 'credbank_threaded.csv')}")
                
                if not threaded_buzzfeed_df.empty:
                    threaded_buzzfeed_df.to_csv(os.path.join(output_dir, "buzzfeed_threaded.csv"), index=False)
                    logger.info(f"Saved threaded BuzzFeed dataset to {os.path.join(output_dir, 'buzzfeed_threaded.csv')}")
        except Exception as e:
            logger.error(f"Error saving threaded datasets: {str(e)}")
    
    return threaded_credbank_df, threaded_buzzfeed_df

def main(
    base_path: str = 'data',
    output_dir: str = None,
    save_csv: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Run the full dataset alignment process.
    
    Args:
        base_path: Base path for all datasets
        output_dir: Output directory for saving results (defaults to base_path/aligned if None)
        save_csv: Whether to save intermediate CSV files
        
    Returns:
        Dictionary mapping dataset names to their aligned DataFrames
    """
    # Set default output directory to base_path/aligned if not specified
    output_dir = output_dir or os.path.join(base_path, 'aligned')
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Starting dataset alignment process...")
    
    # Initialize all DataFrame variables
    buzzfeed_df = pd.DataFrame()
    credbank_df = pd.DataFrame()
    pheme_threads_df = pd.DataFrame()
    credbank_threads_df = pd.DataFrame()
    buzzfeed_threads_df = pd.DataFrame()
    credbank_features = pd.DataFrame()
    buzzfeed_features = pd.DataFrame()
    aligned_result = {}
    
    # Step 1: Extract Twitter threads from BuzzFeed Facebook dataset
    logger.info("Extracting Twitter threads from BuzzFeed Facebook dataset...")
    try:
        buzzfeed_df = extract_twitter_threads_from_buzzfeed(
            base_path=os.path.join(base_path, 'buzzfeed'),
            output_dir=output_dir,
            save_csv=save_csv
        )
    except Exception as e:
        logger.error(f"Error extracting BuzzFeed Twitter threads: {str(e)}")
    
    # Step 2: Align CREDBANK labels
    try:
        credbank_df = align_credbank_labels(
            base_path=os.path.join(base_path, 'credbank'),
            output_dir=output_dir,
            save_csv=save_csv
        )
    except Exception as e:
        logger.error(f"Error aligning CREDBANK labels: {str(e)}")
    
    # Step 3: Capture threaded structure for all datasets
    logger.info("Capturing Twitter's threaded structure...")
    
    # Explicit path to PHEME dataset
    pheme_base_path = os.path.join(base_path, 'pheme')
    
    # Check if pheme directory exists and has the expected structure
    if os.path.exists(pheme_base_path):
        # Try to load from pre-computed CSV files first
        pheme_csv_path = os.path.join(pheme_base_path, 'pheme_raw_dataset.csv')
        if os.path.exists(pheme_csv_path):
            logger.info(f"Loading pre-computed PHEME dataset from {pheme_csv_path}")
            pheme_threads_df = pd.read_csv(pheme_csv_path)
        else:
            try:
                from dataset_pheme import load_pheme_features_dataset
                pheme_threads_df = load_pheme_features_dataset(
                    base_path=os.path.join(pheme_base_path, 'pheme-rnr-dataset'),
                    output_dir=output_dir,
                    save_csv=save_csv
                )
            except Exception as e:
                logger.error(f"Error loading PHEME dataset: {str(e)}")
                logger.info("Continuing without PHEME dataset")
    
    try:
        credbank_threads_df, buzzfeed_threads_df = capture_threaded_structure(
            credbank_df=credbank_df if not credbank_df.empty else None,
            buzzfeed_df=buzzfeed_df if not buzzfeed_df.empty else None,
            base_path=base_path,
            output_dir=output_dir,
            save_csv=save_csv
        )
    except Exception as e:
        logger.error(f"Error capturing threaded structure: {str(e)}")
    
    # Load feature datasets for the threaded versions
    if not credbank_threads_df.empty:
        try:
            from dataset_credbank import load_credbank_threaded_features_dataset
            credbank_features = load_credbank_threaded_features_dataset(
                threaded_dataset=credbank_threads_df,
                base_path=os.path.join(base_path, 'credbank'),
                output_dir=output_dir,
                save_csv=save_csv
            )
            if credbank_features is None or credbank_features.empty:
                logger.warning("CREDBANK features dataset is empty")
                credbank_features = pd.DataFrame()
            else:
                logger.info(f"Generated CREDBANK features: {len(credbank_features)} threads")
        except Exception as e:
            logger.error(f"Error generating CREDBANK features: {str(e)}")
            credbank_features = pd.DataFrame()
    
    if not buzzfeed_threads_df.empty:
        try:
            from dataset_buzzfeed import load_buzzfeed_threaded_features_dataset
            buzzfeed_features = load_buzzfeed_threaded_features_dataset(
                threaded_dataset=buzzfeed_threads_df,
                base_path=os.path.join(base_path, 'buzzfeed'),
                output_dir=output_dir,
                save_csv=save_csv
            )
            if buzzfeed_features is None or buzzfeed_features.empty:
                logger.warning("BuzzFeed features dataset is empty")
                buzzfeed_features = pd.DataFrame()
            else:
                logger.info(f"Generated BuzzFeed features: {len(buzzfeed_features)} threads")
        except Exception as e:
            logger.error(f"Error generating BuzzFeed features: {str(e)}")
            buzzfeed_features = pd.DataFrame()
    
    # Align features across all datasets
    aligned_datasets = {}
    
    if not pheme_threads_df.empty:
        aligned_datasets['pheme'] = pheme_threads_df
    
    if not credbank_features.empty:
        aligned_datasets['credbank'] = credbank_features
        
    if not buzzfeed_features.empty:
        aligned_datasets['buzzfeed'] = buzzfeed_features
    
    # Final alignment of all datasets
    if aligned_datasets:
        try:
            from utils.dataset_alignment import align_datasets
            aligned_result = align_datasets(
                pheme_df=pheme_threads_df if not pheme_threads_df.empty else None,
                buzzfeed_df=buzzfeed_features if not buzzfeed_features.empty else None,
                credbank_df=credbank_features if not credbank_features.empty else None,
                output_dir=output_dir,
                save_csv=save_csv
            )
            logger.info("Dataset alignment complete!")
        except Exception as e:
            logger.error(f"Error in final dataset alignment: {str(e)}")
            # Provide a fallback result if alignment fails
            aligned_result = aligned_datasets
    else:
        logger.warning("No datasets were successfully loaded and aligned.")
        aligned_result = {}
    
    # Summarize the results
    logger.info("\n=== Dataset Alignment Summary ===")
    logger.info(f"BuzzFeed Twitter threads: {len(buzzfeed_threads_df)}")
    logger.info(f"CREDBANK labeled events: {len(credbank_df)}")
    logger.info(f"CREDBANK threaded events: {len(credbank_threads_df)}")
    logger.info(f"BuzzFeed threaded events: {len(buzzfeed_threads_df)}")
    logger.info(f"PHEME dataset size: {len(pheme_threads_df)}")
    logger.info(f"Aligned datasets: {list(aligned_result.keys())}")
    logger.info("===============================")
    
    return aligned_result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Align datasets across BuzzFeed, CREDBANK, and PHEME")
    
    parser.add_argument("--base-path", type=str, default="data",
                        help="Base directory path for datasets")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save the output files")
    parser.add_argument("--no-save", action="store_false", dest="save_csv",
                        help="Don't save intermediate CSVs")
    
    args = parser.parse_args()
    
    main(base_path=args.base_path, output_dir=args.output_dir, save_csv=args.save_csv) 