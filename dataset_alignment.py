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
    
    # Select most shared stories from each political orientation
    left_wing = buzzfeed_df[buzzfeed_df['orientation'] == 'left'].sort_values(
        by='fb_engagement', ascending=False).head(10)
    right_wing = buzzfeed_df[buzzfeed_df['orientation'] == 'right'].sort_values(
        by='fb_engagement', ascending=False).head(10)
    
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
    buzzfeed_threads_df = pd.DataFrame(threads)
    
    # Log statistics
    true_count = len(buzzfeed_threads_df[buzzfeed_threads_df['label'] == 0])
    false_count = len(buzzfeed_threads_df[buzzfeed_threads_df['label'] == 1])
    logger.info(f"Extracted {len(buzzfeed_threads_df)} Twitter threads from BuzzFeed")
    logger.info(f"Label distribution: {true_count} mostly true, {false_count} mostly false")
    
    # Save to CSV if requested
    if save_csv:
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
    
    # Calculate mean accuracy rating for each event
    credbank_df['mean_accuracy'] = credbank_df['ratings'].apply(
        lambda x: np.mean([int(r) for r in eval(x)]) if isinstance(x, str) else np.nan
    )
    
    # Calculate quantiles
    low_quantile = credbank_df['mean_accuracy'].quantile(quantile_threshold)
    high_quantile = credbank_df['mean_accuracy'].quantile(1 - quantile_threshold)
    
    # For reference, paper values:
    # - Low threshold (15% quantile): 1.467
    # - High threshold (85% quantile): 1.9
    logger.info(f"Calculated quantiles: Low ({quantile_threshold*100}%): {low_quantile}, "
                f"High ({(1-quantile_threshold)*100}%): {high_quantile}")
    
    # Apply binary labeling using the convert_credbank_scale function
    credbank_df['label_info'] = credbank_df['ratings'].apply(
        lambda x: convert_credbank_scale(eval(x) if isinstance(x, str) else [])
    )
    
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
    if save_csv:
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
    Capture Twitter's threaded structure for CREDBANK and BuzzFeed datasets.
    
    As described in the paper:
    - For CREDBANK: Identify the most retweeted tweet as the thread root
    - For BuzzFeed: Use popular headline tweets as thread roots
    - Capture replies to construct thread structure similar to PHEME
    - Discard CREDBANK threads with no reactions
    
    Args:
        credbank_df: CREDBANK DataFrame with labels
        buzzfeed_df: BuzzFeed DataFrame with Twitter thread data
        base_path: Base path for all datasets
        output_dir: Output directory for saving results
        save_csv: Whether to save intermediate CSV files
        
    Returns:
        Tuple of (threaded_credbank_df, threaded_buzzfeed_df)
    """
    logger.info("Capturing Twitter's threaded structure...")
    output_dir = output_dir or os.path.join(base_path, 'aligned')
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize thread capture tool
    thread_capture = ThreadCaptureTool(base_path=base_path)
    
    # Process CREDBANK dataset
    threaded_credbank_df = None
    if credbank_df is not None and not credbank_df.empty:
        logger.info("Processing CREDBANK threads...")
        
        # Use the thread capture tool to identify most retweeted tweets and build threads
        threaded_credbank_df = thread_capture.capture_credbank_threads(credbank_df)
        
        # Filter out threads with no reactions
        reaction_count = threaded_credbank_df['num_reactions'].fillna(0)
        threaded_credbank_df = threaded_credbank_df[reaction_count > 0].copy()
        
        # Log statistics
        logger.info(f"CREDBANK threads after filtering: {len(threaded_credbank_df)}")
        positive_count = len(threaded_credbank_df[threaded_credbank_df['label'] == 0])
        negative_count = len(threaded_credbank_df[threaded_credbank_df['label'] == 1])
        logger.info(f"Label distribution: {positive_count} positive, {negative_count} negative")
    
    # Process BuzzFeed dataset
    threaded_buzzfeed_df = None
    if buzzfeed_df is not None and not buzzfeed_df.empty:
        logger.info("Processing BuzzFeed threads...")
        
        # Use the thread capture tool to build threads from headline tweets
        threaded_buzzfeed_df = thread_capture.capture_buzzfeed_threads(buzzfeed_df)
        
        # Log statistics
        logger.info(f"BuzzFeed threads: {len(threaded_buzzfeed_df)}")
        true_count = len(threaded_buzzfeed_df[threaded_buzzfeed_df['label'] == 0])
        false_count = len(threaded_buzzfeed_df[threaded_buzzfeed_df['label'] == 1])
        logger.info(f"Label distribution: {true_count} mostly true, {false_count} mostly false")
    
    # Save results if requested
    if save_csv and (threaded_credbank_df is not None or threaded_buzzfeed_df is not None):
        thread_capture.save_threaded_datasets(
            credbank_threads=threaded_credbank_df,
            buzzfeed_threads=threaded_buzzfeed_df,
            output_dir=output_dir
        )
    
    return threaded_credbank_df, threaded_buzzfeed_df

def main(
    base_path: str = 'data',
    output_dir: str = None,
    save_csv: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Run the full dataset alignment process.
    
    Args:
        base_path: Base directory containing all datasets
        output_dir: Output directory (defaults to base_path/aligned if None)
        save_csv: Whether to save intermediate CSV files
        
    Returns:
        Dictionary of aligned datasets
    """
    logger.info("Starting dataset alignment process...")
    
    # Set default output directory
    output_dir = output_dir or os.path.join(base_path, 'aligned')
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Extract Twitter threads from BuzzFeed Facebook data
    buzzfeed_threads = extract_twitter_threads_from_buzzfeed(
        base_path=os.path.join(base_path, 'buzzfeed'),
        output_dir=output_dir,
        save_csv=save_csv
    )
    
    # Step 2: Align CREDBANK labels
    credbank_labeled = align_credbank_labels(
        base_path=os.path.join(base_path, 'credbank'),
        output_dir=output_dir,
        save_csv=save_csv,
        quantile_threshold=0.15  # Using the 15% quantile as specified in the paper
    )
    
    # Step 3: Capture threaded structure for CREDBANK and BuzzFeed
    threaded_credbank, threaded_buzzfeed = capture_threaded_structure(
        credbank_df=credbank_labeled,
        buzzfeed_df=buzzfeed_threads,
        base_path=base_path,
        output_dir=output_dir,
        save_csv=save_csv
    )
    
    # Load PHEME dataset
    try:
        from dataset_pheme import load_pheme_features_dataset
        pheme_df = load_pheme_features_dataset(
            base_path=os.path.join(base_path, 'pheme'),
            output_dir=output_dir,
            save_csv=save_csv
        )
        logger.info(f"Loaded PHEME dataset: {len(pheme_df)} threads")
    except ImportError:
        logger.error("Could not import PHEME dataset module. Make sure dataset_pheme.py is available.")
        pheme_df = pd.DataFrame()
    
    # Load feature datasets for the threaded versions
    if threaded_credbank is not None and not threaded_credbank.empty:
        try:
            from dataset_credbank import load_credbank_threaded_features_dataset
            credbank_features = load_credbank_threaded_features_dataset(
                threaded_dataset=threaded_credbank,
                base_path=os.path.join(base_path, 'credbank'),
                output_dir=output_dir,
                save_csv=save_csv
            )
            logger.info(f"Generated CREDBANK features: {len(credbank_features)} threads")
        except ImportError:
            logger.error("Could not generate CREDBANK features.")
            credbank_features = pd.DataFrame()
    else:
        credbank_features = pd.DataFrame()
    
    if threaded_buzzfeed is not None and not threaded_buzzfeed.empty:
        try:
            from dataset_buzzfeed import load_buzzfeed_threaded_features_dataset
            buzzfeed_features = load_buzzfeed_threaded_features_dataset(
                threaded_dataset=threaded_buzzfeed,
                base_path=os.path.join(base_path, 'buzzfeed'),
                output_dir=output_dir,
                save_csv=save_csv
            )
            logger.info(f"Generated BuzzFeed features: {len(buzzfeed_features)} threads")
        except ImportError:
            logger.error("Could not generate BuzzFeed features.")
            buzzfeed_features = pd.DataFrame()
    else:
        buzzfeed_features = pd.DataFrame()
    
    # Align features across all datasets
    aligned_datasets = {}
    
    if not pheme_df.empty:
        aligned_datasets['pheme'] = pheme_df
    
    if not credbank_features.empty:
        aligned_datasets['credbank'] = credbank_features
        
    if not buzzfeed_features.empty:
        aligned_datasets['buzzfeed'] = buzzfeed_features
    
    # Final alignment of all datasets
    if aligned_datasets:
        from utils.dataset_alignment import align_datasets
        aligned_result = align_datasets(
            pheme_df=pheme_df if not pheme_df.empty else None,
            buzzfeed_df=buzzfeed_features if not buzzfeed_features.empty else None,
            credbank_df=credbank_features if not credbank_features.empty else None,
            output_dir=output_dir,
            save_csv=save_csv
        )
        logger.info("Dataset alignment complete!")
    else:
        logger.warning("No datasets were successfully loaded and aligned.")
        aligned_result = {}
    
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