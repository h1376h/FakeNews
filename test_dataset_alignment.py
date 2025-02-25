#!/usr/bin/env python3
"""
Test script for dataset alignment code.
This script demonstrates how to use the dataset alignment process
with sample command-line arguments and outputs key statistics.
"""

import argparse
import logging
import sys
from dataset_alignment import (
    extract_twitter_threads_from_buzzfeed,
    align_credbank_labels,
    capture_threaded_structure,
    main as alignment_main
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

def test_full_alignment(base_path='data', output_dir='data/aligned_test', save_csv=True):
    """Run the full alignment process and print statistics."""
    logger.info("Running full dataset alignment test...")
    
    # Run the full alignment process
    aligned_datasets = alignment_main(base_path=base_path, output_dir=output_dir, save_csv=save_csv)
    
    # Print summary statistics for each aligned dataset
    for dataset_name, df in aligned_datasets.items():
        logger.info(f"=== {dataset_name} Summary Statistics ===")
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
        logger.info(f"Number of features: {df.shape[1]}")
        logger.info(f"Top features: {', '.join(df.columns[:10])}")
        logger.info("====================================")
    
    return aligned_datasets

def test_individual_steps(base_path='data', output_dir='data/aligned_test', save_csv=True):
    """Test each individual step of the alignment process."""
    logger.info("Testing individual alignment steps...")
    
    # Step 1: Extract Twitter threads from BuzzFeed Facebook data
    logger.info("\n=== Step 1: BuzzFeed Twitter Thread Extraction ===")
    buzzfeed_threads = extract_twitter_threads_from_buzzfeed(
        base_path=f"{base_path}/buzzfeed",
        output_dir=output_dir,
        save_csv=save_csv
    )
    logger.info(f"Extracted {len(buzzfeed_threads)} Twitter threads from BuzzFeed")
    
    # Step 2: Align CREDBANK labels
    logger.info("\n=== Step 2: CREDBANK Label Alignment ===")
    credbank_labeled = align_credbank_labels(
        base_path=f"{base_path}/credbank",
        output_dir=output_dir,
        save_csv=save_csv,
        quantile_threshold=0.15  # Using the 15% quantile as specified
    )
    logger.info(f"CREDBANK label distribution: {credbank_labeled['label'].value_counts().to_dict()}")
    
    # Step 3: Capture threaded structure
    logger.info("\n=== Step 3: Thread Structure Capture ===")
    threaded_credbank, threaded_buzzfeed = capture_threaded_structure(
        credbank_df=credbank_labeled,
        buzzfeed_df=buzzfeed_threads,
        base_path=base_path,
        output_dir=output_dir,
        save_csv=save_csv
    )
    
    if threaded_credbank is not None:
        logger.info(f"CREDBANK threaded data: {len(threaded_credbank)} threads")
        logger.info(f"CREDBANK thread label distribution: {threaded_credbank['label'].value_counts().to_dict()}")
    
    if threaded_buzzfeed is not None:
        logger.info(f"BuzzFeed threaded data: {len(threaded_buzzfeed)} threads")
        logger.info(f"BuzzFeed thread label distribution: {threaded_buzzfeed['label'].value_counts().to_dict()}")
    
    return {
        "buzzfeed_threads": buzzfeed_threads,
        "credbank_labeled": credbank_labeled,
        "threaded_credbank": threaded_credbank,
        "threaded_buzzfeed": threaded_buzzfeed
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test dataset alignment code")
    
    parser.add_argument("--base-path", type=str, default="data",
                        help="Base directory path for datasets")
    parser.add_argument("--output-dir", type=str, default="data/aligned_test",
                        help="Directory to save the output files")
    parser.add_argument("--no-save", action="store_false", dest="save_csv",
                        help="Don't save intermediate CSVs")
    parser.add_argument("--test-type", type=str, choices=["full", "steps", "both"], default="both",
                        help="Type of test to run (default: both)")
    
    args = parser.parse_args()
    
    # Run the requested tests
    if args.test_type in ["full", "both"]:
        test_full_alignment(base_path=args.base_path, output_dir=args.output_dir, save_csv=args.save_csv)
        
    if args.test_type in ["steps", "both"]:
        test_individual_steps(base_path=args.base_path, output_dir=args.output_dir, save_csv=args.save_csv)
    
    logger.info("Dataset alignment tests completed!") 