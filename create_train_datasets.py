import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union
import warnings

def load_and_combine_datasets(pheme_path: str = 'data/pheme', 
                            credbank_path: str = 'data/credbank',
                            buzzfeed_path: str = 'data/buzzfeed',
                            output_dir: str = 'data/train',
                            save_csv: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and combine PHEME, CREDBANK, and BuzzFeed feature datasets.
    
    Args:
        pheme_path: Path to PHEME dataset directory
        credbank_path: Path to CREDBANK dataset directory
        buzzfeed_path: Path to BuzzFeed dataset directory
        output_dir: Directory to save output files
        save_csv: Whether to save CSV files
        
    Returns:
        Tuple of (paper_features_df, all_features_df)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load PHEME features
    print("Loading PHEME features...")
    pheme_paper = pd.read_csv(os.path.join(pheme_path, 'pheme_paper_features.csv'))
    pheme_all = pd.read_csv(os.path.join(pheme_path, 'pheme_all_features.csv'))
    
    # Load CREDBANK features
    print("\nLoading CREDBANK features...")
    credbank_paper = pd.read_csv(os.path.join(credbank_path, 'credbank_paper_features.csv'))
    credbank_all = pd.read_csv(os.path.join(credbank_path, 'credbank_all_features.csv'))
    
    # Load BuzzFeed features
    print("\nLoading BuzzFeed features...")
    buzzfeed_paper = pd.read_csv(os.path.join(buzzfeed_path, 'buzzfeed_paper_features.csv'))
    buzzfeed_all = pd.read_csv(os.path.join(buzzfeed_path, 'buzzfeed_all_features.csv'))
    
    # Combine datasets
    print("\nCombining datasets...")
    paper_features_df = pd.concat([pheme_paper, credbank_paper, buzzfeed_paper], axis=0, ignore_index=True)
    all_features_df = pd.concat([pheme_all, credbank_all, buzzfeed_all], axis=0, ignore_index=True)
    
    # Save if requested
    if save_csv:
        # Save paper features
        paper_path = os.path.join(output_dir, 'train_paper.csv')
        paper_features_df.to_csv(paper_path, index=False)
        print(f"Saved paper features to: {paper_path}")
        
        # Save all features
        complete_path = os.path.join(output_dir, 'train_complete.csv')
        all_features_df.to_csv(complete_path, index=False)
        print(f"Saved complete features to: {complete_path}")
    
    return paper_features_df, all_features_df

def load_and_combine_threaded_datasets(pheme_path: str = 'data/pheme', 
                                     credbank_path: str = 'data/credbank',
                                     buzzfeed_path: str = 'data/buzzfeed',
                                     output_dir: str = 'data/train',
                                     save_csv: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and combine PHEME, threaded CREDBANK, and threaded BuzzFeed feature datasets.
    
    This uses the thread-captured versions of CREDBANK and BuzzFeed datasets.
    
    For BuzzFeed, the Facebook data is aligned with Twitter through the following process:
    1. Extracting the 10 most shared stories from left-wing pages
    2. Extracting the 10 most shared stories from right-wing pages  
    3. Searching Twitter for these headlines
    4. Keeping the top 3 most retweeted posts for each headline
    5. This results in 35 topics with journalist-provided labels (15 "mostly true", 20 "mostly false")
    
    For CREDBANK, thread structure is created by:
    1. Identifying the most retweeted tweet in each event as the thread root
    2. Collecting replies to this root tweet as children
    3. Discarding threads with no reactions
    
    Args:
        pheme_path: Path to PHEME dataset directory
        credbank_path: Path to CREDBANK dataset directory
        buzzfeed_path: Path to BuzzFeed dataset directory
        output_dir: Directory to save output files
        save_csv: Whether to save CSV files
        
    Returns:
        Tuple of (paper_features_df, all_features_df)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load PHEME features (same as before since PHEME already has thread structure)
    print("Loading PHEME features...")
    pheme_paper = pd.read_csv(os.path.join(pheme_path, 'pheme_paper_features.csv'))
    pheme_all = pd.read_csv(os.path.join(pheme_path, 'pheme_all_features.csv'))
    
    # Load threaded CREDBANK features
    print("\nLoading threaded CREDBANK features...")
    try:
        credbank_paper = pd.read_csv(os.path.join(credbank_path, 'credbank_threaded_paper_features.csv'))
        credbank_all = pd.read_csv(os.path.join(credbank_path, 'credbank_threaded_all_features.csv'))
    except FileNotFoundError as e:
        print(f"Warning: {str(e)}")
        print("Threaded CREDBANK features not found. Run dataset_credbank.py to create them.")
        credbank_paper = pd.DataFrame()
        credbank_all = pd.DataFrame()
    
    # Load threaded BuzzFeed features
    print("\nLoading threaded BuzzFeed features...")
    try:
        buzzfeed_paper = pd.read_csv(os.path.join(buzzfeed_path, 'buzzfeed_threaded_paper_features.csv'))
        buzzfeed_all = pd.read_csv(os.path.join(buzzfeed_path, 'buzzfeed_threaded_all_features.csv'))
    except FileNotFoundError as e:
        print(f"Warning: {str(e)}")
        print("Threaded BuzzFeed features not found. Run dataset_buzzfeed.py to create them.")
        buzzfeed_paper = pd.DataFrame()
        buzzfeed_all = pd.DataFrame()
    
    # Combine datasets (skipping empty dataframes)
    print("\nCombining threaded datasets...")
    dfs_paper = [df for df in [pheme_paper, credbank_paper, buzzfeed_paper] if not df.empty]
    dfs_all = [df for df in [pheme_all, credbank_all, buzzfeed_all] if not df.empty]
    
    if not dfs_paper or not dfs_all:
        print("Warning: No threaded datasets found. Please create them first.")
        return None, None
    
    paper_features_df = pd.concat(dfs_paper, axis=0, ignore_index=True)
    all_features_df = pd.concat(dfs_all, axis=0, ignore_index=True)
    
    # Save if requested
    if save_csv:
        # Save paper features
        paper_path = os.path.join(output_dir, 'train_threaded_paper.csv')
        paper_features_df.to_csv(paper_path, index=False)
        print(f"Saved threaded paper features to: {paper_path}")
        
        # Save all features
        complete_path = os.path.join(output_dir, 'train_threaded_complete.csv')
        all_features_df.to_csv(complete_path, index=False)
        print(f"Saved threaded complete features to: {complete_path}")
    
    return paper_features_df, all_features_df

def main():
    """Main execution function"""
    warnings.filterwarnings('ignore')
    
    # Load and combine regular datasets
    print("Creating regular combined dataset...")
    paper_df, all_df = load_and_combine_datasets(save_csv=True)
    
    print(f"\nRegular combined dataset shapes:")
    print(f"Paper features: {paper_df.shape}")
    print(f"All features: {all_df.shape}")
    
    # Load and combine threaded datasets
    print("\nCreating threaded combined dataset...")
    threaded_paper_df, threaded_all_df = load_and_combine_threaded_datasets(save_csv=True)
    
    if threaded_paper_df is not None and threaded_all_df is not None:
        print(f"\nThreaded combined dataset shapes:")
        print(f"Paper features: {threaded_paper_df.shape}")
        print(f"All features: {threaded_all_df.shape}")
    
    print("\nDataset creation complete!")

if __name__ == "__main__":
    main() 