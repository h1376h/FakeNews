import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union
import warnings

def load_and_combine_datasets(pheme_path: str = 'data/pheme', 
                            credbank_path: str = 'data/credbank',
                            output_dir: str = 'data/train',
                            save_csv: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and combine PHEME and CREDBANK feature datasets.
    
    Args:
        pheme_path: Path to PHEME dataset directory
        credbank_path: Path to CREDBANK dataset directory
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
    
    # Combine datasets
    print("\nCombining datasets...")
    paper_features_df = pd.concat([pheme_paper, credbank_paper], axis=0, ignore_index=True)
    all_features_df = pd.concat([pheme_all, credbank_all], axis=0, ignore_index=True)
    
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

def main():
    """Main execution function"""
    warnings.filterwarnings('ignore')
    
    # Load and combine datasets
    paper_df, complete_df = load_and_combine_datasets(save_csv=True)
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Paper features shape: {paper_df.shape}")
    print(f"Complete features shape: {complete_df.shape}")
    
    # Print source distribution
    print("\nSource Distribution:")
    print(paper_df['source'].value_counts())
    
    # Print label distribution
    print("\nLabel Distribution:")
    print(paper_df['label'].value_counts())

if __name__ == "__main__":
    main() 