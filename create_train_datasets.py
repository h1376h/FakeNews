import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union
import warnings
from utils.feature_consistency import ensure_feature_consistency, align_datasets_features

def load_and_combine_datasets(pheme_path: str = 'data/pheme', 
                            credbank_path: str = 'data/credbank',
                            buzzfeed_path: str = 'data/buzzfeed',
                            output_dir: str = 'data/train',
                            save_csv: bool = True,
                            ensure_consistency: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and combine feature datasets with consistency checks.
    
    Args:
        pheme_path: Path to PHEME dataset directory
        credbank_path: Path to CREDBANK dataset directory
        buzzfeed_path: Path to BuzzFeed dataset directory
        output_dir: Directory to save output files
        save_csv: Whether to save CSV files
        ensure_consistency: Whether to ensure feature consistency
        
    Returns:
        Tuple of (paper_features_df, all_features_df)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = {}
    
    # Load PHEME features if available
    if os.path.exists(os.path.join(pheme_path, 'pheme_paper_features.csv')):
        print("Loading PHEME features...")
        pheme_paper = pd.read_csv(os.path.join(pheme_path, 'pheme_paper_features.csv'))
        pheme_all = pd.read_csv(os.path.join(pheme_path, 'pheme_all_features.csv'))
        datasets['pheme_paper'] = pheme_paper
        datasets['pheme_all'] = pheme_all
    
    # Load CREDBANK features if available
    if os.path.exists(os.path.join(credbank_path, 'credbank_paper_features.csv')):
        print("\nLoading CREDBANK features...")
        credbank_paper = pd.read_csv(os.path.join(credbank_path, 'credbank_paper_features.csv'))
        credbank_all = pd.read_csv(os.path.join(credbank_path, 'credbank_all_features.csv'))
        datasets['credbank_paper'] = credbank_paper
        datasets['credbank_all'] = credbank_all
    
    # Load BuzzFeed features if available
    if os.path.exists(os.path.join(buzzfeed_path, 'buzzfeed_paper_features.csv')):
        print("\nLoading BuzzFeed features...")
        buzzfeed_paper = pd.read_csv(os.path.join(buzzfeed_path, 'buzzfeed_paper_features.csv'))
        buzzfeed_all = pd.read_csv(os.path.join(buzzfeed_path, 'buzzfeed_all_features.csv'))
        datasets['buzzfeed_paper'] = buzzfeed_paper
        datasets['buzzfeed_all'] = buzzfeed_all
    
    # Ensure feature consistency if requested
    if ensure_consistency:
        print("\nEnsuring feature consistency across datasets...")
        
        # Separate paper and all feature datasets
        paper_datasets = {name: df for name, df in datasets.items() if name.endswith('_paper')}
        all_datasets = {name: df for name, df in datasets.items() if name.endswith('_all')}
        
        # Align features
        aligned_paper = align_datasets_features(paper_datasets, os.path.join(output_dir, 'aligned'))
        aligned_all = align_datasets_features(all_datasets, os.path.join(output_dir, 'aligned'))
        
        # Update datasets with aligned versions
        datasets.update(aligned_paper)
        datasets.update(aligned_all)
    
    # Combine datasets
    print("\nCombining datasets...")
    paper_dfs = [df for name, df in datasets.items() if name.endswith('_paper')]
    all_dfs = [df for name, df in datasets.items() if name.endswith('_all')]
    
    paper_features_df = pd.concat(paper_dfs, axis=0, ignore_index=True)
    all_features_df = pd.concat(all_dfs, axis=0, ignore_index=True)
    
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
    
    # Load and combine datasets with consistency checks
    paper_df, complete_df = load_and_combine_datasets(save_csv=True, ensure_consistency=True)
    
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
    
    # Verify feature consistency
    print("\nVerifying final feature consistency...")
    _, paper_issues = ensure_feature_consistency(paper_df, fix_issues=False)
    _, complete_issues = ensure_feature_consistency(complete_df, fix_issues=False)
    
    if paper_issues:
        print(f"Warning: {len(paper_issues)} feature issues found in paper features dataset")
    else:
        print("Paper features dataset is consistent!")
        
    if complete_issues:
        print(f"Warning: {len(complete_issues)} feature issues found in complete features dataset")
    else:
        print("Complete features dataset is consistent!")

if __name__ == "__main__":
    main() 