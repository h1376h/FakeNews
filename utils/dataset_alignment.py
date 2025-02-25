import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union
from textblob import TextBlob
from collections import defaultdict

def align_buzzfeed_threads(article_data: Dict) -> List[Dict]:
    """Extract and align Twitter threads from BuzzFeed articles.
    
    Args:
        article_data: Dictionary containing article data including text and metadata
        
    Returns:
        List of dictionaries containing aligned thread data
    """
    threads = []
    
    # Extract main text and metadata
    main_text = article_data.get('mainText', '')
    veracity = article_data.get('veracity', '')
    orientation = article_data.get('orientation', '')
    
    # Convert veracity to binary label
    label = 1 if veracity in ['mostly false', 'mixture of true and false', 'no factual content'] else 0
    
    # Create thread entry
    thread = {
        'source': 'buzzfeed',
        'text': main_text,
        'label': label,
        'orientation': orientation,
        'veracity': veracity
    }
    
    threads.append(thread)
    return threads

def align_pheme_threads(thread_data: Dict) -> List[Dict]:
    """Extract and align Twitter threads from PHEME dataset.
    
    Args:
        thread_data: Dictionary containing thread data including source tweet and reactions
        
    Returns:
        List of dictionaries containing aligned thread data
    """
    threads = []
    
    # Extract source tweet and metadata
    source_tweet = thread_data.get('source_tweet', {})
    category = thread_data.get('category', '')
    reactions = thread_data.get('reactions', [])
    
    # Convert category to binary label (rumours are considered potentially fake)
    label = 1 if category == 'rumours' else 0
    
    # Calculate disagreement score from reactions
    reaction_texts = [reaction.get('text', '') for reaction in reactions]
    disagreement = calculate_disagreement_score(reaction_texts)
    
    # Create thread entry
    thread = {
        'source': 'pheme',
        'text': source_tweet.get('text', ''),
        'label': label,
        'category': category,
        'disagreement': disagreement,
        'num_reactions': len(reactions)
    }
    
    threads.append(thread)
    return threads

def convert_credbank_scale(ratings: List[int]) -> Tuple[int, float]:
    """Convert CREDBANK's 5-point Likert scale to binary labels and confidence scores.
    
    Args:
        ratings: List of integer ratings from -2 to 2
        
    Returns:
        Tuple of (binary_label, confidence_score)
    """
    # Convert string ratings to integers and then to numpy array
    try:
        ratings = np.array([int(r) for r in ratings])
    except (ValueError, TypeError):
        # Handle invalid ratings by returning default values
        return 0, 0.0
    
    # Calculate mean rating
    mean_rating = np.mean(ratings)
    
    # Convert to binary label (1 for fake, 0 for real)
    binary_label = 1 if mean_rating < 0 else 0
    
    # Calculate confidence score (absolute mean normalized to [0,1])
    confidence = abs(mean_rating) / 2.0
    
    return binary_label, confidence

def calculate_disagreement_score(texts: List[str]) -> float:
    """Calculate disagreement score based on sentiment analysis of reactions.
    
    Args:
        texts: List of reaction texts to analyze
        
    Returns:
        Float indicating disagreement score [0,1]
    """
    if not texts:
        return 0.0
    
    # Calculate sentiment polarity for each text
    sentiments = [TextBlob(text).sentiment.polarity for text in texts]
    
    # Calculate variance of sentiments as disagreement score
    disagreement = np.var(sentiments)
    
    # Normalize to [0,1]
    normalized_disagreement = min(disagreement, 1.0)
    
    return normalized_disagreement

def save_feature_sets(df: pd.DataFrame, output_dir: str, dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Save both paper features and all features for a dataset.
    
    Args:
        df: DataFrame containing all features
        output_dir: Directory to save feature files
        dataset_name: Name of the dataset (pheme, buzzfeed, or credbank)
        
    Returns:
        Tuple of (paper_features_df, all_features_df)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all feature columns
    feature_columns = [col for col in df.columns if any(
        col.startswith(prefix) for prefix in 
        ['structural_', 'user_', 'content_', 'temporal_']
    )]
    
    # Define additional features to exclude from paper features
    additional_features = {
        # Additional content features
        'content_num_info_request', 'content_ratio_info_request',
        'content_num_support', 'content_ratio_support',
        'content_num_disagreement', 'content_ratio_disagreement',
        'content_num_polarity', 'content_num_subjectivity',
        # Additional user features
        'user_source_account_age_at_tweet', 'user_verified_ratio',
        'user_followers_friends_ratio', 'user_interaction_count',
        'user_unique_authors', 'user_avg_interactions_per_author',
        # Additional temporal features
        'temporal_network_density_slope'
    }
    
    # Get paper feature columns (exclude additional features)
    paper_feature_columns = [col for col in feature_columns if col not in additional_features]
    
    # Create DataFrames with source and label columns
    paper_features_df = df[['source', 'label'] + paper_feature_columns].copy()
    all_features_df = df[['source', 'label'] + feature_columns].copy()
    
    # Save both feature sets
    paper_features_path = os.path.join(output_dir, f'{dataset_name}_paper_features.csv')
    all_features_path = os.path.join(output_dir, f'{dataset_name}_all_features.csv')
    
    paper_features_df.to_csv(paper_features_path, index=False)
    all_features_df.to_csv(all_features_path, index=False)
    
    print(f"Saved paper features ({len(paper_feature_columns)} features) to: {paper_features_path}")
    print(f"Saved all features ({len(feature_columns)} features) to: {all_features_path}")
    
    return paper_features_df, all_features_df

def align_datasets(pheme_df: pd.DataFrame = None, buzzfeed_df: pd.DataFrame = None, 
                  credbank_df: pd.DataFrame = None, output_dir: str = 'data/aligned',
                  save_csv: bool = True) -> Dict[str, pd.DataFrame]:
    """Align features across different datasets and save both paper and all feature sets.
    
    Args:
        pheme_df: PHEME dataset features
        buzzfeed_df: BuzzFeed dataset features  
        credbank_df: CREDBANK dataset features
        output_dir: Directory to save aligned datasets
        save_csv: Whether to save CSV files
        
    Returns:
        Dictionary containing aligned DataFrames for paper and all features
    """
    datasets = {}
    
    # Process each dataset if provided
    if pheme_df is not None:
        pheme_paper, pheme_all = save_feature_sets(pheme_df, output_dir, 'pheme')
        datasets['pheme_paper'] = pheme_paper
        datasets['pheme_all'] = pheme_all
        
    if buzzfeed_df is not None:
        buzzfeed_paper, buzzfeed_all = save_feature_sets(buzzfeed_df, output_dir, 'buzzfeed')
        datasets['buzzfeed_paper'] = buzzfeed_paper
        datasets['buzzfeed_all'] = buzzfeed_all
        
    if credbank_df is not None:
        credbank_paper, credbank_all = save_feature_sets(credbank_df, output_dir, 'credbank')
        datasets['credbank_paper'] = credbank_paper
        datasets['credbank_all'] = credbank_all
    
    if save_csv and len(datasets) > 0:
        # Create combined datasets for paper features
        paper_dfs = [df for name, df in datasets.items() if name.endswith('_paper')]
        if paper_dfs:
            combined_paper = pd.concat(paper_dfs, axis=0, ignore_index=True)
            combined_paper_path = os.path.join(output_dir, 'combined_paper_features.csv')
            combined_paper.to_csv(combined_paper_path, index=False)
            print(f"Saved combined paper features to: {combined_paper_path}")
            datasets['combined_paper'] = combined_paper
        
        # Create combined datasets for all features
        all_dfs = [df for name, df in datasets.items() if name.endswith('_all')]
        if all_dfs:
            combined_all = pd.concat(all_dfs, axis=0, ignore_index=True)
            combined_all_path = os.path.join(output_dir, 'combined_all_features.csv')
            combined_all.to_csv(combined_all_path, index=False)
            print(f"Saved combined all features to: {combined_all_path}")
            datasets['combined_all'] = combined_all
    
    return datasets