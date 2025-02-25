import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union
from textblob import TextBlob
from collections import defaultdict

def align_buzzfeed_threads(article_data: Dict, thread_capture_tool=None) -> List[Dict]:
    """Extract and align Twitter threads from BuzzFeed articles.
    
    As described in the paper:
    1. Extract the 10 most shared stories from left-wing pages
    2. Extract the 10 most shared stories from right-wing pages
    3. Search Twitter for these headlines
    4. Keep the top 3 most retweeted posts for each headline
    5. This results in 35 topics with journalist-provided labels (15 "mostly true", 20 "mostly false")
    
    Args:
        article_data: Dictionary containing article data including text and metadata
        thread_capture_tool: ThreadCaptureTool instance for Twitter API access
        
    Returns:
        List of dictionaries containing aligned thread data
    """
    threads = []
    
    # Extract main text and metadata
    title = article_data.get('title', '')
    main_text = article_data.get('mainText', '')
    veracity = article_data.get('veracity', '')
    orientation = article_data.get('orientation', '')
    
    # Convert veracity to binary label (1 for fake, 0 for real)
    # 'mostly false', 'mixture of true and false', 'no factual content' -> 1 (fake)
    # 'mostly true' -> 0 (real)
    label = 1 if veracity in ['mostly false', 'mixture of true and false', 'no factual content'] else 0
    
    # If we have a thread capture tool, use it to search Twitter for headlines
    if thread_capture_tool is not None and hasattr(thread_capture_tool, '_search_twitter_for_headline'):
        # Search Twitter for the headline
        twitter_threads = thread_capture_tool._search_twitter_for_headline(title)
        
        # Keep the top 3 most retweeted posts
        top_tweets = sorted(twitter_threads, key=lambda x: x.get('retweet_count', 0), reverse=True)[:3]
        
        # Create threads from top tweets
        for tweet in top_tweets:
            thread = {
                'source': 'buzzfeed',
                'text': tweet.get('text', ''),
                'headline': title,
                'label': label,
                'orientation': orientation,
                'veracity': veracity,
                'retweet_count': tweet.get('retweet_count', 0),
                'favorite_count': tweet.get('favorite_count', 0),
                'tweet_id': tweet.get('id_str', ''),
                'user_id': tweet.get('user', {}).get('id_str', ''),
                'user_name': tweet.get('user', {}).get('screen_name', '')
            }
            threads.append(thread)
    else:
        # If no thread capture tool or API access, create thread directly from article data
        thread = {
            'source': 'buzzfeed',
            'text': main_text,
            'headline': title,
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

def convert_credbank_scale(ratings: List[int]) -> Tuple[int, float, bool]:
    """Convert CREDBANK's 5-point Likert scale to binary labels based on quantiles.
    
    As described in the paper:
    - The grand mean of CREDBANK's accuracy assessments is 1.7
    - The median is 1.767
    - The 25th and 75th quartiles are 1.6 and 1.867 respectively
    - Events below the 15% quantile (mean rating < 1.467) are labeled as negative (1)
    - Events above the 85% quantile (mean rating > 1.9) are labeled as positive (0)
    - Events between these values are unlabeled and removed
    
    Args:
        ratings: List of integer ratings from -2 to 2
        
    Returns:
        Tuple of (binary_label, confidence_score, is_valid)
        where is_valid is False if the rating falls in the unlabeled range
    """
    # Convert string ratings to integers and then to numpy array
    try:
        ratings = np.array([int(r) for r in ratings])
    except (ValueError, TypeError):
        # Handle invalid ratings by returning default values
        return 0, 0.0, False
    
    # Calculate mean rating
    mean_rating = np.mean(ratings)
    
    # Define quantile thresholds for credibility assessment
    # Bottom 15% quantile = 1.467, Top 15% quantile = 1.9
    low_threshold = 1.467
    high_threshold = 1.9
    
    # Convert to binary label (1 for fake/negative, 0 for real/positive)
    # Also determine if the event should be included in the dataset
    if mean_rating < low_threshold:
        # Below bottom 15% quantile - negative event (low credibility)
        binary_label = 1
        is_valid = True
    elif mean_rating > high_threshold:
        # Above top 15% quantile - positive event (high credibility)
        binary_label = 0
        is_valid = True
    else:
        # In between - unlabeled
        binary_label = -1
        is_valid = False
    
    # Calculate confidence score (absolute distance from neutral value 0)
    confidence = abs(mean_rating) / 2.0
    
    return binary_label, confidence, is_valid

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

def align_credbank_threads(event_data: Dict, thread_capture_tool=None) -> List[Dict]:
    """Extract and align Twitter threads from CREDBANK events.
    
    As described in the paper:
    1. Identify the most retweeted tweet in each event as the thread root
    2. Collect replies to this root tweet as children
    3. Discard threads with no reactions
    
    For label alignment:
    - Events with average accuracy < 1.467 (bottom 15%) are labeled negative (1)
    - Events with average accuracy > 1.9 (top 15%) are labeled positive (0)
    - Events between these values are unlabeled and removed
    
    Args:
        event_data: Dictionary containing event data including tweets and ratings
        thread_capture_tool: ThreadCaptureTool instance for Twitter API access
        
    Returns:
        List of dictionaries containing aligned thread data, empty if the event 
        should be discarded based on rating quantiles
    """
    threads = []
    
    # Extract event data
    topic_terms = event_data.get('topic_terms', '')
    ratings = event_data.get('ratings', [])
    tweets = event_data.get('tweets', [])
    
    # Convert ratings to binary label
    binary_label, confidence, is_valid = convert_credbank_scale(ratings)
    
    # If rating is in the unlabeled range, return empty list
    if not is_valid:
        return []
    
    # Find the most retweeted tweet in the event to use as thread root
    most_retweeted_tweet = None
    most_retweets = -1
    
    for tweet in tweets:
        retweet_count = tweet.get('retweet_count', 0)
        if retweet_count > most_retweets:
            most_retweets = retweet_count
            most_retweeted_tweet = tweet
    
    # If we couldn't find a valid tweet, return empty list
    if most_retweeted_tweet is None:
        return []
    
    # If we have a thread capture tool, use it to get replies
    if thread_capture_tool is not None and hasattr(thread_capture_tool, '_get_twitter_replies'):
        tweet_id = most_retweeted_tweet.get('id_str', '')
        replies = thread_capture_tool._get_twitter_replies(tweet_id)
        
        # If no replies, set an empty list
        if not replies:
            replies = []
        
        # Create thread from root tweet and replies
        thread = {
            'source': 'credbank',
            'text': most_retweeted_tweet.get('text', ''),
            'topic_terms': topic_terms,
            'label': binary_label,
            'confidence': confidence,
            'retweet_count': most_retweets,
            'tweet_id': tweet_id,
            'user_id': most_retweeted_tweet.get('user', {}).get('id_str', ''),
            'user_name': most_retweeted_tweet.get('user', {}).get('screen_name', ''),
            'replies': replies,
            'reply_count': len(replies)
        }
        
        # Only add threads with reactions
        if len(replies) > 0:
            threads.append(thread)
    else:
        # If no thread capture tool or API access, create thread directly from tweet data
        thread = {
            'source': 'credbank',
            'text': most_retweeted_tweet.get('text', ''),
            'topic_terms': topic_terms,
            'label': binary_label,
            'confidence': confidence,
            'retweet_count': most_retweets,
            'tweet_id': most_retweeted_tweet.get('id_str', '')
        }
        threads.append(thread)
    
    return threads