from .base import FeatureExtractor
import pandas as pd
import numpy as np
from datetime import datetime
import networkx as nx
from typing import List, Dict, Any

class BaseUserFeatureExtractor(FeatureExtractor):
    """Base class for user feature extractors.
    
    Supports both original paper features and additional features.
    Original paper features (9):
    1. user_avg_account_age_days: Average account age in days
    2. user_avg_followers_count: Average number of followers per user
    3. user_avg_friends_count: Average number of friends per user
    4. user_avg_statuses_count: Average number of statuses per user
    5. user_num_verified: Number of verified users
    6. user_network_density: Density of user interaction network
    7. user_avg_account_age_at_tweet: Average time between account creation and tweet
    8. user_source_verified: Whether source tweet author is verified
    9. user_source_account_age_days: Source author's account age in days

    Additional features (7):
    1. user_source_account_age_at_tweet: Time between source author's account creation and tweet
    2. user_verified_ratio: Ratio of verified users
    3. user_followers_friends_ratio: Ratio of followers to friends
    4. user_interaction_count: Number of user interactions (mentions/retweets)
    5. user_unique_authors: Number of unique authors in thread
    6. user_avg_interactions_per_author: Average interactions per author
    
    Each feature has multiple calculation methods depending on available data.
    The most accurate method will be used based on what data is provided.
    
    Attributes:
        PAPER_FEATURES: Set of original paper feature names
        EXTRA_FEATURES: Set of additional feature names
    """
    
    PAPER_FEATURES = {
        'user_avg_account_age_days',
        'user_avg_followers_count', 
        'user_avg_friends_count',
        'user_avg_statuses_count',
        'user_num_verified',
        'user_network_density',
        'user_avg_account_age_at_tweet',
        'user_source_verified',
        'user_source_account_age_days'
    }
    
    EXTRA_FEATURES = {
        'user_source_account_age_at_tweet',
        'user_verified_ratio',
        'user_followers_friends_ratio',
        'user_interaction_count',
        'user_unique_authors',
        'user_avg_interactions_per_author'
    }

    def __init__(self, df: pd.DataFrame, include_extra_features: bool = False):
        """Initialize the feature extractor.
        
        Args:
            df: Input DataFrame
            include_extra_features: Whether to include additional features not in the original paper
        """
        super().__init__(df)
        self.include_extra_features = include_extra_features
        self.features_to_extract = self.PAPER_FEATURES.copy()
        if include_extra_features:
            self.features_to_extract.update(self.EXTRA_FEATURES)

    def _calculate_network_density(self, source_user_id: str = None, user_ids: List[str] = None, 
                                 texts: List[str] = None) -> float:
        """Calculate network density based on user interactions.
        
        Args:
            source_user_id: Optional ID of the source tweet author
            user_ids: List of user IDs for each tweet
            texts: List of tweet texts
            
        Returns:
            float: Network density value between 0 and 1
        """
        if not texts:
            return 0.0
            
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add all users as nodes
        all_users = set()
        if source_user_id:
            all_users.add(str(source_user_id))
        if user_ids:
            all_users.update(str(uid) for uid in user_ids if uid)
        G.add_nodes_from(all_users)
        
        # Add edges based on mentions and interactions
        for i, text in enumerate(texts):
            current_user = user_ids[i] if user_ids and i < len(user_ids) else None
            if not current_user:
                continue
                
            # Add edge to source user if this is a reaction
            if source_user_id and i > 0:  # Skip source tweet
                G.add_edge(str(current_user), str(source_user_id))
            
            # Add edges based on mentions (@username)
            mentions = [word.strip('@') for word in str(text).split() if word.startswith('@')]
            for mention in mentions:
                if mention in all_users:  # Only add edge if mentioned user is in our user set
                    G.add_edge(str(current_user), str(mention))
            
            # Add edges for retweets
            if str(text).startswith('RT @'):
                try:
                    retweeted_user = str(text).split('@')[1].split(':')[0].strip()
                    if retweeted_user in all_users:
                        G.add_edge(str(current_user), str(retweeted_user))
                except IndexError:
                    pass
        
        # Calculate network density
        if len(G.nodes) > 1:
            return nx.density(G)
        return 0.0
    
    def _calculate_account_age(self, created_at: datetime, reference_date: datetime = None) -> float:
        """Calculate account age in days.
        
        Args:
            created_at: Account creation datetime
            reference_date: Optional reference date (defaults to now)
            
        Returns:
            float: Account age in days
        """
        if not created_at:
            return 0.0
            
        if not reference_date:
            reference_date = datetime.now()
            
        try:
            age_days = (reference_date - created_at).days
            return max(0.0, age_days)
        except (ValueError, TypeError):
            return 0.0
    
    def _calculate_account_age_at_tweet(self, created_at: datetime, tweet_time: datetime) -> float:
        """Calculate account age at tweet time in days.
        
        Args:
            created_at: Account creation datetime
            tweet_time: Tweet datetime
            
        Returns:
            float: Account age at tweet time in days
        """
        if not created_at or not tweet_time:
            return 0.0
            
        try:
            age_days = (tweet_time - created_at).days
            return max(0.0, age_days)
        except (ValueError, TypeError):
            return 0.0
    
    def _calculate_interaction_metrics(self, texts: List[str]) -> Dict[str, int]:
        """Calculate interaction metrics from tweet texts.
        
        Args:
            texts: List of tweet texts
            
        Returns:
            Dict containing:
            - interaction_count: Total number of interactions
            - mention_count: Number of mentions
            - retweet_count: Number of retweets
        """
        if not texts:
            return {'interaction_count': 0, 'mention_count': 0, 'retweet_count': 0}
            
        mention_count = sum(str(text).count('@') for text in texts)
        retweet_count = sum(1 for text in texts if str(text).startswith('RT @'))
        total_count = len(texts) + mention_count + retweet_count
        
        return {
            'interaction_count': total_count,
            'mention_count': mention_count,
            'retweet_count': retweet_count
        }
    
    def _process_user_features(self,
                             source_user_id: str = None,
                             user_ids: List[str] = None,
                             texts: List[str] = None,
                             followers_counts: List[int] = None,
                             friends_counts: List[int] = None,
                             statuses_counts: List[int] = None,
                             verified_flags: List[bool] = None,
                             account_created_ats: List[datetime] = None,
                             tweet_times: List[datetime] = None) -> Dict[str, float]:
        """Process user features from the provided data.
        
        Args:
            source_user_id: Optional ID of source tweet author
            user_ids: List of user IDs
            texts: List of tweet texts
            followers_counts: List of follower counts
            friends_counts: List of friend counts
            statuses_counts: List of status counts
            verified_flags: List of verification flags
            account_created_ats: List of account creation dates
            tweet_times: List of tweet timestamps
            
        Returns:
            Dictionary containing user features based on include_extra_features setting
        """
        # Initialize empty features dictionary
        features = {}
        
        # Calculate unique authors
        unique_users = set()
        if source_user_id:
            unique_users.add(str(source_user_id))
        if user_ids:
            unique_users.update(str(uid) for uid in user_ids if uid)
        
        if 'user_unique_authors' in self.features_to_extract:
            features['user_unique_authors'] = len(unique_users)
        
        # Calculate network density
        if 'user_network_density' in self.features_to_extract:
            features['user_network_density'] = self._calculate_network_density(
                source_user_id=source_user_id,
                user_ids=user_ids,
                texts=texts
            )
        
        # Calculate interaction metrics
        if texts:
            interaction_metrics = self._calculate_interaction_metrics(texts)
            if 'user_interaction_count' in self.features_to_extract:
                features['user_interaction_count'] = interaction_metrics['interaction_count']
            if 'user_avg_interactions_per_author' in self.features_to_extract and len(unique_users) > 0:
                features['user_avg_interactions_per_author'] = (
                    interaction_metrics['interaction_count'] / len(unique_users)
                )
        
        # Calculate user metric averages if available
        if followers_counts:
            if 'user_avg_followers_count' in self.features_to_extract:
                features['user_avg_followers_count'] = np.mean(followers_counts)
        if friends_counts:
            if 'user_avg_friends_count' in self.features_to_extract:
                features['user_avg_friends_count'] = np.mean(friends_counts)
            if 'user_followers_friends_ratio' in self.features_to_extract and features.get('user_avg_followers_count'):
                features['user_followers_friends_ratio'] = (
                    features['user_avg_followers_count'] / max(1, features['user_avg_friends_count'])
                )
        if statuses_counts:
            if 'user_avg_statuses_count' in self.features_to_extract:
                features['user_avg_statuses_count'] = np.mean(statuses_counts)
        
        # Calculate verification metrics if available
        if verified_flags:
            if 'user_num_verified' in self.features_to_extract:
                features['user_num_verified'] = sum(1 for v in verified_flags if v)
            if 'user_verified_ratio' in self.features_to_extract:
                features['user_verified_ratio'] = np.mean([int(v) for v in verified_flags])
            if 'user_source_verified' in self.features_to_extract:
                features['user_source_verified'] = int(verified_flags[0]) if verified_flags else 0
        
        # Calculate account age metrics if available
        if account_created_ats and tweet_times:
            account_ages = []
            age_at_tweets = []
            
            for created_at, tweet_time in zip(account_created_ats, tweet_times):
                if created_at and tweet_time:
                    account_ages.append(self._calculate_account_age(created_at, reference_date=tweet_time))
                    age_at_tweets.append(self._calculate_account_age_at_tweet(created_at, tweet_time))
            
            if account_ages:
                if 'user_avg_account_age_days' in self.features_to_extract:
                    features['user_avg_account_age_days'] = np.mean(account_ages)
                if 'user_source_account_age_days' in self.features_to_extract:
                    features['user_source_account_age_days'] = account_ages[0] if account_ages else 0
            
            if age_at_tweets:
                if 'user_avg_account_age_at_tweet' in self.features_to_extract:
                    features['user_avg_account_age_at_tweet'] = np.mean(age_at_tweets)
                if 'user_source_account_age_at_tweet' in self.features_to_extract:
                    features['user_source_account_age_at_tweet'] = age_at_tweets[0] if age_at_tweets else 0
        
        return features
    
    def extract_features(self) -> pd.DataFrame:
        """Extract user features from the dataset.
        
        Returns:
            DataFrame with user features based on include_extra_features setting
            
        Raises:
            NotImplementedError: If not implemented by subclass
            ValueError: If feature validation fails
        """
        raise NotImplementedError("Subclasses must implement extract_features method")