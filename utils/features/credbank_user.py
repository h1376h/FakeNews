from .base_user import BaseUserFeatureExtractor
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

"""User-based features for CREDBANK dataset."""
class CredbankUserFeatureExtractor(BaseUserFeatureExtractor):
    """Features based on user characteristics in CREDBANK dataset.
    
    Extracts user features from CREDBANK dataset tweets. Supports both original paper features
    and additional features based on the include_extra_features parameter.
    
    Original paper features (9):
    - user_avg_account_age_days
    - user_avg_followers_count
    - user_avg_friends_count 
    - user_avg_statuses_count
    - user_num_verified
    - user_network_density
    - user_avg_account_age_at_tweet
    - user_source_verified
    - user_source_account_age_days
    
    Additional features when include_extra_features=True (7):
    - user_source_account_age_at_tweet
    - user_verified_ratio
    - user_followers_friends_ratio
    - user_interaction_count
    - user_unique_authors
    - user_avg_interactions_per_author
    
    Note: Since CREDBANK dataset has limited user metadata, some features use default values.
    """
    
    def _parse_credbank_date(self, date_str: str) -> datetime:
        """Parse CREDBANK's date format to datetime object."""
        try:
            return datetime.strptime(str(date_str), '%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            return None
    
    def extract_features(self) -> pd.DataFrame:
        """Extract user-based features from the CREDBANK dataset."""
        df = self.df.copy()
        
        # Initialize feature columns with NaN
        df = self._initialize_feature_columns(df, list(self.features_to_extract))
        
        # Process each row
        for idx, row in df.iterrows():
            # Get user IDs and texts
            authors = row.get('authors', [])
            texts = row.get('tweet_texts', [])
            created_at_times = row.get('created_at_times', [])
            
            if not isinstance(authors, list):
                authors = []
            if not isinstance(texts, list):
                texts = []
            if not isinstance(created_at_times, list):
                created_at_times = []
            
            # Parse timestamps
            tweet_times = []
            for t in created_at_times:
                timestamp = self._parse_credbank_date(str(t))
                if timestamp:
                    tweet_times.append(timestamp)
            
            # Get user metrics (using default values since CREDBANK doesn't have this data)
            num_tweets = len(tweet_times)
            followers_counts = [100] * num_tweets  # Default value
            friends_counts = [50] * num_tweets     # Default value
            statuses_counts = [200] * num_tweets   # Default value
            verified_flags = [False] * num_tweets  # Default value
            
            # Get account creation dates (using Twitter's founding date as default)
            twitter_founding = datetime(2006, 3, 21)
            account_created_ats = [twitter_founding] * num_tweets
            
            # Get user IDs from tweet list
            user_ids = []
            for tweet_tuple in row.get('tweet_list', []):
                user_id = None
                for part in tweet_tuple:
                    if 'user_id=' in part:
                        user_id = part.split('user_id=')[1]
                        break
                user_ids.append(user_id or f'unknown_{len(user_ids)}')
            
            # Process user features
            features = self._process_user_features(
                user_ids=user_ids,
                texts=texts,
                followers_counts=followers_counts,
                friends_counts=friends_counts,
                statuses_counts=statuses_counts,
                verified_flags=verified_flags,
                account_created_ats=account_created_ats,
                tweet_times=tweet_times
            )
            
            # Store features
            for col, value in features.items():
                df.at[idx, col] = value
        
        # Handle missing values
        df = self._handle_missing_values(df, list(self.features_to_extract))
        
        return df 