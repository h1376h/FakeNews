from .base_user import BaseUserFeatureExtractor
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

class BuzzFeedUserFeatureExtractor(BaseUserFeatureExtractor):
    """Extract user-related features from BuzzFeed articles and tweets.
    
    Extracts user features from BuzzFeed dataset tweets. Supports both original paper features
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
    """
    
    def _parse_buzzfeed_date(self, date_str: str) -> datetime:
        """Parse BuzzFeed's date format to datetime object."""
        try:
            return datetime.strptime(str(date_str), '%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            return None
    
    def extract_features(self) -> pd.DataFrame:
        """Extract user-related features from the BuzzFeed dataset."""
        # Initialize feature columns
        df = self.df.copy()
        df = self._initialize_feature_columns(df, list(self.features_to_extract))
        
        # Process each row
        for idx, row in self.df.iterrows():
            # Get source user data
            source_user_id = str(row.get('user_screen_name', ''))
            source_created_at = self._parse_buzzfeed_date(row.get('user_created_at', ''))
            source_tweet_time = self._parse_buzzfeed_date(row.get('created_at', ''))
            
            # Get reaction user data
            user_ids = [str(name) for name in row.get('reaction_user_screen_names', [])]
            texts = [row.get('text', '')] + row.get('reaction_texts', [])
            
            # Get user metrics
            followers_counts = [row.get('user_followers_count', 0)]
            friends_counts = [row.get('user_friends_count', 0)]
            statuses_counts = [row.get('user_statuses_count', 0)]
            verified_flags = [row.get('user_verified', False)]
            
            # Add reaction user metrics
            followers_counts.extend(row.get('reaction_user_followers_counts', []))
            friends_counts.extend(row.get('reaction_user_friends_counts', []))
            statuses_counts.extend(row.get('reaction_user_statuses_counts', []))
            verified_flags.extend(row.get('reaction_user_verified', []))
            
            # Get timestamps
            tweet_times = []
            if source_tweet_time:
                tweet_times.append(source_tweet_time)
            
            for time_str in row.get('reaction_created_at', []):
                reaction_time = self._parse_buzzfeed_date(time_str)
                if reaction_time:
                    tweet_times.append(reaction_time)
            
            # Get account creation dates
            account_created_ats = []
            if source_created_at:
                account_created_ats.append(source_created_at)
            
            for created_at in row.get('reaction_user_created_at', []):
                reaction_created = self._parse_buzzfeed_date(created_at)
                if reaction_created:
                    account_created_ats.append(reaction_created)
            
            # Process user features
            features = self._process_user_features(
                source_user_id=source_user_id,
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