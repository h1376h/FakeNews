from .base_user import BaseUserFeatureExtractor
import pandas as pd
from datetime import datetime
from typing import List, Dict

"""User-based features for PHEME dataset."""
class PhemeUserFeatureExtractor(BaseUserFeatureExtractor):
    """Features based on user characteristics in PHEME dataset.
    
    Extracts user features from PHEME dataset tweets. Supports both original paper features
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
    
    def _parse_pheme_date(self, date_str: str) -> datetime:
        """Parse PHEME's date format to datetime object."""
        try:
            return datetime.strptime(date_str, '%a %b %d %H:%M:%S +0000 %Y')
        except (ValueError, TypeError):
            return None
    
    def extract_features(self) -> pd.DataFrame:
        """Extract user-based features from the PHEME dataset."""
        df = self.df.copy()
        
        # Initialize feature columns with NaN
        df = self._initialize_feature_columns(df, list(self.features_to_extract))
        
        # Process each row
        for idx, row in df.iterrows():
            # Get source user data
            source_user_id = row['source_tweet_user_id_str']
            source_created_at = self._parse_pheme_date(row['source_tweet_user_created_at'])
            source_tweet_time = self._parse_pheme_date(row['source_tweet_created_at'])
            
            # Get reaction user data
            user_ids = row.get('reaction_user_ids', [])
            texts = [row['source_tweet_text']] + row['reaction_texts']
            
            # Get user metrics
            followers_counts = [row['source_tweet_user_followers_count']]
            friends_counts = [row['source_tweet_user_friends_count']]
            statuses_counts = [row['source_tweet_user_statuses_count']]
            verified_flags = [row['source_tweet_user_verified']]
            
            # Add reaction user metrics (using source tweet user data since we don't have reaction user data)
            num_reactions = len(row['reaction_created_at'])
            followers_counts.extend([row['source_tweet_user_followers_count']] * num_reactions)
            friends_counts.extend([row['source_tweet_user_friends_count']] * num_reactions)
            statuses_counts.extend([row['source_tweet_user_statuses_count']] * num_reactions)
            verified_flags.extend([row['source_tweet_user_verified']] * num_reactions)
            
            # Get timestamps
            tweet_times = []
            if source_tweet_time:
                tweet_times.append(source_tweet_time)
            
            for time_str in row['reaction_created_at']:
                reaction_time = self._parse_pheme_date(time_str)
                if reaction_time:
                    tweet_times.append(reaction_time)
            
            # Get account creation dates
            account_created_ats = []
            if source_created_at:
                account_created_ats.append(source_created_at)
                account_created_ats.extend([source_created_at] * num_reactions)
            
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