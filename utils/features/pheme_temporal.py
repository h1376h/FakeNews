from .base_temporal import BaseTemporalFeatureExtractor
import pandas as pd
from datetime import datetime
from typing import Any, List, Dict

"""Temporal features for PHEME dataset."""
class PhemeTemporalFeatureExtractor(BaseTemporalFeatureExtractor):
    """Features based on temporal aspects of tweets in PHEME dataset.
    
    Args:
        df: DataFrame containing the PHEME dataset
        include_additional: Whether to include additional features not in the paper
    """
    
    def __init__(self, df: pd.DataFrame, include_additional: bool = False):
        """Initialize the PHEME temporal feature extractor."""
        super().__init__(df, include_additional)
    
    def _parse_pheme_date(self, date_str: str) -> datetime:
        """Parse PHEME's date format to datetime object."""
        try:
            return datetime.strptime(date_str, '%a %b %d %H:%M:%S +0000 %Y')
        except (ValueError, TypeError):
            return None
    
    def extract_features(self) -> pd.DataFrame:
        """Extract temporal features from the PHEME dataset."""
        df = self.df.copy()
        
        # Initialize temporal columns with NaN
        df = self._initialize_feature_columns(df, list(self.TEMPORAL_FEATURES))
        
        for idx, row in df.iterrows():
            # Get timestamps
            timestamps = []
            source_time = self._parse_pheme_date(row['source_tweet_created_at'])
            if source_time:
                timestamps.append(source_time)
            
            for time_str in row['reaction_created_at']:
                reaction_time = self._parse_pheme_date(time_str)
                if reaction_time:
                    timestamps.append(reaction_time)
            
            if not timestamps:
                continue
            
            # Get user metrics
            followers_counts = [row['source_tweet_user_followers_count']]
            friends_counts = [row['source_tweet_user_friends_count']]
            statuses_counts = [row['source_tweet_user_statuses_count']]
            
            # Add reaction user metrics (using source tweet user data since we don't have reaction user data)
            num_reactions = len(row['reaction_created_at'])
            followers_counts.extend([row['source_tweet_user_followers_count']] * num_reactions)
            friends_counts.extend([row['source_tweet_user_friends_count']] * num_reactions)
            statuses_counts.extend([row['source_tweet_user_statuses_count']] * num_reactions)
            
            # Get account creation dates
            account_created_ats = []
            source_created = self._parse_pheme_date(row['source_tweet_user_created_at'])
            if source_created:
                account_created_ats.append(source_created)
                account_created_ats.extend([source_created] * num_reactions)
            
            # Get texts and user IDs
            texts = [row['source_tweet_text']] + row['reaction_texts']
            user_ids = [row['source_tweet_user_id_str']] + row.get('reaction_user_ids', [])
            
            # Process temporal features
            features = self._process_temporal_features(
                timestamps=timestamps,
                followers_counts=followers_counts,
                friends_counts=friends_counts,
                statuses_counts=statuses_counts,
                account_created_ats=account_created_ats,
                texts=texts,
                user_ids=user_ids
            )
            
            # Store features
            for col, value in features.items():
                df.at[idx, col] = value
        
        # Handle missing values
        df = self._handle_missing_values(df, list(self.TEMPORAL_FEATURES))

        return df 