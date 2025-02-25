from .base_temporal import BaseTemporalFeatureExtractor
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

"""Temporal features for CREDBANK dataset."""
class CredbankTemporalFeatureExtractor(BaseTemporalFeatureExtractor):
    """Features based on temporal aspects of tweets in CREDBANK dataset.
    
    Args:
        df: DataFrame containing the CREDBANK dataset
        include_additional: Whether to include additional features not in the paper
    """
    
    def __init__(self, df: pd.DataFrame, include_additional: bool = False):
        """Initialize the CREDBANK temporal feature extractor."""
        super().__init__(df, include_additional)
    
    def _parse_credbank_date(self, date_str: str) -> datetime:
        """Parse CREDBANK's date format to datetime object."""
        try:
            return datetime.strptime(str(date_str), '%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            return None
    
    def extract_features(self) -> pd.DataFrame:
        """Extract temporal features from the CREDBANK dataset."""
        df = self.df.copy()
        
        # Initialize temporal columns with NaN
        df = self._initialize_feature_columns(df, list(self.TEMPORAL_FEATURES))
        
        # Process each row
        for idx, row in df.iterrows():
            # Get timestamps
            timestamps = []
            for t in row.get('created_at_times', []):
                timestamp = self._parse_credbank_date(str(t))
                if timestamp:
                    timestamps.append(timestamp)
            
            if not timestamps:
                continue
            
            # Get texts
            texts = row.get('Reasons', [])
            if not isinstance(texts, list):
                texts = []
            
            # Get user metrics (using default values since CREDBANK doesn't have this data)
            num_tweets = len(timestamps)
            followers_counts = [100] * num_tweets  # Default value
            friends_counts = [50] * num_tweets     # Default value
            statuses_counts = [200] * num_tweets   # Default value
            
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