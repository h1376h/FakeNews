from .base_temporal import BaseTemporalFeatureExtractor
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import re

"""Temporal features for BuzzFeed dataset."""
class BuzzFeedTemporalFeatureExtractor(BaseTemporalFeatureExtractor):
    """Extract temporal features from BuzzFeed articles and tweets
    
    Args:
        df: DataFrame containing the BuzzFeed dataset
        include_additional: Whether to include additional features not in the paper
    """
    
    def __init__(self, df: pd.DataFrame, include_additional: bool = False):
        """Initialize the BuzzFeed temporal feature extractor."""
        super().__init__(df, include_additional)
    
    def _parse_buzzfeed_date(self, date_str: str) -> datetime:
        """Parse BuzzFeed's date format to datetime object."""
        try:
            return datetime.strptime(str(date_str), '%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            return None
    
    def extract_features(self) -> pd.DataFrame:
        """Extract temporal features from the BuzzFeed dataset."""
        df = self.df.copy()
        
        # Initialize feature columns
        df = self._initialize_feature_columns(df, list(self.TEMPORAL_FEATURES))
        
        # Process each row
        for idx, row in df.iterrows():
            # Get timestamps
            timestamps = []
            source_time = self._parse_buzzfeed_date(row.get('created_at', ''))
            if source_time:
                timestamps.append(source_time)
            
            for time_str in row.get('reaction_created_at', []):
                reaction_time = self._parse_buzzfeed_date(time_str)
                if reaction_time:
                    timestamps.append(reaction_time)
            
            if not timestamps:
                continue
            
            # Get user metrics
            followers_counts = [row.get('user_followers_count', 0)]
            friends_counts = [row.get('user_friends_count', 0)]
            statuses_counts = [row.get('user_statuses_count', 0)]
            
            # Add reaction user metrics
            followers_counts.extend(row.get('reaction_user_followers_counts', []))
            friends_counts.extend(row.get('reaction_user_friends_counts', []))
            statuses_counts.extend(row.get('reaction_user_statuses_counts', []))
            
            # Get account creation dates
            account_created_ats = []
            source_created = self._parse_buzzfeed_date(row.get('user_created_at', ''))
            if source_created:
                account_created_ats.append(source_created)
            
            for created_at in row.get('reaction_user_created_at', []):
                reaction_created = self._parse_buzzfeed_date(created_at)
                if reaction_created:
                    account_created_ats.append(reaction_created)
            
            # Get texts and user IDs
            texts = [row.get('text', '')] + row.get('reaction_texts', [])
            user_ids = [str(row.get('user_screen_name', ''))]
            user_ids.extend([str(name) for name in row.get('reaction_user_screen_names', [])])
            
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