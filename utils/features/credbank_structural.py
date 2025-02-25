from .base_structural import BaseStructuralFeatureExtractor
import pandas as pd
from datetime import datetime
from typing import List

"""Structural features for CREDBANK dataset."""
class CredbankStructuralFeatureExtractor(BaseStructuralFeatureExtractor):
    """Features based on the structural properties of tweets in CREDBANK dataset."""
    
    def _parse_credbank_date(self, date_str: str) -> datetime:
        """Parse CREDBANK's date format to datetime object."""
        try:
            return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            return None
    
    def extract_features(self) -> pd.DataFrame:
        """Extract structural features from the CREDBANK dataset."""
        df = self.df.copy()
        
        # Initialize structural columns with NaN
        df = self._initialize_feature_columns(df, list(self.STRUCTURAL_FEATURES))
        
        # Process each row
        for idx, row in df.iterrows():
            # Get tweet texts from Reasons field
            texts = row.get('Reasons', [])
            if not isinstance(texts, list):
                texts = []
            
            # Parse timestamps
            timestamps = []
            for t in row.get('created_at_times', []):
                timestamp = self._parse_credbank_date(str(t))
                if timestamp:
                    timestamps.append(timestamp)
            
            # Get tweet IDs and reply-to IDs for conversation depth
            tweet_list = row.get('tweet_list', [])
            tweet_ids = []
            reply_to_ids = []
            
            if tweet_list:
                for tweet_tuple in tweet_list:
                    tweet_id = None
                    reply_to = None
                    
                    # Parse tweet tuple to extract ID and in-reply-to information
                    for part in tweet_tuple:
                        if 'ID=' in part:
                            tweet_id = part.split('ID=')[1]
                        elif 'in_reply_to_status_id=' in part:
                            reply_to = part.split('in_reply_to_status_id=')[1]
                    
                    if tweet_id:
                        tweet_ids.append(tweet_id)
                        reply_to_ids.append(reply_to or '')
            
            # Process features
            features = self._process_structural_features(
                texts=texts,
                timestamps=timestamps,
                tweet_ids=tweet_ids,
                reply_to_ids=reply_to_ids
            )
            
            # Store features
            for col, value in features.items():
                df.at[idx, col] = value
        
        # Handle missing values
        df = self._handle_missing_values(df, list(self.STRUCTURAL_FEATURES))

        return df 