from .base_structural import BaseStructuralFeatureExtractor
import pandas as pd
from datetime import datetime
from typing import List

"""Structural features for BuzzFeed dataset."""
class BuzzFeedStructuralFeatureExtractor(BaseStructuralFeatureExtractor):
    """Extract structural features from BuzzFeed articles and tweets"""
    
    def _parse_buzzfeed_date(self, date_str: str) -> datetime:
        """Parse BuzzFeed's date format to datetime object."""
        try:
            return datetime.strptime(str(date_str), '%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            return None
    
    def extract_features(self) -> pd.DataFrame:
        """Extract structural features from the BuzzFeed dataset."""
        df = self.df.copy()
        
        # Initialize feature columns
        df = self._initialize_feature_columns(df, list(self.STRUCTURAL_FEATURES))
        
        # Process each row
        for idx, row in df.iterrows():
            # Get all tweets and their metadata
            texts = [row.get('mainText', '')] + (row.get('reaction_texts', []) or [])
            
            # Parse timestamps
            timestamps = []
            reaction_timestamps = row.get('reaction_timestamps', [])
            for timestamp in reaction_timestamps:
                parsed_time = self._parse_buzzfeed_date(timestamp)
                if parsed_time:
                    timestamps.append(parsed_time)
            
            for t in row.get('reaction_created_at', []):
                reaction_time = self._parse_buzzfeed_date(t)
                if reaction_time:
                    timestamps.append(reaction_time)
            
            # Get tweet IDs and reply-to IDs for conversation depth
            tweet_ids = [str(row.get('id', ''))]
            reply_to_ids = ['']  # Source tweet has no reply-to
            
            try:
                tweet_ids.extend([str(id) for id in row.get('reaction_id', [])])
                reply_to_ids.extend([str(id) for id in row.get('reaction_in_reply_to_status_id', [])])
            except Exception:
                pass
            
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