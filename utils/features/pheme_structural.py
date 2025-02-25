from .base_structural import BaseStructuralFeatureExtractor
import pandas as pd
from datetime import datetime
from typing import List

"""Structural features for PHEME dataset."""
class PhemeStructuralFeatureExtractor(BaseStructuralFeatureExtractor):
    """Features based on the structural properties of tweets in PHEME dataset."""
    
    def _parse_pheme_date(self, date_str: str) -> datetime:
        """Parse PHEME's date format to datetime object."""
        try:
            return datetime.strptime(str(date_str), '%a %b %d %H:%M:%S +0000 %Y')
        except (ValueError, TypeError):
            return None
    
    def extract_features(self) -> pd.DataFrame:
        """Extract structural features from the PHEME dataset."""
        df = self.df.copy()
        
        # Initialize feature columns with NaN
        df = self._initialize_feature_columns(df, list(self.STRUCTURAL_FEATURES))
        
        # Process each row
        for idx, row in df.iterrows():
            # Get all tweets and their metadata
            texts = [row['source_tweet_text']] + row['reaction_texts']
            
            # Parse timestamps
            timestamps = []
            if row['source_tweet_created_at']:
                source_time = self._parse_pheme_date(row['source_tweet_created_at'])
                if source_time:
                    timestamps.append(source_time)
            
            for t in row['reaction_created_at']:
                reaction_time = self._parse_pheme_date(t)
                if reaction_time:
                    timestamps.append(reaction_time)
            
            # Get tweet IDs and reply-to IDs for conversation depth
            tweet_ids = [str(row['source_tweet_id'])]
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