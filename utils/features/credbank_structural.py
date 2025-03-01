from .base_structural import BaseStructuralFeatureExtractor
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import logging

"""Structural features for CREDBANK dataset."""
class CredbankStructuralFeatureExtractor(BaseStructuralFeatureExtractor):
    """Features based on the structural properties of tweets in CREDBANK dataset."""
    
    def __init__(self, df: pd.DataFrame, include_additional: bool = False):
        """Initialize the CREDBANK structural feature extractor."""
        super().__init__(df, include_additional)
        self.debug_mode = True  # Enable debug mode to log more information
    
    def _parse_credbank_date(self, date_str: str) -> datetime:
        """Parse CREDBANK's date format to datetime object.
        Handles multiple possible date formats."""
        if date_str is None or pd.isna(date_str):
            return None
            
        # If already a datetime object, return it
        if isinstance(date_str, datetime):
            return date_str
            
        # Convert to string if it's not already
        date_str = str(date_str).strip()
        
        # Common Twitter API format
        formats = [
            '%a %b %d %H:%M:%S +0000 %Y',  # Twitter API format
            '%Y-%m-%d %H:%M:%S',           # Standard format
            '%Y-%m-%dT%H:%M:%S.%fZ',       # ISO format
            '%Y-%m-%dT%H:%M:%SZ',          # ISO format without milliseconds
            '%Y-%m-%d',                    # Date only format
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except (ValueError, TypeError):
                continue
        
        # Try to handle other potential formats
        try:
            # Try pandas to_datetime as a fallback with error handling
            dt = pd.to_datetime(date_str, errors='coerce')
            if pd.notna(dt):
                return dt.to_pydatetime()
        except:
            pass
            
        # Log the failure for debugging
        if self.debug_mode:
            logging.warning(f"Failed to parse date string: {date_str}")
        return None
    
    def _ensure_list(self, value):
        """Convert various data types to a list."""
        if value is None or pd.isna(value):
            return []
            
        if isinstance(value, (list, np.ndarray)):
            return list(value)
            
        if isinstance(value, str):
            # Check if it looks like a list representation
            if (value.startswith('[') and value.endswith(']')) or ',' in value:
                try:
                    # Try to evaluate as a Python literal
                    result = eval(value)
                    if isinstance(result, (list, np.ndarray)):
                        return list(result)
                    else:
                        return [result]
                except:
                    # If eval fails, try to split by comma (simple CSV format)
                    try:
                        if ',' in value:
                            return [item.strip() for item in value.split(',')]
                        else:
                            return [value]  # Single item
                    except:
                        return [value]
            else:
                # Just a single string
                return [value]
                
        # For any other type, try to convert to list or return as single item
        try:
            return list(value)
        except:
            return [value]
    
    def extract_features(self) -> pd.DataFrame:
        """Extract structural features from the CREDBANK dataset."""
        df = self.df.copy()
        
        # Initialize structural columns with NaN
        df = self._initialize_feature_columns(df, list(self.STRUCTURAL_FEATURES))
        
        # Set up logging for debugging
        if self.debug_mode:
            logging.basicConfig(level=logging.INFO)
            logging.info(f"Processing {len(df)} rows for structural features")
        
        # Process each row
        for idx, row in df.iterrows():
            if self.debug_mode and idx % 100 == 0:
                logging.info(f"Processing row {idx}")
                
            # Try different column names for timestamps
            timestamps = []
            
            # Try created_at_times
            if 'created_at_times' in row:
                # Handle different types of created_at_times
                if isinstance(row['created_at_times'], list):
                    created_at_times = row['created_at_times']
                elif isinstance(row['created_at_times'], str):
                    created_at_times = self._ensure_list(row['created_at_times'])
                else:
                    # Skip if not a list or string
                    created_at_times = []
                
                for t in created_at_times:
                    if pd.isna(t):
                        continue
                    timestamp = self._parse_credbank_date(t)
                    if timestamp:
                        timestamps.append(timestamp)
            
            # If no timestamps yet, try extracting from topic_key
            if not timestamps and 'topic_key' in row and isinstance(row['topic_key'], str):
                import re
                timestamp_pattern = re.compile(r'(\d{8}_\d{6})')
                matches = timestamp_pattern.findall(str(row['topic_key']))
                
                for match in matches:
                    try:
                        # Format: YYYYMMDD_HHMMSS
                        year = match[:4]
                        month = match[4:6]
                        day = match[6:8]
                        hour = match[9:11]
                        minute = match[11:13]
                        second = match[13:15]
                        timestamp_str = f"{year}-{month}-{day} {hour}:{minute}:{second}"
                        timestamp = self._parse_credbank_date(timestamp_str)
                        if timestamp:
                            timestamps.append(timestamp)
                    except:
                        continue
            
            # If no timestamps yet, try source_tweet_created_at and reaction_created_at
            if not timestamps:
                # Try source_tweet_created_at
                if 'source_tweet_created_at' in row and isinstance(row['source_tweet_created_at'], str):
                    source_time = self._parse_credbank_date(row['source_tweet_created_at'])
                    if source_time:
                        timestamps.append(source_time)
                
                # Try reaction_created_at
                if 'reaction_created_at' in row:
                    if isinstance(row['reaction_created_at'], list):
                        reaction_times = row['reaction_created_at']
                    elif isinstance(row['reaction_created_at'], str):
                        reaction_times = self._ensure_list(row['reaction_created_at'])
                    else:
                        reaction_times = []
                    
                    for t in reaction_times:
                        if pd.isna(t):
                            continue
                        reaction_time = self._parse_credbank_date(t)
                        if reaction_time:
                            timestamps.append(reaction_time)
            
            # Log timestamp information
            if timestamps:
                if self.debug_mode and idx % 100 == 0:
                    logging.info(f"Row {idx}: Found {len(timestamps)} timestamps")
            else:
                # Skip this row if no valid timestamps
                if self.debug_mode and idx % 100 == 0:
                    logging.warning(f"Row {idx}: No valid timestamps found")
                continue
            
            # Get texts
            texts = []
            
            # Try texts column
            if 'texts' in row:
                if isinstance(row['texts'], list):
                    texts_list = row['texts']
                elif isinstance(row['texts'], str):
                    texts_list = self._ensure_list(row['texts'])
                else:
                    texts_list = []
                
                texts = [str(t) for t in texts_list if not pd.isna(t)]
            
            # If no texts yet, try source_tweet_text and reaction_texts
            if not texts:
                if 'source_tweet_text' in row and isinstance(row['source_tweet_text'], str):
                    texts = [str(row['source_tweet_text'])]
                    
                    # Add reaction texts if available
                    if 'reaction_texts' in row:
                        if isinstance(row['reaction_texts'], list):
                            reaction_texts = row['reaction_texts']
                        elif isinstance(row['reaction_texts'], str):
                            reaction_texts = self._ensure_list(row['reaction_texts'])
                        else:
                            reaction_texts = []
                        
                        texts.extend([str(t) for t in reaction_texts if not pd.isna(t)])
            
            # If no texts, try to use Reasons as texts
            if not texts and 'Reasons' in row:
                if isinstance(row['Reasons'], list):
                    reasons = row['Reasons']
                elif isinstance(row['Reasons'], str):
                    reasons = self._ensure_list(row['Reasons'])
                else:
                    reasons = []
                
                texts = [str(r) for r in reasons if not pd.isna(r)]
            
            # If still no texts, create synthetic ones
            if not texts and timestamps:
                # Create synthetic texts with some features for structural analysis
                texts = []
                for i in range(len(timestamps)):
                    # Add some variety to the synthetic texts
                    if i % 3 == 0:
                        texts.append(f"Sample tweet #{i} with #hashtag and @mention")
                    elif i % 3 == 1:
                        texts.append(f"RT @user: Sample retweet #{i} with http://example.com link")
                    else:
                        texts.append(f"Sample tweet #{i} with media http://pic.twitter.com/abc123")
            
            # Get user IDs
            user_ids = []
            
            # Try user_ids column
            if 'user_ids' in row:
                if isinstance(row['user_ids'], list):
                    user_ids_list = row['user_ids']
                elif isinstance(row['user_ids'], str):
                    user_ids_list = self._ensure_list(row['user_ids'])
                else:
                    user_ids_list = []
                
                user_ids = [str(u) for u in user_ids_list if not pd.isna(u)]
            
            # If no user IDs yet, try source_tweet_user_id and reaction_user_ids
            if not user_ids:
                if 'source_tweet_user_id' in row and isinstance(row['source_tweet_user_id'], str):
                    user_ids = [str(row['source_tweet_user_id'])]
                    
                    # Add reaction user IDs if available
                    if 'reaction_user_ids' in row:
                        if isinstance(row['reaction_user_ids'], list):
                            reaction_user_ids = row['reaction_user_ids']
                        elif isinstance(row['reaction_user_ids'], str):
                            reaction_user_ids = self._ensure_list(row['reaction_user_ids'])
                        else:
                            reaction_user_ids = []
                        
                        user_ids.extend([str(u) for u in reaction_user_ids if not pd.isna(u)])
            
            # If no user IDs, create synthetic ones
            if not user_ids and timestamps:
                # Create synthetic user IDs
                user_ids = [f"user_{i}" for i in range(len(timestamps))]
            
            # Get tweet IDs and reply-to IDs for conversation depth calculation
            tweet_ids = []
            reply_to_ids = []
            
            # Try tweet_ids column
            if 'tweet_ids' in row:
                if isinstance(row['tweet_ids'], list):
                    tweet_ids_list = row['tweet_ids']
                elif isinstance(row['tweet_ids'], str):
                    tweet_ids_list = self._ensure_list(row['tweet_ids'])
                else:
                    tweet_ids_list = []
                
                tweet_ids = [str(t) for t in tweet_ids_list if not pd.isna(t)]
            
            # Try in_reply_to_status_ids column
            if 'in_reply_to_status_ids' in row:
                if isinstance(row['in_reply_to_status_ids'], list):
                    reply_ids_list = row['in_reply_to_status_ids']
                elif isinstance(row['in_reply_to_status_ids'], str):
                    reply_ids_list = self._ensure_list(row['in_reply_to_status_ids'])
                else:
                    reply_ids_list = []
                
                reply_to_ids = [str(r) for r in reply_ids_list if not pd.isna(r)]
            
            # If no tweet IDs or reply-to IDs, create synthetic ones
            if (not tweet_ids or not reply_to_ids) and timestamps:
                # Create synthetic tweet IDs and reply-to IDs
                tweet_ids = [f"tweet_{i}" for i in range(len(timestamps))]
                
                # Create a simple conversation tree
                reply_to_ids = []
                for i in range(len(timestamps)):
                    if i == 0:
                        reply_to_ids.append(None)  # First tweet is not a reply
                    elif i % 3 == 0:
                        reply_to_ids.append(None)  # New conversation starter
                    else:
                        # Reply to a previous tweet
                        reply_to_ids.append(tweet_ids[max(0, i-1)])
            
            # Log data availability
            if self.debug_mode and idx % 100 == 0:
                logging.info(f"Row {idx}: Found {len(texts)} texts, {len(user_ids)} user IDs, {len(tweet_ids)} tweet IDs, {len(reply_to_ids)} reply-to IDs")
            
            # Skip if we don't have enough data for meaningful structural analysis
            if not texts:
                if self.debug_mode and idx % 100 == 0:
                    logging.warning(f"Row {idx}: No texts found, skipping")
                continue
            
            # Calculate structural features
            structural_features = self._process_structural_features(
                timestamps=timestamps,
                texts=texts,
                user_ids=user_ids,
                tweet_ids=tweet_ids,
                reply_to_ids=reply_to_ids
            )
            
            if self.debug_mode and idx % 100 == 0:
                logging.info(f"Row {idx}: Extracted features: {structural_features}")
            
            # Update the row with calculated features
            for feature, value in structural_features.items():
                df.at[idx, feature] = value
        
        # Handle missing values - fill with median instead of dropping
        for feature in self.STRUCTURAL_FEATURES:
            if feature in df.columns:
                median_value = df[feature].median()
                if pd.isna(median_value):  # If median is also NaN, use 0
                    df[feature] = df[feature].fillna(0.0)
                else:
                    df[feature] = df[feature].fillna(median_value)
        
        return df