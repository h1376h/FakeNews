from .base_user import BaseUserFeatureExtractor
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import logging

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
    
    def __init__(self, df: pd.DataFrame, include_additional: bool = False):
        """Initialize the CREDBANK user feature extractor."""
        super().__init__(df, include_additional)
    
    def _parse_credbank_date(self, date_str: str) -> datetime:
        """Parse CREDBANK's date format to datetime object."""
        if date_str is None or pd.isna(date_str):
            return None
            
        # If already a datetime object, return it
        if isinstance(date_str, datetime):
            return date_str
            
        date_str = str(date_str).strip()
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
            return pd.to_datetime(date_str, errors='coerce').to_pydatetime()
        except:
            pass
            
        # Log the failure for debugging
        logging.debug(f"Failed to parse date string: {date_str}")
        return None
    
    def extract_features(self) -> pd.DataFrame:
        """Extract user-based features from the CREDBANK dataset."""
        df = self.df.copy()
        
        # Initialize user columns with NaN
        df = self._initialize_feature_columns(df, list(self.features_to_extract))
        
        # Process each row
        for idx, row in df.iterrows():
            # Get tweet times
            tweet_times = []
            
            # Try created_at_times
            if 'created_at_times' in row:
                created_at_times = row['created_at_times']
                if isinstance(created_at_times, (list, np.ndarray)):
                    for t in created_at_times:
                        if pd.isna(t):
                            continue
                        timestamp = self._parse_credbank_date(t)
                        if timestamp:
                            tweet_times.append(timestamp)
                elif isinstance(created_at_times, str):
                    try:
                        times_list = eval(created_at_times)
                        if isinstance(times_list, (list, np.ndarray)):
                            for t in times_list:
                                if pd.isna(t):
                                    continue
                                timestamp = self._parse_credbank_date(t)
                                if timestamp:
                                    tweet_times.append(timestamp)
                    except:
                        timestamp = self._parse_credbank_date(created_at_times)
                        if timestamp:
                            tweet_times.append(timestamp)
            
            # Try source_tweet_created_at
            if 'source_tweet_created_at' in row and not pd.isna(row['source_tweet_created_at']):
                source_time = self._parse_credbank_date(row['source_tweet_created_at'])
                if source_time:
                    tweet_times.append(source_time)
            
            # Try reaction_created_at
            if 'reaction_created_at' in row and not pd.isna(row['reaction_created_at']):
                reaction_times = row['reaction_created_at']
                if isinstance(reaction_times, (list, np.ndarray)):
                    for t in reaction_times:
                        if pd.isna(t):
                            continue
                        timestamp = self._parse_credbank_date(t)
                        if timestamp:
                            tweet_times.append(timestamp)
                elif isinstance(reaction_times, str):
                    try:
                        times_list = eval(reaction_times)
                        if isinstance(times_list, (list, np.ndarray)):
                            for t in times_list:
                                if pd.isna(t):
                                    continue
                                timestamp = self._parse_credbank_date(t)
                                if timestamp:
                                    tweet_times.append(timestamp)
                    except:
                        timestamp = self._parse_credbank_date(reaction_times)
                        if timestamp:
                            tweet_times.append(timestamp)
            
            # Get user metrics - try different column formats
            num_tweets = len(tweet_times) if tweet_times else 1
            
            # Initialize with default values
            followers_counts = []
            friends_counts = []
            statuses_counts = []
            verified_flags = []
            
            # Try source tweet user metrics
            if 'source_tweet_user_followers_count' in row:
                source_followers = row['source_tweet_user_followers_count']
                if not pd.isna(source_followers):
                    followers_counts.append(source_followers)
            
            if 'source_tweet_user_friends_count' in row:
                source_friends = row['source_tweet_user_friends_count']
                if not pd.isna(source_friends):
                    friends_counts.append(source_friends)
            
            if 'source_tweet_user_statuses_count' in row:
                source_statuses = row['source_tweet_user_statuses_count']
                if not pd.isna(source_statuses):
                    statuses_counts.append(source_statuses)
            
            if 'source_tweet_user_verified' in row:
                source_verified = row['source_tweet_user_verified']
                if not pd.isna(source_verified):
                    verified_flags.append(bool(source_verified))
            
            # Try arrays of user metrics
            if 'followers_counts' in row:
                followers_data = row['followers_counts']
                if isinstance(followers_data, (list, np.ndarray)):
                    followers_counts.extend([f for f in followers_data if not pd.isna(f)])
                elif isinstance(followers_data, str):
                    try:
                        followers_list = eval(followers_data)
                        if isinstance(followers_list, (list, np.ndarray)):
                            followers_counts.extend([f for f in followers_list if not pd.isna(f)])
                    except:
                        pass
            
            if 'friends_counts' in row:
                friends_data = row['friends_counts']
                if isinstance(friends_data, (list, np.ndarray)):
                    friends_counts.extend([f for f in friends_data if not pd.isna(f)])
                elif isinstance(friends_data, str):
                    try:
                        friends_list = eval(friends_data)
                        if isinstance(friends_list, (list, np.ndarray)):
                            friends_counts.extend([f for f in friends_list if not pd.isna(f)])
                    except:
                        pass
            
            if 'statuses_counts' in row:
                statuses_data = row['statuses_counts']
                if isinstance(statuses_data, (list, np.ndarray)):
                    statuses_counts.extend([s for s in statuses_data if not pd.isna(s)])
                elif isinstance(statuses_data, str):
                    try:
                        statuses_list = eval(statuses_data)
                        if isinstance(statuses_list, (list, np.ndarray)):
                            statuses_counts.extend([s for s in statuses_list if not pd.isna(s)])
                    except:
                        pass
            
            if 'verified_flags' in row:
                verified_data = row['verified_flags']
                if isinstance(verified_data, (list, np.ndarray)):
                    verified_flags.extend([bool(v) for v in verified_data if not pd.isna(v)])
                elif isinstance(verified_data, str):
                    try:
                        verified_list = eval(verified_data)
                        if isinstance(verified_list, (list, np.ndarray)):
                            verified_flags.extend([bool(v) for v in verified_list if not pd.isna(v)])
                    except:
                        pass
            
            # Try reaction user metrics
            if 'reaction_user_followers_counts' in row:
                reaction_followers = row['reaction_user_followers_counts']
                if isinstance(reaction_followers, (list, np.ndarray)):
                    followers_counts.extend([f for f in reaction_followers if not pd.isna(f)])
                elif isinstance(reaction_followers, str):
                    try:
                        followers_list = eval(reaction_followers)
                        if isinstance(followers_list, (list, np.ndarray)):
                            followers_counts.extend([f for f in followers_list if not pd.isna(f)])
                    except:
                        pass
            
            if 'reaction_user_friends_counts' in row:
                reaction_friends = row['reaction_user_friends_counts']
                if isinstance(reaction_friends, (list, np.ndarray)):
                    friends_counts.extend([f for f in reaction_friends if not pd.isna(f)])
                elif isinstance(reaction_friends, str):
                    try:
                        friends_list = eval(reaction_friends)
                        if isinstance(friends_list, (list, np.ndarray)):
                            friends_counts.extend([f for f in friends_list if not pd.isna(f)])
                    except:
                        pass
            
            if 'reaction_user_statuses_counts' in row:
                reaction_statuses = row['reaction_user_statuses_counts']
                if isinstance(reaction_statuses, (list, np.ndarray)):
                    statuses_counts.extend([s for s in reaction_statuses if not pd.isna(s)])
                elif isinstance(reaction_statuses, str):
                    try:
                        statuses_list = eval(reaction_statuses)
                        if isinstance(statuses_list, (list, np.ndarray)):
                            statuses_counts.extend([s for s in statuses_list if not pd.isna(s)])
                    except:
                        pass
            
            if 'reaction_user_verified_flags' in row:
                reaction_verified = row['reaction_user_verified_flags']
                if isinstance(reaction_verified, (list, np.ndarray)):
                    verified_flags.extend([bool(v) for v in reaction_verified if not pd.isna(v)])
                elif isinstance(reaction_verified, str):
                    try:
                        verified_list = eval(reaction_verified)
                        if isinstance(verified_list, (list, np.ndarray)):
                            verified_flags.extend([bool(v) for v in verified_list if not pd.isna(v)])
                    except:
                        pass
            
            if len(verified_flags) < num_tweets:
                verified_flags.extend([False] * (num_tweets - len(verified_flags)))
            
            # Get account creation dates (using Twitter's founding date as default)
            twitter_founding = datetime(2006, 3, 21)
            account_created_ats = []
            
            # Try source_tweet_user_created_at
            if 'source_tweet_user_created_at' in row and not pd.isna(row['source_tweet_user_created_at']):
                source_created = self._parse_credbank_date(row['source_tweet_user_created_at'])
                if source_created:
                    account_created_ats.append(source_created)
            
            # Try account_created_ats array
            if 'account_created_ats' in row:
                account_created_data = row['account_created_ats']
                if isinstance(account_created_data, (list, np.ndarray)):
                    for t in account_created_data:
                        if pd.isna(t):
                            continue
                        created_at = self._parse_credbank_date(t)
                        if created_at:
                            account_created_ats.append(created_at)
                elif isinstance(account_created_data, str):
                    try:
                        created_list = eval(account_created_data)
                        if isinstance(created_list, (list, np.ndarray)):
                            for t in created_list:
                                if pd.isna(t):
                                    continue
                                created_at = self._parse_credbank_date(t)
                                if created_at:
                                    account_created_ats.append(created_at)
                    except:
                        created_at = self._parse_credbank_date(account_created_data)
                        if created_at:
                            account_created_ats.append(created_at)
            
            # Fill with default values if needed
            if not account_created_ats:
                account_created_ats = [twitter_founding] * num_tweets
            elif len(account_created_ats) < num_tweets:
                account_created_ats.extend([account_created_ats[0] if account_created_ats else twitter_founding] * 
                                          (num_tweets - len(account_created_ats)))
            
            # Get user IDs
            user_ids = []
            source_user_id = row.get('source_tweet_user_id')
            if source_user_id and not pd.isna(source_user_id):
                user_ids.append(str(source_user_id))
            
            # Get reaction user IDs
            reaction_user_ids = []
            if 'reaction_user_ids' in row:
                reaction_data = row['reaction_user_ids']
                if isinstance(reaction_data, (list, np.ndarray)):
                    reaction_user_ids = [str(u) for u in reaction_data if not pd.isna(u)]
                elif isinstance(reaction_data, str):
                    try:
                        user_ids_list = eval(reaction_data)
                        if isinstance(user_ids_list, (list, np.ndarray)):
                            reaction_user_ids = [str(u) for u in user_ids_list if not pd.isna(u)]
                    except:
                        pass
            
            user_ids.extend(reaction_user_ids)
            
            # If we still don't have user IDs, try user_ids column
            if not user_ids and 'user_ids' in row:
                user_ids_data = row['user_ids']
                if isinstance(user_ids_data, (list, np.ndarray)):
                    user_ids = [str(u) for u in user_ids_data if not pd.isna(u)]
                elif isinstance(user_ids_data, str):
                    try:
                        user_ids_list = eval(user_ids_data)
                        if isinstance(user_ids_list, (list, np.ndarray)):
                            user_ids = [str(u) for u in user_ids_list if not pd.isna(u)]
                        else:
                            user_ids = [str(user_ids_data)]
                    except:
                        user_ids = [str(user_ids_data)]
            
            # Process user features with available data
            features = self._process_user_features(
                followers_counts=followers_counts if followers_counts else None,
                friends_counts=friends_counts if friends_counts else None,
                statuses_counts=statuses_counts if statuses_counts else None,
                verified_flags=verified_flags if verified_flags else None,
                account_created_ats=account_created_ats if account_created_ats else None,
                tweet_times=tweet_times if tweet_times else None,
                user_ids=user_ids if user_ids else None
            )
            
            # Store features
            for col, value in features.items():
                df.at[idx, col] = value
        
        # Handle missing values
        df = self._handle_missing_values(df, list(self.features_to_extract))
        
        return df 