from .base import FeatureExtractor
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from sklearn.linear_model import LinearRegression, HuberRegressor
import logging

class BaseTemporalFeatureExtractor(FeatureExtractor):
    """Base class for temporal feature extractors with enhanced validation and computation.
    
    Supports 7 paper features and 1 additional feature:
    Paper features (39-45):
    1. temporal_account_age_slope: Trend in user account ages over time
    2. temporal_followers_count_slope: Trend in follower counts over time
    3. temporal_statuses_count_slope: Trend in status counts over time
    4. temporal_tweets_per_minute_slope: Trend in tweet frequency over time
    5. temporal_friends_count_slope: Trend in friend counts over time
    6. temporal_interaction_slope: Trend in user interactions over time
    7. temporal_age_tweet_diff_slope: Trend in time between tweets over time
    
    Additional features:
    8. temporal_network_density_slope: Trend in network density over time
    
    Each feature has multiple calculation methods depending on available data.
    The most accurate method will be used based on what data is provided.
    
    Attributes:
        PAPER_FEATURES: Set of temporal features from the original paper
        ADDITIONAL_FEATURES: Set of additional temporal features
    """
    
    PAPER_FEATURES = {
        'temporal_account_age_slope',
        'temporal_followers_count_slope',
        'temporal_statuses_count_slope',
        'temporal_tweets_per_minute_slope',
        'temporal_friends_count_slope',
        'temporal_interaction_slope',
        'temporal_age_tweet_diff_slope'
    }
    
    ADDITIONAL_FEATURES = {
        'temporal_network_density_slope'
    }
    
    @property
    def TEMPORAL_FEATURES(self):
        """Get all temporal features based on include_additional flag."""
        features = self.PAPER_FEATURES.copy()
        if getattr(self, '_include_additional', False):
            features.update(self.ADDITIONAL_FEATURES)
        return features
    
    def __init__(self, df: pd.DataFrame, include_additional: bool = False):
        """Initialize the feature extractor.
        
        Args:
            df: DataFrame containing the dataset
            include_additional: Whether to include additional features not in the paper
        """
        super().__init__(df)
        self._include_additional = include_additional
    
    def _calculate_temporal_slope(self, times: List[datetime], values: List[float], 
                                use_robust: bool = True) -> float:
        """Calculate the slope of a feature's values over time in log space.
        
        This method fits a regression to the log-transformed values over time
        to capture exponential growth/decay patterns in the data. It can use
        either standard linear regression or robust regression (Huber) to handle
        outliers better.
        
        Args:
            times: List of datetime objects representing when each value was observed
            values: List of feature values corresponding to each time
            use_robust: Whether to use robust regression (Huber) instead of standard OLS
            
        Returns:
            float: Slope of the regression in log space. A positive slope indicates
                  increasing values over time, while a negative slope indicates decreasing values.
                  Zero is returned if there are insufficient data points or an error occurs.
        """
        if not times or not values or len(times) < 2 or len(values) < 2:
            return 0.0
            
        try:
            # Ensure times are datetime objects
            valid_times = [t for t in times if isinstance(t, datetime)]
            if len(valid_times) < 2:
                return 0.0
                
            # Remove any invalid values and sort by time
            valid_pairs = [(t, v) for t, v in zip(valid_times, values) 
                          if t is not None and v is not None and not np.isnan(v) and isinstance(t, datetime) and v > 0]
            if len(valid_pairs) < 2:
                # If we don't have enough positive values, try without log transform
                valid_pairs = [(t, v) for t, v in zip(valid_times, values) 
                              if t is not None and v is not None and not np.isnan(v) and isinstance(t, datetime)]
                if len(valid_pairs) < 2:
                    return 0.0
                    
            valid_pairs.sort(key=lambda x: x[0])  # Sort by timestamp
            times, values = zip(*valid_pairs)
            
            # Convert times to seconds since the first timestamp
            reference_time = times[0]
            time_seconds = [(t - reference_time).total_seconds() for t in times]
            
            # Check if all values are the same
            if all(v == values[0] for v in values):
                return 0.0  # No change over time
                
            # Check if all times are the same
            if all(t == time_seconds[0] for t in time_seconds):
                return 0.0  # No time difference
                
            # Prepare data for regression
            X = np.array(time_seconds).reshape(-1, 1)
            
            # Check if all values are positive for log transform
            all_positive = all(v > 0 for v in values)
            
            if all_positive:
                # Log transform for exponential trends
                y = np.log(np.array(values))
            else:
                # Use raw values if log transform not possible
                y = np.array(values)
            
            # Fit regression model
            if use_robust and len(valid_pairs) >= 3:  # Huber needs at least 3 points
                try:
                    model = HuberRegressor(epsilon=1.35, max_iter=100)
                    model.fit(X, y)
                    slope = model.coef_[0]
                except:
                    # Fall back to standard linear regression
                    model = LinearRegression()
                    model.fit(X, y)
                    slope = model.coef_[0]
            else:
                model = LinearRegression()
                model.fit(X, y)
                slope = model.coef_[0]
            
            # Scale the slope to be per day instead of per second
            seconds_per_day = 86400
            slope_per_day = slope * seconds_per_day
            
            # Limit extreme values
            max_slope = 10.0  # Maximum reasonable slope value
            if abs(slope_per_day) > max_slope:
                slope_per_day = max_slope * (1 if slope_per_day > 0 else -1)
                
            return slope_per_day
            
        except Exception as e:
            logging.warning(f"Error calculating temporal slope: {str(e)}")
            return 0.0
    
    def _calculate_network_density(self, 
                                 user_ids: Optional[List[str]] = None,
                                 texts: Optional[List[str]] = None,
                                 unique_users_count: Optional[int] = None,
                                 interaction_pairs: Optional[Set[Tuple[str, str]]] = None) -> float:
        """Calculate network density based on available user interaction data.
        
        This method supports multiple ways to calculate network density:
        1. From user IDs and texts (most detailed)
        2. From pre-calculated interaction pairs
        3. From unique user count only (least detailed)
        
        Args:
            user_ids: Optional list of user IDs for each tweet
            texts: Optional list of tweet texts to analyze for mentions/interactions
            unique_users_count: Optional count of unique users if detailed data unavailable
            interaction_pairs: Optional pre-calculated set of user interaction pairs
            
        Returns:
            float: Network density value between 0 and 1
        """
        if interaction_pairs is not None:
            # Use pre-calculated interaction pairs
            num_users = len({user for pair in interaction_pairs for user in pair})
            if num_users > 1:
                max_possible = num_users * (num_users - 1)
                return len(interaction_pairs) / max_possible
            return 0.0
            
        if user_ids and texts:
            # Calculate from detailed data
            unique_users = set(user_id for user_id in user_ids if user_id)
            interactions = set()
            
            for user_id, text in zip(user_ids, texts):
                if not user_id or not text:
                    continue
                    
                # Add mentions as interactions
                mentions = [word.strip('@') for word in str(text).split() 
                          if word.startswith('@') and len(word) > 1]
                for mention in mentions:
                    if mention in unique_users and mention != user_id:  # Avoid self-mentions
                        interactions.add((user_id, mention))
                
                # Add retweet interactions
                if str(text).startswith('RT @'):
                    try:
                        retweeted_user = str(text).split('@')[1].split(':')[0].strip()
                        if retweeted_user in unique_users and retweeted_user != user_id:
                            interactions.add((user_id, retweeted_user))
                    except IndexError:
                        pass
            
            num_users = len(unique_users)
            if num_users > 1:
                max_possible = num_users * (num_users - 1)
                return len(interactions) / max_possible
            return 0.0
            
        if unique_users_count and unique_users_count > 1:
            # Estimate from unique user count using a more sophisticated model
            # Assume power-law distribution of interactions
            num_users = unique_users_count
            expected_interactions = np.power(num_users, 1.5)  # Based on typical social network scaling
            max_possible = num_users * (num_users - 1)
            return min(1.0, expected_interactions / max_possible)
            
        return 0.0
    
    def _calculate_interaction_slope(self,
                                  timestamps: List[datetime],
                                  texts: Optional[List[str]] = None,
                                  interaction_counts: Optional[List[int]] = None,
                                  unique_users_by_time: Optional[List[int]] = None) -> float:
        """Calculate interaction slope using available data.
        
        This method supports multiple ways to calculate interaction trends:
        1. From tweet texts (most detailed)
        2. From pre-calculated interaction counts
        3. From unique users over time (least detailed)
        
        Args:
            timestamps: List of tweet timestamps
            texts: Optional list of tweet texts to analyze for interactions
            interaction_counts: Optional pre-calculated interaction counts
            unique_users_by_time: Optional counts of unique users at each time
            
        Returns:
            float: Slope of interaction trend
        """
        if texts:
            # Calculate from texts with more sophisticated interaction counting
            counts = []
            current_interactions = 0
            
            for text in texts:
                if not text:
                    counts.append(current_interactions)
                    continue
                    
                # Count mentions (weighted by uniqueness)
                mentions = [word.strip('@') for word in str(text).split() 
                          if word.startswith('@') and len(word) > 1]
                unique_mentions = len(set(mentions))
                
                # Count retweets and quotes
                is_retweet = 2 if str(text).startswith('RT @') else 0  # Weight retweets more
                has_quote = 1 if '"@' in str(text) else 0
                
                # Count reply indicators
                is_reply = 1 if str(text).strip().startswith('@') else 0
                
                current_interactions += unique_mentions + is_retweet + has_quote + is_reply
                counts.append(current_interactions)
                
            return self._calculate_temporal_slope(timestamps, counts, use_robust=True)
            
        if interaction_counts:
            # Use pre-calculated counts with robust regression
            return self._calculate_temporal_slope(timestamps, interaction_counts, use_robust=True)
            
        if unique_users_by_time:
            # Estimate from unique users with power-law scaling
            counts = []
            current_interactions = 0
            prev_users = 0
            
            for users in unique_users_by_time:
                new_users = max(0, users - prev_users)
                # Use power-law scaling for interaction estimation
                if new_users > 0:
                    current_interactions += int(np.power(new_users, 1.5))
                counts.append(current_interactions)
                prev_users = users
                
            return self._calculate_temporal_slope(timestamps, counts)
            
        return 0.0
    
    def _process_temporal_features(self, 
                                 timestamps: List[datetime],
                                 followers_counts: Optional[List[int]] = None,
                                 friends_counts: Optional[List[int]] = None,
                                 statuses_counts: Optional[List[int]] = None,
                                 account_created_ats: Optional[List[datetime]] = None,
                                 texts: Optional[List[str]] = None,
                                 user_ids: Optional[List[str]] = None,
                                 unique_users_by_time: Optional[List[int]] = None,
                                 interaction_counts: Optional[List[int]] = None,
                                 interaction_pairs: Optional[Set[Tuple[str, str]]] = None) -> Dict[str, float]:
        """Process temporal features from the dataset.
        
        This method calculates all temporal features based on the provided data.
        It handles missing data gracefully and uses the most accurate calculation
        method available based on the provided inputs.
        
        Args:
            timestamps: List of tweet timestamps
            followers_counts: List of follower counts corresponding to each timestamp
            friends_counts: List of friend counts corresponding to each timestamp
            statuses_counts: List of status counts corresponding to each timestamp
            account_created_ats: List of account creation dates
            texts: List of tweet texts
            user_ids: List of user IDs
            unique_users_by_time: List of cumulative unique users at each timestamp
            interaction_counts: List of interaction counts at each timestamp
            interaction_pairs: Set of user interaction pairs
            
        Returns:
            Dictionary mapping feature names to their calculated values
        """
        features = {}
        
        # Ensure we have timestamps
        if not timestamps or len(timestamps) < 2:
            # Return zeros for all features if we don't have enough timestamps
            for feature in self.TEMPORAL_FEATURES:
                features[feature] = 0.0
            return features
        
        # Sort all data by timestamp
        data = list(zip(timestamps, 
                       followers_counts or [None] * len(timestamps),
                       friends_counts or [None] * len(timestamps),
                       statuses_counts or [None] * len(timestamps),
                       account_created_ats or [None] * len(timestamps) if account_created_ats else [None] * len(timestamps),
                       texts or [None] * len(timestamps) if texts else [None] * len(timestamps),
                       user_ids or [None] * len(timestamps) if user_ids else [None] * len(timestamps)))
        data.sort(key=lambda x: x[0])  # Sort by timestamp
        
        # Unpack sorted data
        timestamps = [d[0] for d in data]
        followers_counts = [d[1] for d in data if d[1] is not None] if any(d[1] is not None for d in data) else None
        friends_counts = [d[2] for d in data if d[2] is not None] if any(d[2] is not None for d in data) else None
        statuses_counts = [d[3] for d in data if d[3] is not None] if any(d[3] is not None for d in data) else None
        account_created_ats = [d[4] for d in data if d[4] is not None] if any(d[4] is not None for d in data) else None
        texts = [d[5] for d in data if d[5] is not None] if any(d[5] is not None for d in data) else None
        user_ids = [d[6] for d in data if d[6] is not None] if any(d[6] is not None for d in data) else None
        
        # Calculate temporal features
        
        # 1. Account age slope
        if account_created_ats and len(account_created_ats) >= 2:
            # Calculate account ages in days at each timestamp
            account_ages = []
            valid_pairs = []
            
            for i, (ts, ac) in enumerate(zip(timestamps[:len(account_created_ats)], account_created_ats)):
                if ts and ac and isinstance(ts, datetime) and isinstance(ac, datetime):
                    age_days = (ts - ac).total_seconds() / (24 * 3600)  # Convert to days
                    if age_days >= 0:  # Ensure account was created before the tweet
                        account_ages.append(age_days)
                        valid_pairs.append((ts, age_days))
            
            if len(valid_pairs) >= 2:
                valid_timestamps, valid_ages = zip(*valid_pairs)
                features['temporal_account_age_slope'] = self._calculate_temporal_slope(
                    valid_timestamps, valid_ages
                )
            else:
                features['temporal_account_age_slope'] = 0.0
        else:
            features['temporal_account_age_slope'] = 0.0
        
        # 2. Followers count slope
        if followers_counts and len(followers_counts) >= 2:
            # Match timestamps with follower counts
            valid_pairs = []
            for i, (ts, fc) in enumerate(zip(timestamps[:len(followers_counts)], followers_counts)):
                if ts and fc is not None and isinstance(ts, datetime) and fc >= 0:
                    valid_pairs.append((ts, fc))
            
            if len(valid_pairs) >= 2:
                valid_timestamps, valid_counts = zip(*valid_pairs)
                features['temporal_followers_count_slope'] = self._calculate_temporal_slope(
                    valid_timestamps, valid_counts
                )
            else:
                features['temporal_followers_count_slope'] = 0.0
        else:
            features['temporal_followers_count_slope'] = 0.0
        
        # 3. Statuses count slope
        if statuses_counts and len(statuses_counts) >= 2:
            # Match timestamps with status counts
            valid_pairs = []
            for i, (ts, sc) in enumerate(zip(timestamps[:len(statuses_counts)], statuses_counts)):
                if ts and sc is not None and isinstance(ts, datetime) and sc >= 0:
                    valid_pairs.append((ts, sc))
            
            if len(valid_pairs) >= 2:
                valid_timestamps, valid_counts = zip(*valid_pairs)
                features['temporal_statuses_count_slope'] = self._calculate_temporal_slope(
                    valid_timestamps, valid_counts
                )
            else:
                features['temporal_statuses_count_slope'] = 0.0
        else:
            features['temporal_statuses_count_slope'] = 0.0
        
        # 4. Tweets per minute slope
        if len(timestamps) >= 2:
            # Calculate cumulative tweet counts at each timestamp
            tweet_counts = list(range(1, len(timestamps) + 1))
            
            features['temporal_tweets_per_minute_slope'] = self._calculate_temporal_slope(
                timestamps, tweet_counts
            )
        else:
            features['temporal_tweets_per_minute_slope'] = 0.0
        
        # 5. Friends count slope
        if friends_counts and len(friends_counts) >= 2:
            # Match timestamps with friend counts
            valid_pairs = []
            for i, (ts, fc) in enumerate(zip(timestamps[:len(friends_counts)], friends_counts)):
                if ts and fc is not None and isinstance(ts, datetime) and fc >= 0:
                    valid_pairs.append((ts, fc))
            
            if len(valid_pairs) >= 2:
                valid_timestamps, valid_counts = zip(*valid_pairs)
                features['temporal_friends_count_slope'] = self._calculate_temporal_slope(
                    valid_timestamps, valid_counts
                )
            else:
                features['temporal_friends_count_slope'] = 0.0
        else:
            features['temporal_friends_count_slope'] = 0.0
        
        # 6. Interaction slope
        if texts and user_ids:
            features['temporal_interaction_slope'] = self._calculate_interaction_slope(
                timestamps, texts=texts, unique_users_by_time=unique_users_by_time,
                interaction_counts=interaction_counts
            )
        else:
            features['temporal_interaction_slope'] = 0.0
        
        # 7. Age-tweet difference slope
        if account_created_ats and len(account_created_ats) >= 2:
            # Calculate time differences between consecutive tweets
            tweet_diffs = []
            valid_pairs = []
            
            for i in range(1, min(len(timestamps), len(account_created_ats))):
                if timestamps[i] and timestamps[i-1] and isinstance(timestamps[i], datetime) and isinstance(timestamps[i-1], datetime):
                    diff_minutes = (timestamps[i] - timestamps[i-1]).total_seconds() / 60
                    if diff_minutes > 0:
                        tweet_diffs.append(diff_minutes)
                        valid_pairs.append((timestamps[i], diff_minutes))
            
            if len(valid_pairs) >= 2:
                valid_timestamps, valid_diffs = zip(*valid_pairs)
                features['temporal_age_tweet_diff_slope'] = self._calculate_temporal_slope(
                    valid_timestamps, valid_diffs
                )
            else:
                features['temporal_age_tweet_diff_slope'] = 0.0
        else:
            features['temporal_age_tweet_diff_slope'] = 0.0
        
        # 8. Network density slope (additional feature)
        if self._include_additional and user_ids:
            # Calculate network density at each timestamp
            densities = []
            valid_pairs = []
            
            # Calculate cumulative unique users at each timestamp
            unique_users_set = set()
            for i, (ts, uid) in enumerate(zip(timestamps, user_ids)):
                if uid:
                    unique_users_set.add(uid)
                if ts and isinstance(ts, datetime):
                    density = self._calculate_network_density(
                        user_ids=list(unique_users_set),
                        unique_users_count=len(unique_users_set),
                        interaction_pairs=interaction_pairs
                    )
                    densities.append(density)
                    valid_pairs.append((ts, density))
            
            if len(valid_pairs) >= 2:
                valid_timestamps, valid_densities = zip(*valid_pairs)
                features['temporal_network_density_slope'] = self._calculate_temporal_slope(
                    valid_timestamps, valid_densities
                )
            else:
                features['temporal_network_density_slope'] = 0.0
        elif 'temporal_network_density_slope' in self.TEMPORAL_FEATURES:
            features['temporal_network_density_slope'] = 0.0
        
        return features
    
    def extract_features(self) -> pd.DataFrame:
        """Extract temporal features from the dataset.
        
        Returns:
            DataFrame with temporal features. If include_additional=True, includes all 8 features:
            Paper features (39-45):
            1. temporal_account_age_slope: Trend in user account ages
            2. temporal_followers_count_slope: Trend in follower counts
            3. temporal_statuses_count_slope: Trend in status counts
            4. temporal_tweets_per_minute_slope: Trend in tweet frequency
            5. temporal_friends_count_slope: Trend in friend counts
            6. temporal_interaction_slope: Trend in user interactions
            7. temporal_age_tweet_diff_slope: Trend in time between tweets
            
            Additional features (if include_additional=True):
            8. temporal_network_density_slope: Trend in network density
        """
        raise NotImplementedError("Subclasses must implement extract_features()")

    def _handle_missing_values(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """Handle missing values in feature columns.
        
        Args:
            df: DataFrame to process
            feature_columns: List of feature column names to handle
            
        Returns:
            DataFrame with missing values handled
        """
        # For each feature column, fill missing values with the median
        for col in feature_columns:
            if col in df.columns:
                # Calculate median value
                median_value = df[col].median()
                
                # If median is NaN (all values are NaN), use 0.0
                if pd.isna(median_value):
                    df[col] = df[col].fillna(0.0)
                else:
                    df[col] = df[col].fillna(median_value)
        
        return df