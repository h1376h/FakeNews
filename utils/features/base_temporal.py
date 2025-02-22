from .base import FeatureExtractor
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from sklearn.linear_model import LinearRegression, HuberRegressor

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
        if not times or not values or len(times) < 2:
            return 0.0
            
        try:
            # Remove any invalid values
            valid_pairs = [(t, v) for t, v in zip(times, values) 
                          if t is not None and v is not None and not np.isnan(v)]
            if len(valid_pairs) < 2:
                return 0.0
                
            times, values = zip(*valid_pairs)
            
            # Convert times to minutes from first tweet
            base_time = min(times)
            minutes = [(t - base_time).total_seconds() / 60.0 for t in times]
            
            # Convert to numpy arrays and reshape for sklearn
            X = np.array(minutes).reshape(-1, 1)
            
            # Handle zero/negative values in log space
            min_positive = min([v for v in values if v > 0]) if any(v > 0 for v in values) else 1.0
            y = np.log([max(v, min_positive/10) for v in values])
            
            # Scale the features for better convergence
            X_mean = np.mean(X)
            X_std = np.std(X) if np.std(X) != 0 else 1
            X_scaled = (X - X_mean) / X_std
            
            # Use robust regression if requested and we have enough points
            if use_robust and len(minutes) >= 3:
                model = HuberRegressor(
                    epsilon=1.35,  # Default epsilon for 95% statistical efficiency
                    max_iter=200,  # Increase max iterations
                    tol=1e-3,     # Slightly relax tolerance
                    warm_start=True  # Use warm start for better convergence
                )
            else:
                model = LinearRegression()
                
            model.fit(X_scaled, y)
            
            # Get slope and handle potential errors
            slope = model.coef_[0] if hasattr(model, 'coef_') else 0.0
            # Adjust slope for the scaling we applied
            slope = slope / X_std
            return float(slope) if not np.isnan(slope) else 0.0
            
        except Exception:
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
        """Process temporal features for a thread.
        
        This method calculates all 8 temporal features by analyzing how various metrics
        change over time throughout the thread. It supports multiple calculation methods
        for each feature based on available data.
        
        Args:
            timestamps: List of tweet timestamps
            followers_counts: Optional list of follower counts for each tweet
            friends_counts: Optional list of friend counts for each tweet
            statuses_counts: Optional list of status counts for each tweet
            account_created_ats: Optional list of account creation dates
            texts: Optional list of tweet texts
            user_ids: Optional list of user IDs
            unique_users_by_time: Optional list of unique user counts at each timestamp
            interaction_counts: Optional pre-calculated interaction counts
            interaction_pairs: Optional pre-calculated set of user interaction pairs
            
        Returns:
            Dictionary containing exactly 8 temporal features
        """
        features = {}
        
        if not timestamps or len(timestamps) < 2:
            return {
                'temporal_account_age_slope': 0.0,
                'temporal_followers_count_slope': 0.0,
                'temporal_statuses_count_slope': 0.0,
                'temporal_tweets_per_minute_slope': 0.0,
                'temporal_friends_count_slope': 0.0,
                'temporal_interaction_slope': 0.0,
                'temporal_age_tweet_diff_slope': 0.0,
                'temporal_network_density_slope': 0.0
            }
            
        # Remove any invalid timestamps
        valid_indices = [i for i, t in enumerate(timestamps) if t is not None]
        if len(valid_indices) < 2:
            return features
            
        # Sort all data by timestamp
        sorted_indices = sorted(valid_indices, key=lambda k: timestamps[k])
        timestamps = [timestamps[i] for i in sorted_indices]
        
        # Calculate tweets per minute slope using time differences
        time_diffs = [(timestamps[i] - timestamps[0]).total_seconds() / 60.0 
                      for i in range(len(timestamps))]
        features['temporal_tweets_per_minute_slope'] = self._calculate_temporal_slope(
            timestamps, 
            list(range(1, len(timestamps) + 1)),
            use_robust=True
        )
        
        # Calculate user metric slopes if data is available
        if followers_counts:
            valid_counts = [followers_counts[i] for i in sorted_indices 
                          if followers_counts[i] is not None]
            if valid_counts:
                features['temporal_followers_count_slope'] = self._calculate_temporal_slope(
                    timestamps[:len(valid_counts)], 
                    valid_counts,
                    use_robust=True
                )
            
        if friends_counts:
            valid_counts = [friends_counts[i] for i in sorted_indices 
                          if friends_counts[i] is not None]
            if valid_counts:
                features['temporal_friends_count_slope'] = self._calculate_temporal_slope(
                    timestamps[:len(valid_counts)], 
                    valid_counts,
                    use_robust=True
                )
            
        if statuses_counts:
            valid_counts = [statuses_counts[i] for i in sorted_indices 
                          if statuses_counts[i] is not None]
            if valid_counts:
                features['temporal_statuses_count_slope'] = self._calculate_temporal_slope(
                    timestamps[:len(valid_counts)], 
                    valid_counts,
                    use_robust=True
                )
        
        # Calculate account age features if creation dates are available
        if account_created_ats:
            account_created_ats = [account_created_ats[i] for i in sorted_indices]
            
            # Account age slope
            valid_pairs = [(t, c) for t, c in zip(timestamps, account_created_ats) 
                          if c is not None]
            if valid_pairs:
                times, created_ats = zip(*valid_pairs)
                account_ages = [(t - c).days for t, c in zip(times, created_ats)]
                features['temporal_account_age_slope'] = self._calculate_temporal_slope(
                    times, 
                    account_ages,
                    use_robust=True
                )
            
            # Age at tweet difference slope
            age_tweet_diffs = []
            valid_times = []
            for i in range(1, len(timestamps)):
                if account_created_ats[i] is not None:
                    diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600
                    if diff > 0:  # Only include positive time differences
                        age_tweet_diffs.append(diff)
                        valid_times.append(timestamps[i])
            
            if age_tweet_diffs:
                features['temporal_age_tweet_diff_slope'] = self._calculate_temporal_slope(
                    valid_times, 
                    age_tweet_diffs,
                    use_robust=True
                )
        
        # Calculate interaction slope using best available data
        if texts or interaction_counts or unique_users_by_time:
            texts_sorted = [texts[i] for i in sorted_indices] if texts else None
            counts_sorted = [interaction_counts[i] for i in sorted_indices] if interaction_counts else None
            users_sorted = [unique_users_by_time[i] for i in sorted_indices] if unique_users_by_time else None
            
            features['temporal_interaction_slope'] = self._calculate_interaction_slope(
                timestamps=timestamps,
                texts=texts_sorted,
                interaction_counts=counts_sorted,
                unique_users_by_time=users_sorted
            )
        
        # Calculate network density slope using best available data
        if user_ids and texts:
            # Use detailed data
            user_ids = [user_ids[i] for i in sorted_indices]
            texts = [texts[i] for i in sorted_indices]
            
            density_values = []
            for i in range(1, len(timestamps) + 1):
                density = self._calculate_network_density(
                    user_ids=user_ids[:i],
                    texts=texts[:i]
                )
                density_values.append(density)
                
        elif unique_users_by_time:
            # Use unique user counts
            density_values = []
            for users in unique_users_by_time:
                density = self._calculate_network_density(unique_users_count=users)
                density_values.append(density)
                
        elif interaction_pairs:
            # Use pre-calculated interaction pairs
            density_values = []
            for i in range(len(timestamps)):
                density = self._calculate_network_density(interaction_pairs=interaction_pairs)
                density_values.append(density)
                
        else:
            density_values = []
            
        if density_values:
            features['temporal_network_density_slope'] = self._calculate_temporal_slope(
                timestamps, 
                density_values,
                use_robust=True
            )
        
        # Ensure all features are present with default values
        default_features = {
            'temporal_account_age_slope': 0.0,
            'temporal_followers_count_slope': 0.0,
            'temporal_statuses_count_slope': 0.0,
            'temporal_tweets_per_minute_slope': 0.0,
            'temporal_friends_count_slope': 0.0,
            'temporal_interaction_slope': 0.0,
            'temporal_age_tweet_diff_slope': 0.0,
            'temporal_network_density_slope': 0.0
        }
        default_features.update(features)
        return default_features
    
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