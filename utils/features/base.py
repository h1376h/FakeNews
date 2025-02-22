import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Set
from abc import ABC, abstractmethod

class FeatureExtractor(ABC):
    """Abstract base class for feature extraction with enhanced error handling and validation."""
    
    VALID_FILL_STRATEGIES = {'zero', 'mean', 'median'}
    
    def __init__(self, df: pd.DataFrame, fill_strategy: Optional[Union[str, Dict[str, Union[str, float]]]] = None):
        """Initialize feature extractor with input validation.
        
        Args:
            df: DataFrame to extract features from
            fill_strategy: Strategy for handling missing values. Can be:
                - None: Leave missing values as NaN (default)
                - 'zero': Fill all missing values with 0
                - Dict mapping column names to fill strategies ('zero', 'mean', 'median') or specific values
                
        Raises:
            ValueError: If df is empty
            TypeError: If df is not a pandas DataFrame
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if df.empty:
            raise ValueError("Input DataFrame is empty")
                    
        self.df = df.copy()
        self.fill_strategy = fill_strategy
        self.feature_columns: Set[str] = set()
    
    def _initialize_feature_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Initialize feature columns with NaN values and track them.
        
        Args:
            df: DataFrame to initialize columns in
            columns: List of column names to initialize
            
        Returns:
            DataFrame with initialized columns
            
        Raises:
            ValueError: If columns list is empty
        """
        if not columns:
            raise ValueError("No feature columns specified")
            
        for col in columns:
            df[col] = np.nan
            self.feature_columns.add(col)
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Handle missing values according to the specified strategy with error handling.
        
        Args:
            df: DataFrame to handle missing values in
            columns: List of column names to process
            
        Returns:
            DataFrame with handled missing values
            
        Raises:
            ValueError: If any column in columns does not exist in df
        """
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
        
        if self.fill_strategy is None:
            return df
        
        try:
            if isinstance(self.fill_strategy, str):
                if self.fill_strategy == 'zero':
                    df[columns] = df[columns].fillna(0)
                elif self.fill_strategy == 'mean':
                    for col in columns:
                        df[col] = df[col].fillna(df[col].mean())
                elif self.fill_strategy == 'median':
                    for col in columns:
                        df[col] = df[col].fillna(df[col].median())
            
            elif isinstance(self.fill_strategy, dict):
                for col in columns:
                    strategy = self.fill_strategy.get(col)
                    if strategy is None:
                        continue
                        
                    if isinstance(strategy, (int, float)):
                        df[col] = df[col].fillna(strategy)
                    elif strategy == 'zero':
                        df[col] = df[col].fillna(0)
                    elif strategy == 'mean':
                        df[col] = df[col].fillna(df[col].mean())
                    elif strategy == 'median':
                        df[col] = df[col].fillna(df[col].median())
            
            # Validate features before returning
            self.validate_features(df)
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error handling missing values: {str(e)}")
        
    @abstractmethod
    def extract_features(self) -> pd.DataFrame:
        """Extract features from the dataset.
        
        This method must be implemented by subclasses to extract their specific features.
        The implementation should:
        1. Initialize feature columns using _initialize_feature_columns
        2. Process the data to extract features
        3. Handle missing values using _handle_missing_values
        4. Return the DataFrame with all required features
        
        Returns:
            DataFrame containing the extracted features
            
        Raises:
            NotImplementedError: If not implemented by subclass
            ValueError: If feature validation fails
        """
        raise NotImplementedError("Subclasses must implement extract_features method")

    def validate_features(self, df: pd.DataFrame) -> None:
        """Validate that all required features were extracted.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If any feature columns are missing or contain all NaN values
        """
        missing_features = self.feature_columns - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")
            
        nan_features = [col for col in self.feature_columns 
                       if df[col].isna().all()]
        if nan_features:
            raise ValueError(f"Features contain all NaN values: {nan_features}")