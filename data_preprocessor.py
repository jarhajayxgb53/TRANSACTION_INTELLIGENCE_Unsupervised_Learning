"""
Data Preprocessor Module
=======================
Handles data cleaning, preprocessing, and feature scaling.

Author: Data Science Team
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    A class to handle data preprocessing, including handling missing values,
    encoding categorical variables, and feature scaling.
    
    Attributes:
        df (pd.DataFrame): Input dataframe
        df_processed (pd.DataFrame): Processed dataframe
        scaler (sklearn.preprocessing): Fitted scaler object
        numerical_cols (list): List of numerical columns
        categorical_cols (list): List of categorical columns
    """
    
    def __init__(self, df):
        """
        Initialize the DataPreprocessor.
        
        Args:
            df (pd.DataFrame): Input dataframe
        """
        self.df = df.copy()
        self.df_processed = None
        self.scaler = None
        self.numerical_cols = None
        self.categorical_cols = None
        self._identify_column_types()
    
    def _identify_column_types(self):
        """Identify numerical and categorical columns."""
        self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        logger.info(f"Identified {len(self.numerical_cols)} numerical columns")
        logger.info(f"Identified {len(self.categorical_cols)} categorical columns")
    
    def handle_missing_values(self, strategy='mean', threshold=0.5):
        """
        Handle missing values in the dataset.
        
        Args:
            strategy (str): Strategy for imputation - 'mean', 'median', 'most_frequent', 'knn'
            threshold (float): Drop columns with missing percentage > threshold (0-1)
            
        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        if self.df_processed is None:
            self.df_processed = self.df.copy()
        
        logger.info(f"Handling missing values using '{strategy}' strategy")
        
        # Drop columns with high missing percentage
        missing_pct = self.df_processed.isnull().sum() / len(self.df_processed)
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        
        if cols_to_drop:
            logger.info(f"Dropping columns with >{threshold*100:.0f}% missing: {cols_to_drop}")
            self.df_processed = self.df_processed.drop(columns=cols_to_drop)
        
        # Impute missing values
        if self.df_processed[self.numerical_cols].isnull().sum().sum() > 0:
            if strategy == 'knn':
                imputer = KNNImputer(n_neighbors=5)
            else:
                imputer = SimpleImputer(strategy=strategy)
            
            self.df_processed[self.numerical_cols] = imputer.fit_transform(
                self.df_processed[self.numerical_cols]
            )
            logger.info(f"Missing values imputed using {strategy} strategy")
        
        return self.df_processed
    
    def handle_outliers(self, method='iqr', threshold=3.0):
        """
        Handle outliers in numerical columns.
        
        Args:
            method (str): Method for outlier detection - 'iqr' or 'zscore'
            threshold (float): Threshold for zscore method (default: 3.0)
            
        Returns:
            pd.DataFrame: Dataframe with outliers handled (capped or flagged)
        """
        if self.df_processed is None:
            self.df_processed = self.df.copy()
        
        logger.info(f"Detecting outliers using {method} method")
        
        for col in self.numerical_cols:
            if method == 'iqr':
                Q1 = self.df_processed[col].quantile(0.25)
                Q3 = self.df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing
                self.df_processed[col] = self.df_processed[col].clip(lower_bound, upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((self.df_processed[col] - self.df_processed[col].mean()) / 
                                 self.df_processed[col].std())
                self.df_processed.loc[z_scores > threshold, col] = \
                    self.df_processed[col].median()
        
        logger.info("Outliers handled")
        return self.df_processed
    
    def encode_categorical_variables(self, method='onehot'):
        """
        Encode categorical variables.
        
        Args:
            method (str): Encoding method - 'onehot' or 'label'
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical variables
        """
        if self.df_processed is None:
            self.df_processed = self.df.copy()
        
        if len(self.categorical_cols) == 0:
            logger.info("No categorical columns to encode")
            return self.df_processed
        
        logger.info(f"Encoding categorical variables using {method} method")
        
        if method == 'onehot':
            self.df_processed = pd.get_dummies(
                self.df_processed,
                columns=self.categorical_cols,
                drop_first=True,
                dummy_na=False
            )
            logger.info(f"One-hot encoding applied. New shape: {self.df_processed.shape}")
        
        # Update numerical columns list
        self._identify_column_types()
        
        return self.df_processed
    
    def scale_features(self, method='standard'):
        """
        Scale numerical features.
        
        Args:
            method (str): Scaling method - 'standard', 'minmax', or 'robust'
            
        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        if self.df_processed is None:
            self.df_processed = self.df.copy()
        
        logger.info(f"Scaling features using {method} method")
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        scaled_data = self.scaler.fit_transform(self.df_processed[self.numerical_cols])
        self.df_processed[self.numerical_cols] = scaled_data
        
        logger.info("Features scaled successfully")
        logger.info(f"Scaled data statistics:\n{self.df_processed.describe()}")
        
        return self.df_processed
    
    def preprocess_pipeline(self, handle_outliers=True, scale_method='standard'):
        """
        Execute the complete preprocessing pipeline.
        
        Args:
            handle_outliers (bool): Whether to handle outliers
            scale_method (str): Method for scaling features
            
        Returns:
            pd.DataFrame: Fully preprocessed dataframe
        """
        logger.info("="*80)
        logger.info("STARTING PREPROCESSING PIPELINE")
        logger.info("="*80)
        
        # Step 1: Handle missing values
        self.handle_missing_values(strategy='mean', threshold=0.5)
        
        # Step 2: Handle outliers
        if handle_outliers:
            self.handle_outliers(method='iqr')
        
        # Step 3: Encode categorical variables
        if len(self.categorical_cols) > 0:
            self.encode_categorical_variables(method='onehot')
        
        # Step 4: Scale features
        self.scale_features(method=scale_method)
        
        logger.info("="*80)
        logger.info("PREPROCESSING PIPELINE COMPLETED")
        logger.info(f"Final shape: {self.df_processed.shape}")
        logger.info("="*80)
        
        return self.df_processed
    
    def get_processed_data(self):
        """
        Get the processed dataframe.
        
        Returns:
            pd.DataFrame: Processed dataframe
        """
        if self.df_processed is None:
            raise ValueError("Data not yet processed. Run preprocessing pipeline first.")
        
        return self.df_processed.copy()
    
    def get_scaler(self):
        """
        Get the fitted scaler object.
        
        Returns:
            sklearn.preprocessing: Fitted scaler object
        """
        if self.scaler is None:
            raise ValueError("Scaler not yet fitted. Run scale_features first.")
        
        return self.scaler
    
    def inverse_transform(self, data):
        """
        Inverse transform scaled data.
        
        Args:
            data (pd.DataFrame or np.ndarray): Scaled data
            
        Returns:
            np.ndarray: Original scale data
        """
        if self.scaler is None:
            raise ValueError("Scaler not yet fitted.")
        
        return self.scaler.inverse_transform(data)


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader('transactions.parquet')
    df = loader.load_data()
    
    # Preprocess data
    preprocessor = DataPreprocessor(df)
    df_processed = preprocessor.preprocess_pipeline(handle_outliers=True, scale_method='standard')
    
    print(f"\nProcessed data shape: {df_processed.shape}")
    print(f"Processed data statistics:\n{df_processed.describe()}")
