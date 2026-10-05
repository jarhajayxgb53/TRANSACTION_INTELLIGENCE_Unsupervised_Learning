"""
Data Loader Module
==================
Handles loading and initial exploration of the transactions parquet dataset.

Author: Data Science Team
Date: 2024
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    A class to handle loading and exploring transaction data from parquet files.
    
    Attributes:
        file_path (str): Path to the parquet file
        df (pd.DataFrame): Loaded dataframe
    """
    
    def __init__(self, file_path='transactions.parquet'):
        """
        Initialize the DataLoader.
        
        Args:
            file_path (str): Path to the parquet file. Default: 'transactions.parquet'
        """
        self.file_path = file_path
        self.df = None
        
    def load_data(self):
        """
        Load data from parquet file.
        
        Returns:
            pd.DataFrame: Loaded dataframe
            
        Raises:
            FileNotFoundError: If the parquet file doesn't exist
        """
        try:
            self.df = pd.read_parquet(self.file_path)
            logger.info(f"Successfully loaded data from {self.file_path}")
            logger.info(f"Dataset shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            raise
    
    def get_basic_info(self):
        """
        Get basic information about the dataset.
        
        Returns:
            dict: Dictionary containing basic dataset information
        """
        if self.df is None:
            self.load_data()
        
        info = {
            'rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'dtypes': self.df.dtypes.to_dict(),
            'memory_usage_mb': self.df.memory_usage().sum() / 1024**2,
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicate_rows': self.df.duplicated().sum()
        }
        
        return info
    
    def explore_data(self):
        """
        Perform exploratory data analysis.
        
        Returns:
            dict: Dictionary containing exploration results
        """
        if self.df is None:
            self.load_data()
        
        logger.info("="*80)
        logger.info("EXPLORATORY DATA ANALYSIS")
        logger.info("="*80)
        
        # Basic stats
        logger.info(f"\n1. DATASET OVERVIEW")
        logger.info(f"   Rows: {self.df.shape[0]:,}")
        logger.info(f"   Columns: {self.df.shape[1]}")
        logger.info(f"   Memory Usage: {self.df.memory_usage().sum() / 1024**2:.2f} MB")
        
        # Missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            logger.info(f"\n2. MISSING VALUES")
            for col, count in missing[missing > 0].items():
                logger.info(f"   {col}: {count} ({count/len(self.df)*100:.2f}%)")
        else:
            logger.info(f"\n2. MISSING VALUES: None found!")
        
        # Data types
        logger.info(f"\n3. DATA TYPES")
        for col, dtype in self.df.dtypes.items():
            logger.info(f"   {col}: {dtype}")
        
        # Duplicates
        duplicates = self.df.duplicated().sum()
        logger.info(f"\n4. DUPLICATE ROWS: {duplicates}")
        
        # Statistical summary
        logger.info(f"\n5. STATISTICAL SUMMARY")
        logger.info(f"\n{self.df.describe().to_string()}")
        
        return {
            'info': self.get_basic_info(),
            'describe': self.df.describe(),
            'dtypes': self.df.dtypes
        }
    
    def get_numerical_columns(self):
        """
        Get names of numerical columns.
        
        Returns:
            list: List of numerical column names
        """
        if self.df is None:
            self.load_data()
        
        return self.df.select_dtypes(include=[np.number]).columns.tolist()
    
    def get_categorical_columns(self):
        """
        Get names of categorical columns.
        
        Returns:
            list: List of categorical column names
        """
        if self.df is None:
            self.load_data()
        
        return self.df.select_dtypes(include=['object']).columns.tolist()
    
    def get_data(self):
        """
        Get the loaded dataframe.
        
        Returns:
            pd.DataFrame: The loaded dataframe
        """
        if self.df is None:
            self.load_data()
        
        return self.df.copy()


if __name__ == "__main__":
    # Example usage
    loader = DataLoader('transactions.parquet')
    df = loader.load_data()
    info = loader.explore_data()
    
    print(f"\nNumerical columns: {loader.get_numerical_columns()}")
    print(f"Categorical columns: {loader.get_categorical_columns()}")
