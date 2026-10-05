"""
Dimensionality Reduction Module
================================
Handles dimensionality reduction using PCA, t-SNE, and other techniques.

Author: Data Science Team
Date: 2024
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DimensionalityReducer:
    """
    A class to perform dimensionality reduction on high-dimensional data.
    
    Attributes:
        X (np.ndarray or pd.DataFrame): Input data
        pca_full (sklearn.decomposition.PCA): Full PCA model
        pca_optimal (sklearn.decomposition.PCA): Optimal PCA model
        X_pca (np.ndarray): PCA-transformed data
        X_pca_2d (np.ndarray): 2D PCA-transformed data for visualization
        X_tsne (np.ndarray): t-SNE transformed data
    """
    
    def __init__(self, X):
        """
        Initialize the DimensionalityReducer.
        
        Args:
            X (np.ndarray or pd.DataFrame): Input data (should be scaled)
        """
        self.X = X
        self.pca_full = None
        self.pca_optimal = None
        self.X_pca = None
        self.X_pca_2d = None
        self.X_pca_3d = None
        self.X_tsne = None
    
    def apply_pca_full(self):
        """
        Apply PCA to all components to analyze variance.
        
        Returns:
            tuple: (X_pca_full, pca_model)
        """
        logger.info("Applying PCA to all components")
        
        self.pca_full = PCA()
        X_pca_full = self.pca_full.fit_transform(self.X)
        
        logger.info(f"PCA applied. Shape: {X_pca_full.shape}")
        logger.info(f"Total variance: {self.pca_full.explained_variance_ratio_.sum():.4f}")
        
        return X_pca_full, self.pca_full
    
    def get_optimal_components(self, variance_threshold=0.95):
        """
        Find optimal number of components for given variance threshold.
        
        Args:
            variance_threshold (float): Target cumulative variance (0-1)
            
        Returns:
            int: Optimal number of components
        """
        if self.pca_full is None:
            self.apply_pca_full()
        
        cumsum_var = np.cumsum(self.pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumsum_var >= variance_threshold) + 1
        
        logger.info(f"For {variance_threshold*100:.0f}% variance: {n_components} components needed")
        logger.info(f"Actual variance explained: {cumsum_var[n_components-1]:.4f}")
        
        return n_components
    
    def apply_pca_optimal(self, n_components=None, variance_threshold=0.95):
        """
        Apply PCA with optimal number of components.
        
        Args:
            n_components (int): Number of components. If None, uses variance threshold
            variance_threshold (float): Target cumulative variance if n_components is None
            
        Returns:
            np.ndarray: PCA-transformed data
        """
        if n_components is None:
            n_components = self.get_optimal_components(variance_threshold)
        
        logger.info(f"Applying PCA with {n_components} components")
        
        self.pca_optimal = PCA(n_components=min(n_components, self.X.shape[1]))
        self.X_pca = self.pca_optimal.fit_transform(self.X)
        
        logger.info(f"PCA completed. Output shape: {self.X_pca.shape}")
        logger.info(f"Explained variance: {self.pca_optimal.explained_variance_ratio_.sum():.4f}")
        
        return self.X_pca
    
    def apply_pca_2d(self):
        """
        Apply PCA to reduce to 2D for visualization.
        
        Returns:
            np.ndarray: 2D PCA-transformed data
        """
        logger.info("Applying PCA for 2D visualization")
        
        pca_2d = PCA(n_components=2)
        self.X_pca_2d = pca_2d.fit_transform(self.X)
        
        logger.info(f"2D PCA completed. Shape: {self.X_pca_2d.shape}")
        logger.info(f"Explained variance: {pca_2d.explained_variance_ratio_.sum():.4f}")
        logger.info(f"  PC1: {pca_2d.explained_variance_ratio_[0]:.4f}")
        logger.info(f"  PC2: {pca_2d.explained_variance_ratio_[1]:.4f}")
        
        return self.X_pca_2d
    
    def apply_pca_3d(self):
        """
        Apply PCA to reduce to 3D for interactive visualization.
        
        Returns:
            np.ndarray: 3D PCA-transformed data
        """
        logger.info("Applying PCA for 3D visualization")
        
        pca_3d = PCA(n_components=3)
        self.X_pca_3d = pca_3d.fit_transform(self.X)
        
        logger.info(f"3D PCA completed. Shape: {self.X_pca_3d.shape}")
        logger.info(f"Explained variance: {pca_3d.explained_variance_ratio_.sum():.4f}")
        logger.info(f"  PC1: {pca_3d.explained_variance_ratio_[0]:.4f}")
        logger.info(f"  PC2: {pca_3d.explained_variance_ratio_[1]:.4f}")
        logger.info(f"  PC3: {pca_3d.explained_variance_ratio_[2]:.4f}")
        
        return self.X_pca_3d
    
    def get_pca_variance_explained(self):
        """
        Get the explained variance ratio for each PCA component.
        
        Returns:
            tuple: (explained_variance_ratio, cumsum_variance)
        """
        if self.pca_full is None:
            self.apply_pca_full()
        
        explained_var = self.pca_full.explained_variance_ratio_
        cumsum_var = np.cumsum(explained_var)
        
        return explained_var, cumsum_var
    
    def get_pca_components(self):
        """
        Get the PCA components (loadings).
        
        Returns:
            np.ndarray: PCA components
        """
        if self.pca_optimal is None:
            raise ValueError("PCA not yet applied. Run apply_pca_optimal first.")
        
        return self.pca_optimal.components_
    
    def apply_tsne(self, n_components=2, perplexity=30, random_state=42):
        """
        Apply t-SNE for dimensionality reduction and visualization.
        
        Args:
            n_components (int): Number of components (usually 2 or 3)
            perplexity (float): Perplexity parameter (5-50)
            random_state (int): Random seed for reproducibility
            
        Returns:
            np.ndarray: t-SNE transformed data
        """
        logger.info(f"Applying t-SNE with {n_components} components")
        logger.info("This may take a while for large datasets...")
        
        # Use PCA-reduced data if available to speed up t-SNE
        data_for_tsne = self.X_pca if self.X_pca is not None else self.X
        
        tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                   random_state=random_state, n_iter=1000, verbose=1)
        self.X_tsne = tsne.fit_transform(data_for_tsne)
        
        logger.info(f"t-SNE completed. Output shape: {self.X_tsne.shape}")
        
        return self.X_tsne
    
    def get_explained_variance_dataframe(self, top_n=20):
        """
        Get explained variance as a dataframe.
        
        Args:
            top_n (int): Number of top components to return
            
        Returns:
            pd.DataFrame: Dataframe with variance information
        """
        if self.pca_full is None:
            self.apply_pca_full()
        
        explained_var = self.pca_full.explained_variance_ratio_[:top_n]
        cumsum_var = np.cumsum(explained_var)
        
        df = pd.DataFrame({
            'Component': [f'PC{i+1}' for i in range(len(explained_var))],
            'Explained_Variance': explained_var,
            'Cumulative_Variance': cumsum_var
        })
        
        return df
    
    def get_transformed_data(self, method='pca_optimal'):
        """
        Get transformed data.
        
        Args:
            method (str): Method to use - 'pca_full', 'pca_optimal', 'pca_2d', or 'tsne'
            
        Returns:
            np.ndarray: Transformed data
        """
        if method == 'pca_full':
            if self.pca_full is None:
                return self.apply_pca_full()[0]
            return self.pca_full.transform(self.X)
        
        elif method == 'pca_optimal':
            if self.X_pca is None:
                return self.apply_pca_optimal()
            return self.X_pca
        
        elif method == 'pca_2d':
            if self.X_pca_2d is None:
                return self.apply_pca_2d()
            return self.X_pca_2d
        
        elif method == 'tsne':
            if self.X_tsne is None:
                return self.apply_tsne()
            return self.X_tsne
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def inverse_transform_pca(self, X_transformed):
        """
        Inverse transform PCA-transformed data.
        
        Args:
            X_transformed (np.ndarray): PCA-transformed data
            
        Returns:
            np.ndarray: Original space data
        """
        if self.pca_optimal is None:
            raise ValueError("PCA not yet applied. Run apply_pca_optimal first.")
        
        return self.pca_optimal.inverse_transform(X_transformed)
    
    def summary(self):
        """Print a summary of the dimensionality reduction results."""
        logger.info("="*80)
        logger.info("DIMENSIONALITY REDUCTION SUMMARY")
        logger.info("="*80)
        
        if self.pca_optimal is not None:
            logger.info(f"\nPCA Optimal:")
            logger.info(f"  Input shape: {self.X.shape}")
            logger.info(f"  Output shape: {self.X_pca.shape}")
            logger.info(f"  Components: {self.pca_optimal.n_components}")
            logger.info(f"  Explained variance: {self.pca_optimal.explained_variance_ratio_.sum():.4f}")
        
        if self.X_pca_2d is not None:
            logger.info(f"\nPCA 2D:")
            logger.info(f"  Output shape: {self.X_pca_2d.shape}")
        
        if self.X_tsne is not None:
            logger.info(f"\nt-SNE:")
            logger.info(f"  Output shape: {self.X_tsne.shape}")


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    from data_preprocessor import DataPreprocessor
    
    # Load and preprocess data
    loader = DataLoader('transactions.parquet')
    df = loader.load_data()
    preprocessor = DataPreprocessor(df)
    df_processed = preprocessor.preprocess_pipeline()
    
    # Apply dimensionality reduction
    reducer = DimensionalityReducer(df_processed)
    X_pca = reducer.apply_pca_optimal(variance_threshold=0.95)
    X_pca_2d = reducer.apply_pca_2d()
    
    # Get variance information
    var_df = reducer.get_explained_variance_dataframe()
    print(var_df)
    
    # Print summary
    reducer.summary()
