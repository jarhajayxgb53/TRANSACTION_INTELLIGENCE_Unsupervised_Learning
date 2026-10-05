"""
Unsupervised Learning Models Module
===================================
Implements K-Means, Hierarchical Clustering, DBSCAN, GMM, and Isolation Forest.

Author: Data Science Team
Date: 2024
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KMeansClustering:
    """K-Means clustering model."""
    
    def __init__(self, n_clusters=3, random_state=42, n_init=10):
        """
        Initialize K-Means clustering.
        
        Args:
            n_clusters (int): Number of clusters
            random_state (int): Random seed
            n_init (int): Number of initializations
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.model = None
        self.labels = None
        self.centers = None
    
    def fit(self, X):
        """
        Fit K-Means model.
        
        Args:
            X (np.ndarray or pd.DataFrame): Input data
            
        Returns:
            self
        """
        logger.info(f"Fitting K-Means with {self.n_clusters} clusters")
        
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state,
                           n_init=self.n_init)
        self.labels = self.model.fit_predict(X)
        self.centers = self.model.cluster_centers_
        
        logger.info(f"K-Means fitted. Inertia: {self.model.inertia_:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Predict cluster labels.
        
        Args:
            X (np.ndarray or pd.DataFrame): Input data
            
        Returns:
            np.ndarray: Cluster labels
        """
        if self.model is None:
            raise ValueError("Model not yet fitted. Run fit() first.")
        
        return self.model.predict(X)
    
    def get_labels(self):
        """Get cluster labels."""
        return self.labels
    
    def get_centers(self):
        """Get cluster centers."""
        return self.centers
    
    def get_inertia(self):
        """Get inertia value."""
        return self.model.inertia_


class HierarchicalClustering:
    """Hierarchical clustering model."""
    
    def __init__(self, n_clusters=3, linkage='ward'):
        """
        Initialize Hierarchical clustering.
        
        Args:
            n_clusters (int): Number of clusters
            linkage (str): Linkage criterion - 'ward', 'complete', 'average', 'single'
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.model = None
        self.labels = None
    
    def fit(self, X):
        """
        Fit Hierarchical clustering model.
        
        Args:
            X (np.ndarray or pd.DataFrame): Input data
            
        Returns:
            self
        """
        logger.info(f"Fitting Hierarchical clustering with {self.linkage} linkage")
        
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
        self.labels = self.model.fit_predict(X)
        
        logger.info(f"Hierarchical clustering fitted. Clusters: {len(np.unique(self.labels))}")
        
        return self
    
    def predict(self, X):
        """
        Predict cluster labels (requires refitting with new data).
        
        Args:
            X (np.ndarray or pd.DataFrame): Input data
            
        Returns:
            np.ndarray: Cluster labels
        """
        # Hierarchical clustering doesn't support predict for new data
        logger.warning("Hierarchical clustering doesn't support predict. Refitting required.")
        return self.fit(X).labels
    
    def get_labels(self):
        """Get cluster labels."""
        return self.labels


class DBSCANClustering:
    """DBSCAN clustering model."""
    
    def __init__(self, eps=0.5, min_samples=5):
        """
        Initialize DBSCAN clustering.
        
        Args:
            eps (float): Maximum distance between samples
            min_samples (int): Minimum number of samples in a neighborhood
        """
        self.eps = eps
        self.min_samples = min_samples
        self.model = None
        self.labels = None
    
    def find_optimal_eps(self, X, k=5):
        """
        Find optimal epsilon using k-distance graph.
        
        Args:
            X (np.ndarray or pd.DataFrame): Input data
            k (int): k for k-NN distance
            
        Returns:
            float: Suggested epsilon value
        """
        logger.info(f"Finding optimal epsilon using k={k}-distance graph")
        
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(X)
        distances, _ = neighbors.kneighbors(X)
        distances = np.sort(distances[:, -1])
        
        # Use 90th percentile as epsilon
        suggested_eps = np.percentile(distances, 90)
        
        logger.info(f"Suggested epsilon: {suggested_eps:.4f}")
        
        return suggested_eps, distances
    
    def fit(self, X):
        """
        Fit DBSCAN model.
        
        Args:
            X (np.ndarray or pd.DataFrame): Input data
            
        Returns:
            self
        """
        logger.info(f"Fitting DBSCAN with eps={self.eps}, min_samples={self.min_samples}")
        
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels = self.model.fit_predict(X)
        
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = list(self.labels).count(-1)
        
        logger.info(f"DBSCAN fitted. Clusters: {n_clusters}, Noise points: {n_noise}")
        
        return self
    
    def predict(self, X):
        """
        Predict cluster labels (limited support).
        
        Args:
            X (np.ndarray or pd.DataFrame): Input data
            
        Returns:
            np.ndarray: Cluster labels (-1 for noise)
        """
        if self.model is None:
            raise ValueError("Model not yet fitted. Run fit() first.")
        
        # DBSCAN doesn't have a standard predict method for new data
        return self.model.fit_predict(X)
    
    def get_labels(self):
        """Get cluster labels."""
        return self.labels
    
    def get_noise_points(self):
        """Get indices of noise points."""
        return np.where(self.labels == -1)[0]


class GaussianMixtureModel:
    """Gaussian Mixture Model for clustering."""
    
    def __init__(self, n_components=3, covariance_type='full', random_state=42):
        """
        Initialize Gaussian Mixture Model.
        
        Args:
            n_components (int): Number of mixture components
            covariance_type (str): Type of covariance - 'full', 'tied', 'diag', 'spherical'
            random_state (int): Random seed
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = None
        self.labels = None
        self.probabilities = None
    
    def find_optimal_components(self, X, max_components=10):
        """
        Find optimal number of components using BIC and AIC.
        
        Args:
            X (np.ndarray or pd.DataFrame): Input data
            max_components (int): Maximum number of components to test
            
        Returns:
            dict: Dictionary with optimal components and scores
        """
        logger.info(f"Finding optimal components (testing 1-{max_components})")
        
        bic_scores = []
        aic_scores = []
        
        for n in range(1, max_components + 1):
            gmm = GaussianMixture(n_components=n, covariance_type=self.covariance_type,
                                 random_state=self.random_state, n_init=10)
            gmm.fit(X)
            bic_scores.append(gmm.bic(X))
            aic_scores.append(gmm.aic(X))
        
        optimal_bic = np.argmin(bic_scores) + 1
        optimal_aic = np.argmin(aic_scores) + 1
        
        logger.info(f"Optimal components (BIC): {optimal_bic}")
        logger.info(f"Optimal components (AIC): {optimal_aic}")
        
        return {
            'bic_scores': bic_scores,
            'aic_scores': aic_scores,
            'optimal_bic': optimal_bic,
            'optimal_aic': optimal_aic
        }
    
    def fit(self, X):
        """
        Fit Gaussian Mixture Model.
        
        Args:
            X (np.ndarray or pd.DataFrame): Input data
            
        Returns:
            self
        """
        logger.info(f"Fitting GMM with {self.n_components} components")
        
        self.model = GaussianMixture(n_components=self.n_components,
                                    covariance_type=self.covariance_type,
                                    random_state=self.random_state, n_init=10)
        self.labels = self.model.fit_predict(X)
        self.probabilities = self.model.predict_proba(X)
        
        logger.info(f"GMM fitted. BIC: {self.model.bic(X):.4f}, AIC: {self.model.aic(X):.4f}")
        
        return self
    
    def predict(self, X):
        """
        Predict cluster labels.
        
        Args:
            X (np.ndarray or pd.DataFrame): Input data
            
        Returns:
            np.ndarray: Cluster labels
        """
        if self.model is None:
            raise ValueError("Model not yet fitted. Run fit() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict cluster probabilities.
        
        Args:
            X (np.ndarray or pd.DataFrame): Input data
            
        Returns:
            np.ndarray: Cluster membership probabilities
        """
        if self.model is None:
            raise ValueError("Model not yet fitted. Run fit() first.")
        
        return self.model.predict_proba(X)
    
    def get_labels(self):
        """Get cluster labels."""
        return self.labels
    
    def get_probabilities(self):
        """Get cluster membership probabilities."""
        return self.probabilities


class IsolationForestModel:
    """Isolation Forest for anomaly detection."""
    
    def __init__(self, contamination=0.1, random_state=42):
        """
        Initialize Isolation Forest.
        
        Args:
            contamination (float): Expected proportion of outliers (0-0.5)
            random_state (int): Random seed
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.labels = None
        self.scores = None
    
    def fit(self, X):
        """
        Fit Isolation Forest model.
        
        Args:
            X (np.ndarray or pd.DataFrame): Input data
            
        Returns:
            self
        """
        logger.info(f"Fitting Isolation Forest with contamination={self.contamination}")
        
        self.model = IsolationForest(contamination=self.contamination,
                                    random_state=self.random_state)
        self.labels = self.model.fit_predict(X)
        self.scores = self.model.score_samples(X)
        
        n_anomalies = (self.labels == -1).sum()
        logger.info(f"Isolation Forest fitted. Anomalies detected: {n_anomalies}")
        
        return self
    
    def predict(self, X):
        """
        Predict anomalies.
        
        Args:
            X (np.ndarray or pd.DataFrame): Input data
            
        Returns:
            np.ndarray: Anomaly labels (-1 for anomaly, 1 for normal)
        """
        if self.model is None:
            raise ValueError("Model not yet fitted. Run fit() first.")
        
        return self.model.predict(X)
    
    def decision_function(self, X):
        """
        Get anomaly scores.
        
        Args:
            X (np.ndarray or pd.DataFrame): Input data
            
        Returns:
            np.ndarray: Anomaly scores
        """
        if self.model is None:
            raise ValueError("Model not yet fitted. Run fit() first.")
        
        return self.model.score_samples(X)
    
    def get_labels(self):
        """Get anomaly labels."""
        return self.labels
    
    def get_scores(self):
        """Get anomaly scores."""
        return self.scores


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    from data_preprocessor import DataPreprocessor
    
    # Load and preprocess data
    loader = DataLoader('transactions.parquet')
    df = loader.load_data()
    preprocessor = DataPreprocessor(df)
    X = preprocessor.preprocess_pipeline()
    
    # Test K-Means
    kmeans = KMeansClustering(n_clusters=5)
    kmeans.fit(X)
    print(f"K-Means inertia: {kmeans.get_inertia():.4f}")
    
    # Test DBSCAN
    dbscan = DBSCANClustering()
    eps, distances = dbscan.find_optimal_eps(X)
    dbscan.eps = eps
    dbscan.fit(X)
    
    # Test GMM
    gmm = GaussianMixtureModel(n_components=5)
    gmm.fit(X)
    
    # Test Isolation Forest
    iso_forest = IsolationForestModel(contamination=0.1)
    iso_forest.fit(X)
