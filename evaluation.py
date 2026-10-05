"""
Model Evaluation Module
=======================
Provides metrics and evaluation functions for clustering models.

Author: Data Science Team
Date: 2024
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClusteringEvaluator:
    """
    A class to evaluate clustering models using various metrics.
    
    Metrics included:
    - Silhouette Score
    - Davies-Bouldin Index
    - Calinski-Harabasz Index
    - Adjusted Rand Index (if ground truth available)
    - Normalized Mutual Information (if ground truth available)
    """
    
    def __init__(self, X, labels, true_labels=None):
        """
        Initialize the ClusteringEvaluator.
        
        Args:
            X (np.ndarray or pd.DataFrame): Input data
            labels (np.ndarray): Predicted cluster labels
            true_labels (np.ndarray, optional): True cluster labels for comparison
        """
        self.X = X
        self.labels = labels
        self.true_labels = true_labels
        self.metrics = {}
    
    def compute_silhouette_score(self):
        """
        Compute silhouette score.
        
        Returns:
            float: Silhouette score (-1 to 1, higher is better)
        """
        # Filter out noise points (label == -1) if present
        valid_mask = self.labels != -1
        if valid_mask.sum() < 2:
            logger.warning("Not enough valid clusters for silhouette score")
            return None
        
        # Need at least 2 unique labels
        unique_labels = np.unique(self.labels[valid_mask])
        if len(unique_labels) < 2:
            logger.warning("Need at least 2 clusters for silhouette score")
            return None
        
        score = silhouette_score(self.X[valid_mask], self.labels[valid_mask])
        self.metrics['silhouette_score'] = score
        
        logger.info(f"Silhouette Score: {score:.4f}")
        
        return score
    
    def compute_davies_bouldin_index(self):
        """
        Compute Davies-Bouldin Index.
        
        Returns:
            float: Davies-Bouldin Index (0 or higher, lower is better)
        """
        # Filter out noise points
        valid_mask = self.labels != -1
        if valid_mask.sum() < 2:
            logger.warning("Not enough valid clusters for Davies-Bouldin Index")
            return None
        
        unique_labels = np.unique(self.labels[valid_mask])
        if len(unique_labels) < 2:
            logger.warning("Need at least 2 clusters for Davies-Bouldin Index")
            return None
        
        score = davies_bouldin_score(self.X[valid_mask], self.labels[valid_mask])
        self.metrics['davies_bouldin_index'] = score
        
        logger.info(f"Davies-Bouldin Index: {score:.4f}")
        
        return score
    
    def compute_calinski_harabasz_index(self):
        """
        Compute Calinski-Harabasz Index.
        
        Returns:
            float: Calinski-Harabasz Index (higher is better)
        """
        # Filter out noise points
        valid_mask = self.labels != -1
        if valid_mask.sum() < 2:
            logger.warning("Not enough valid clusters for Calinski-Harabasz Index")
            return None
        
        unique_labels = np.unique(self.labels[valid_mask])
        if len(unique_labels) < 2:
            logger.warning("Need at least 2 clusters for Calinski-Harabasz Index")
            return None
        
        score = calinski_harabasz_score(self.X[valid_mask], self.labels[valid_mask])
        self.metrics['calinski_harabasz_index'] = score
        
        logger.info(f"Calinski-Harabasz Index: {score:.4f}")
        
        return score
    
    def compute_adjusted_rand_index(self):
        """
        Compute Adjusted Rand Index (requires true labels).
        
        Returns:
            float: Adjusted Rand Index (-1 to 1, higher is better)
        """
        if self.true_labels is None:
            logger.warning("True labels not provided. Cannot compute Adjusted Rand Index.")
            return None
        
        score = adjusted_rand_score(self.true_labels, self.labels)
        self.metrics['adjusted_rand_index'] = score
        
        logger.info(f"Adjusted Rand Index: {score:.4f}")
        
        return score
    
    def compute_normalized_mutual_information(self):
        """
        Compute Normalized Mutual Information (requires true labels).
        
        Returns:
            float: Normalized Mutual Information (0 to 1, higher is better)
        """
        if self.true_labels is None:
            logger.warning("True labels not provided. Cannot compute NMI.")
            return None
        
        score = normalized_mutual_info_score(self.true_labels, self.labels)
        self.metrics['normalized_mutual_information'] = score
        
        logger.info(f"Normalized Mutual Information: {score:.4f}")
        
        return score
    
    def compute_all_metrics(self):
        """
        Compute all available metrics.
        
        Returns:
            dict: Dictionary with all metrics
        """
        logger.info("=" * 80)
        logger.info("COMPUTING ALL EVALUATION METRICS")
        logger.info("=" * 80)
        
        self.compute_silhouette_score()
        self.compute_davies_bouldin_index()
        self.compute_calinski_harabasz_index()
        
        if self.true_labels is not None:
            self.compute_adjusted_rand_index()
            self.compute_normalized_mutual_information()
        
        logger.info("=" * 80)
        
        return self.metrics
    
    def get_cluster_statistics(self):
        """
        Get statistics about the clusters.
        
        Returns:
            dict: Dictionary with cluster statistics
        """
        unique_labels = np.unique(self.labels)
        n_clusters = len(unique_labels[unique_labels != -1])
        n_noise = (self.labels == -1).sum()
        
        stats = {
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'noise_percentage': n_noise / len(self.labels) * 100,
            'cluster_sizes': {},
            'cluster_percentages': {}
        }
        
        for label in unique_labels:
            if label != -1:
                size = (self.labels == label).sum()
                stats['cluster_sizes'][f'Cluster_{int(label)}'] = size
                stats['cluster_percentages'][f'Cluster_{int(label)}'] = size / len(self.labels) * 100
        
        logger.info(f"\nCluster Statistics:")
        logger.info(f"  Number of clusters: {n_clusters}")
        logger.info(f"  Noise points: {n_noise} ({n_noise/len(self.labels)*100:.2f}%)")
        logger.info(f"  Cluster sizes:")
        for cluster, size in stats['cluster_sizes'].items():
            pct = stats['cluster_percentages'][cluster]
            logger.info(f"    {cluster}: {size} ({pct:.2f}%)")
        
        return stats
    
    def get_silhouette_samples(self):
        """
        Compute silhouette coefficient for each sample.
        
        Returns:
            np.ndarray: Silhouette coefficients for each sample
        """
        valid_mask = self.labels != -1
        if valid_mask.sum() < 2:
            logger.warning("Not enough valid clusters")
            return None
        
        unique_labels = np.unique(self.labels[valid_mask])
        if len(unique_labels) < 2:
            logger.warning("Need at least 2 clusters for silhouette samples")
            return None
        
        samples = silhouette_samples(self.X[valid_mask], self.labels[valid_mask])
        
        return samples
    
    def get_metrics_dataframe(self):
        """
        Get metrics as a pandas DataFrame.
        
        Returns:
            pd.DataFrame: Metrics dataframe
        """
        if not self.metrics:
            self.compute_all_metrics()
        
        df = pd.DataFrame(list(self.metrics.items()), columns=['Metric', 'Value'])
        df['Value'] = df['Value'].round(4)
        
        return df
    
    def print_summary(self):
        """Print a summary of all metrics."""
        if not self.metrics:
            self.compute_all_metrics()
        
        logger.info("=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 80)
        
        for metric, value in self.metrics.items():
            if value is not None:
                logger.info(f"{metric:.<50} {value:.4f}")
        
        logger.info("=" * 80)


class ModelComparison:
    """
    A class to compare multiple clustering models.
    """
    
    def __init__(self):
        """Initialize ModelComparison."""
        self.results = {}
    
    def evaluate_model(self, model_name, X, labels, true_labels=None):
        """
        Evaluate a single model.
        
        Args:
            model_name (str): Name of the model
            X (np.ndarray or pd.DataFrame): Input data
            labels (np.ndarray): Predicted cluster labels
            true_labels (np.ndarray, optional): True cluster labels
        """
        logger.info(f"\nEvaluating model: {model_name}")
        
        evaluator = ClusteringEvaluator(X, labels, true_labels)
        metrics = evaluator.compute_all_metrics()
        stats = evaluator.get_cluster_statistics()
        
        self.results[model_name] = {
            'metrics': metrics,
            'stats': stats,
            'evaluator': evaluator
        }
    
    def get_comparison_dataframe(self):
        """
        Get comparison results as a DataFrame.
        
        Returns:
            pd.DataFrame: Comparison dataframe
        """
        data = []
        
        for model_name, result in self.results.items():
            row = {'Model': model_name}
            row.update(result['metrics'])
            row['n_clusters'] = result['stats']['n_clusters']
            row['noise_pct'] = result['stats']['noise_percentage']
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Round numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(4)
        
        return df
    
    def print_comparison(self):
        """Print model comparison table."""
        df = self.get_comparison_dataframe()
        
        logger.info("\n" + "=" * 120)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 120)
        logger.info(f"\n{df.to_string(index=False)}")
        logger.info("\n" + "=" * 120)


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    from data_preprocessor import DataPreprocessor
    from unsupervised_models import KMeansClustering
    
    # Load and preprocess data
    loader = DataLoader('transactions.parquet')
    df = loader.load_data()
    preprocessor = DataPreprocessor(df)
    X = preprocessor.preprocess_pipeline()
    
    # Fit model and evaluate
    kmeans = KMeansClustering(n_clusters=5)
    kmeans.fit(X)
    
    evaluator = ClusteringEvaluator(X, kmeans.get_labels())
    evaluator.compute_all_metrics()
    evaluator.print_summary()
