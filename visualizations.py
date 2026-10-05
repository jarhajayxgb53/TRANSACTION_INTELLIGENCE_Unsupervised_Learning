"""
Visualizations Module
=====================
Provides visualization functions for clustering analysis.

Author: Data Science Team
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ClusteringVisualizer:
    """
    A class to create visualizations for clustering results.
    """
    
    @staticmethod
    def plot_elbow_curve(inertias, k_range, output_path=None):
        """
        Plot elbow curve for K-Means.
        
        Args:
            inertias (list): List of inertia values
            k_range (range): Range of k values tested
            output_path (str, optional): Path to save the figure
        """
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Inertia', fontsize=12)
        plt.title('Elbow Method - K-Means', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved elbow curve to {output_path}")
        
        plt.close()
    
    @staticmethod
    def plot_silhouette_curve(silhouette_scores, k_range, optimal_k, output_path=None):
        """
        Plot silhouette scores vs k.
        
        Args:
            silhouette_scores (list): List of silhouette scores
            k_range (range): Range of k values tested
            optimal_k (int): Optimal number of clusters
            output_path (str, optional): Path to save the figure
        """
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        plt.axvline(x=optimal_k, color='g', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved silhouette curve to {output_path}")
        
        plt.close()
    
    @staticmethod
    def plot_bic_aic_curves(bic_scores, aic_scores, optimal_bic, output_path=None):
        """
        Plot BIC and AIC scores for GMM.
        
        Args:
            bic_scores (list): List of BIC scores
            aic_scores (list): List of AIC scores
            optimal_bic (int): Optimal components by BIC
            output_path (str, optional): Path to save the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        n_components_range = range(1, len(bic_scores) + 1)
        
        axes[0].plot(n_components_range, bic_scores, 'bo-', linewidth=2, markersize=8)
        axes[0].axvline(x=optimal_bic, color='g', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Number of Components', fontsize=12)
        axes[0].set_ylabel('BIC Score', fontsize=12)
        axes[0].set_title('BIC Score - Gaussian Mixture Models', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(n_components_range, aic_scores, 'ro-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Number of Components', fontsize=12)
        axes[1].set_ylabel('AIC Score', fontsize=12)
        axes[1].set_title('AIC Score - Gaussian Mixture Models', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved BIC/AIC curves to {output_path}")
        
        plt.close()
    
    @staticmethod
    def plot_clustering_results_2d(X_pca_2d, labels, title, output_path=None):
        """
        Plot clustering results in 2D space.
        
        Args:
            X_pca_2d (np.ndarray): 2D PCA-transformed data
            labels (np.ndarray): Cluster labels
            title (str): Plot title
            output_path (str, optional): Path to save the figure
        """
        plt.figure(figsize=(10, 8))
        
        scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=labels,
                            cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        plt.xlabel('PC1', fontsize=12)
        plt.ylabel('PC2', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved clustering visualization to {output_path}")
        
        plt.close()
    
    @staticmethod
    def plot_cluster_comparison_2d(results_dict, X_pca_2d, output_path=None):
        """
        Plot multiple clustering results for comparison.
        
        Args:
            results_dict (dict): Dictionary with model names and labels
            X_pca_2d (np.ndarray): 2D PCA-transformed data
            output_path (str, optional): Path to save the figure
        """
        n_models = len(results_dict)
        n_cols = min(n_models, 2)
        n_rows = (n_models + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
        
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        colormaps = ['viridis', 'plasma', 'cool', 'twilight']
        
        for idx, (model_name, labels) in enumerate(results_dict.items()):
            scatter = axes[idx].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=labels,
                                       cmap=colormaps[idx % len(colormaps)], s=50,
                                       alpha=0.6, edgecolors='black', linewidth=0.5)
            axes[idx].set_title(model_name, fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('PC1')
            axes[idx].set_ylabel('PC2')
            plt.colorbar(scatter, ax=axes[idx])
        
        # Hide extra subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {output_path}")
        
        plt.close()
    
    @staticmethod
    def plot_cluster_characteristics(df_clustered, cluster_column, features, output_path=None):
        """
        Plot mean values of features for each cluster.
        
        Args:
            df_clustered (pd.DataFrame): Data with cluster labels
            cluster_column (str): Name of cluster column
            features (list): List of features to plot
            output_path (str, optional): Path to save the figure
        """
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = np.array(axes).flatten()
        
        n_clusters = df_clustered[cluster_column].max() + 1
        
        for idx, feature in enumerate(features):
            cluster_means = []
            for cluster in range(n_clusters):
                cluster_data = df_clustered[df_clustered[cluster_column] == cluster]
                cluster_means.append(cluster_data[feature].mean())
            
            colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
            axes[idx].bar(range(n_clusters), cluster_means, color=colors)
            axes[idx].set_xlabel('Cluster', fontsize=11)
            axes[idx].set_ylabel('Mean Value', fontsize=11)
            axes[idx].set_title(f'{feature} by Cluster', fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Hide extra subplots
        for idx in range(len(features), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved cluster characteristics to {output_path}")
        
        plt.close()
    
    @staticmethod
    def plot_correlation_matrix(df, output_path=None):
        """
        Plot correlation matrix heatmap.
        
        Args:
            df (pd.DataFrame): Input dataframe
            output_path (str, optional): Path to save the figure
        """
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   fmt='.2f')
        plt.title('Correlation Matrix of Numerical Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved correlation matrix to {output_path}")
        
        plt.close()
    
    @staticmethod
    def plot_anomaly_scores(scores, threshold=None, output_path=None):
        """
        Plot distribution of anomaly scores.
        
        Args:
            scores (np.ndarray): Anomaly scores
            threshold (float, optional): Anomaly threshold
            output_path (str, optional): Path to save the figure
        """
        plt.figure(figsize=(12, 6))
        plt.hist(scores, bins=50, edgecolor='black', alpha=0.7)
        
        if threshold is not None:
            plt.axvline(x=threshold, color='red', linestyle='--',
                       label=f'Threshold: {threshold:.4f}', linewidth=2)
            plt.legend()
        
        plt.xlabel('Anomaly Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Anomaly Scores - Isolation Forest', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved anomaly scores plot to {output_path}")
        
        plt.close()
    
    @staticmethod
    def plot_pca_variance(explained_variance, cumsum_variance, output_path=None):
        """
        Plot PCA variance explained.
        
        Args:
            explained_variance (np.ndarray): Explained variance ratio
            cumsum_variance (np.ndarray): Cumulative explained variance
            output_path (str, optional): Path to save the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        n_components = min(20, len(explained_variance))
        
        axes[0].plot(range(1, n_components + 1), explained_variance[:n_components],
                    'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Principal Component', fontsize=12)
        axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
        axes[0].set_title('Scree Plot - PCA', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(range(1, n_components + 1), cumsum_variance[:n_components],
                    'ro-', linewidth=2, markersize=8)
        axes[1].axhline(y=0.95, color='g', linestyle='--', label='95% variance')
        axes[1].set_xlabel('Number of Components', fontsize=12)
        axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
        axes[1].set_title('Cumulative Explained Variance - PCA', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved PCA variance plot to {output_path}")
        
        plt.close()
    
    @staticmethod
    def plot_distribution(df, columns=None, output_path=None):
        """
        Plot distributions of numerical features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (list, optional): Columns to plot. Defaults to all numerical.
            output_path (str, optional): Path to save the figure
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = 3
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = np.array(axes).flatten()
        
        for idx, col in enumerate(columns):
            axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_title(col, fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
        
        for idx in range(len(columns), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Feature Distributions', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved distribution plot to {output_path}")
        
        plt.close()
    
    @staticmethod
    def plot_boxplots(df, columns=None, output_path=None):
        """
        Plot boxplots for numerical features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (list, optional): Columns to plot. Defaults to all numerical.
            output_path (str, optional): Path to save the figure
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = 3
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = np.array(axes).flatten()
        
        for idx, col in enumerate(columns):
            axes[idx].boxplot(df[col].dropna())
            axes[idx].set_title(col, fontsize=11, fontweight='bold')
        
        for idx in range(len(columns), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Feature Boxplots', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved boxplot to {output_path}")
        
        plt.close()
    
    @staticmethod
    def plot_interactive_clusters_2d(X_pca_2d, labels, title='Interactive 2D Cluster Visualization', hover_df=None, output_path=None):
        """
        Create an interactive 2D scatter plot using Plotly.
        
        Args:
            X_pca_2d (np.ndarray): 2D coordinates (e.g., from PCA)
            labels (np.ndarray): Cluster labels
            title (str): Plot title
            hover_df (pd.DataFrame, optional): Additional feature columns for tooltips
            output_path (str, optional): Path to save interactive HTML file
        """
        plot_df = pd.DataFrame({
            'PC1': X_pca_2d[:, 0],
            'PC2': X_pca_2d[:, 1],
            'Cluster': [f'Cluster {l}' if l != -1 else 'Noise' for l in labels]
        })
        
        hover_cols = []
        if hover_df is not None:
            for col in hover_df.columns:
                plot_df[col] = hover_df[col].values
                hover_cols.append(col)
        
        fig = px.scatter(
            plot_df,
            x='PC1',
            y='PC2',
            color='Cluster',
            title=title,
            hover_data=hover_cols if hover_cols else None,
            template='plotly_white',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig.update_traces(marker=dict(size=7, opacity=0.75, line=dict(width=0.5, color='DarkSlateGrey')))
        fig.update_layout(
            title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}},
            legend_title_text='Clusters',
            width=1000,
            height=700
        )
        
        if output_path:
            fig.write_html(output_path, include_plotlyjs='cdn')
            logger.info(f"Saved interactive 2D cluster visualization to {output_path}")
            
        return fig
    
    @staticmethod
    def plot_interactive_clusters_3d(X_pca_3d, labels, title='Interactive 3D Cluster Visualization', hover_df=None, output_path=None):
        """
        Create an interactive 3D scatter plot using Plotly.
        
        Args:
            X_pca_3d (np.ndarray): 3D coordinates (from PCA)
            labels (np.ndarray): Cluster labels
            title (str): Plot title
            hover_df (pd.DataFrame, optional): Additional feature columns for tooltips
            output_path (str, optional): Path to save interactive HTML file
        """
        plot_df = pd.DataFrame({
            'PC1': X_pca_3d[:, 0],
            'PC2': X_pca_3d[:, 1],
            'PC3': X_pca_3d[:, 2],
            'Cluster': [f'Cluster {l}' if l != -1 else 'Noise' for l in labels]
        })
        
        hover_cols = []
        if hover_df is not None:
            for col in hover_df.columns:
                plot_df[col] = hover_df[col].values
                hover_cols.append(col)
        
        fig = px.scatter_3d(
            plot_df,
            x='PC1',
            y='PC2',
            z='PC3',
            color='Cluster',
            title=title,
            hover_data=hover_cols if hover_cols else None,
            template='plotly_dark',
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig.update_traces(marker=dict(size=4, opacity=0.8))
        fig.update_layout(
            title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}},
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3'
            ),
            legend_title_text='Clusters',
            width=1100,
            height=750
        )
        
        if output_path:
            fig.write_html(output_path, include_plotlyjs='cdn')
            logger.info(f"Saved interactive 3D cluster visualization to {output_path}")
            
        return fig
    
    @staticmethod
    def plot_interactive_cluster_comparison(results_dict, X_pca_2d, output_path=None):
        """
        Create side-by-side interactive comparison subplots using Plotly.
        
        Args:
            results_dict (dict): Dictionary with model names and labels
            X_pca_2d (np.ndarray): 2D coordinates
            output_path (str, optional): Path to save interactive HTML file
        """
        models = list(results_dict.keys())
        n_models = len(models)
        n_cols = min(n_models, 2)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[f'{m} Clustering' for m in models],
            horizontal_spacing=0.08,
            vertical_spacing=0.1
        )
        
        for idx, (model_name, labels) in enumerate(results_dict.items()):
            row = (idx // n_cols) + 1
            col = (idx % n_cols) + 1
            
            trace = go.Scatter(
                x=X_pca_2d[:, 0],
                y=X_pca_2d[:, 1],
                mode='markers',
                marker=dict(
                    color=labels,
                    colorscale='Viridis',
                    size=5,
                    opacity=0.7,
                    showscale=(idx == 0)
                ),
                text=[f"Model: {model_name}<br>Cluster: {l}<br>PC1: {x:.2f}<br>PC2: {y:.2f}"
                      for l, x, y in zip(labels, X_pca_2d[:, 0], X_pca_2d[:, 1])],
                hoverinfo='text',
                name=model_name
            )
            fig.add_trace(trace, row=row, col=col)
            fig.update_xaxes(title_text='PC1', row=row, col=col)
            fig.update_yaxes(title_text='PC2', row=row, col=col)
        
        fig.update_layout(
            title_text='Interactive Multi-Model Clustering Comparison (PCA 2D)',
            title_x=0.5,
            template='plotly_white',
            width=1200,
            height=500 * n_rows,
            showlegend=False
        )
        
        if output_path:
            fig.write_html(output_path, include_plotlyjs='cdn')
            logger.info(f"Saved interactive comparison visualization to {output_path}")
            
        return fig
    
    @staticmethod
    def plot_interactive_anomaly_scores(scores, labels, output_path=None):
        """
        Plot interactive distribution of anomaly scores using Plotly.
        
        Args:
            scores (np.ndarray): Anomaly scores
            labels (np.ndarray): Anomaly labels (-1 for anomaly, 1 for normal)
            output_path (str, optional): Path to save interactive HTML file
        """
        plot_df = pd.DataFrame({
            'Anomaly_Score': scores,
            'Status': ['Anomaly' if l == -1 else 'Normal' for l in labels]
        })
        
        fig = px.histogram(
            plot_df,
            x='Anomaly_Score',
            color='Status',
            nbins=60,
            barmode='overlay',
            title='Interactive Distribution of Anomaly Scores (Isolation Forest)',
            color_discrete_map={'Normal': '#2ecc71', 'Anomaly': '#e74c3c'},
            template='plotly_white',
            opacity=0.75
        )
        
        threshold = scores[labels == -1].max() if (labels == -1).sum() > 0 else scores.min()
        fig.add_vline(
            x=threshold,
            line_dash='dash',
            line_color='red',
            annotation_text=f'Threshold ({threshold:.4f})',
            annotation_position='top left'
        )
        
        fig.update_layout(
            title_x=0.5,
            xaxis_title='Anomaly Score',
            yaxis_title='Transaction Count',
            width=1000,
            height=600
        )
        
        if output_path:
            fig.write_html(output_path, include_plotlyjs='cdn')
            logger.info(f"Saved interactive anomaly distribution to {output_path}")
            
        return fig


if __name__ == "__main__":
    # Example usage
    logger.info("ClusteringVisualizer module loaded successfully")
