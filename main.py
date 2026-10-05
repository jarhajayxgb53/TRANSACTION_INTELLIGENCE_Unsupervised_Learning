

import logging
import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd

# Import custom modules
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from dimensionality_reduction import DimensionalityReducer
from unsupervised_models import (
    KMeansClustering,
    HierarchicalClustering,
    DBSCANClustering,
    GaussianMixtureModel,
    IsolationForestModel
)
from evaluation import ClusteringEvaluator, ModelComparison
from visualizations import ClusteringVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unsupervised_learning_project.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class UnsupervisedLearningPipeline:
    """
    Main pipeline class that orchestrates the entire unsupervised learning workflow.
    """
    
    def __init__(self, data_path='transactions.parquet'):
        """
        Initialize the pipeline.
        
        Args:
            data_path (str): Path to the parquet file
        """
        self.data_path = data_path
        self.data = None
        self.X_processed = None
        self.X_scaled = None
        self.reducer = None
        self.models = {}
        self.evaluations = {}
        self.comparator = ModelComparison()
        
        logger.info("\n" + "=" * 100)
        logger.info("UNSUPERVISED LEARNING PIPELINE INITIALIZED")
        logger.info("=" * 100)
    
    def load_and_explore(self):
        """Load and explore the dataset."""
        logger.info("\n[STEP 1] LOADING AND EXPLORING DATA")
        logger.info("-" * 100)
        
        loader = DataLoader(self.data_path)
        self.data = loader.load_data()
        loader.explore_data()
        
        # Generate EDA visualizations
        visualizer = ClusteringVisualizer()
        visualizer.plot_correlation_matrix(self.data, output_path='correlation_matrix.png')
        visualizer.plot_distribution(self.data, output_path='feature_distributions.png')
        visualizer.plot_boxplots(self.data, output_path='feature_boxplots.png')
        
        logger.info("✓ Data loaded and explored")
    
    def preprocess(self, handle_outliers=True, scale_method='standard'):
        """Preprocess and scale the data."""
        logger.info("\n[STEP 2] DATA PREPROCESSING")
        logger.info("-" * 100)
        
        preprocessor = DataPreprocessor(self.data)
        self.X_processed = preprocessor.preprocess_pipeline(
            handle_outliers=handle_outliers,
            scale_method=scale_method
        )
        
        logger.info("✓ Data preprocessed and scaled")
    
    def reduce_dimensionality(self, variance_threshold=0.95):
        """Apply dimensionality reduction."""
        logger.info("\n[STEP 3] DIMENSIONALITY REDUCTION")
        logger.info("-" * 100)
        
        self.reducer = DimensionalityReducer(self.X_processed)
        self.reducer.apply_pca_full()
        self.reducer.apply_pca_optimal(variance_threshold=variance_threshold)
        self.reducer.apply_pca_2d()
        self.reducer.apply_pca_3d()
        self.reducer.summary()
        
        # Visualize PCA results
        explained_var, cumsum_var = self.reducer.get_pca_variance_explained()
        visualizer = ClusteringVisualizer()
        visualizer.plot_pca_variance(explained_var, cumsum_var,
                                    output_path='pca_variance.png')
        
        logger.info("✓ Dimensionality reduction applied")
    
    def train_kmeans(self, max_k=10):
        """Train K-Means with optimal k."""
        logger.info("\n[STEP 4.1] K-MEANS CLUSTERING")
        logger.info("-" * 100)
        
        from sklearn.metrics import silhouette_score
        
        # Find optimal k using elbow method and silhouette score
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeansClustering(n_clusters=k)
            kmeans.fit(self.X_processed)
            inertias.append(kmeans.get_inertia())
            score = silhouette_score(self.X_processed, kmeans.labels)
            silhouette_scores.append(score)
        
        optimal_k = list(k_range)[silhouette_scores.index(max(silhouette_scores))]
        
        # Generate elbow and silhouette plots
        visualizer = ClusteringVisualizer()
        visualizer.plot_elbow_curve(inertias, k_range, output_path='elbow_curve.png')
        visualizer.plot_silhouette_curve(silhouette_scores, k_range, optimal_k,
                                        output_path='silhouette_scores.png')
        
        # Train optimal model
        kmeans_optimal = KMeansClustering(n_clusters=optimal_k)
        kmeans_optimal.fit(self.X_processed)
        
        self.models['kmeans'] = {
            'model': kmeans_optimal,
            'labels': kmeans_optimal.labels,
            'optimal_k': optimal_k
        }
        
        # Visualize clusters
        visualizer.plot_clustering_results_2d(
            self.reducer.X_pca_2d, kmeans_optimal.labels,
            f'K-Means Clustering (k={optimal_k})',
            output_path='kmeans_clusters_2d.png'
        )
        
        # Evaluate
        self.comparator.evaluate_model('K-Means', self.X_processed, kmeans_optimal.labels)
        
        logger.info(f"✓ K-Means trained with k={optimal_k}")
    
    def train_hierarchical(self):
        """Train Hierarchical Clustering."""
        logger.info("\n[STEP 4.2] HIERARCHICAL CLUSTERING")
        logger.info("-" * 100)
        
        optimal_k = self.models['kmeans']['optimal_k']  # Use same k as K-Means
        hierarchical = HierarchicalClustering(n_clusters=optimal_k, linkage='ward')
        hierarchical.fit(self.X_processed)
        
        self.models['hierarchical'] = {
            'model': hierarchical,
            'labels': hierarchical.labels
        }
        
        # Visualize clusters
        visualizer = ClusteringVisualizer()
        visualizer.plot_clustering_results_2d(
            self.reducer.X_pca_2d, hierarchical.labels,
            f'Hierarchical Clustering (k={optimal_k})',
            output_path='hierarchical_clusters_2d.png'
        )
        
        # Evaluate
        self.comparator.evaluate_model('Hierarchical', self.X_processed, hierarchical.labels)
        
        logger.info("✓ Hierarchical clustering trained")
    
    def train_dbscan(self):
        """Train DBSCAN."""
        logger.info("\n[STEP 4.3] DBSCAN CLUSTERING")
        logger.info("-" * 100)
        
        dbscan = DBSCANClustering()
        eps, distances = dbscan.find_optimal_eps(self.X_processed, k=5)
        dbscan.eps = eps
        dbscan.fit(self.X_processed)
        
        self.models['dbscan'] = {
            'model': dbscan,
            'labels': dbscan.labels,
            'eps': eps
        }
        
        # Visualize clusters
        visualizer = ClusteringVisualizer()
        visualizer.plot_clustering_results_2d(
            self.reducer.X_pca_2d, dbscan.labels,
            f'DBSCAN Clustering (eps={eps:.4f})',
            output_path='dbscan_clusters_2d.png'
        )
        
        # Evaluate
        self.comparator.evaluate_model('DBSCAN', self.X_processed, dbscan.labels)
        
        logger.info("✓ DBSCAN trained")
    
    def train_gmm(self, max_components=10):
        """Train Gaussian Mixture Model."""
        logger.info("\n[STEP 4.4] GAUSSIAN MIXTURE MODELS")
        logger.info("-" * 100)
        
        gmm_selector = GaussianMixtureModel(n_components=2)
        results = gmm_selector.find_optimal_components(self.X_processed, max_components)
        optimal_components = results['optimal_bic']
        
        # Plot BIC/AIC curves
        visualizer = ClusteringVisualizer()
        visualizer.plot_bic_aic_curves(
            results['bic_scores'], results['aic_scores'],
            optimal_components, output_path='gmm_bic_aic.png'
        )
        
        gmm_optimal = GaussianMixtureModel(n_components=optimal_components)
        gmm_optimal.fit(self.X_processed)
        
        self.models['gmm'] = {
            'model': gmm_optimal,
            'labels': gmm_optimal.labels,
            'optimal_components': optimal_components
        }
        
        # Visualize clusters
        visualizer.plot_clustering_results_2d(
            self.reducer.X_pca_2d, gmm_optimal.labels,
            f'GMM Clustering (n={optimal_components})',
            output_path='gmm_clusters_2d.png'
        )
        
        # Evaluate
        self.comparator.evaluate_model('GMM', self.X_processed, gmm_optimal.labels)
        
        logger.info(f"✓ GMM trained with {optimal_components} components")
    
    def train_isolation_forest(self, contamination=0.1):
        """Train Isolation Forest for anomaly detection."""
        logger.info("\n[STEP 4.5] ISOLATION FOREST - ANOMALY DETECTION")
        logger.info("-" * 100)
        
        iso_forest = IsolationForestModel(contamination=contamination)
        iso_forest.fit(self.X_processed)
        
        self.models['isolation_forest'] = {
            'model': iso_forest,
            'labels': iso_forest.labels,
            'scores': iso_forest.scores
        }
        
        # Visualize anomaly scores
        visualizer = ClusteringVisualizer()
        visualizer.plot_anomaly_scores(iso_forest.scores,
                                      output_path='anomaly_scores.png')
        
        # Visualize anomalies in 2D
        visualizer.plot_clustering_results_2d(
            self.reducer.X_pca_2d, iso_forest.labels,
            'Isolation Forest - Anomaly Detection',
            output_path='isolation_forest_2d.png'
        )
        
        logger.info("✓ Isolation Forest trained")
    
    def visualize_comparison(self):
        """Generate comparison visualizations."""
        logger.info("\n[STEP 5] GENERATING COMPARISON VISUALIZATIONS")
        logger.info("-" * 100)
        
        visualizer = ClusteringVisualizer()
        
        # Cluster comparison
        results_dict = {}
        if 'kmeans' in self.models:
            results_dict['K-Means'] = self.models['kmeans']['labels']
        if 'hierarchical' in self.models:
            results_dict['Hierarchical'] = self.models['hierarchical']['labels']
        if 'dbscan' in self.models:
            results_dict['DBSCAN'] = self.models['dbscan']['labels']
        if 'gmm' in self.models:
            results_dict['GMM'] = self.models['gmm']['labels']
        
        if results_dict:
            # Static comparison plot
            visualizer.plot_cluster_comparison_2d(results_dict, self.reducer.X_pca_2d,
                                                 output_path='clustering_comparison_2d.png')
            # Interactive multi-model comparison
            visualizer.plot_interactive_cluster_comparison(
                results_dict, self.reducer.X_pca_2d,
                output_path='interactive_clustering_comparison.html'
            )
        
        # Interactive 2D & 3D cluster visualizations (using optimal K-Means)
        if 'kmeans' in self.models:
            k = self.models['kmeans']['optimal_k']
            visualizer.plot_interactive_clusters_2d(
                self.reducer.X_pca_2d, self.models['kmeans']['labels'],
                title=f'Interactive K-Means Clustering (k={k})',
                hover_df=self.data,
                output_path='interactive_clusters_2d.html'
            )
            if hasattr(self.reducer, 'X_pca_3d') and self.reducer.X_pca_3d is not None:
                visualizer.plot_interactive_clusters_3d(
                    self.reducer.X_pca_3d, self.models['kmeans']['labels'],
                    title=f'Interactive 3D K-Means Clustering (k={k})',
                    hover_df=self.data,
                    output_path='interactive_clusters_3d.html'
                )
        
        # Interactive anomaly distribution
        if 'isolation_forest' in self.models:
            visualizer.plot_interactive_anomaly_scores(
                self.models['isolation_forest']['scores'],
                self.models['isolation_forest']['labels'],
                output_path='interactive_anomaly_scores.html'
            )
        
        logger.info("✓ Comparison visualizations generated (static PNG and interactive HTML)")
    
    def generate_abstract(self):
        """Generate an academic abstract summarizing the analysis."""
        logger.info("\n[STEP 6] GENERATING ABSTRACT")
        logger.info("-" * 100)
        
        # Gather information for abstract
        n_samples = len(self.data)
        n_features = self.data.shape[1]
        n_processed_features = self.X_processed.shape[1]
        
        # Get comparison data
        comparison_df = self.comparator.get_comparison_dataframe()
        
        # Build abstract
        abstract_lines = []
        abstract_lines.append("=" * 80)
        abstract_lines.append("ACADEMIC ABSTRACT")
        abstract_lines.append("Unsupervised Learning Analysis of Transaction Data")
        abstract_lines.append("=" * 80)
        abstract_lines.append("")
        
        abstract_lines.append("BACKGROUND:")
        abstract_lines.append(
            f"Transaction data analysis is critical for identifying patterns, "
            f"detecting anomalies, and understanding customer behavior in financial "
            f"systems. This study applies unsupervised learning techniques to a "
            f"transaction dataset containing {n_samples:,} records and "
            f"{n_features} features to discover latent structures and anomalous "
            f"patterns without prior labeling."
        )
        abstract_lines.append("")
        
        abstract_lines.append("OBJECTIVE:")
        abstract_lines.append(
            f"To identify meaningful clusters and anomalies in transaction data "
            f"using multiple unsupervised learning algorithms, evaluate their "
            f"effectiveness through established metrics, and provide actionable "
            f"business insights."
        )
        abstract_lines.append("")
        
        abstract_lines.append("METHODOLOGY:")
        abstract_lines.append(
            f"The dataset was preprocessed by handling missing values, capping "
            f"outliers using the IQR method, encoding categorical variables, and "
            f"applying StandardScaler normalization, yielding {n_processed_features} "
            f"features. PCA was applied for dimensionality reduction, retaining 95% "
            f"of explained variance. Five unsupervised learning algorithms were "
            f"implemented: K-Means Clustering, Hierarchical Clustering (Ward linkage), "
            f"DBSCAN, Gaussian Mixture Models (GMM), and Isolation Forest for anomaly "
            f"detection."
        )
        abstract_lines.append("")
        
        abstract_lines.append("KEY FINDINGS:")
        
        # Add model-specific findings
        if 'kmeans' in self.models:
            k = self.models['kmeans']['optimal_k']
            abstract_lines.append(f"  - K-Means identified {k} optimal clusters via silhouette analysis.")
        
        if 'gmm' in self.models:
            n_comp = self.models['gmm']['optimal_components']
            abstract_lines.append(f"  - GMM selected {n_comp} components based on BIC optimization.")
        
        if 'dbscan' in self.models:
            eps = self.models['dbscan']['eps']
            n_noise = (self.models['dbscan']['labels'] == -1).sum()
            abstract_lines.append(
                f"  - DBSCAN (eps={eps:.4f}) identified {n_noise} noise points "
                f"({n_noise/n_samples*100:.1f}% of data)."
            )
        
        if 'isolation_forest' in self.models:
            n_anomalies = (self.models['isolation_forest']['labels'] == -1).sum()
            abstract_lines.append(
                f"  - Isolation Forest detected {n_anomalies} anomalies "
                f"({n_anomalies/n_samples*100:.1f}% of transactions)."
            )
        
        # Add metrics summary
        abstract_lines.append("")
        abstract_lines.append("EVALUATION METRICS:")
        abstract_lines.append(f"\n{comparison_df.to_string(index=False)}")
        abstract_lines.append("")
        
        abstract_lines.append("CONCLUSIONS:")
        abstract_lines.append(
            f"The analysis reveals distinct transaction patterns that can be "
            f"leveraged for customer segmentation, fraud detection, and targeted "
            f"marketing strategies. The multi-algorithm approach provides robust "
            f"validation of discovered clusters, while anomaly detection highlights "
            f"potentially fraudulent or unusual transactions requiring further "
            f"investigation."
        )
        abstract_lines.append("")
        
        abstract_lines.append("FUTURE WORK:")
        abstract_lines.append(
            f"Recommendations include: (1) incorporating temporal features for "
            f"time-series clustering, (2) applying semi-supervised learning with "
            f"domain expert labels, (3) developing real-time anomaly detection "
            f"systems, and (4) combining clustering insights with supervised "
            f"classification for predictive modeling."
        )
        abstract_lines.append("")
        abstract_lines.append("=" * 80)
        
        abstract_text = "\n".join(abstract_lines)
        
        # Save abstract in plain text
        with open('ABSTRACT.txt', 'w', encoding='utf-8') as f:
            f.write(abstract_text)
        
        # Save abstract in Markdown format
        md_content = f"""# Academic Abstract
## Unsupervised Learning Analysis of Transaction Data

### Background
Transaction data analysis is critical for identifying patterns, detecting anomalies, and understanding customer behavior in financial systems. This study applies unsupervised learning techniques to a transaction dataset containing {n_samples:,} records and {n_features} features to discover latent structures and anomalous patterns without prior labeling.

### Objective
To identify meaningful clusters and anomalies in transaction data using multiple unsupervised learning algorithms, evaluate their effectiveness through established metrics, and provide actionable business insights.

### Methodology
The dataset was preprocessed by handling missing values, capping outliers using the IQR method, encoding categorical variables, and applying StandardScaler normalization, yielding {n_processed_features} features. PCA was applied for dimensionality reduction, retaining 95% of explained variance. Five unsupervised learning algorithms were implemented:
1. **K-Means Clustering** (centroid-based clustering with elbow & silhouette optimization)
2. **Hierarchical Clustering** (agglomerative with Ward linkage)
3. **DBSCAN** (density-based spatial clustering with k-distance epsilon estimation)
4. **Gaussian Mixture Models (GMM)** (probabilistic clustering with BIC/AIC component selection)
5. **Isolation Forest** (tree-based anomaly detection)

### Key Findings
"""
        if 'kmeans' in self.models:
            k = self.models['kmeans']['optimal_k']
            md_content += f"- **K-Means**: Identified {k} optimal clusters via silhouette analysis.\n"
        if 'gmm' in self.models:
            n_comp = self.models['gmm']['optimal_components']
            md_content += f"- **GMM**: Selected {n_comp} components based on Bayesian Information Criterion (BIC) optimization.\n"
        if 'dbscan' in self.models:
            eps = self.models['dbscan']['eps']
            n_noise = (self.models['dbscan']['labels'] == -1).sum()
            md_content += f"- **DBSCAN** (eps={eps:.4f}): Identified {n_noise} noise points ({n_noise/n_samples*100:.1f}% of data).\n"
        if 'isolation_forest' in self.models:
            n_anomalies = (self.models['isolation_forest']['labels'] == -1).sum()
            md_content += f"- **Isolation Forest**: Detected {n_anomalies} anomalies ({n_anomalies/n_samples*100:.1f}% of transactions).\n"

        md_content += f"""
### Model Evaluation & Benchmark Metrics

| Model | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index | Clusters | Noise % |
|---|---|---|---|---|---|
"""
        for _, row in comparison_df.iterrows():
            sil = f"{row.get('silhouette_score', 0):.4f}" if pd.notnull(row.get('silhouette_score')) else "N/A"
            db = f"{row.get('davies_bouldin_index', 0):.4f}" if pd.notnull(row.get('davies_bouldin_index')) else "N/A"
            ch = f"{row.get('calinski_harabasz_index', 0):.2f}" if pd.notnull(row.get('calinski_harabasz_index')) else "N/A"
            cl = f"{int(row.get('n_clusters', 0))}" if pd.notnull(row.get('n_clusters')) else "N/A"
            npct = f"{row.get('noise_pct', 0):.2f}%" if pd.notnull(row.get('noise_pct')) else "0.00%"
            md_content += f"| {row['Model']} | {sil} | {db} | {ch} | {cl} | {npct} |\n"

        md_content += """
### Conclusions
The analysis reveals distinct transaction patterns that can be leveraged for customer segmentation, fraud detection, and targeted marketing strategies. The multi-algorithm approach provides robust validation of discovered clusters, while anomaly detection highlights potentially fraudulent or unusual transactions requiring further investigation.

### Future Work
Recommendations include:
1. Incorporating temporal features for time-series clustering.
2. Applying semi-supervised learning with domain expert labels.
3. Developing real-time anomaly detection systems.
4. Combining clustering insights with supervised classification for predictive modeling.
"""
        with open('ABSTRACT.md', 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info("✓ Abstract generated and saved to ABSTRACT.txt and ABSTRACT.md")
        logger.info(f"\n{abstract_text}")
    
    def save_data_outputs(self):
        """Save preprocessed data, clustered data, and comparison metrics."""
        logger.info("\n[STEP 7] SAVING DATA OUTPUTS AND MODEL RESULTS")
        logger.info("-" * 100)
        
        # 1. Save preprocessed dataset
        if self.X_processed is not None:
            self.X_processed.to_parquet('transactions_preprocessed.parquet', index=False)
            self.X_processed.to_csv('transactions_preprocessed.csv', index=False)
            logger.info("✓ Saved preprocessed dataset to transactions_preprocessed.parquet and .csv")
            
        # 2. Save full dataset with cluster assignments and anomaly flags
        if self.data is not None:
            clustered_df = self.data.copy()
            if self.reducer is not None and getattr(self.reducer, 'X_pca_3d', None) is not None:
                clustered_df['pca_1'] = np.round(self.reducer.X_pca_3d[:, 0], 4)
                clustered_df['pca_2'] = np.round(self.reducer.X_pca_3d[:, 1], 4)
                clustered_df['pca_3'] = np.round(self.reducer.X_pca_3d[:, 2], 4)
            elif self.reducer is not None and getattr(self.reducer, 'X_pca_2d', None) is not None:
                clustered_df['pca_1'] = np.round(self.reducer.X_pca_2d[:, 0], 4)
                clustered_df['pca_2'] = np.round(self.reducer.X_pca_2d[:, 1], 4)
            
            for model_key in ['kmeans', 'hierarchical', 'dbscan', 'gmm']:
                if model_key in self.models:
                    clustered_df[f'{model_key}_cluster'] = self.models[model_key]['labels']
            
            if 'isolation_forest' in self.models:
                clustered_df['anomaly_label'] = self.models['isolation_forest']['labels']
                clustered_df['anomaly_score'] = np.round(self.models['isolation_forest']['scores'], 4)
            
            clustered_df.to_parquet('transactions_clustered.parquet', index=False)
            clustered_df.to_csv('transactions_clustered.csv', index=False)
            logger.info("✓ Saved clustered results to transactions_clustered.parquet and .csv")
            
        # 3. Save comparison metrics
        comparison_df = self.comparator.get_comparison_dataframe()
        comparison_df.to_csv('model_comparison.csv', index=False)
        logger.info("✓ Saved model comparison metrics to model_comparison.csv")
    
    def print_summary(self):
        """Print summary of the analysis."""
        logger.info("\n" + "=" * 100)
        logger.info("ANALYSIS SUMMARY")
        logger.info("=" * 100)
        
        logger.info("\nDataset Statistics:")
        logger.info(f"  Samples: {len(self.data):,}")
        logger.info(f"  Original Features: {len(self.data.columns)}")
        logger.info(f"  Processed Features: {self.X_processed.shape[1]}")
        
        logger.info("\nModels Trained:")
        for model_name in self.models.keys():
            logger.info(f"  ✓ {model_name.upper()}")
        
        logger.info("\nModel Comparison:")
        self.comparator.print_comparison()
        
        logger.info("\n" + "=" * 100)
    
    def run_full_pipeline(self):
        """Execute the complete pipeline."""
        start_time = datetime.now()
        
        try:
            self.load_and_explore()
            self.preprocess()
            self.reduce_dimensionality()
            self.train_kmeans()
            self.train_hierarchical()
            self.train_dbscan()
            self.train_gmm()
            self.train_isolation_forest()
            self.visualize_comparison()
            self.generate_abstract()
            self.save_data_outputs()
            self.print_summary()
            
            elapsed = datetime.now() - start_time
            
            logger.info("\n" + "=" * 100)
            logger.info(f"✓ PIPELINE COMPLETED SUCCESSFULLY in {elapsed}")
            logger.info("=" * 100)
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    # Initialize and run pipeline
    pipeline = UnsupervisedLearningPipeline(data_path='transactions.parquet')
    pipeline.run_full_pipeline()
