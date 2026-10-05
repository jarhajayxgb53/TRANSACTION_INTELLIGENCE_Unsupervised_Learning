# Unsupervised Learning on Transactions Dataset

## Project Overview

This project performs comprehensive unsupervised learning analysis on a transactions dataset. The analysis implements multiple clustering algorithms, evaluates their performance, and generates actionable insights through pattern discovery and anomaly detection.

### Key Features
- ✅ Data exploration and preprocessing
- ✅ Dimensionality reduction (PCA, t-SNE)
- ✅ Multiple clustering algorithms (K-Means, Hierarchical, DBSCAN, GMM)
- ✅ Anomaly detection (Isolation Forest)
- ✅ Comprehensive model evaluation and comparison
- ✅ Professional visualizations
- ✅ Academic abstract generation

---

## Project Structure

```
.
├── data_loader.py                 # Load and explore data
├── data_preprocessor.py           # Data cleaning and scaling
├── dimensionality_reduction.py    # PCA and t-SNE
├── unsupervised_models.py         # Clustering algorithms
├── evaluation.py                  # Model evaluation metrics
├── visualizations.py              # Plotting functions
├── main.py                        # Pipeline orchestration
├── unsupervised_learning_project.ipynb  # Jupyter notebook
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Setup Instructions

1. **Clone or download the project**
   ```bash
   cd unsupervised-learning-project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place your data**
   - Ensure `transactions.parquet` is in the project directory
   - Or update the data path in `main.py`

---

## Usage

### Run the Streamlit Dashboard

From the project directory, install the dependencies and launch the interactive dashboard:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard loads `transactions.parquet` from the project directory by default.

### Option 1: Run Python Script (Recommended)

```bash
python main.py
```

This will execute the complete pipeline:
1. Load and explore data
2. Preprocess and scale features
3. Apply dimensionality reduction
4. Train all clustering models
5. Evaluate and compare models
6. Generate visualizations
7. Print analysis summary

### Option 2: Use Jupyter Notebook

```bash
jupyter notebook unsupervised_learning_project.ipynb
```

The notebook provides step-by-step execution with detailed explanations and inline visualizations.

### Option 3: Use Individual Modules

```python
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from unsupervised_models import KMeansClustering
from evaluation import ClusteringEvaluator

# Load data
loader = DataLoader('transactions.parquet')
df = loader.load_data()

# Preprocess
preprocessor = DataPreprocessor(df)
X = preprocessor.preprocess_pipeline()

# Train model
kmeans = KMeansClustering(n_clusters=5)
kmeans.fit(X)

# Evaluate
evaluator = ClusteringEvaluator(X, kmeans.get_labels())
metrics = evaluator.compute_all_metrics()
```

---

## Module Documentation

### `data_loader.py`
Handles data loading and initial exploration.

**Key Classes:**
- `DataLoader`: Load parquet files and perform EDA

**Example:**
```python
from data_loader import DataLoader

loader = DataLoader('transactions.parquet')
df = loader.load_data()
loader.explore_data()
```

### `data_preprocessor.py`
Performs data cleaning and feature scaling.

**Key Classes:**
- `DataPreprocessor`: Handle missing values, outliers, encoding, and scaling

**Example:**
```python
from data_preprocessor import DataPreprocessor

preprocessor = DataPreprocessor(df)
X = preprocessor.preprocess_pipeline()
```

### `dimensionality_reduction.py`
Applies PCA and t-SNE for dimensionality reduction.

**Key Classes:**
- `DimensionalityReducer`: PCA, t-SNE transformation

**Example:**
```python
from dimensionality_reduction import DimensionalityReducer

reducer = DimensionalityReducer(X)
X_pca = reducer.apply_pca_optimal(variance_threshold=0.95)
X_2d = reducer.apply_pca_2d()
```

### `unsupervised_models.py`
Implements clustering algorithms and anomaly detection.

**Key Classes:**
- `KMeansClustering`: K-Means clustering
- `HierarchicalClustering`: Hierarchical clustering
- `DBSCANClustering`: DBSCAN clustering
- `GaussianMixtureModel`: Gaussian Mixture Models
- `IsolationForestModel`: Isolation Forest anomaly detection

**Example:**
```python
from unsupervised_models import KMeansClustering

kmeans = KMeansClustering(n_clusters=5)
kmeans.fit(X)
labels = kmeans.predict(X_new)
```

### `evaluation.py`
Provides evaluation metrics and model comparison.

**Key Classes:**
- `ClusteringEvaluator`: Compute clustering metrics
- `ModelComparison`: Compare multiple models

**Metrics Computed:**
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Adjusted Rand Index (with ground truth)
- Normalized Mutual Information (with ground truth)

**Example:**
```python
from evaluation import ClusteringEvaluator

evaluator = ClusteringEvaluator(X, labels)
metrics = evaluator.compute_all_metrics()
evaluator.print_summary()
```

### `visualizations.py`
Creates professional visualizations of results.

**Key Classes:**
- `ClusteringVisualizer`: Various plotting functions

**Available Plots:**
- Elbow curve
- Silhouette scores
- BIC/AIC curves
- 2D cluster visualization
- Cluster comparison
- Cluster characteristics
- Correlation matrix
- Anomaly scores
- PCA variance

**Example:**
```python
from visualizations import ClusteringVisualizer

visualizer = ClusteringVisualizer()
visualizer.plot_elbow_curve(inertias, k_range, output_path='elbow.png')
visualizer.plot_clustering_results_2d(X_2d, labels, 'K-Means Results')
```

---

## Algorithms Explained

### K-Means Clustering
- **Purpose**: Partition data into k clusters
- **Method**: Iterative optimization of cluster centroids
- **Best for**: Well-separated, spherical clusters
- **Hyperparameters**: n_clusters (k)

### Hierarchical Clustering
- **Purpose**: Create hierarchical cluster structure
- **Method**: Agglomerative (bottom-up) or divisive (top-down)
- **Best for**: Understanding cluster hierarchy
- **Hyperparameters**: n_clusters, linkage method

### DBSCAN (Density-Based Spatial Clustering)
- **Purpose**: Find clusters of arbitrary shape
- **Method**: Density-based approach
- **Best for**: Non-convex clusters, outlier detection
- **Hyperparameters**: eps (ε), min_samples

### Gaussian Mixture Models (GMM)
- **Purpose**: Model data as mixture of Gaussians
- **Method**: Probabilistic clustering
- **Best for**: Soft clustering, probability estimates
- **Hyperparameters**: n_components

### Isolation Forest
- **Purpose**: Detect anomalies/outliers
- **Method**: Isolation-based anomaly detection
- **Best for**: Anomaly detection with minimal labels
- **Hyperparameters**: contamination rate

---

## Evaluation Metrics

### Silhouette Score
- **Range**: -1 to 1
- **Interpretation**: 
  - > 0.5: Well-separated clusters
  - 0.3-0.5: Reasonable separation
  - < 0.3: Overlapping clusters

### Davies-Bouldin Index
- **Range**: 0 to ∞
- **Interpretation**: Lower is better (more compact, separated clusters)

### Calinski-Harabasz Index
- **Range**: 0 to ∞
- **Interpretation**: Higher is better (more dense, well-separated clusters)

---

## Output Files

The pipeline generates the following comprehensive output files:

### Visualizations (Static & Interactive)
- **Interactive HTML Visualizations**:
  - `interactive_clusters_3d.html`: Rotatable 3D PCA cluster scatter plot with rich hover tooltips
  - `interactive_clusters_2d.html`: Interactive 2D PCA cluster visualization
  - `interactive_clustering_comparison.html`: Multi-panel synchronized interactive comparison of all models
  - `interactive_anomaly_scores.html`: Interactive distribution of Isolation Forest anomaly scores
- **Static PNG Visualizations**:
  - `pca_variance.png`: PCA scree plot and cumulative explained variance
  - `clustering_comparison_2d.png`: 2D comparison of K-Means, Hierarchical, DBSCAN, and GMM
  - `kmeans_clusters_2d.png`: Optimal K-Means clusters in 2D PCA space
  - `hierarchical_clusters_2d.png`: Ward Hierarchical clustering results
  - `dbscan_clusters_2d.png`: Density-based clusters and noise points
  - `gmm_clusters_2d.png`: Gaussian Mixture Models components
  - `isolation_forest_2d.png`: Anomaly distribution in 2D PCA space
  - `anomaly_scores.png`: Isolation Forest anomaly score histogram
  - `elbow_curve.png` & `silhouette_scores.png`: K-Means hyperparameter optimization curves
  - `gmm_bic_aic.png`: GMM model selection via BIC/AIC curves
  - `correlation_matrix.png`, `feature_distributions.png`, `feature_boxplots.png`: EDA charts

### Processed Datasets & Model Results
- `transactions_preprocessed.parquet` & `transactions_preprocessed.csv`: Cleaned, imputed, outlier-capped, and normalized data
- `transactions_clustered.parquet` & `transactions_clustered.csv`: Full transactions dataset enriched with PCA dimensions (`pca_1`, `pca_2`, `pca_3`), cluster assignments from all algorithms, and Isolation Forest anomaly flags & scores
- `model_comparison.csv`: Comprehensive evaluation benchmark metrics across all models

### Academic Reports & Documentation
- `ABSTRACT.md`: Formatted Markdown research abstract with benchmark tables and findings
- `ABSTRACT.txt`: Plain-text academic abstract
- `unsupervised_learning_project.ipynb`: Fully executed Jupyter notebook with inline plots and analysis
- `unsupervised_learning_project.log`: Complete execution logs

---

## Working with Your Data

### Data Format
- Supported: Parquet files (`.parquet`)
- Format: Rows = samples, Columns = features
- Numerical features are processed by default
- Categorical features are automatically encoded

### Data Preparation
1. Ensure data is in parquet format
2. Place file in project directory
3. Update file path in `main.py` if needed

### Expected Output
```
Dataset Shape: (n_samples, n_features)
✓ Data loaded and explored
✓ Data preprocessed and scaled
✓ Dimensionality reduction applied
✓ All models trained
✓ Visualizations generated
```

---

## Tips and Best Practices

### General
1. **Data Quality**: Clean data before clustering
2. **Scaling**: Always scale numerical features
3. **Feature Selection**: Remove irrelevant features
4. **Domain Knowledge**: Use business context to interpret results

### Model Selection
1. **K-Means**: Start with this for quick results
2. **DBSCAN**: Use for non-convex clusters
3. **GMM**: Use when you need soft assignments
4. **Hierarchical**: Use for understanding cluster relationships

### Hyperparameter Tuning
1. **K-Means k**: Use elbow method or silhouette score
2. **DBSCAN eps**: Use k-distance graph
3. **GMM components**: Use BIC or AIC
4. **Sample size**: Larger datasets may need different parameters

### Interpretation
1. Always look at multiple metrics
2. Validate results with domain experts
3. Check cluster characteristics
4. Investigate outliers and anomalies

---

## Troubleshooting

### Issue: File not found
**Solution**: Ensure `transactions.parquet` is in the working directory

### Issue: Memory error with large datasets
**Solution**: Use sampling or PCA reduction before clustering

### Issue: All points are noise in DBSCAN
**Solution**: Increase eps value or decrease min_samples

### Issue: Poor cluster quality
**Solution**: 
- Try different number of clusters
- Use different preprocessing method
- Investigate feature importance

---

## Contributing

To extend this project:

1. Add new models in `unsupervised_models.py`
2. Add new metrics in `evaluation.py`
3. Add new visualizations in `visualizations.py`
4. Update `main.py` to include new components

---

## References

### Papers and Books
- [Scikit-learn Clustering Documentation](https://scikit-learn.org/stable/modules/clustering.html)
- [Unsupervised Learning Handbook](https://www.deeplearningbook.org/)
- [K-Means++ Initialization](https://arxiv.org/abs/1203.6402)

### Libraries Used
- pandas: Data manipulation
- scikit-learn: Machine learning algorithms
- matplotlib/seaborn: Visualization
- numpy: Numerical computing

---

## License

This project is provided as-is for educational and research purposes.

---

## Contact & Support

For questions or issues:
1. Check the troubleshooting section
2. Review module docstrings
3. Consult the Jupyter notebook for detailed explanations

---

## Version History

### v1.0 (Current)
- Initial release
- 5 clustering algorithms
- Comprehensive evaluation framework
- Professional visualizations
- Academic abstract generation

---

## Abstract Example

After running the pipeline, an abstract will be generated summarizing:
- Dataset characteristics
- Methodology used
- Key findings
- Model comparison results
- Recommendations
- Future work directions

See `ABSTRACT.txt` for the generated abstract.

---

**Last Updated**: 2024
**Status**: Production Ready
