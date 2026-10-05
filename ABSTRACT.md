# Academic Abstract
## Unsupervised Learning Analysis of Transaction Data

### Background
Transaction data analysis is critical for identifying patterns, detecting anomalies, and understanding customer behavior in financial systems. This study applies unsupervised learning techniques to a transaction dataset containing 5,000 records and 11 features to discover latent structures and anomalous patterns without prior labeling.

### Objective
To identify meaningful clusters and anomalies in transaction data using multiple unsupervised learning algorithms, evaluate their effectiveness through established metrics, and provide actionable business insights.

### Methodology
The dataset was preprocessed by handling missing values, capping outliers using the IQR method, encoding categorical variables, and applying StandardScaler normalization, yielding 11 features. PCA was applied for dimensionality reduction, retaining 95% of explained variance. Five unsupervised learning algorithms were implemented:
1. **K-Means Clustering** (centroid-based clustering with elbow & silhouette optimization)
2. **Hierarchical Clustering** (agglomerative with Ward linkage)
3. **DBSCAN** (density-based spatial clustering with k-distance epsilon estimation)
4. **Gaussian Mixture Models (GMM)** (probabilistic clustering with BIC/AIC component selection)
5. **Isolation Forest** (tree-based anomaly detection)

### Key Findings
- **K-Means**: Identified 3 optimal clusters via silhouette analysis.
- **GMM**: Selected 10 components based on Bayesian Information Criterion (BIC) optimization.
- **DBSCAN** (eps=1.8329): Identified 187 noise points (3.7% of data).
- **Isolation Forest**: Detected 500 anomalies (10.0% of transactions).

### Model Evaluation & Benchmark Metrics

| Model | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index | Clusters | Noise % |
|---|---|---|---|---|---|
| K-Means | 0.2995 | 1.3411 | 2364.04 | 3 | 0.00% |
| Hierarchical | 0.2962 | 1.3398 | 2335.62 | 3 | 0.00% |
| DBSCAN | 0.2117 | 1.0029 | 1107.55 | 3 | 3.74% |
| GMM | 0.0550 | 4.9885 | 800.20 | 10 | 0.00% |

### Conclusions
The analysis reveals distinct transaction patterns that can be leveraged for customer segmentation, fraud detection, and targeted marketing strategies. The multi-algorithm approach provides robust validation of discovered clusters, while anomaly detection highlights potentially fraudulent or unusual transactions requiring further investigation.

### Future Work
Recommendations include:
1. Incorporating temporal features for time-series clustering.
2. Applying semi-supervised learning with domain expert labels.
3. Developing real-time anomaly detection systems.
4. Combining clustering insights with supervised classification for predictive modeling.
