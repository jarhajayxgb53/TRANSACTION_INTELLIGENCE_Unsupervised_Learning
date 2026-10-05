"""
Generate synthetic transactions.parquet dataset for the unsupervised learning project.
Creates a realistic transaction dataset with multiple features and natural cluster structure.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

n_samples = 5000

# Create cluster-based transaction data (4 natural segments)
# Cluster 0: Small retail purchases (frequent, low amount)
# Cluster 1: Medium business transactions
# Cluster 2: Large corporate transfers
# Cluster 3: International/premium transactions

cluster_sizes = [2000, 1500, 800, 700]
cluster_labels = np.repeat([0, 1, 2, 3], cluster_sizes)

# Transaction Amount
amounts = np.concatenate([
    np.random.lognormal(mean=3.0, sigma=0.8, size=cluster_sizes[0]),    # ~$20-100
    np.random.lognormal(mean=5.5, sigma=0.6, size=cluster_sizes[1]),    # ~$200-500
    np.random.lognormal(mean=7.5, sigma=0.5, size=cluster_sizes[2]),    # ~$1500-3000
    np.random.lognormal(mean=6.5, sigma=1.0, size=cluster_sizes[3]),    # ~$500-2000
])

# Transaction Frequency (per month)
frequency = np.concatenate([
    np.random.poisson(lam=15, size=cluster_sizes[0]),
    np.random.poisson(lam=8, size=cluster_sizes[1]),
    np.random.poisson(lam=3, size=cluster_sizes[2]),
    np.random.poisson(lam=5, size=cluster_sizes[3]),
])

# Account Balance
balance = np.concatenate([
    np.random.normal(loc=2000, scale=800, size=cluster_sizes[0]),
    np.random.normal(loc=15000, scale=5000, size=cluster_sizes[1]),
    np.random.normal(loc=100000, scale=30000, size=cluster_sizes[2]),
    np.random.normal(loc=50000, scale=15000, size=cluster_sizes[3]),
])
balance = np.abs(balance)

# Transaction Duration (seconds)
duration = np.concatenate([
    np.random.exponential(scale=5, size=cluster_sizes[0]),
    np.random.exponential(scale=15, size=cluster_sizes[1]),
    np.random.exponential(scale=30, size=cluster_sizes[2]),
    np.random.exponential(scale=20, size=cluster_sizes[3]),
])

# Number of Items
n_items = np.concatenate([
    np.random.poisson(lam=3, size=cluster_sizes[0]),
    np.random.poisson(lam=5, size=cluster_sizes[1]),
    np.random.poisson(lam=2, size=cluster_sizes[2]),
    np.random.poisson(lam=8, size=cluster_sizes[3]),
])
n_items = np.clip(n_items, 1, None)

# Discount Applied (%)
discount = np.concatenate([
    np.random.uniform(0, 10, size=cluster_sizes[0]),
    np.random.uniform(5, 15, size=cluster_sizes[1]),
    np.random.uniform(0, 5, size=cluster_sizes[2]),
    np.random.uniform(10, 25, size=cluster_sizes[3]),
])

# Customer Age
age = np.concatenate([
    np.random.normal(loc=28, scale=8, size=cluster_sizes[0]),
    np.random.normal(loc=42, scale=10, size=cluster_sizes[1]),
    np.random.normal(loc=50, scale=8, size=cluster_sizes[2]),
    np.random.normal(loc=35, scale=12, size=cluster_sizes[3]),
])
age = np.clip(age, 18, 80).astype(int)

# Days Since Last Transaction
days_since = np.concatenate([
    np.random.exponential(scale=3, size=cluster_sizes[0]),
    np.random.exponential(scale=7, size=cluster_sizes[1]),
    np.random.exponential(scale=15, size=cluster_sizes[2]),
    np.random.exponential(scale=10, size=cluster_sizes[3]),
])

# Customer Tenure (months)
tenure = np.concatenate([
    np.random.normal(loc=12, scale=6, size=cluster_sizes[0]),
    np.random.normal(loc=36, scale=12, size=cluster_sizes[1]),
    np.random.normal(loc=60, scale=18, size=cluster_sizes[2]),
    np.random.normal(loc=24, scale=10, size=cluster_sizes[3]),
])
tenure = np.clip(tenure, 1, 120).astype(int)

# Credit Score
credit_score = np.concatenate([
    np.random.normal(loc=650, scale=50, size=cluster_sizes[0]),
    np.random.normal(loc=720, scale=40, size=cluster_sizes[1]),
    np.random.normal(loc=780, scale=30, size=cluster_sizes[2]),
    np.random.normal(loc=700, scale=45, size=cluster_sizes[3]),
])
credit_score = np.clip(credit_score, 300, 850).astype(int)

# Risk Score (0-100)
risk_score = np.concatenate([
    np.random.beta(a=2, b=5, size=cluster_sizes[0]) * 100,
    np.random.beta(a=2, b=8, size=cluster_sizes[1]) * 100,
    np.random.beta(a=1, b=10, size=cluster_sizes[2]) * 100,
    np.random.beta(a=3, b=4, size=cluster_sizes[3]) * 100,
])

# Add some anomalies (inject 50 unusual transactions)
n_anomalies = 50
anomaly_idx = np.random.choice(n_samples, n_anomalies, replace=False)
amounts[anomaly_idx] *= np.random.uniform(5, 20, size=n_anomalies)
risk_score[anomaly_idx] = np.random.uniform(80, 100, size=n_anomalies)

# Create DataFrame
df = pd.DataFrame({
    'transaction_amount': np.round(amounts, 2),
    'transaction_frequency': frequency,
    'account_balance': np.round(balance, 2),
    'transaction_duration': np.round(duration, 2),
    'num_items': n_items,
    'discount_pct': np.round(discount, 2),
    'customer_age': age,
    'days_since_last_txn': np.round(days_since, 2),
    'customer_tenure_months': tenure,
    'credit_score': credit_score,
    'risk_score': np.round(risk_score, 2),
})

# Shuffle to remove cluster ordering
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Introduce some missing values (~2% missing in select columns)
for col in ['discount_pct', 'days_since_last_txn', 'credit_score']:
    mask = np.random.random(n_samples) < 0.02
    df.loc[mask, col] = np.nan

# Save as parquet
df.to_parquet('transactions.parquet', index=False, engine='pyarrow')

print(f"Dataset created: {df.shape}")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nSample:\n{df.head()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nSaved to transactions.parquet")
