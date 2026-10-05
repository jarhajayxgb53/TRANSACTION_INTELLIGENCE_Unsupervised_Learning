"""
Transaction Clustering & Anomaly Detection - Streamlit Web Application
======================================================================
Interactive interface providing:
1. Single-transaction customer segmentation and real-time fraud risk screening.
2. Dynamic 2D PCA cluster space projection highlighting the analyzed record.
3. Batch CSV/Parquet scoring with downloadable enriched predictions.
4. Macro-level population analytics and rotatable 3D PCA cluster visualization.
"""

import os
import io
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# Import custom project modules
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from unsupervised_models import KMeansClustering, GaussianMixtureModel, IsolationForestModel

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Transaction Intelligence & Anomaly Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling and visual polish
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 10px;
    }
    .badge-normal {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-anomaly {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0px 0px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Cached Pipeline Loader with Dedicated PCA Transformers
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing machine learning models and dataset...")
def load_full_pipeline(data_path="transactions.parquet"):
    """
    Loads dataset, fits preprocessor, dedicated PCA 2D/3D transformers,
    KMeans, GMM, and Isolation Forest.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset '{data_path}' was not found.")

    # 1. Load data
    loader = DataLoader(data_path)
    df_raw = loader.load_data()

    # 2. Preprocess training data
    preprocessor = DataPreprocessor(df_raw)
    X_processed = preprocessor.preprocess_pipeline(handle_outliers=True, scale_method="standard")

    # 3. Fit dedicated 2D & 3D PCA models for projection and plotting
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca_2d = pca_2d.fit_transform(X_processed)

    pca_3d = PCA(n_components=3, random_state=42)
    X_pca_3d = pca_3d.fit_transform(X_processed)

    # 4. Train K-Means (4 clusters)
    kmeans = KMeansClustering(n_clusters=4, random_state=42)
    kmeans.fit(X_processed)

    # 5. Train GMM for confidence probabilities
    gmm = GaussianMixtureModel(n_components=4, random_state=42)
    gmm.fit(X_processed)

    # 6. Train Isolation Forest (5% anomaly threshold)
    iso_forest = IsolationForestModel(contamination=0.05, random_state=42)
    iso_forest.fit(X_processed)

    expected_features = X_processed.columns.tolist()

    # Create background dataframe for population plots
    df_population = df_raw.copy()
    df_population["cluster_id"] = kmeans.labels
    df_population["is_anomaly"] = iso_forest.labels == -1
    df_population["anomaly_score"] = iso_forest.scores
    df_population["pca_x"] = X_pca_2d[:, 0]
    df_population["pca_y"] = X_pca_2d[:, 1]
    df_population["pca_z"] = X_pca_3d[:, 2]

    return {
        "df_raw": df_raw,
        "preprocessor": preprocessor,
        "pca_2d": pca_2d,
        "pca_3d": pca_3d,
        "kmeans": kmeans,
        "gmm": gmm,
        "iso_forest": iso_forest,
        "X_processed": X_processed,
        "expected_features": expected_features,
        "df_population": df_population
    }

try:
    pipeline = load_full_pipeline("transactions.parquet")
except Exception as err:
    st.error(f"❌ Failed to load pipeline: {err}")
    st.stop()

# Segment Personas Definition
CLUSTER_PERSONAS = {
    0: {
        "name": "Everyday Consumer / Retail",
        "description": "Frequent low-value everyday purchases, mobile/POS channel, low risk profile.",
        "badge": "🛒 Retail Shopper"
    },
    1: {
        "name": "Medium Commercial Business",
        "description": "Moderate transaction frequency with healthy account balance and stable tenure.",
        "badge": "🏢 Commercial Tier"
    },
    2: {
        "name": "Corporate Wholesale & Institutional",
        "description": "High ticket amounts, high balances, lower monthly frequency, institutional accounts.",
        "badge": "💼 Enterprise Account"
    },
    3: {
        "name": "International & Premium Cardholder",
        "description": "High travel distances, platinum card tier, higher discounts and irregular timing.",
        "badge": "✈️ Premium Travel"
    }
}

# -----------------------------------------------------------------------------
# 3. Sidebar System Status
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ System Status")
    st.success("All Models Active & Cached")
    st.metric("Total Population", f"{len(pipeline['df_raw']):,} records")
    st.metric("Feature Dimensions", f"{len(pipeline['expected_features'])} features")
    st.metric("Active Segments", f"{pipeline['kmeans'].n_clusters} clusters")

    st.divider()
    st.markdown("### 📖 Segment Directory")
    # Show personas for available clusters
    n_clusters = pipeline['kmeans'].n_clusters
    for cid in range(n_clusters):
        if cid in CLUSTER_PERSONAS:
            st.markdown(f"**Cluster {cid}**: {CLUSTER_PERSONAS[cid]['badge']}")
        else:
            st.markdown(f"**Cluster {cid}**: Segment {cid}")

    st.divider()
    st.caption("Transactions Intelligence Platform • Streamlit & Scikit-Learn")

# -----------------------------------------------------------------------------
# 4. Header & Top-Level Tabs
# -----------------------------------------------------------------------------
st.title("💳 Transaction Intelligence & Anomaly Dashboard")
st.caption("Enterprise behavioral clustering, customer segmentation, and automated fraud risk assessment.")

tab_single, tab_batch, tab_analytics = st.tabs([
    "🔍 Single Transaction Analyzer",
    "📁 Batch CSV Processing",
    "📊 Global Population Analytics"
])

# =============================================================================
# TAB 1: SINGLE TRANSACTION ANALYZER
# =============================================================================
with tab_single:
    st.subheader("1. Enter Transaction Parameters")

    with st.form("single_tx_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 💰 Financial Parameters")
            transaction_amount = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=250000.0, value=250.0, step=25.0)
            account_balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=1000000.0, value=15000.0, step=500.0)
            num_items = st.number_input("Number of Items", min_value=1, max_value=50, value=3, step=1)
            discount_pct = st.slider("Discount Applied (%)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
            transaction_duration = st.number_input("Transaction Duration (sec)", min_value=1.0, max_value=300.0, value=18.0, step=1.0)

        with col2:
            st.markdown("#### 👤 Customer Profile")
            customer_age = st.number_input("Customer Age", min_value=18, max_value=90, value=35, step=1)
            customer_tenure_months = st.number_input("Customer Tenure (months)", min_value=1, max_value=120, value=24, step=1)
            credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=710, step=5)
            transaction_frequency = st.slider("Monthly Tx Frequency", min_value=1, max_value=60, value=12, step=1)
            days_since_last_txn = st.slider("Days Since Last Tx", min_value=0.0, max_value=90.0, value=4.0, step=1.0)
            risk_score = st.slider("Risk Score (0-100)", min_value=0.0, max_value=100.0, value=25.0, step=0.5)

        submitted = st.form_submit_button("🚀 Analyze Transaction & Predict Segment", width='stretch')

    if submitted:
        raw_input_df = pd.DataFrame([{
            "transaction_amount": transaction_amount,
            "transaction_frequency": transaction_frequency,
            "account_balance": account_balance,
            "transaction_duration": transaction_duration,
            "num_items": num_items,
            "discount_pct": discount_pct,
            "customer_age": customer_age,
            "days_since_last_txn": days_since_last_txn,
            "customer_tenure_months": customer_tenure_months,
            "credit_score": credit_score,
            "risk_score": risk_score
        }])

        encoded_input = pd.get_dummies(raw_input_df, drop_first=True)
        aligned_input = encoded_input.reindex(columns=pipeline["expected_features"], fill_value=0)
        scaler = pipeline["preprocessor"].get_scaler()
        X_scaled = scaler.transform(aligned_input)
        X_scaled_df = pd.DataFrame(X_scaled, columns=pipeline["expected_features"])

        cluster_id = int(pipeline["kmeans"].predict(X_scaled_df)[0])
        persona = CLUSTER_PERSONAS.get(cluster_id, {"name": f"Segment {cluster_id}", "description": "", "badge": "Segment"})
        cluster_probs = pipeline["gmm"].predict_proba(X_scaled_df)[0]
        confidence = cluster_probs[cluster_id] * 100

        is_anomaly = bool(pipeline["iso_forest"].predict(X_scaled_df)[0] == -1)
        raw_anomaly_score = float(pipeline["iso_forest"].model.decision_function(X_scaled_df)[0])
        normalized_risk = float(np.clip((0.15 - raw_anomaly_score) / 0.30 * 100, 0, 100))

        user_pca_2d = pipeline["pca_2d"].transform(X_scaled_df)[0]

        st.divider()
        st.subheader("2. Real-Time Analysis Results")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown(f"### {persona['badge']}")
            st.markdown(f"**Persona Title:** {persona['name']}")
            st.info(persona["description"])
            st.metric(label="Assigned Cluster", value=f"Cluster #{cluster_id}")
            st.metric(label="Cluster Membership Confidence", value=f"{confidence:.1f}%")
            st.progress(float(confidence / 100))

        with res_col2:
            st.markdown("### 🛡️ Fraud & Anomaly Screening")
            if is_anomaly:
                st.error("⚠️ **SUSPICIOUS TRANSACTION DETECTED**\n\nDiverges significantly from customer behavioral baselines.")
            else:
                st.success("✅ **NORMAL TRANSACTION PATTERN**\n\nTransaction fits cleanly within expected customer behavior.")

            st.metric(
                label="Risk Rating",
                value=f"{normalized_risk:.1f} / 100",
                delta="High Risk" if is_anomaly else "Nominal Risk",
                delta_color="inverse"
            )
            st.caption(f"Raw Decision Score: `{raw_anomaly_score:.4f}` (Values < 0 trigger anomaly alerts)")

        st.divider()
        st.subheader("3. Visual Cluster Projection & Distribution")

        vcol1, vcol2 = st.columns([3, 2])

        with vcol1:
            st.markdown("#### 🗺️ 2D PCA Cluster Map (Your Transaction Highlighted)")
            bg_sample = pipeline["df_population"].sample(n=min(1200, len(pipeline["df_population"])), random_state=42)
            
            fig_2d = px.scatter(
                bg_sample,
                x="pca_x",
                y="pca_y",
                color=bg_sample["cluster_id"].astype(str),
                opacity=0.45,
                hover_data=["transaction_amount", "transaction_frequency", "account_balance"],
                labels={"color": "Cluster ID", "pca_x": "PCA Component 1", "pca_y": "PCA Component 2"},
                title="Population Clusters vs. Your Current Transaction"
            )
            
            fig_2d.add_trace(go.Scatter(
                x=[user_pca_2d[0]],
                y=[user_pca_2d[1]],
                mode="markers+text",
                marker=dict(symbol="star", size=20, color="#FF0055", line=dict(width=2, color="white")),
                name="⭐ Current Transaction",
                text=["⭐ YOUR TRANSACTION"],
                textposition="top center"
            ))
            fig_2d.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_2d, use_container_width=True)

        with vcol2:
            st.markdown("#### 📉 Anomaly Score Distribution")
            fig_dist = px.histogram(
                pipeline["df_population"],
                x="anomaly_score",
                nbins=40,
                color="is_anomaly",
                color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
                labels={"is_anomaly": "Is Anomaly?", "anomaly_score": "Isolation Forest Score"},
                title="Population Score Distribution"
            )
            fig_dist.add_vline(
                x=raw_anomaly_score,
                line_dash="dash",
                line_color="#FF0055",
                annotation_text="Your Tx",
                annotation_position="top left"
            )
            st.plotly_chart(fig_dist, use_container_width=True)

# =============================================================================
# TAB 2: BATCH CSV PROCESSING
# =============================================================================
with tab_batch:
    st.subheader("📁 Batch File Scoring & Screening")
    st.markdown("Upload a `.csv` or `.parquet` file containing transactions to perform batch clustering and anomaly screening.")
    st.caption("Required columns: transaction_amount, transaction_frequency, account_balance, transaction_duration, num_items, discount_pct, customer_age, days_since_last_txn, customer_tenure_months, credit_score, risk_score")

    uploaded_file = st.file_uploader("Upload transactions dataset", type=["csv", "parquet"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".parquet"):
                batch_df = pd.read_parquet(uploaded_file)
            else:
                batch_df = pd.read_csv(uploaded_file)

            st.success(f"✓ File successfully uploaded: `{uploaded_file.name}` ({len(batch_df):,} records)")

            core_cols = ["transaction_amount", "account_balance", "transaction_frequency", "transaction_duration", "num_items", "discount_pct", "customer_age", "days_since_last_txn", "customer_tenure_months", "credit_score", "risk_score"]
            missing_cols = [c for c in core_cols if c not in batch_df.columns]

            if missing_cols:
                st.error(f"❌ Uploaded file is missing required columns: `{missing_cols}`. Please check schema.")
            else:
                if st.button("⚡ Run Batch Analysis on Uploaded File", width='stretch'):
                    with st.spinner("Processing batch records through pipeline..."):
                        batch_encoded = pd.get_dummies(batch_df, drop_first=True)
                        batch_aligned = batch_encoded.reindex(columns=pipeline["expected_features"], fill_value=0)

                        scaler = pipeline["preprocessor"].get_scaler()
                        batch_scaled = scaler.transform(batch_aligned)

                        batch_clusters = pipeline["kmeans"].predict(batch_scaled)
                        batch_anomaly_preds = pipeline["iso_forest"].predict(batch_scaled)
                        batch_anomaly_scores = pipeline["iso_forest"].model.decision_function(batch_scaled)

                        results_df = batch_df.copy()
                        results_df["predicted_cluster"] = batch_clusters
                        results_df["is_anomaly"] = batch_anomaly_preds == -1
                        results_df["anomaly_score"] = np.round(batch_anomaly_scores, 4)

                        total_uploaded = len(results_df)
                        total_anomalies = int(results_df["is_anomaly"].sum())
                        anomaly_rate = (total_anomalies / total_uploaded) * 100

                        mcol1, mcol2, mcol3 = st.columns(3)
                        mcol1.metric("Total Records Processed", f"{total_uploaded:,}")
                        mcol2.metric("Anomalies Flagged", f"{total_anomalies:,}")
                        mcol3.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")

                        st.markdown("#### 📋 Batch Predictions Preview")
                        st.dataframe(results_df.head(100), width='stretch')

                        csv_buffer = io.StringIO()
                        results_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="📥 Download Full Scored Dataset as CSV",
                            data=csv_buffer.getvalue(),
                            file_name=f"scored_transactions_{uploaded_file.name.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            width='stretch'
                        )

        except Exception as e:
            st.error(f"❌ Error processing uploaded file: {str(e)}")
    else:
        st.info("💡 Tip: You can test this batch uploader using the sample `transactions.parquet` file in your project folder.")

# =============================================================================
# TAB 3: GLOBAL POPULATION ANALYTICS
# =============================================================================
with tab_analytics:
    st.subheader("📊 Global Population Analytics & Cluster Discovery")
    st.markdown("Macro-level analysis across the entire training dataset.")

    acol1, acol2 = st.columns([3, 2])

    with acol1:
        st.markdown("#### 🌐 3D Interactive PCA Cluster Space")
        sample_3d = pipeline["df_population"].sample(n=min(1000, len(pipeline["df_population"])), random_state=42)
        fig_3d = px.scatter_3d(
            sample_3d,
            x="pca_x",
            y="pca_y",
            z="pca_z",
            color=sample_3d["cluster_id"].astype(str),
            opacity=0.7,
            labels={"color": "Cluster ID", "pca_x": "PC 1", "pca_y": "PC 2", "pca_z": "PC 3"},
            title="3D PCA Customer Segments"
        )
        fig_3d.update_layout(scene=dict(aspectmode='cube'), margin=dict(l=0, r=0, b=0, t=30))
        st.plotly_chart(fig_3d, width='stretch')

    with acol2:
        st.markdown("#### 👥 Segment Distribution Breakdown")
        cluster_counts = pipeline["df_population"]["cluster_id"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster ID", "Record Count"]

        # Handle both 3 and 4 cluster scenarios
        def get_persona_badge(cid):
            if cid in CLUSTER_PERSONAS:
                return CLUSTER_PERSONAS[cid]["badge"]
            return f"Cluster {cid}"

        cluster_counts["Persona"] = cluster_counts["Cluster ID"].map(get_persona_badge)

        fig_bar = px.bar(
            cluster_counts,
            x="Persona",
            y="Record Count",
            color="Cluster ID",
            text="Record Count",
            title="Transactions by Persona Tier"
        )
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, width='stretch')

    st.divider()
    st.markdown("#### 🏆 Model Benchmark & Performance Metrics")
    if os.path.exists("model_comparison.csv"):
        comp_df = pd.read_csv("model_comparison.csv")
        st.dataframe(comp_df, width='stretch')
    else:
        st.caption("Model benchmark comparison table will appear once main.py is run.")
