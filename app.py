import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# --- PAGE CONFIG ---
st.set_page_config(page_title="Unsupervised Learning Workbench", page_icon="🧪", layout="wide")

st.title("🧪 Unsupervised Learning Workbench v4.1")


# --- STEP 1: DATA LOADING (Synthetic "Mall Customers" Generator) ---
@st.cache_data
def load_data():
    # Generate 240 data points to mimic the famous "Mall Customers" dataset
    # This creates realistic clusters so t-SNE and PCA look good.
    np.random.seed(42)

    # Cluster 1: Low Income, Low Spending (The "Sensible Savers")
    c1 = pd.DataFrame({
        'AnnualIncome': np.random.randint(15, 30, 40),
        'SpendingScore': np.random.randint(5, 25, 40),
        'Age': np.random.randint(40, 60, 40)
    })

    # Cluster 2: Low Income, High Spending (The "Careless Youth")
    c2 = pd.DataFrame({
        'AnnualIncome': np.random.randint(15, 30, 40),
        'SpendingScore': np.random.randint(70, 90, 40),
        'Age': np.random.randint(18, 30, 40)
    })

    # Cluster 3: Mid Income, Mid Spending (The "Average Joes")
    c3 = pd.DataFrame({
        'AnnualIncome': np.random.randint(40, 70, 80),
        'SpendingScore': np.random.randint(40, 60, 80),
        'Age': np.random.randint(30, 50, 80)
    })

    # Cluster 4: High Income, Low Spending (The "Wealthy Savers")
    c4 = pd.DataFrame({
        'AnnualIncome': np.random.randint(70, 130, 40),
        'SpendingScore': np.random.randint(5, 25, 40),
        'Age': np.random.randint(35, 55, 40)
    })

    # Cluster 5: High Income, High Spending (The "VIPs")
    c5 = pd.DataFrame({
        'AnnualIncome': np.random.randint(70, 130, 40),
        'SpendingScore': np.random.randint(75, 95, 40),
        'Age': np.random.randint(25, 45, 40)
    })

    # Combine into one big dataset
    df = pd.concat([c1, c2, c3, c4, c5], ignore_index=True)
    df['CustomerID'] = range(1, len(df) + 1)

    return df


df = load_data()

# --- PREPROCESSING ---
features = df[['AnnualIncome', 'SpendingScore', 'Age']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# --- NAVIGATION ---
st.sidebar.header("🛠️ Tool Selector")
app_mode = st.sidebar.selectbox("Choose Activity",
                                ["1. Cluster Analysis (K-Means/DBSCAN)", "2. Dimensionality Reduction (PCA/t-SNE)"])

# --- DATASET VIEWER (Global Option) ---
if st.sidebar.checkbox("📄 Show Generated Dataset"):
    st.subheader("Raw Data (First 10 Rows)")
    st.dataframe(df.head(10))
    st.caption(f"Total Records: {len(df)} | Columns: AnnualIncome, SpendingScore, Age")


# --- HELPER: COLOR PALETTE ---
def get_colors(n):
    # Standard bright colors for clusters
    base = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6']
    return base[:n] if n <= len(base) else base


# ==========================================
# MODE 1: CLUSTERING
# ==========================================
if app_mode == "1. Cluster Analysis (K-Means/DBSCAN)":
    st.subheader("🔍 Cluster Analysis")
    st.markdown("Group data points based on **Income, Spending Score, AND Age**.")

    algo_type = st.sidebar.radio("Select Algorithm", ["K-Means", "DBSCAN"])

    # Run Algorithm
    if algo_type == "K-Means":
        k = st.sidebar.slider("Number of Clusters (k)", 2, 8, 5)  # Default k=5 fits the synthetic data best
        model = KMeans(n_clusters=k, init='k-means++', random_state=42)
        df['Cluster'] = model.fit_predict(features_scaled)
    else:
        eps = st.sidebar.slider("Epsilon", 0.1, 2.0, 0.5, 0.1)
        min_samples = st.sidebar.slider("Min Samples", 2, 10, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        df['Cluster'] = model.fit_predict(features_scaled)

    # Plotting (2D Slice)
    fig, ax = plt.subplots(figsize=(10, 6))
    unique_clusters = sorted(df['Cluster'].unique())
    colors = get_colors(len(unique_clusters))

    for i, cluster in enumerate(unique_clusters):
        cluster_data = df[df['Cluster'] == cluster]
        c = 'black' if cluster == -1 else colors[i % len(colors)]
        label = 'Noise' if cluster == -1 else f'Cluster {cluster}'
        marker = 'x' if cluster == -1 else 'o'

        ax.scatter(cluster_data['AnnualIncome'], cluster_data['SpendingScore'], c=[c], s=50, label=label, marker=marker,
                   edgecolors='white', linewidth=0.5)

    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Spending Score")
    ax.set_title(f"{algo_type} Results (Income vs Score)")
    ax.legend()
    st.pyplot(fig)

# ==========================================
# MODE 2: DIMENSIONALITY REDUCTION
# ==========================================
else:
    st.subheader("📉 Dimensionality Reduction")
    st.markdown("Flatten 3D data (Income, Score, Age) into 2D to visualize global structure.")

    # 1. Choose Reduction Method
    dim_algo = st.sidebar.radio("Method", ["PCA", "t-SNE"])

    # 2. Choose Color Overlay
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🎨 Visualization Overlay**")
    color_by = st.sidebar.selectbox("Color Points By:", ["No Color", "K-Means Clusters", "DBSCAN Clusters"])

    # Calculate Colors based on selection
    if color_by == "K-Means Clusters":
        k = st.sidebar.slider("K (for Coloring)", 2, 8, 5)
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(features_scaled)
        title_suffix = f" (Colored by K-Means, k={k})"
    elif color_by == "DBSCAN Clusters":
        model = DBSCAN(eps=0.5, min_samples=5)
        labels = model.fit_predict(features_scaled)
        title_suffix = " (Colored by DBSCAN)"
    else:
        labels = [0] * len(df)  # Dummy labels
        title_suffix = ""

    # Run Dimensionality Reduction
    if dim_algo == "PCA":
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(features_scaled)
        x_col, y_col = 'PCA1', 'PCA2'
    else:
        # NOTE: Higher Perplexity (e.g. 30) works better for this larger dataset
        perplexity = st.sidebar.slider("Perplexity", 5, 50, 30)
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        coords = reducer.fit_transform(features_scaled)
        x_col, y_col = 't-SNE1', 't-SNE2'

    # Create Plot DataFrame
    df_viz = pd.DataFrame(coords, columns=[x_col, y_col])
    df_viz['Label'] = labels

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    unique_labels = sorted(df_viz['Label'].unique())
    colors = get_colors(len(unique_labels))

    for i, lbl in enumerate(unique_labels):
        cluster_data = df_viz[df_viz['Label'] == lbl]
        if color_by == "No Color":
            c = 'teal'
            label = None
        else:
            c = 'black' if lbl == -1 else colors[i % len(colors)]
            label = 'Noise' if lbl == -1 else f'Cluster {lbl}'

        marker = 'x' if lbl == -1 else 'o'
        ax.scatter(cluster_data[x_col], cluster_data[y_col], c=[c], s=50, label=label, marker=marker,
                   edgecolors='black', alpha=0.8, linewidth=0.5)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{dim_algo} Visualization{title_suffix}")
    if color_by != "No Color": ax.legend()
    ax.grid(True, alpha=0.3)

    col1, col2 = st.columns([3, 1])
    col1.pyplot(fig)
    col2.info("""
    **Interpretation:**
    * **PCA:** Shows global spread. If you see overlapping colors, the clusters might be similar in some dimensions.
    * **t-SNE:** Shows local groups. You should see distinct islands of color here with Perplexity ~30.
    """)