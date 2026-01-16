import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# --- PAGE CONFIG ---
st.set_page_config(page_title="Unsupervised Learning Workbench", page_icon="🧪", layout="wide")

st.title("🧪 Unsupervised Learning Workbench v4.0")


# --- STEP 1: DATA LOADING (3D Data) ---
@st.cache_data
def load_data():
    data = {
        'CustomerID': range(1, 46),
        'AnnualIncome': [
            15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5,
            20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24, 24.5,
            25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5,
            30, 30.5, 31, 31.5, 32, 32.5, 33, 33.5, 34, 34.5,
            35, 80, 85, 90, 95
        ],
        'SpendingScore': [
            39, 81, 6, 77, 40, 76, 6, 94, 3, 72,
            14, 99, 15, 79, 10, 87, 4, 92, 5, 88,
            39, 81, 6, 77, 40, 76, 6, 94, 3, 72,
            14, 99, 15, 79, 10, 87, 4, 92, 5, 88,
            35, 80, 85, 90, 92
        ],
        'Age': [
            19, 21, 20, 23, 31, 22, 35, 23, 64, 30,
            67, 35, 58, 24, 37, 22, 35, 20, 52, 35,
            25, 20, 29, 31, 40, 28, 33, 41, 55, 30,
            67, 35, 58, 24, 37, 22, 35, 20, 52, 35,
            30, 45, 48, 50, 52
        ]
    }
    return pd.DataFrame(data)


df = load_data()

# --- PREPROCESSING ---
features = df[['AnnualIncome', 'SpendingScore', 'Age']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# --- NAVIGATION ---
st.sidebar.header("🛠️ Tool Selector")
app_mode = st.sidebar.selectbox("Choose Activity",
                                ["1. Cluster Analysis (K-Means/DBSCAN)", "2. Dimensionality Reduction (PCA/t-SNE)"])


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
        k = st.sidebar.slider("Number of Clusters (k)", 2, 6, 3)
        model = KMeans(n_clusters=k, init='k-means++', random_state=42)
        df['Cluster'] = model.fit_predict(features_scaled)
    else:
        eps = st.sidebar.slider("Epsilon", 0.1, 2.0, 0.5, 0.1)
        min_samples = st.sidebar.slider("Min Samples", 2, 10, 3)
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

        ax.scatter(cluster_data['AnnualIncome'], cluster_data['SpendingScore'], c=[c], s=100, label=label,
                   marker=marker, edgecolors='white')

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

    # 2. Choose Color Overlay (The New Feature!)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🎨 Visualization Overlay**")
    color_by = st.sidebar.selectbox("Color Points By:", ["No Color", "K-Means Clusters", "DBSCAN Clusters"])

    # Calculate Colors based on selection
    if color_by == "K-Means Clusters":
        k = st.sidebar.slider("K (for Coloring)", 2, 6, 3)
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(features_scaled)
        title_suffix = f" (Colored by K-Means, k={k})"
    elif color_by == "DBSCAN Clusters":
        model = DBSCAN(eps=0.5, min_samples=3)
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
        perplexity = st.sidebar.slider("Perplexity", 2, 10, 3)
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
        ax.scatter(cluster_data[x_col], cluster_data[y_col], c=[c], s=100, label=label, marker=marker,
                   edgecolors='black', alpha=0.8)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{dim_algo} Visualization{title_suffix}")
    if color_by != "No Color": ax.legend()
    ax.grid(True, alpha=0.3)

    col1, col2 = st.columns([3, 1])
    col1.pyplot(fig)
    col2.info("""
    **Interpretation:**
    If you see distinct colored groups here, it means the clusters found in 3D space are well-separated even when flattened to 2D.
    """)