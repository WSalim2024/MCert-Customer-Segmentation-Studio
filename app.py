import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# --- PAGE CONFIG ---
st.set_page_config(page_title="Customer Segmentation Studio", page_icon="🛍️", layout="wide")

st.title("🛍️ Customer Segmentation Studio")
st.markdown("""
> **Goal:** Group customers based on their **Annual Income** and **Spending Score**.
> **Algorithms:**
> * **K-Means:** Best for specific number of round clusters.
> * **DBSCAN:** Best for finding outliers and arbitrary shapes.
""")


# --- STEP 1: DATA LOADING ---
@st.cache_data
def load_data():
    # Using the dataset from the activity
    data = {
        # FIX: range(1, 25) creates 24 IDs to match the 24 data points below
        'CustomerID': range(1, 25),
        'AnnualIncome': [15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5,
                         20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24, 24.5,
                         35, 80, 85, 90],
        'SpendingScore': [39, 81, 6, 77, 40, 76, 6, 94, 3, 72,
                          14, 99, 15, 79, 10, 87, 4, 92, 5, 88,
                          35, 80, 85, 90]
    }
    return pd.DataFrame(data)


df = load_data()

# --- STEP 2: SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Configuration")
model_type = st.sidebar.radio("Select Algorithm", ["K-Means", "DBSCAN"])
show_raw = st.sidebar.checkbox("Show Raw Data", value=False)

# --- STEP 3: PREPROCESSING ---
features = df[['AnnualIncome', 'SpendingScore']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# --- MODEL LOGIC ---
if model_type == "K-Means":
    # === K-MEANS LOGIC ===
    k_clusters = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=6, value=3)

    # Elbow Method
    st.subheader("1. Determine Optimal Clusters (Elbow Method)")
    c1, c2 = st.columns([2, 1])
    with c1:
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(features_scaled)
            wcss.append(kmeans.inertia_)
        fig_elbow, ax = plt.subplots(figsize=(8, 3))
        ax.plot(range(1, 11), wcss, marker='o', linestyle='--', color='teal')
        ax.set_xlabel('k')
        ax.set_ylabel('WCSS')
        st.pyplot(fig_elbow)

    # Fit Model
    model = KMeans(n_clusters=k_clusters, init='k-means++', random_state=42)
    df['Cluster'] = model.fit_predict(features_scaled)

else:
    # === DBSCAN LOGIC ===
    st.sidebar.markdown("---")
    eps_val = st.sidebar.slider("Epsilon (Radius)", 0.1, 2.0, 0.5, 0.1)
    min_samples_val = st.sidebar.slider("Min Samples", 2, 10, 3)

    st.subheader("1. Parameter Tuning")
    st.info(f"""
    **Epsilon ({eps_val}):** Max distance between two points to be neighbors.
    **Min Samples ({min_samples_val}):** Min neighbors needed to form a 'Core' cluster.
    """)

    # Fit Model
    model = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    df['Cluster'] = model.fit_predict(features_scaled)

# --- VISUALIZATION ---
st.subheader(f"2. {model_type} Cluster Visualization")
c_viz, c_stats = st.columns([2, 1])

with c_viz:
    fig, ax = plt.subplots(figsize=(10, 6))

    # Handle Outliers for DBSCAN (Cluster -1)
    unique_clusters = sorted(df['Cluster'].unique())
    colors = sns.color_palette("bright", len(unique_clusters))

    for i, cluster in enumerate(unique_clusters):
        cluster_data = df[df['Cluster'] == cluster]

        # Color logic: Black for noise (-1), other colors for clusters
        color = 'black' if cluster == -1 else colors[i]
        label = 'Noise (Outliers)' if cluster == -1 else f'Cluster {cluster}'
        marker = 'x' if cluster == -1 else 'o'

        ax.scatter(
            cluster_data['AnnualIncome'],
            cluster_data['SpendingScore'],
            s=100,
            c=[color],
            label=label,
            marker=marker,
            edgecolors='white'
        )

    ax.set_title(f'Customer Segments ({model_type})')
    ax.set_xlabel('Annual Income')
    ax.set_ylabel('Spending Score')
    ax.legend()
    st.pyplot(fig)

with c_stats:
    st.write("### Cluster Statistics")
    # Count sizes
    counts = df['Cluster'].value_counts().sort_index()
    st.dataframe(counts.rename("Count"))

    if -1 in df['Cluster'].values:
        st.warning(f"⚠️ Detected {len(df[df['Cluster'] == -1])} Noise Points (Outliers)!")

    if show_raw:
        st.dataframe(df)