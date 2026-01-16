import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- PAGE CONFIG ---
st.set_page_config(page_title="Customer Segmentation Studio", page_icon="🛍️", layout="wide")

st.title("🛍️ Customer Segmentation Studio")
st.markdown("""
> **Goal:** Group customers based on their **Annual Income** and **Spending Score**.
> **Algorithm:** K-Means Clustering (Unsupervised Learning).
""")


# --- STEP 1 & 2: DATA LOADING ---
@st.cache_data
def load_data():
    # Creating the dummy dataset from the activity
    data = {
        'CustomerID': range(1, 21),
        'AnnualIncome': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                         25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
        'SpendingScore': [39, 81, 6, 77, 40, 76, 6, 94, 3, 72,
                          14, 99, 15, 79, 10, 87, 4, 92, 5, 88]
    }
    return pd.DataFrame(data)


df = load_data()

# Sidebar Controls
st.sidebar.header("⚙️ Configuration")
show_raw = st.sidebar.checkbox("Show Raw Data", value=True)
k_clusters = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=6, value=3)

# --- STEP 3: PREPROCESSING ---
# We only use Income and Spending Score for clustering
features = df[['AnnualIncome', 'SpendingScore']]

# Normalize the data (StandardScaler)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# --- STEP 4: ELBOW METHOD (Helper) ---
st.subheader("1. Determine Optimal Clusters (The Elbow Method)")
col1, col2 = st.columns([2, 1])

with col1:
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(features_scaled)
        wcss.append(kmeans.inertia_)

    # Plot Elbow
    fig_elbow, ax_elbow = plt.subplots(figsize=(10, 4))
    ax_elbow.plot(range(1, 11), wcss, marker='o', linestyle='--', color='teal')
    ax_elbow.set_title('The Elbow Method')
    ax_elbow.set_xlabel('Number of Clusters')
    ax_elbow.set_ylabel('WCSS (Error)')
    ax_elbow.grid(True)
    st.pyplot(fig_elbow)

with col2:
    st.info("""
    **How to read this:**
    Look for the "Elbow" or "Knee" in the line.

    This is the point where adding more clusters stops significantly reducing the error.

    *For this dataset, the elbow is usually around k=3 or k=5.*
    """)

# --- STEP 5: K-MEANS CLUSTERING ---
st.subheader(f"2. Cluster Visualization (k={k_clusters})")

# Apply K-Means
kmeans = KMeans(n_clusters=k_clusters, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(features_scaled)

# --- STEP 6: VISUALIZATION ---
# Create 2 Columns for Viz and Stats
c_viz, c_stats = st.columns([2, 1])

with c_viz:
    fig_cluster, ax_cluster = plt.subplots(figsize=(10, 6))

    # Scatter plot with colors based on cluster
    # We map clusters to a color palette
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']

    for i in range(k_clusters):
        cluster_data = df[df['Cluster'] == i]
        ax_cluster.scatter(
            cluster_data['AnnualIncome'],
            cluster_data['SpendingScore'],
            s=100,
            c=colors[i],
            label=f'Cluster {i + 1}'
        )

    # Plot Centroids (Inverse transform to get real values back)
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    ax_cluster.scatter(centroids[:, 0], centroids[:, 1], s=300, c='yellow', marker='*', label='Centroids',
                       edgecolors='black')

    ax_cluster.set_title('Customer Groups')
    ax_cluster.set_xlabel('Annual Income (k$)')
    ax_cluster.set_ylabel('Spending Score (1-100)')
    ax_cluster.legend()
    ax_cluster.grid(True, alpha=0.3)

    st.pyplot(fig_cluster)

with c_stats:
    st.write("### Cluster Insights")
    # Calculate average stats per cluster
    avg_df = df.groupby('Cluster')[['AnnualIncome', 'SpendingScore']].mean()
    st.dataframe(avg_df.style.format("{:.1f}"))

    st.markdown("---")
    if show_raw:
        st.write("### Raw Data with Labels")
        st.dataframe(df, height=200)

# --- BUSINESS INTERPRETATION ---
st.subheader("3. Business Interpretation")
st.markdown("""
The AI has grouped your customers. Here is how a Marketing Manager would read this:
* **Cluster with Low Income, High Spending:** "Impulse Buyers" -> *Target with budget-friendly deals.*
* **Cluster with High Income, High Spending:** "VIPs" -> *Target with luxury items and exclusive memberships.*
* **Cluster with High Income, Low Spending:** "Savers" -> *Target with high-quality, value-focused marketing.*
""")