<div align="center">

# 🧪 Unsupervised Learning Workbench

### **A Comprehensive Dashboard for Clustering & Dimensionality Reduction**

*From Customer Segmentation to Full-Scale ML Experimentation*

---

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

[![GitHub](https://img.shields.io/badge/GitHub-WSalim2024-181717?style=flat-square&logo=github)](https://github.com/WSalim2024)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com)

<br>

[**Features**](#-key-features--modes) · [**Installation**](#-installation) · [**Tech Stack**](#-tech-stack) · [**Roadmap**](#-future-roadmap)

</div>

---

## 📋 Table of Contents

- [Project Evolution](#-project-evolution)
- [Key Features & Modes](#-key-features--modes)
- [Screenshots](#-screenshots)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Directory Structure](#-directory-structure)
- [Future Roadmap](#-future-roadmap)
- [Author](#-author)

---

## 📖 Project Evolution

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           THE EVOLUTION STORY                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│     v1.0                      v2.0                      v3.0                    │
│   ┌─────────┐              ┌─────────┐              ┌─────────┐                 │
│   │Customer │              │  Multi  │              │Unsuper- │                 │
│   │Segment- │    ───►      │  Model  │    ───►      │ vised   │                 │
│   │ation    │              │ Engine  │              │Learning │                 │
│   │  Tool   │              │         │              │Workbench│                 │
│   └─────────┘              └─────────┘              └─────────┘                 │
│                                                                                 │
│   • K-Means only           • + DBSCAN               • + PCA                     │
│   • 2D data                • Outlier Detection      • + t-SNE                   │
│   • Basic viz              • Multi-algorithm        • 3D → 2D reduction         │
│                                                     • Educational focus         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

What started as a simple **Customer Segmentation Tool** has evolved into a comprehensive **Unsupervised Learning Workbench**.

This project now serves as an **educational dashboard** designed to compare different Machine Learning techniques on **high-dimensional data**:

| Dimension | Feature | Description |
|:---------:|:--------|:------------|
| **X₁** | Annual Income | Customer's yearly earnings ($k) |
| **X₂** | Spending Score | Purchase behavior metric (1-100) |
| **X₃** | Age | Customer age in years |

### The Challenge

With **3 dimensions**, traditional 2D scatter plots can't show the complete picture. This workbench solves that problem by offering:

1. **Clustering Algorithms** — Group similar customers together
2. **Dimensionality Reduction** — Flatten 3D data into interpretable 2D views

---

## ✨ Key Features & Modes

The Workbench is organized into two powerful modes:

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         WORKBENCH ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                        ┌─────────────────────────┐                              │
│                        │   🧪 UNSUPERVISED       │                              │
│                        │   LEARNING WORKBENCH    │                              │
│                        └───────────┬─────────────┘                              │
│                                    │                                            │
│                    ┌───────────────┴───────────────┐                            │
│                    │                               │                            │
│                    ▼                               ▼                            │
│         ┌─────────────────────┐       ┌─────────────────────┐                   │
│         │  🔍 MODE A          │       │  📉 MODE B          │                   │
│         │  CLUSTER ANALYSIS   │       │  DIMENSIONALITY     │                   │
│         │                     │       │  REDUCTION          │                   │
│         │  • K-Means          │       │                     │                   │
│         │  • DBSCAN           │       │  • PCA              │                   │
│         │                     │       │  • t-SNE            │                   │
│         └─────────────────────┘       └─────────────────────┘                   │
│                                                                                 │
│         "WHO belongs together?"       "HOW can we SEE the data?"               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

---

### 🔍 Mode A: Cluster Analysis

**Purpose:** Group customers into meaningful segments based on behavioral similarity.

<table>
<tr>
<td width="50%">

#### ⚙️ K-Means Clustering
*Geometric Partitioning*

**How it works:**
Divides data into exactly $k$ groups by minimizing within-cluster distances.

**Interactive Controls:**
- 🎚️ **Clusters ($k$):** Slider from 2 to 10
- 📈 **Elbow Method:** Visual guide for optimal $k$

**Best for:**
- Well-separated, spherical clusters
- When you know the approximate number of segments

</td>
<td width="50%">

#### 🌐 DBSCAN Clustering
*Density-Based Spatial Clustering*

**How it works:**
Finds dense regions and marks sparse points as outliers (noise).

**Interactive Controls:**
- 🎚️ **Epsilon ($\varepsilon$):** Neighborhood radius
- 🎚️ **Min Samples:** Minimum points for dense region

**Best for:**
- Irregular-shaped clusters
- Automatic outlier detection
- Unknown number of segments

</td>
</tr>
</table>

#### Visualization Features

| Feature | Description |
|---------|-------------|
| 🎨 **Auto Color Mapping** | Each cluster gets a distinct color |
| ⭐ **Centroid Markers** | K-Means cluster centers highlighted |
| ⚠️ **Noise Visualization** | DBSCAN outliers shown in distinct color |
| 📊 **2D Scatterplots** | Income vs Spending with cluster overlay |

#### Outlier Detection (DBSCAN)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        DBSCAN NOISE DETECTION                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Spending                                                                      │
│   Score        ●●●                                                              │
│      │        ●●●●●         ◆◆◆                                                 │
│      │         ●●●●        ◆◆◆◆◆                                                │
│      │          ●●          ◆◆◆         ★ ← Outlier (Noise)                     │
│      │                                                                          │
│      │                              ★ ← Outlier (Noise)                         │
│      │     ■■■■                                                                 │
│      │    ■■■■■■                                                                │
│      │     ■■■■         ★ ← Outlier (Noise)                                     │
│      │                                                                          │
│      └──────────────────────────────────────────────────────────────────────    │
│                              Annual Income                                      │
│                                                                                 │
│   Legend:  ● Cluster 1   ◆ Cluster 2   ■ Cluster 3   ★ Noise (Outliers)        │
│                                                                                 │
│   💡 Outliers = Customers with unusual behavior → Investigate individually     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### 📉 Mode B: Dimensionality Reduction

**The Problem:**

Our dataset has **3 dimensions** (Income, Spending Score, Age). Human eyes can only perceive 2D effectively. How do we visualize 3D data?

**The Solution:**

Flatten the data from 3D → 2D while preserving meaningful structure.

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     THE DIMENSIONALITY PROBLEM                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│        3D DATA (Hard to visualize)              2D PROJECTION (Easy to see)    │
│        ────────────────────────────             ───────────────────────────     │
│                                                                                 │
│              Age                                                                │
│               │    ● ●                                  ● ●                     │
│               │  ●     ●                              ●     ●                   │
│               │    ●                                    ●       ●               │
│               │        ● ───────────────────►             ●   ●                 │
│              /│\      ●                              ●  ●    ●                  │
│             / │ \   ●                                  ●   ●                    │
│            /  │  \                                       ●                      │
│           ────┼────── Spending                                                  │
│          /    │     Score                           PC1 / t-SNE₁                │
│       Income                                                                    │
│                                                                                 │
│        "I can't see patterns!"                  "Now I see the clusters!"       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

<table>
<tr>
<td width="50%">

#### 📐 PCA (Principal Component Analysis)
*Linear Transformation*

**How it works:**
Finds the directions (principal components) that capture the most variance in the data.

**Interpretation:**
- **Global Structure** — Shows overall data spread
- **Variance Explained** — Quantifies information retained
- **Linear relationships** preserved

**Best for:**
- Understanding overall data distribution
- Feature importance analysis
- Fast computation

</td>
<td width="50%">

#### 🌀 t-SNE (t-Distributed SNE)
*Non-Linear Embedding*

**How it works:**
Preserves local neighborhoods — points close in 3D stay close in 2D.

**Interpretation:**
- **Local Clusters** — Reveals groupings
- **Non-linear patterns** captured
- **Perplexity** controls neighborhood size

**Best for:**
- Discovering hidden clusters
- Visualizing complex relationships
- Exploratory data analysis

</td>
</tr>
</table>

#### PCA vs t-SNE: When to Use Which?

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PCA vs t-SNE COMPARISON                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                  PCA                                    t-SNE                   │
│         "The Big Picture View"                 "The Neighborhood View"          │
│         ─────────────────────────              ─────────────────────────        │
│                                                                                 │
│         ┌─────────────────────┐                ┌─────────────────────┐          │
│         │    ●                │                │  ●●●      ◆◆◆      │          │
│         │      ●  ●           │                │ ●●●●●    ◆◆◆◆◆     │          │
│         │    ●   ●  ●   ●     │                │  ●●●      ◆◆◆      │          │
│         │  ●    ●    ●    ●   │                │                     │          │
│         │    ●     ●   ●      │                │     ■■■■            │          │
│         │       ●   ●         │                │    ■■■■■■           │          │
│         │         ●           │                │     ■■■■            │          │
│         └─────────────────────┘                └─────────────────────┘          │
│                                                                                 │
│         ✅ Preserves global spread             ✅ Reveals tight clusters        │
│         ✅ Fast computation                    ✅ Non-linear relationships      │
│         ✅ Interpretable axes                  ❌ Slower computation            │
│         ❌ May miss local clusters             ❌ Axes not interpretable        │
│                                                                                 │
│         Use first to understand               Use second to find                │
│         overall structure                     hidden groupings                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📸 Screenshots

<div align="center">

### Mode A: Cluster Analysis

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         [SCREENSHOT PLACEHOLDER]                                │
│                                                                                 │
│                    🔍 K-Means & DBSCAN Clustering Results                       │
│                                                                                 │
│                         Add image: assets/clustering.png                        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Mode B: Dimensionality Reduction

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         [SCREENSHOT PLACEHOLDER]                                │
│                                                                                 │
│                    📉 PCA vs t-SNE Projection Comparison                        │
│                                                                                 │
│                         Add image: assets/dim_reduction.png                     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

*Screenshots will be added after deployment.*

</div>

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology | Components | Purpose |
|:-----:|:----------:|:-----------|:--------|
| 🐍 | **Python 3.10** | — | Core programming language |
| 🖥️ | **Streamlit** | — | Interactive web dashboard |
| 🤖 | **Scikit-Learn** | `KMeans` | Partition-based clustering |
| | | `DBSCAN` | Density-based clustering |
| | | `PCA` | Linear dimensionality reduction |
| | | `TSNE` | Non-linear embedding |
| | | `StandardScaler` | Feature normalization |
| 📊 | **Matplotlib** | — | Optimized visualizations |

</div>

### Why These Choices?

| Technology | Rationale |
|------------|-----------|
| **Streamlit** | Rapid prototyping, no frontend code needed |
| **Scikit-Learn** | Industry-standard ML library with consistent API |
| **Matplotlib** | Memory-optimized for Streamlit deployment |

---

## 📥 Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/WSalim2024/MCert-Customer-Segmentation-Studio.git

# Navigate to project directory
cd MCert-Customer-Segmentation-Studio

# Install dependencies
pip install pandas matplotlib scikit-learn streamlit

# Launch the application
streamlit run app.py
```

### Access the Dashboard

Once launched, open your browser:

```
Local URL: http://localhost:8501
```

---

## 📁 Directory Structure

```
Unsupervised-Learning-Workbench/
│
├── 📄 app.py                    # Main Streamlit application
├── 📄 README.md                 # Project documentation
└── 📄 .gitignore                # Git ignore rules
```

---

## 🚀 Future Roadmap

The Workbench continues to evolve. Here's what's planned:

<div align="center">

| Phase | Feature | Status |
|:-----:|:--------|:------:|
| 🔮 | **Hierarchical Clustering** — Dendrogram visualization | Planned |
| 🔮 | **UMAP** — Faster alternative to t-SNE | Planned |
| 🔮 | **Silhouette Analysis** — Cluster quality metrics | Planned |
| 🔮 | **Data Upload** — Custom CSV file support | Planned |
| 🔮 | **Export Results** — Download cluster assignments | Planned |
| 🔮 | **3D Visualization** — Interactive Plotly 3D scatter | Planned |

</div>

### Contribution Ideas

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         POTENTIAL ENHANCEMENTS                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   CLUSTERING                    REDUCTION                   EVALUATION          │
│   ──────────                    ─────────                   ──────────          │
│                                                                                 │
│   • Agglomerative              • UMAP                      • Silhouette Score   │
│   • Mean-Shift                 • MDS                       • Davies-Bouldin     │
│   • Spectral                   • Isomap                    • Calinski-Harabasz  │
│   • OPTICS                     • LLE                       • Elbow Automation   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 👨‍💻 Author

<div align="center">

### **Waqar Salim**

*Master's Student & IT Professional*

---

[![GitHub](https://img.shields.io/badge/GitHub-WSalim2024-181717?style=for-the-badge&logo=github)](https://github.com/WSalim2024)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/waqar-salim/)

---

**Built with 🧪 experimentation, 📊 data science, and 🎯 purpose**

*Unsupervised Learning Workbench — See the Unseen Patterns in Your Data*

---

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   "The goal is to turn data into information, and information into insight." ║
║                                                        — Carly Fiorina        ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

</div>
