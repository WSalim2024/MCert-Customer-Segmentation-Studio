<div align="center">

# 🧪 Unsupervised Learning Workbench

### **Version 4.0 — The Unified Update**

*Cross-Reference Clustering with Dimensionality Reduction for Complete Visual Validation*

---

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

[![GitHub](https://img.shields.io/badge/GitHub-WSalim2024-181717?style=flat-square&logo=github)](https://github.com/WSalim2024)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com)

<br>

[**Features**](#-feature-breakdown) · [**Installation**](#-quick-start) · [**Tech Stack**](#-technical-implementation) · [**Screenshots**](#-screenshots)

</div>

---

## 📋 Table of Contents

- [Project Evolution](#-project-evolution)
- [What's New in v4.0](#-whats-new-in-v40)
- [Feature Breakdown](#-feature-breakdown)
- [Screenshots](#-screenshots)
- [Technical Implementation](#-technical-implementation)
- [Quick Start](#-quick-start)
- [Directory Structure](#-directory-structure)
- [Author](#-author)

---

## 📖 Project Evolution

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        THE EVOLUTION TO v4.0                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   v1.0              v2.0              v3.0              v4.0                    │
│  ┌──────┐          ┌──────┐          ┌──────┐          ┌──────┐                 │
│  │Basic │   ───►   │Multi │   ───►   │Dim   │   ───►   │UNIFIED│                │
│  │K-Means│          │Model │          │Reduc-│          │ANALYSIS│               │
│  │      │          │Engine│          │tion  │          │       │                │
│  └──────┘          └──────┘          └──────┘          └──────┘                 │
│                                                                                 │
│  • Single          • +DBSCAN         • +PCA            • Cluster +              │
│    algorithm       • Outlier         • +t-SNE            Reduction              │
│  • 2D data           detection       • 3D→2D             INTEGRATION            │
│                                        projection       • Visual                │
│                                                           validation            │
│                                                                                 │
│                                                         ▲                       │
│                                                         │                       │
│                                              ┌──────────┴──────────┐            │
│                                              │   THE BREAKTHROUGH   │            │
│                                              │   Cross-reference    │            │
│                                              │   clusters with      │            │
│                                              │   reduced dimensions │            │
│                                              └─────────────────────┘            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

What began as a simple **Customer Segmentation Tool** has matured into a complete **Educational Workbench** for Unsupervised Machine Learning.

### The Journey

| Version | Codename | Key Innovation |
|:-------:|:---------|:---------------|
| v1.0 | *Foundation* | Basic K-Means clustering |
| v2.0 | *Multi-Engine* | Added DBSCAN with outlier detection |
| v3.0 | *Visualization* | PCA & t-SNE dimensionality reduction |
| **v4.0** | **Unified Update** | **Cross-referenced analysis** |

---

## 🚀 What's New in v4.0

### The Key Innovation: Unified Analysis

Previous versions treated **Clustering** and **Dimensionality Reduction** as separate operations. Version 4.0 bridges this gap with **cross-referenced analysis**.

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         THE UNIFIED ANALYSIS CONCEPT                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   BEFORE v4.0 (Isolated Analysis)                                               │
│   ────────────────────────────────                                              │
│                                                                                 │
│   ┌─────────────────┐         ┌─────────────────┐                               │
│   │  TAB 1          │         │  TAB 2          │                               │
│   │  Clustering     │    ✗    │  Reduction      │     No connection!            │
│   │  (K-Means)      │◄───────►│  (PCA/t-SNE)    │     Results in silos.         │
│   │                 │         │                 │                               │
│   │  🔴🔵🟢🟡        │         │  ●●●●●●●●●      │                               │
│   └─────────────────┘         └─────────────────┘                               │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                 │
│   v4.0 (Unified Analysis)                                                       │
│   ───────────────────────                                                       │
│                                                                                 │
│   ┌─────────────────┐         ┌─────────────────┐                               │
│   │  TAB 1          │         │  TAB 2          │                               │
│   │  Clustering     │────────►│  Reduction      │     CONNECTED!                │
│   │  (K-Means)      │ Labels  │  (PCA/t-SNE)    │     See clusters in           │
│   │                 │ passed  │                 │     reduced space.            │
│   │  🔴🔵🟢🟡        │         │  🔴🔵🟢🟡        │                               │
│   └─────────────────┘         └─────────────────┘                               │
│                                                                                 │
│   🎯 KEY INSIGHT: Validate if clusters that are mathematically distinct        │
│                   in 3D are also VISUALLY distinct in 2D                        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

### Why This Matters

| Question | How v4.0 Answers It |
|----------|---------------------|
| "Are my K-Means clusters actually separated?" | Project them onto PCA/t-SNE and see visually |
| "Does DBSCAN's grouping make sense?" | Overlay DBSCAN labels on t-SNE to validate density regions |
| "Which reduction method shows my clusters better?" | Compare PCA vs t-SNE with same cluster coloring |

---

## 🎨 Feature Breakdown

The Workbench is organized into two integrated tabs:

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         WORKBENCH ARCHITECTURE v4.0                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                        ┌───────────────────────────┐                            │
│                        │  🧪 UNSUPERVISED LEARNING │                            │
│                        │       WORKBENCH v4.0      │                            │
│                        └─────────────┬─────────────┘                            │
│                                      │                                          │
│                    ┌─────────────────┴─────────────────┐                        │
│                    │                                   │                        │
│                    ▼                                   ▼                        │
│         ┌─────────────────────┐           ┌─────────────────────┐              │
│         │  🔍 TAB 1           │           │  📉 TAB 2           │              │
│         │  CLUSTER ANALYSIS   │──────────►│  DIMENSIONALITY     │              │
│         │                     │  Labels   │  REDUCTION          │              │
│         │  • K-Means          │  passed   │                     │              │
│         │  • DBSCAN           │  to Tab 2 │  • PCA              │              │
│         │                     │           │  • t-SNE            │              │
│         │  Output: Cluster    │           │  • "Color By..." 🆕 │              │
│         │  assignments        │           │                     │              │
│         └─────────────────────┘           └─────────────────────┘              │
│                                                                                 │
│                              UNIFIED ANALYSIS FLOW                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

---

### 🔍 Tab 1: Cluster Analysis

**Purpose:** Group customers into meaningful segments based on behavioral similarity in 3D space.

#### Input Data

| Dimension | Feature | Range | Description |
|:---------:|:--------|:------|:------------|
| **X₁** | Annual Income | $15k - $137k | Customer's yearly earnings |
| **X₂** | Spending Score | 1 - 99 | Purchase behavior metric |
| **X₃** | Age | 18 - 70 | Customer age in years |

#### Available Algorithms

<table>
<tr>
<td width="50%">

##### ⚙️ K-Means Clustering

*Geometric Partitioning*

**How it works:**
Divides customers into exactly $k$ groups by minimizing within-cluster variance.

**Interactive Controls:**
```
┌─────────────────────────────┐
│  Number of Clusters (k)     │
│  [2]────────●────────[10]   │
│            k = 5            │
└─────────────────────────────┘
```

**Output:** Cluster labels (0, 1, 2, ... k-1)

</td>
<td width="50%">

##### 🌐 DBSCAN Clustering

*Density-Based Spatial Clustering*

**How it works:**
Finds dense regions automatically; sparse points become outliers.

**Interactive Controls:**
```
┌─────────────────────────────┐
│  Epsilon (ε)                │
│  [0.1]───────●───────[2.0]  │
│            ε = 0.5          │
├─────────────────────────────┤
│  Min Samples                │
│  [2]─────────●───────[20]   │
│          min = 5            │
└─────────────────────────────┘
```

**Output:** Cluster labels + Noise (-1)

</td>
</tr>
</table>

#### Visualization

2D projection showing **Annual Income vs Spending Score** with:
- 🎨 Color-coded cluster assignments
- ⭐ Centroid markers (K-Means)
- ⚠️ Noise points highlighted (DBSCAN)

---

### 📉 Tab 2: Dimensionality Reduction *(The v4.0 Star)*

**Purpose:** Flatten 3D data into 2D while preserving meaningful structure — now with **cluster overlay capability**.

#### The 3D → 2D Problem

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     WHY DIMENSIONALITY REDUCTION?                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│        3D DATA                                     2D PROJECTION                │
│        ────────                                    ──────────────               │
│                                                                                 │
│           Age                                                                   │
│            │      ●                                      ●  ●                   │
│            │    ●   ●                                  ●  ●  ●                  │
│            │  ●       ●                               ●      ●                  │
│            │    ●   ●        ═══════════►                  ●                    │
│           /│\     ●                                   ●  ●    ●                 │
│          / │ \                                         ●  ●                     │
│         /  │  \                                                                 │
│     Income─┴───Spending                              Component 1                │
│                                                                                 │
│   😵 "I can't visualize                          😊 "Now I can see              │
│       3 dimensions!"                                  the patterns!"            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Available Algorithms

<table>
<tr>
<td width="50%">

##### 📐 PCA (Principal Component Analysis)

*Linear Transformation*

**Interpretation:**
- Preserves **global structure**
- Shows overall data spread
- Axes represent directions of maximum variance

**Best for:**
- Understanding overall distribution
- Fast, deterministic results
- Interpretable components

</td>
<td width="50%">

##### 🌀 t-SNE (t-Distributed SNE)

*Non-Linear Embedding*

**Interpretation:**
- Preserves **local neighborhoods**
- Points close in 3D stay close in 2D
- Reveals hidden cluster structure

**Best for:**
- Discovering tight groupings
- Non-linear relationships
- Exploratory visualization

</td>
</tr>
</table>

---

#### 🆕 NEW FEATURE: "Color Points By..." Dropdown

This is the **breakthrough feature** of v4.0. The dropdown allows users to overlay clustering results onto the dimensionality reduction plot.

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    "COLOR POINTS BY..." DROPDOWN                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────────────────────────────────┐                                   │
│   │  Color Points By:          [▼]         │                                   │
│   ├─────────────────────────────────────────┤                                   │
│   │  ○ None (Single Color)                 │ ← Default: all points same color  │
│   │  ● K-Means Clusters                    │ ← Overlay K-Means labels          │
│   │  ○ DBSCAN Clusters                     │ ← Overlay DBSCAN labels           │
│   └─────────────────────────────────────────┘                                   │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                 │
│   VISUAL RESULT:                                                                │
│                                                                                 │
│   None (Single Color)        K-Means Colored          DBSCAN Colored           │
│   ───────────────────        ───────────────          ──────────────           │
│                                                                                 │
│      ●  ●  ●                   🔴  🔴  🔵                🟢  🟢  🔵              │
│    ●  ●  ●  ●               🔴  🔴  🔵  🔵            🟢  🟢  🔵  🔵            │
│      ●  ●  ●                   🔴  🔵  🔵                🟢  🔵  ⚫              │
│    ●  ●  ●  ●               🟡  🟡  🟢  🟢            🟡  🟡  🟡  ⚫            │
│      ●  ●  ●                   🟡  🟡  🟢                🟡  🟡  ⚫              │
│                                                              ▲                  │
│   "Just the shape"          "Clusters visible!"       Noise shown (⚫)          │
│                                                                                 │
│   🎯 INSIGHT: If colors cluster together in 2D, your 3D clustering is valid!   │
│              If colors are scattered, clusters may be overlapping.              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

#### How to Interpret the Unified View

| Observation | Meaning | Action |
|-------------|---------|--------|
| 🟢 **Colors form tight groups** | Clusters are well-separated in reduced space | ✅ Clustering is valid |
| 🟡 **Colors partially mixed** | Some cluster overlap exists | ⚠️ Consider adjusting $k$ or $\varepsilon$ |
| 🔴 **Colors completely scattered** | Clusters don't translate to 2D | ❌ Re-evaluate clustering parameters |
| ⚫ **Noise points isolated** (DBSCAN) | Outliers genuinely different | ✅ DBSCAN working correctly |

---

## 📸 Screenshots

<div align="center">

### Tab 1: Cluster Analysis

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         [SCREENSHOT PLACEHOLDER]                                │
│                                                                                 │
│                    🔍 K-Means & DBSCAN Clustering Interface                     │
│                                                                                 │
│                         Add image: assets/tab1_clustering.png                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Tab 2: Dimensionality Reduction with Cluster Overlay

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         [SCREENSHOT PLACEHOLDER]                                │
│                                                                                 │
│                    📉 PCA/t-SNE with "Color Points By..." Feature               │
│                                                                                 │
│                         Add image: assets/tab2_unified.png                      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### The Unified Analysis in Action

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         [SCREENSHOT PLACEHOLDER]                                │
│                                                                                 │
│                    🎯 K-Means Clusters Projected onto t-SNE Map                 │
│                                                                                 │
│                         Add image: assets/unified_analysis.png                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

*Screenshots will be added after deployment.*

</div>

---

## 🛠️ Technical Implementation

### Performance Optimization: "Lite" Architecture

Version 4.0 introduces a streamlined architecture optimized for **Streamlit Cloud deployment**.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         ARCHITECTURE COMPARISON                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   BEFORE (Heavy)                           AFTER (Lite)                         │
│   ──────────────                           ────────────                         │
│                                                                                 │
│   ┌─────────────────┐                      ┌─────────────────┐                  │
│   │ Seaborn         │ ───► Removed         │ Matplotlib      │ ✅ Retained     │
│   │ (Heavy styling) │                      │ (Core plotting) │                  │
│   └─────────────────┘                      └─────────────────┘                  │
│                                                                                 │
│   Memory: ~150MB                           Memory: ~80MB                        │
│   Load time: 3-4s                          Load time: 1-2s                      │
│   Cloud issues: ⚠️ Yes                     Cloud issues: ✅ None                │
│                                                                                 │
│   💡 Result: Faster loads, no memory errors, same visual quality               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Tech Stack

<div align="center">

| Layer | Technology | Version | Purpose |
|:-----:|:----------:|:-------:|:--------|
| 🐍 | **Python** | 3.10 | Core runtime |
| 🖥️ | **Streamlit** | 1.28+ | Interactive dashboard |
| 🤖 | **Scikit-Learn** | 1.3+ | ML algorithms |
| | | `KMeans` | Partition clustering |
| | | `DBSCAN` | Density clustering |
| | | `PCA` | Linear reduction |
| | | `TSNE` | Non-linear embedding |
| | | `StandardScaler` | Feature normalization |
| 📊 | **Matplotlib** | 3.7+ | Optimized visualizations |
| 📋 | **Pandas** | 2.0+ | Data manipulation |

</div>

### Why These Choices?

| Decision | Rationale |
|----------|-----------|
| **Matplotlib over Seaborn** | Lower memory footprint, faster rendering on Streamlit Cloud |
| **StandardScaler** | Essential for distance-based algorithms (K-Means, DBSCAN, t-SNE) |
| **Session State for Labels** | Enables cross-tab communication without data duplication |

---

## 🚀 Quick Start

### Installation

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

```
Local URL: http://localhost:8501
```

### Recommended Workflow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         RECOMMENDED ANALYSIS FLOW                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   STEP 1                    STEP 2                    STEP 3                    │
│   ──────                    ──────                    ──────                    │
│                                                                                 │
│   🔍 Tab 1: Cluster         📉 Tab 2: Reduce          🎯 Validate               │
│                                                                                 │
│   Run K-Means or     ───►   Select PCA or      ───►   Check if clusters        │
│   DBSCAN on 3D data         t-SNE algorithm           are visually separated   │
│                                                                                 │
│   Adjust k or ε             Set "Color By..."         Well-separated? ✅        │
│   until satisfied           to your clustering        Overlapping? Adjust ⚠️   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
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

## 🔮 Future Roadmap

| Phase | Feature | Description | Status |
|:-----:|:--------|:------------|:------:|
| v4.1 | **UMAP Integration** | Faster alternative to t-SNE | 🔜 Planned |
| v4.2 | **Hierarchical Clustering** | Dendrogram visualization | 🔜 Planned |
| v4.3 | **Silhouette Scores** | Quantitative cluster validation | 🔜 Planned |
| v5.0 | **Custom Data Upload** | User CSV file support | 🔜 Planned |

## 👨‍💻 Author

<div align="center">

### **Waqar Salim**

*Master's Student & IT Professional*

---

[![GitHub](https://img.shields.io/badge/GitHub-WSalim2024-181717?style=for-the-badge&logo=github)](https://github.com/WSalim2024)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/waqar-salim/)

---

**Built with 🧪 experimentation, 🔗 integration, and 🎯 precision**

*Unsupervised Learning Workbench v4.0 — Unified Analysis for Complete Understanding*

---

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   "Clustering tells you WHO belongs together.                                 ║
║    Dimensionality reduction shows you WHY."                                   ║
║                                                                               ║
║    v4.0 bridges both — for the first time.                                    ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

</div>
