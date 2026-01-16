<div align="center">

# 🛍️ Customer Segmentation Studio

### **Know Your Customers. Target With Precision.**

*An Interactive Data Science Dashboard for Unsupervised Customer Clustering*

---

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<br>

[**Features**](#-key-features) · [**Installation**](#-installation) · [**Usage**](#-usage) · [**The Science**](#-the-science)

<br>

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   "Stop marketing to everyone. Start marketing to the right ones."           ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Demo](#-demo)
- [Installation](#-installation)
- [Usage](#-usage)
- [The Science](#-the-science)
- [Business Insights](#-business-insights)
- [Tech Stack](#-tech-stack)
- [License](#-license)

---

## 🚀 Overview

**Customer Segmentation Studio** transforms raw customer data into actionable marketing intelligence. Using **unsupervised machine learning**, this interactive dashboard automatically groups customers based on their **Annual Income** and **Spending Score** — revealing hidden patterns that drive smarter business decisions.

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        FROM RAW DATA TO MARKETING GOLD                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│     RAW DATA                  K-MEANS                    ACTIONABLE             │
│     ─────────                 ───────                    SEGMENTS               │
│                                                          ────────               │
│   ┌───────────┐            ┌───────────┐            ┌───────────────┐           │
│   │ CustomerID│            │           │            │ 💎 Big Spenders│           │
│   │ Income    │    ───►    │  CLUSTER  │    ───►    │ 💰 Affluent    │           │
│   │ Spending  │            │  ANALYSIS │            │ 🎯 Targets     │           │
│   │ ...       │            │           │            │ 💵 Savers      │           │
│   └───────────┘            └───────────┘            └───────────────┘           │
│                                                                                 │
│   "Who are my              "Find natural             "Here's who to            │
│    customers?"              groupings"                target and how"           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

### Why Customer Segmentation?

| Without Segmentation | With Segmentation |
|---------------------|-------------------|
| One-size-fits-all marketing | Personalized campaigns per segment |
| Wasted ad spend on wrong audiences | Focused spend on high-value targets |
| Generic messaging that converts poorly | Tailored messaging that resonates |
| No understanding of customer diversity | Clear view of distinct customer types |

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🤖 K-Means Clustering
- **Scikit-Learn Implementation** — Industry-standard algorithm
- **Automatic Grouping** — Discovers natural customer segments
- **Centroid Visualization** — See the "center" of each cluster
- **Scalable** — Handles thousands of customers

</td>
<td width="50%">

### 📐 Elbow Method Optimization
- **Visual $k$ Selection** — Find the optimal cluster count
- **WCSS Plot** — Within-Cluster Sum of Squares curve
- **Clear Elbow Detection** — Where diminishing returns begin
- **Guidance** — Recommendations for cluster selection

</td>
</tr>
<tr>
<td width="50%">

### 🎛️ Interactive Controls
- **Dynamic $k$ Slider** — Adjust clusters (2-10) in real-time
- **Instant Updates** — Watch segmentation change live
- **No Coding Required** — Built for non-technical users
- **Sidebar Configuration** — Clean, intuitive interface

</td>
<td width="50%">

### 📊 Rich Visualizations
- **2D Scatter Plots** — Income vs. Spending Score
- **Color-Coded Clusters** — Distinct segment visualization
- **Centroid Markers** — Cluster center identification
- **Matplotlib/Seaborn** — Publication-quality graphics

</td>
</tr>
</table>

---

## 🖼️ Demo

<div align="center">

### Screenshots

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│                         [SCREENSHOT PLACEHOLDER]                                │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                         │   │
│   │                      📊 CLUSTER VISUALIZATION                           │   │
│   │                                                                         │   │
│   │                           ●  ●                                          │   │
│   │                         ●  ●  ●     ▲                                   │   │
│   │        Spending         ●  ●  ●  ●                    ◆  ◆              │   │
│   │          Score        ●  ●  ●  ●              ◆  ◆  ◆  ◆               │   │
│   │            │                ▲                   ◆  ◆  ◆                 │   │
│   │            │      ■  ■  ■                                               │   │
│   │            │    ■  ■  ■  ■  ■                                           │   │
│   │            │      ■  ■  ▲                    ★  ★  ★                    │   │
│   │            │                               ★  ★  ★  ★                   │   │
│   │            └────────────────────────────────────▲──────────────────     │   │
│   │                                                                         │   │
│   │                         Annual Income ($k)                              │   │
│   │                                                                         │   │
│   │   Legend: ● Cluster 1  ◆ Cluster 2  ■ Cluster 3  ★ Cluster 4  ▲ Centroid│   │
│   │                                                                         │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│                    Add your screenshot: assets/demo.png                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

*Replace this placeholder with an actual screenshot of your running application.*

**To add a screenshot:**
```markdown
![Customer Segmentation Demo](assets/demo.png)
```

</div>

---

## 📥 Installation

### Prerequisites

| Requirement | Version | Installation |
|-------------|---------|--------------|
| **Python** | 3.10+ | [python.org](https://python.org/downloads) |
| **pip** | Latest | Included with Python |
| **Git** | Any | [git-scm.com](https://git-scm.com/downloads) |

### Step-by-Step Setup

#### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Customer-Segmentation-Studio.git
cd Customer-Segmentation-Studio
```

#### Step 2: Create Virtual Environment

<table>
<tr>
<th>🐧 Linux / 🍎 macOS</th>
<th>🪟 Windows</th>
</tr>
<tr>
<td>

```bash
# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate
```

</td>
<td>

```powershell
# Create virtual environment
python -m venv venv

# Activate environment
.\venv\Scripts\activate
```

</td>
</tr>
</table>

#### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

### requirements.txt

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

#### Step 4: Verify Installation

```bash
python -c "
import streamlit
import sklearn
import seaborn
print('✅ All dependencies installed successfully!')
print(f'   Streamlit: {streamlit.__version__}')
print(f'   Scikit-Learn: {sklearn.__version__}')
print(f'   Seaborn: {seaborn.__version__}')
"
```

---

## ▶️ Usage

### Launch the Application

```bash
streamlit run app.py
```

### Expected Output

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Using the Dashboard

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        🛍️ CUSTOMER SEGMENTATION STUDIO                          │
├────────────────────┬────────────────────────────────────────────────────────────┤
│                    │                                                            │
│   📁 SIDEBAR       │                    📊 MAIN CANVAS                          │
│                    │                                                            │
│  ┌──────────────┐  │    ┌────────────────────────────────────────────────────┐  │
│  │ ⚙️ Settings   │  │    │                                                    │  │
│  │              │  │    │              CLUSTER SCATTER PLOT                  │  │
│  │ Number of    │  │    │                                                    │  │
│  │ Clusters (k) │  │    │     Customers plotted by Income vs Spending       │  │
│  │              │  │    │     Color-coded by assigned cluster                │  │
│  │ [2]───●───[10]│  │    │     Centroids marked with ★                       │  │
│  │       ▲      │  │    │                                                    │  │
│  │    k = 5     │  │    └────────────────────────────────────────────────────┘  │
│  │              │  │                                                            │
│  └──────────────┘  │    ┌────────────────────────────────────────────────────┐  │
│                    │    │                                                    │  │
│  ┌──────────────┐  │    │              ELBOW METHOD CHART                    │  │
│  │ 📈 Show      │  │    │                                                    │  │
│  │ Elbow Chart  │  │    │     WCSS vs. Number of Clusters                    │  │
│  │ [✓]          │  │    │     Find the "elbow" for optimal k                 │  │
│  └──────────────┘  │    │                                                    │  │
│                    │    └────────────────────────────────────────────────────┘  │
│                    │                                                            │
└────────────────────┴────────────────────────────────────────────────────────────┘
```

### Workflow

1. **Load Data** — The app uses customer data with Income and Spending columns
2. **View Elbow Chart** — Identify the optimal number of clusters
3. **Adjust $k$ Slider** — Set your desired cluster count
4. **Analyze Segments** — Review the cluster visualization and business insights
5. **Export Results** — Download segmented customer data

---

## 🔬 The Science

### K-Means Clustering Algorithm

**K-Means** is an unsupervised machine learning algorithm that partitions data into $k$ distinct clusters based on feature similarity.

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         HOW K-MEANS WORKS                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  STEP 1: INITIALIZE              STEP 2: ASSIGN                                 │
│  ──────────────────              ─────────────────                              │
│                                                                                 │
│    ●  ●     ●  ●                   ●  ●     ●  ●                                │
│  ●  ★  ●  ●                      ●  ★  ●  ●                                     │
│    ●     ●  ●                      ●     ●  ●     ← Each point assigned        │
│       ●        ●                      ●        ●     to NEAREST centroid       │
│    ●     ★  ●                      ●     ★  ●                                   │
│  ●  ●  ●                         ●  ●  ●                                        │
│                                                                                 │
│  Randomly place k                 Assign each customer                          │
│  centroids (★)                    to closest centroid                           │
│                                                                                 │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                 │
│  STEP 3: UPDATE                  STEP 4: REPEAT                                 │
│  ──────────────────              ────────────────                               │
│                                                                                 │
│    ●  ●     ●  ●                   ●  ●     ◆  ◆                                │
│  ●   ●  ●  ●                     ●   ●  ◆  ◆                                    │
│    ●  ★  ●  ●                      ●  ★  ◆  ◆    ← Final stable clusters       │
│       ●        ●                      ●     ★  ◆                                │
│    ●  ●   ★ ●                      ■  ■   ★ ●                                   │
│  ●  ●  ●                         ■  ■  ■                                        │
│                                                                                 │
│  Move centroids to               Repeat until centroids                         │
│  cluster MEAN position           stop moving (convergence)                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

#### The Mathematics

**Objective Function (Minimize):**

$$J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$

Where:
- $k$ = number of clusters
- $C_i$ = set of points in cluster $i$
- $\mu_i$ = centroid (mean) of cluster $i$
- $||x - \mu_i||^2$ = squared Euclidean distance

**Algorithm Steps:**

```python
from sklearn.cluster import KMeans

# Initialize and fit K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)

# Get cluster assignments and centroids
labels = kmeans.labels_           # Cluster ID for each customer
centroids = kmeans.cluster_centers_  # Center of each cluster
```

---

### The Elbow Method

**Problem:** How do we choose the optimal number of clusters ($k$)?

**Solution:** The Elbow Method visualizes the trade-off between cluster count and model fit.

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           THE ELBOW METHOD                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  WCSS (Within-Cluster Sum of Squares)                                           │
│    │                                                                            │
│    │                                                                            │
│ 800├──●                                                                         │
│    │    \                                                                       │
│    │     \                                                                      │
│ 600├      \                                                                     │
│    │       \                                                                    │
│    │        \                                                                   │
│ 400├         ●                                                                  │
│    │          \                                                                 │
│    │           \    ← THE ELBOW                                                 │
│ 200├            ●────●────●────●────●────●                                      │
│    │                 ▲                                                          │
│    │            Optimal k                                                       │
│  0 ├────┬────┬────┬────┬────┬────┬────┬────┬────                                │
│    1    2    3    4    5    6    7    8    9    10                              │
│                                                                                 │
│                    Number of Clusters (k)                                       │
│                                                                                 │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                 │
│  INTERPRETATION:                                                                │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                                                                           │  │
│  │  • WCSS decreases as k increases (more clusters = tighter fit)            │  │
│  │  • The "elbow" is where adding more clusters gives DIMINISHING RETURNS    │  │
│  │  • In this example: k=5 is optimal (elbow point)                          │  │
│  │  • Beyond k=5: marginal WCSS reduction doesn't justify complexity         │  │
│  │                                                                           │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

#### Implementation

```python
import matplotlib.pyplot as plt

# Calculate WCSS for different k values
wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # inertia_ = WCSS

# Plot the Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal k')
plt.axvline(x=5, color='r', linestyle='--', label='Optimal k=5')
plt.legend()
plt.show()
```

---

## 💼 Business Insights

The dashboard automatically interprets clusters and suggests marketing strategies:

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      CUSTOMER SEGMENT INTERPRETATION                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                           High Spending Score                                   │
│                                  ▲                                              │
│                                  │                                              │
│         ┌────────────────────────┼────────────────────────┐                     │
│         │                        │                        │                     │
│         │   🎯 CAREFUL           │   💎 BIG SPENDERS      │                     │
│         │                        │                        │                     │
│         │   Low Income,          │   High Income,         │                     │
│         │   High Spending        │   High Spending        │                     │
│         │                        │                        │                     │
│         │   Strategy:            │   Strategy:            │                     │
│         │   Budget-friendly      │   Premium products,    │                     │
│         │   options, loyalty     │   VIP treatment,       │                     │
│         │   programs             │   exclusive offers     │                     │
│         │                        │                        │                     │
│  Low ───┼────────────────────────┼────────────────────────┼─── High             │
│ Income  │                        │                        │   Income            │
│         │   💵 SAVERS            │   💰 AFFLUENT          │                     │
│         │                        │                        │                     │
│         │   Low Income,          │   High Income,         │                     │
│         │   Low Spending         │   Low Spending         │                     │
│         │                        │                        │                     │
│         │   Strategy:            │   Strategy:            │                     │
│         │   Value deals,         │   Investment products, │                     │
│         │   essentials focus,    │   premium savings,     │                     │
│         │   discounts            │   upsell potential     │                     │
│         │                        │                        │                     │
│         └────────────────────────┼────────────────────────┘                     │
│                                  │                                              │
│                                  ▼                                              │
│                           Low Spending Score                                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

### Segment Profiles

| Segment | Income | Spending | Size | Strategy |
|---------|--------|----------|------|----------|
| 💎 **Big Spenders** | High | High | ~20% | VIP programs, premium products |
| 💰 **Affluent Savers** | High | Low | ~25% | Investment offers, upselling |
| 🎯 **Careful Spenders** | Low | High | ~20% | Loyalty rewards, payment plans |
| 💵 **Budget Conscious** | Low | Low | ~35% | Discounts, value bundles |

---

## 🛠️ Tech Stack

<div align="center">

| Component | Technology | Version | Purpose |
|:---------:|:----------:|:-------:|:--------|
| **🖥️ Frontend** | Streamlit | 1.28+ | Interactive dashboard UI |
| **🐍 Runtime** | Python | 3.10 | Core programming language |
| **📊 Data** | Pandas | 2.0+ | Data manipulation |
| **🔢 Numerical** | NumPy | 1.24+ | Array operations |
| **🤖 ML** | Scikit-Learn | 1.3+ | K-Means clustering |
| **📈 Plotting** | Matplotlib | 3.7+ | Base visualizations |
| **🎨 Styling** | Seaborn | 0.12+ | Enhanced chart aesthetics |

</div>

---

## 📁 Project Structure

```
Customer-Segmentation-Studio/
│
├── 📄 app.py                    # Main Streamlit application
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # This documentation
│
├── 📁 data/
│   └── customers.csv            # Sample customer dataset
│
├── 📁 src/
│   ├── clustering.py            # K-Means implementation
│   ├── visualization.py         # Plotting functions
│   └── business_logic.py        # Segment interpretation
│
└── 📁 assets/
    └── demo.png                 # Screenshot for README
```

---

## 📄 License

<div align="center">

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

See [LICENSE](LICENSE) for full details.

</div>

---

## 👨‍💻 Author

<div align="center">

### **Waqar Salim**

*Master's Student & IT Professional*

---

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/yourusername)

---

**Built with 📊 data, 🤖 algorithms, and ☕ caffeine**

*Customer Segmentation Studio — Because every customer is unique, but some are more profitable.*

---

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   "In God we trust. All others must bring data."                              ║
║                                               — W. Edwards Deming              ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

</div>
****