<div align="center">

# 🛍️ Customer Segmentation Studio

### **v2.0 — Multi-Model Clustering Engine**

*Transform Raw Customer Data into Actionable Marketing Intelligence*

---

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

[![GitHub](https://img.shields.io/badge/GitHub-WSalim2024-181717?style=flat-square&logo=github)](https://github.com/WSalim2024)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com)

<br>

[**Features**](#-key-features) · [**Installation**](#-installation) · [**How It Works**](#-how-it-works) · [**Tech Stack**](#-tech-stack)

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Screenshots](#-screenshots)
- [Installation](#-installation)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Directory Structure](#-directory-structure)
- [Author](#-author)

---

## 🚀 Project Overview

**Customer Segmentation Studio v2.0** is an interactive Data Science Dashboard that solves a real-world business problem: **Customer Segmentation**.

Built for Marketing Managers and Business Analysts, this tool groups customers based on **Annual Income** vs **Spending Score** using advanced **Unsupervised Machine Learning** — no coding required.

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           THE BUSINESS PROBLEM                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   "We have 1,000 customers. How do we know who to target with which campaign?" │
│                                                                                 │
│   ┌───────────────┐         ┌───────────────┐         ┌───────────────┐        │
│   │   RAW DATA    │   ───►  │   ML ENGINE   │   ───►  │   SEGMENTS    │        │
│   │               │         │               │         │               │        │
│   │  CustomerID   │         │  • K-Means    │         │ 💎 VIP Clients │        │
│   │  Income       │         │  • DBSCAN     │         │ 🎯 Targets     │        │
│   │  Spending     │         │               │         │ 💵 Savers      │        │
│   │  ...          │         │               │         │ ⚠️ Outliers    │        │
│   └───────────────┘         └───────────────┘         └───────────────┘        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

### What's New in v2.0?

| Feature | v1.0 | v2.0 |
|---------|:----:|:----:|
| K-Means Clustering | ✅ | ✅ |
| DBSCAN Clustering | ❌ | ✅ |
| Outlier Detection | ❌ | ✅ |
| Multi-Model Switching | ❌ | ✅ |
| Interactive Hyperparameter Tuning | Basic | Advanced |

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🔄 Multi-Model Engine
Switch between two powerful clustering algorithms in real-time:
- **K-Means** — Standard geometric partitioning
- **DBSCAN** — Density-based spatial clustering

*Choose the right tool for your data characteristics.*

</td>
<td width="50%">

### 📐 K-Means Mode
Includes an interactive **Elbow Method** graph to mathematically determine the optimal number of clusters ($k$).

- Visual WCSS curve
- Clear elbow point detection
- Adjustable $k$ slider (2-10)

</td>
</tr>
<tr>
<td width="50%">

### 🔍 DBSCAN Mode
Features automatic **Noise Detection** to identify and isolate outliers that don't fit any group.

- Anomaly identification
- No pre-defined cluster count
- Discovers arbitrary-shaped clusters

</td>
<td width="50%">

### 🎛️ Interactive Tuning
Sidebar sliders for dynamic hyperparameter adjustment:

- **K-Means:** Number of clusters ($k$)
- **DBSCAN:** Epsilon ($\varepsilon$) and Min Samples

</td>
</tr>
</table>

### 💼 Business Logic Engine

Automatically interprets clusters into actionable marketing insights:

| Segment | Characteristics | Recommended Action |
|---------|----------------|-------------------|
| 💎 **VIP Customers** | High Income, High Spending | Premium services, exclusive offers |
| 🎯 **Target Prospects** | High Income, Low Spending | Upselling campaigns |
| 💵 **Budget Shoppers** | Low Income, High Spending | Loyalty programs, payment plans |
| 📊 **Standard Customers** | Average metrics | General promotions |
| ⚠️ **Outliers** (DBSCAN) | Anomalous behavior | Individual analysis |

---

## 📸 Screenshots

<div align="center">

### Elbow Method Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│                         [SCREENSHOT PLACEHOLDER]                                │
│                                                                                 │
│                    📈 Elbow Method - Optimal K Selection                        │
│                                                                                 │
│                         Add image: assets/elbow_curve.png                       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### K-Means Cluster Plot

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│                         [SCREENSHOT PLACEHOLDER]                                │
│                                                                                 │
│                    🎯 K-Means Clustering Results                                │
│                                                                                 │
│                         Add image: assets/kmeans_clusters.png                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### DBSCAN with Outlier Detection

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│                         [SCREENSHOT PLACEHOLDER]                                │
│                                                                                 │
│                    🔍 DBSCAN Clustering with Noise Points                       │
│                                                                                 │
│                         Add image: assets/dbscan_clusters.png                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

*Screenshots will be added after deployment.*

</div>

---

## 📥 Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/WSalim2024/Customer-Segmentation-Studio.git

# Navigate to project directory
cd Customer-Segmentation-Studio

# Install dependencies
pip install pandas matplotlib seaborn scikit-learn streamlit

# Launch the application
streamlit run app.py
```

### Access the Dashboard

Once launched, open your browser and navigate to:

```
http://localhost:8501
```

---

## 🔬 How It Works

This dashboard offers two distinct clustering approaches. Here's a simple explanation for non-technical users:

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    K-MEANS vs DBSCAN: A SIMPLE COMPARISON                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│         K-MEANS                                    DBSCAN                       │
│    "Geometric Grouping"                      "Density Grouping"                 │
│    ─────────────────────                     ─────────────────────              │
│                                                                                 │
│    Think of it as:                           Think of it as:                    │
│    Dividing a pizza into                     Finding crowded areas              │
│    equal slices                              at a party                         │
│                                                                                 │
│         ┌─────────┐                              ●●●    ●                       │
│        /    |     \                             ●●●●●                           │
│       /  ●  | ●    \                            ●●●●     ◆◆◆                    │
│      /  ●●  |  ●●   \                                   ◆◆◆◆                    │
│     /───────┼────────\                           ★       ◆◆                     │
│     \  ●●   |   ●●   /                        (noise)                           │
│      \ ●    |    ●  /                                                           │
│       \     |      /                                                            │
│        \____|_____/                                                             │
│                                                                                 │
│    ✅ You decide how many                    ✅ Algorithm decides               │
│       groups (k)                                how many groups                 │
│                                                                                 │
│    ✅ Equal-sized, round                     ✅ Any shape, any size             │
│       clusters                                  clusters                        │
│                                                                                 │
│    ❌ Cannot detect                          ✅ Automatically finds             │
│       outliers                                  outliers (noise)                │
│                                                                                 │
│    Best for: Well-separated,                 Best for: Irregular shapes,        │
│    spherical customer groups                 finding anomalies                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</div>

### K-Means: The Geometric Approach

**How it works:** Divides customers into exactly $k$ groups by minimizing the distance between each customer and their group's center point.

**Key Parameter:**
- $k$ = Number of clusters (you choose using the Elbow Method)

**Best when:** You know roughly how many segments you want and your data forms round, well-separated groups.

---

### DBSCAN: The Density Approach

**How it works:** Finds areas where customers are "crowded together" and groups them. Points in sparse areas are marked as **outliers** (noise).

**Key Parameters:**
- $\varepsilon$ (Epsilon) = How close points must be to be considered neighbors
- Min Samples = Minimum points needed to form a dense region

**Best when:** You don't know how many segments exist, your groups have irregular shapes, or you want to identify unusual customers.

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology | Purpose |
|:-----:|:----------:|:--------|
| 🐍 | **Python 3.10** | Core programming language |
| 🖥️ | **Streamlit** | Interactive web dashboard |
| 🤖 | **Scikit-Learn** | KMeans, DBSCAN, StandardScaler |
| 📊 | **Matplotlib** | Base visualizations |
| 🎨 | **Seaborn** | Enhanced chart aesthetics |

</div>

---

## 📁 Directory Structure

```
Customer-Segmentation-Studio/
│
├── 📄 app.py                    # Main Streamlit application
├── 📄 README.md                 # Project documentation
└── 📄 .gitignore                # Git ignore rules
```

---

## 👨‍💻 Author

<div align="center">

### **Waqar Salim**

*Master's Student & IT Professional*

---

[![GitHub](https://img.shields.io/badge/GitHub-WSalim2024-181717?style=for-the-badge&logo=github)](https://github.com/WSalim2024)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com)

---

**Built with 📊 data science, 🤖 machine learning, and ☕ dedication**

*Customer Segmentation Studio v2.0 — Know Your Customers. Target With Precision.*

</div>
