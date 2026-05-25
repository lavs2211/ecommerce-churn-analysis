# E-Commerce Customer Churn & Retention Analytics Engine

## 📂 Project Architecture Layout
```text
raju/
│
├── data/
│   ├── processed/
│   │   ├── customer_lifecycle_matrix.csv     # RFM tiers and explicit churn states
│   │   └── engineered_churn_features.csv     # Mapped leakage-free historical ML training data
│   └── Online Retail.xlsv - Online Retail.csv # Raw transnational database
│
├── notebooks/
│   ├── 01_retention_engine.py                # Time-series cohorts and RFM segmentation script
│   ├── 02_retention_visualizer.py            # Static seaborn chart and heatmap generation
│   └── 03_churn_predictor.py                 # Random Forest classifier model script
│
├── outputs/
│   ├── cohort_retention_heatmap.png          # Saved month-over-month cohort matrix
│   ├── customer_segment_distributions.png    # Portfolio cluster size metrics plot
│   └── ml_feature_importances.png            # Random forest feature weights chart
│
└── README.md                                 # Executive presentation guide
---

### Step 3: Insert the Executive Content Template

Directly underneath your directory layout map, paste this structured executive outline template. Be sure to fill in your exact Machine Learning scores ($0.7433$ ROC-AUC) so recruiters can instantly spot your metrics without reading through the source scripts:

```markdown
## 📊 Executive Project Summary
In non-subscription e-commerce settings, customer churn is implicit—buyers do not cancel a subscription package; they simply drift away over time. This data science project builds a custom analytics engine to isolate natural consumer purchase gaps, segment buyers into strategic RFM categories, deploy an executive dashboard interface, and train a machine learning algorithm to flag high-value at-risk shoppers before they leave.

---

## 🛠️ Data Engineering & Analytical Steps

### 1. Retention Window Calculation & RFM Profile Engine (`01_retention_engine.py`)
* Processes raw transaction lines, drops invalid rows, and calculates natural consumer transaction tempos.
* Analyzes data frequency splits to reveal: **Median Order Gap (P50) is 28 days**, **Upper Quadrant (P75) is 58 days**, and **Critical Inactivity (P90) is 110 days**.
* Uses empirical insight to establish a data-driven **90-day inactivity threshold** to classify true profile churn.
* Computes Recency, Frequency, and Monetary scores using quintiles to place buyers into lifecycle groups (e.g., VIP, New Trialist, At Risk, Hibernating).

### 2. Business Intelligence & Dashboard Visualizer (`02_retention_visualizer.py`)
* Computes month-over-month cohort transaction counts and saves a high-resolution seaborn tracking heatmap.
* Establishes a complete database baseline linked directly to an enterprise **Power BI Dashboard Layout**, allowing corporate teams to interactively segment lists on the fly.

### 3. Predictive Machine Learning Engine (`03_churn_predictor.py`)
* Implements a rigorous chronological **Time-Series Split** to eliminate data leakage.
* Extracts consumer features leading up to a 90-day anchor point, evaluating their transaction presence in the final 90 days to capture real target event metrics.
* Fits a **Scikit-Learn Random Forest Classifier** to project account drops.

---

## 🤖 Predictive Machine Learning Metrics

### 🎯 Random Forest Model Results
* **ROC-AUC Performance Score:** `0.7433` (Demonstrates strong mathematical ability to separate repeat buyers from churn risks).
* **Precision (Class 1 - Churners):** `0.73` (When the system flags a customer profile as a churn risk, it is correct **73% of the time**).
* **Recall (Class 1 - Churners):** `0.72` (Successfully captures **72% of all actual customer attrition** events across the testing window).

### 📈 Core Business Insights
The generated feature coefficients indicate that **Recency** (days since last purchase) and customer **Tenure** are the primary mathematical indicators of user departure risk, while operational metrics like **ReturnAmount** provide significant warning signs for high-value buyer drops.

---

## 🎨 Processed Analytical Graphics Showcase

### 🌡️ 1. Month-over-Month Customer Retention Cohorts Heatmap
The heatmap captures customer drop-off velocities across distinct registration months over a 12-month timeline:
![Cohort Heatmap](outputs/cohort_retention_heatmap.png)

### 📊 2. Portfolio Customer Distribution Matrix
Breaks down the overall unique profile counts sorted across our custom RFM lifecycle tiers:
![Portfolio Segments](outputs/customer_segment_distributions.png)

### 🧠 3. Machine Learning Behavioral Feature Importance Weights
Highlights exactly what user metric carries the highest predictive weight when identifying churn risk:
![ML Feature Importances](outputs/ml_feature_importances.png)