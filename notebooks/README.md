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