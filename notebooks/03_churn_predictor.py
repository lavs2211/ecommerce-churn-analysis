import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def run_predictive_churn_engine():
    # 1. Coordinate file system directories
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    
    # Target the exact filename your workspace is using
    FILENAME = 'Online Retail.xlsv - Online Retail.csv'
    RAW_DATA_PATH = os.path.join(PROJECT_ROOT, FILENAME)
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("🔄 ML Step 1: Loading raw rows and isolating time-split windows...")
    df = pd.read_csv(RAW_DATA_PATH)
    df = df.dropna(subset=['CustomerID'])
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df = df.dropna(subset=['InvoiceDate'])
    
    # Calculate revenue line item totals
    df['TotalSpend'] = df['Quantity'] * df['UnitPrice']
    
    # Establish Time Split to prevent Data Leakage
    max_date = df['InvoiceDate'].max()
    cutoff_date = max_date - pd.Timedelta(days=90)
    print(f"   -> Max Date in Dataset: {max_date}")
    print(f"   -> 90-Day ML Cutoff Anchor: {cutoff_date}")
    
    df_obs = df[df['InvoiceDate'] <= cutoff_date].copy()
    df_target = df[df['InvoiceDate'] > cutoff_date].copy()
    
    print("\n⚡ ML Step 2: Feature Engineering across the Historical Observation Window...")
    # Isolate successful purchases vs cancellations inside historical window
    obs_purchases = df_obs[df_obs['Quantity'] > 0]
    obs_returns = df_obs[df_obs['Quantity'] < 0]
    
    # Compute behavioral features
    recency = obs_purchases.groupby('CustomerID')['InvoiceDate'].max().apply(lambda x: (cutoff_date - x).days)
    frequency = obs_purchases.groupby('CustomerID')['InvoiceNo'].nunique()
    monetary = obs_purchases.groupby('CustomerID')['TotalSpend'].sum()
    total_qty = obs_purchases.groupby('CustomerID')['Quantity'].sum()
    unique_prods = obs_purchases.groupby('CustomerID')['StockCode'].nunique()
    first_date = obs_purchases.groupby('CustomerID')['InvoiceDate'].min()
    tenure = first_date.apply(lambda x: (cutoff_date - x).days)
    
    # Compute return frictions
    return_count = obs_returns.groupby('CustomerID')['InvoiceNo'].nunique()
    return_amount = obs_returns.groupby('CustomerID')['TotalSpend'].sum().abs()
    
    # Consolidate feature matrices
    features_df = pd.DataFrame(index=df_obs['CustomerID'].unique())
    features_df['Recency'] = recency
    features_df['Frequency'] = frequency
    features_df['Monetary'] = monetary
    features_df['AvgOrderValue'] = monetary / frequency
    features_df['TotalQuantity'] = total_qty
    features_df['UniqueProducts'] = unique_prods
    features_df['Tenure'] = tenure
    features_df['ReturnCount'] = return_count
    features_df['ReturnAmount'] = return_amount
    
    # Clean structural nulls for customers who never returned items
    features_df = features_df.fillna(0)
    features_df = features_df.reset_index().rename(columns={'index': 'CustomerID'})
    
    print("🎯 ML Step 3: Mapping ground-truth future labels from Target Window...")
    # Find customers who made valid purchases in the final 90 days
    target_active_customers = df_target[df_target['Quantity'] > 0]['CustomerID'].unique()
    
    # If they are not in target_active_customers, they Churned (1), else Retained (0)
    features_df['Churn'] = np.where(features_df['CustomerID'].isin(target_active_customers), 1, 0)
    
    print(f"   -> Modeled Dataset Size: {len(features_df)} unique customer profiles.")
    print(f"   -> Churn Event Balance Split %:\n{features_df['Churn'].value_counts(normalize=True) * 100}")
    
    print("\n🤖 ML Step 4: Training Random Forest Classifier Model...")
    X = features_df.drop(columns=['CustomerID', 'Churn'])
    y = features_df['Churn']
    
    # Train-test split (80% train, 20% validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
    rf.fit(X_train, y_train)
    
    # Predict outcomes
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    print("\n📝 --- Model Performance Report ---")
    print(f"ROC-AUC Performance Score: {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Matrix Metrics:\n", classification_report(y_test, y_pred))
    
    print("📈 ML Step 5: Extracting and Saving Behavioral Feature Importances...")
    importances = rf.feature_importances_
    feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values('Importance', ascending=True)
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='magma', hue='Feature', legend=False)
    plt.title('What Drives E-Commerce Customer Churn? (Feature Importance Mapping)', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Random Forest Relative Importance Weight', fontsize=10)
    plt.ylabel('Engineered Customer Metric', fontsize=10)
    plt.tight_layout()
    
    chart_path = os.path.join(OUTPUT_DIR, 'ml_feature_importances.png')
    plt.savefig(chart_path, dpi=300)
    plt.close()
    
    # Export clean ML features for presentation
    os.makedirs(os.path.join(PROJECT_ROOT, 'data', 'processed'), exist_ok=True)
    features_df.to_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'engineered_churn_features.csv'), index=False)
    print(f"✅ Feature Importance plot saved straight to: {chart_path}")
    print("🎉 Predictive Churn Model completed successfully!")

if __name__ == "__main__":
    run_predictive_churn_engine()