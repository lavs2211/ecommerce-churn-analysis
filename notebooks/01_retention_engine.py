import pandas as pd
import numpy as np
import os

def run_statistical_retention_engine(data_path):
    print("🔄 Step 1: Loading raw data and establishing constraints...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Target data file not found at {data_path}")
        
    df = pd.read_csv(data_path)
    
    # Structural Data Cleaning
    df = df.dropna(subset=['CustomerID'])
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df = df.dropna(subset=['InvoiceDate'])
    
    # Engineer Total Spend per line transaction
    df['TotalSpend'] = df['Quantity'] * df['UnitPrice']
    
    print("📊 Step 2: Isolating natural repeat purchase intervals...")
    # Isolate valid base purchases (Exclude returns for purchase rhythm analysis)
    df_purchases = df[df['Quantity'] > 0]
    
    # Group unique purchase days per customer to find the real purchasing cadence
    df_days = df_purchases[['CustomerID', 'InvoiceDate']].copy()
    
    # Ensure the column is explicitly registered as a Datetime format
    df_days['InvoiceDay'] = pd.to_datetime(df_days['InvoiceDate'].dt.date)
    
    df_days = df_days[['CustomerID', 'InvoiceDay']].drop_duplicates().sort_values(['CustomerID', 'InvoiceDay'])
    
    # Compute the differences between consecutive purchase days
    df_days['PrevInvoiceDay'] = df_days.groupby('CustomerID')['InvoiceDay'].shift(1)
    df_days['DaysBetween'] = (df_days['InvoiceDay'] - df_days['PrevInvoiceDay']).dt.days
    
    # Statistical Quantile Check to define Churn Window
    p50 = df_days['DaysBetween'].median()
    p75 = df_days['DaysBetween'].quantile(0.75)
    p90 = df_days['DaysBetween'].quantile(0.90)
    print(f"   -> Median Order Gap (P50): {p50:.1f} days")
    print(f"   -> Upper Quadrant Gap (P75): {p75:.1f} days")
    print(f"   -> Critical Inactivity Marker (P90): {p90:.1f} days")
    
    # Empirically established threshold (90-day window)
    CHURN_WINDOW_DAYS = 90
    print(f"   -> Data-Driven Churn Logic Rule: Active if Recency <= {CHURN_WINDOW_DAYS} days, else Churned.")

    print("\n⚡ Step 3: Compiling RFM metrics per customer matrix...")
    max_date = df['InvoiceDate'].max()
    
    # Grouping customer history
    customer_rfm = df_purchases.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (max_date - x.max()).days, # Recency
        'InvoiceNo': 'nunique',                             # Frequency
        'TotalSpend': 'sum'                                 # Monetary
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalSpend': 'Monetary'})
    
    print("📈 Step 4: Applying statistical quintiles (1-5 scoring)...")
    # Handling data distribution skew: Using qcut to bin into equal-sized buckets (quintiles)
    customer_rfm['R_Score'] = pd.qcut(customer_rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1])
    
    # Frequency has repeated values, use rank method to resolve duplicate bin edge conflicts
    customer_rfm['F_Score'] = pd.qcut(customer_rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
    customer_rfm['M_Score'] = pd.qcut(customer_rfm['Monetary'], q=5, labels=[1, 2, 3, 4, 5])
    
    # Combine individual scores into an RFM segment ID
    customer_rfm['RFM_Class'] = customer_rfm['R_Score'].astype(str) + customer_rfm['F_Score'].astype(str) + customer_rfm['M_Score'].astype(str)

    print("🎯 Step 5: Mapping lifecycle segments and setting churn statuses...")
    # Define Explicit Churn Target Field
    customer_rfm['Churn_Status'] = np.where(customer_rfm['Recency'] > CHURN_WINDOW_DAYS, 'Churned', 'Active')
    
    # Map high-level descriptive segments based on RFM Scores
    def segment_mapping(row):
        r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
        
        if r >= 4 and f >= 4 and m >= 4:
            return 'VIP / Core Loyalist'
        elif r <= 2 and f >= 3:
            return 'At Risk / High-Value Lapsing'
        elif r >= 3 and f == 1:
            return 'New / Recent Trialist'
        elif r <= 2 and f <= 2:
            return 'Hibernating / Lost'
        else:
            return 'Average Mid-Tier Core'
            
    customer_rfm['Lifecycle_Segment'] = customer_rfm.apply(segment_mapping, axis=1)
    
    # Save processed analytical baseline
    os.makedirs('data/processed', exist_ok=True)
    customer_rfm.to_csv('data/processed/customer_lifecycle_matrix.csv')
    print("✅ Step 6: Processed metrics successfully saved to 'data/processed/customer_lifecycle_matrix.csv'")
    
    # Save processed analytical baseline
    os.makedirs('data/processed', exist_ok=True)
    customer_rfm.to_csv('data/processed/customer_lifecycle_matrix.csv')
    print("✅ Step 6: Processed metrics successfully saved to 'data/processed/customer_lifecycle_matrix.csv'")
    
    # 🌟 ADD THIS BLOCK TO PRINT RESULTS TO YOUR SCREEN
    print("\n--- Project Baseline Snapshot ---")
    print("\n📊 Customer Counts per Segment:")
    print(customer_rfm['Lifecycle_Segment'].value_counts())
    print("\n📉 Portfolio Churn Split Rate:")
    print(customer_rfm['Churn_Status'].value_counts(normalize=True) * 100)
    if __name__ == "__main__":
        # 1. Dynamically locate the directory where THIS script file is saved (notebooks/)
     SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Step up one directory out of notebooks/ into the main workspace root (raju/)
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    
    # 3. Target the correct raw data path matching your exact filename spelling
    ROBUST_DATA_PATH = os.path.join(PROJECT_ROOT, 'Online Retail.xlsv - Online Retail.csv')
    
    # 4. Trigger the core pipeline
    run_statistical_retention_engine(ROBUST_DATA_PATH)
    
    # 4. Trigger the core pipeline
    run_statistical_retention_engine(ROBUST_DATA_PATH)
    
    # 4. Trigger the engine with our verified path location
    run_statistical_retention_engine(ROBUST_DATA_PATH)