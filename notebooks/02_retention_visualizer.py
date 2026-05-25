import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def build_retention_visualizations():
    # 1. Establish project directory routes
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Points to notebooks/
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)              # Points to raju/
    
    # MATCH THE EXACT FILE STRING FOUND BY POWERSHELL
    FILENAME = 'Online Retail.xlsv - Online Retail.csv'
    
    # Define potential paths to locate it safely
    PATH_OPTIONS = [
        os.path.join(PROJECT_ROOT, FILENAME),                       # raju/
        os.path.join(PROJECT_ROOT, 'data', FILENAME),                # raju/data/
        os.path.join(PROJECT_ROOT, 'data', 'raw', FILENAME),         # raju/data/raw/
        os.path.join(SCRIPT_DIR, FILENAME),                         # raju/notebooks/
        os.path.join(os.getcwd(), FILENAME)                         # Terminal Active Directory
    ]
    
    RAW_DATA_PATH = None
    for path_option in PATH_OPTIONS:
        if os.path.exists(path_option):
            RAW_DATA_PATH = path_option
            break
            
    if RAW_DATA_PATH is None:
        raise FileNotFoundError(f"❌ Could not find '{FILENAME}' anywhere in your project workspace paths.\n"
                                f"Checked locations:\n" + "\n".join(PATH_OPTIONS))
                                
    # Update baseline matrix locations built by script 1
    MATRIX_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'customer_lifecycle_matrix.csv')
    if not os.path.exists(MATRIX_DATA_PATH):
        # Fallbacks if folder structure is flat or saved in root
        PATH_OPTIONS_MATRIX = [
            os.path.join(PROJECT_ROOT, 'customer_lifecycle_matrix.csv'),
            os.path.join(SCRIPT_DIR, 'customer_lifecycle_matrix.csv'),
            os.path.join(PROJECT_ROOT, 'data', 'customer_lifecycle_matrix.csv'),
            os.path.join(PROJECT_ROOT, 'data', 'processed', 'customer_lifecycle_matrix.csv')
        ]
        for opt in PATH_OPTIONS_MATRIX:
            if os.path.exists(opt):
                MATRIX_DATA_PATH = opt
                break

    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"🎯 Successfully located raw data at: {RAW_DATA_PATH}")
    print(f"🎯 Successfully located matrix data at: {MATRIX_DATA_PATH}")
    print("季 Visualizer Step 1: Generating Cohort Heatmap Layout...")
    
    # Load raw file
    df = pd.read_csv(RAW_DATA_PATH)
    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]
    
    # Parse dates securely and drop any corrupted date rows completely
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df = df.dropna(subset=['InvoiceDate'])
    
    # Calculate acquisition and transactional months
    df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')
    df['CohortMonth'] = df.groupby('CustomerID')['InvoiceMonth'].transform('min')
    
    # Group coordinates and pivot
    cohort_counts = df.groupby(['CohortMonth', 'InvoiceMonth']).agg({'CustomerID': 'nunique'}).reset_index()
    
    # Extract the integer months directly from the Period objects to avoid NaT math crashes
    cohort_counts['CohortIndex'] = cohort_counts.apply(
        lambda row: (row['InvoiceMonth'].year - row['CohortMonth'].year) * 12 + (row['InvoiceMonth'].month - row['CohortMonth'].month), 
        axis=1
    )
    
    cohort_pivot = cohort_counts.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')
    
    # Calculate cohort percentage drops
    cohort_sizes = cohort_pivot.iloc[:, 0]
    retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0)
    
    # Render Cohort Heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(retention_matrix, annot=True, fmt='.1%', cmap='YlGnBu', vmin=0.0, vmax=0.5, cbar_kws={'label': 'Retention Percentage'})
    plt.title('E-Commerce Month-over-Month Customer Retention Cohorts', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Cohort Index (Months Since Acquisition)', fontsize=11)
    plt.ylabel('Acquisition Cohort Month', fontsize=11)
    plt.tight_layout()
    
    heatmap_path = os.path.join(OUTPUT_DIR, 'cohort_retention_heatmap.png')
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"✅ Saved Cohort Heatmap layout directly to: {heatmap_path}")
    
    print("\n📊 Visualizer Step 2: Generating Customer Lifecycle Segment Distributions...")
    # Read matrix outputs built by your previous script
    lifecycle_df = pd.read_csv(MATRIX_DATA_PATH)
    
    # Count profiles across segments and sort by volume descending
    segment_counts = lifecycle_df['Lifecycle_Segment'].value_counts().reset_index()
    segment_counts.columns = ['Lifecycle_Segment', 'Total_Customers']
    segment_counts = segment_counts.sort_values(by='Total_Customers', ascending=False)
    
    # Render Segment Distribution Bar Chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Total_Customers', y='Lifecycle_Segment', data=segment_counts, palette='viridis', hue='Lifecycle_Segment', legend=False)
    plt.title('Customer Segment Counts Across Portfolio Lifecycle Groups', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Total Volumetric Profile Count', fontsize=11)
    plt.ylabel('Lifecycle Segment Categorization', fontsize=11)
    
    # Annotate absolute values onto the bars
    for index, value in enumerate(segment_counts['Total_Customers']):
        plt.text(value + 15, index, f" {int(value)}", va='center', fontweight='bold', color='black')
        
    plt.tight_layout()
    distribution_chart_path = os.path.join(OUTPUT_DIR, 'customer_segment_distributions.png')
    plt.savefig(distribution_chart_path, dpi=300)
    plt.close()
    print(f"✅ Saved Portfolio Segment chart directly to: {distribution_chart_path}")
    print("\n🎉 Phase 5 Visualization scripts finished running successfully!")

if __name__ == "__main__":
    build_retention_visualizations()