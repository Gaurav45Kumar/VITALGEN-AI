import pandas as pd
import os

def ingest_data(tabular_path: str):
    print("📥 Data Ingestion Agent: Loading structured medical data...")
    df = pd.read_csv(tabular_path)
    print(f"   Loaded {len(df)} real patient records from {tabular_path}")
    return df
