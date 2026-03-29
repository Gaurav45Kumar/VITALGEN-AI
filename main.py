import os
import pandas as pd
from agents.orchestrator import run_vitalgen_pipeline

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 VitalGen: Privacy-Preserving Synthetic Medical Data Platform")
    print("=" * 60)

    # Create necessary folders
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    tabular_path = "data/mimic_sample.csv"
    
    # Create sample data if it doesn't exist
    if not os.path.exists(tabular_path):
        print("📊 Creating sample medical data for demonstration...")
        
        # Safe way to create DataFrame with equal length columns
        n_samples = 100
        data = {
            'patient_id': list(range(1, n_samples + 1)),
            'age': [45, 62, 33, 71, 55, 28, 67, 39] * 13,
            'blood_pressure': [120, 140, 110, 160, 130, 115, 155, 125] * 13,
            'heart_rate': [78, 92, 65, 110, 85, 70, 105, 80] * 13,
            'smoking': ['Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No'] * 13,
            'diabetes': [0, 1, 0, 1, 0, 0, 1, 0] * 13,
            'pneumonia': [0, 1, 0, 1, 0, 0, 1, 0] * 13
        }
        
        # Truncate all columns to exactly 100 rows
        for key in data:
            data[key] = data[key][:n_samples]
        
        df = pd.DataFrame(data)
        df.to_csv(tabular_path, index=False)
        print(f"   ✅ Created sample dataset with {len(df)} patient records")

    image_dir = "data/chest_xrays/"
    os.makedirs(image_dir, exist_ok=True)

    # Run the full pipeline
    run_vitalgen_pipeline(tabular_path, image_dir, num_synthetic=800)

    print("\n🎉 VitalGen execution completed successfully!")
    print("📁 Synthetic data saved as 'synthetic_patients.csv'")