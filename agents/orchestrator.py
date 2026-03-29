import os
from agents.data_ingestion import ingest_data
from agents.training_agent import ModelTrainingAgent
from agents.generation_agent import GenerationAgent
from models.fidelity_checker import tstr_evaluation

def run_vitalgen_pipeline(tabular_path: str, image_dir: str, num_synthetic: int = 800):
    print("\n🧬 Starting VitalGen Agent Workflow...\n")
    
    # Agent 1: Data Ingestion
    real_data = ingest_data(tabular_path)
    
    # Agent 2: Training (with Privacy)
    trainer = ModelTrainingAgent()
    tabular_model, tabular_gen = trainer.train_tabular(real_data)
    diffusion_model = trainer.train_diffusion(image_dir)
    
    # Agent 3: Generation
    generator = GenerationAgent()
    synthetic_tabular = generator.generate_tabular(tabular_model, num_synthetic)
    synthetic_images = generator.generate_images(diffusion_model, 5)
    
    # Agent 4: Validation
    print("\n📊 Validation Agent: Running TSTR evaluation...")
    tstr_evaluation(real_data, synthetic_tabular, target_col='pneumonia')
    
    # Export synthetic data only
    synthetic_tabular.to_csv("synthetic_patients.csv", index=False)
    print(f"\n✅ Export Control Agent: Saved {len(synthetic_tabular)} synthetic records to synthetic_patients.csv")
    print(f"🖼️  Generated {synthetic_images.shape[0]} synthetic medical images")
    
    print("\n🎉 All agents completed! Privacy preserved. Synthetic data ready.")
