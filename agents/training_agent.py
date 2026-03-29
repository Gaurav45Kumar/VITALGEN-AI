from models.tabular_generator import TabularVitalGen
from models.diffusion_generator import train_diffusion
from agents.privacy_agent import PrivacyEnforcementAgent

class ModelTrainingAgent:
    def __init__(self):
        self.privacy_agent = PrivacyEnforcementAgent()

    def train_tabular(self, real_data, epochs=150):
        self.privacy_agent.protect_training()
        print("🤖 Training Agent: Training Tabular CTGAN model...")
        tabular_gen = TabularVitalGen()
        model = tabular_gen.train(real_data, epochs)
        return model, tabular_gen

    def train_diffusion(self, image_dir):
        print("🖼️ Training Agent: Training Diffusion model for medical images...")
        return train_diffusion(image_dir)
