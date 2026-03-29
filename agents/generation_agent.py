from models.tabular_generator import TabularVitalGen
from models.diffusion_generator import generate_synthetic_images

class GenerationAgent:
    def generate_tabular(self, tabular_model, num_samples=1000):
        """Generate synthetic tabular data using the trained synthesizer"""
        print(f"🔄 Generation Agent: Generating {num_samples} synthetic patient records...")
        # Fixed: Use .sample() instead of .generate()
        return tabular_model.synthesizer.sample(num_samples)

    def generate_images(self, diffusion_model, num_samples=5):
        print(f"🖼️ Generation Agent: Generating {num_samples} synthetic medical images...")
        return generate_synthetic_images(diffusion_model, num_samples)
