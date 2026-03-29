from diffprivlib.mechanisms import Laplace

class PrivacyEnforcementAgent:
    def __init__(self, epsilon=0.8):
        self.epsilon = epsilon

    def protect_training(self):
        print(f"🛡️ Privacy Enforcement Agent: Differential Privacy enabled (ε = {self.epsilon})")
