import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

class SimpleDDPM(nn.Module):
    def __init__(self, img_size=64, timesteps=200):
        super().__init__()
        self.img_size = img_size
        self.timesteps = timesteps
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x, t):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        alpha = 1.0 - 0.02 * (t / self.timesteps)
        return torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise, noise

def train_diffusion(data_dir, epochs=5, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleDDPM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    
    images = []
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img_path = os.path.join(data_dir, filename)
                img = Image.open(img_path).convert('L')
                images.append(transform(img))
            except Exception:
                continue
    
    if len(images) == 0:
        print("⚠️ No medical images found. Skipping diffusion training.")
        return None
    
    dataset = torch.stack(images)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Diffusion Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch = batch.to(device)
            t = torch.randint(0, model.timesteps, (batch.size(0),)).float().to(device)
            noisy, noise = model.add_noise(batch, t)
            pred = model(noisy.unsqueeze(1), t)
            loss = F.mse_loss(pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    torch.save(model.state_dict(), "models/diffusion_model.pth")
    print("✅ Diffusion model trained and saved!")
    return model

def generate_synthetic_images(model, num_samples=5, device="cpu"):
    if model is None:
        print("⚠️ Using dummy synthetic images (no real images provided).")
        return torch.randn(num_samples, 1, 64, 64)
    
    model.eval()
    with torch.no_grad():
        samples = torch.randn(num_samples, 1, 64, 64).to(device)
        for t in range(199, -1, -1):
            pred = model(samples, torch.full((num_samples,), t, device=device, dtype=torch.float32))
            samples = samples - 0.02 * pred
        return samples.cpu()
