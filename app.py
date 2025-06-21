# app.py
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Generator model structure (same as during training)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(100 + 10, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        img = self.model(x)
        return img.view(-1, 1, 28, 28)

# Load model
device = torch.device("cpu")
generator = Generator().to(device)
generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.eval()

# Streamlit UI
st.title("🖊️ Handwritten Digit Generator")
digit = st.selectbox("Choose a digit to generate (0-9)", list(range(10)))

if st.button("Generate Images"):
    z = torch.randn(5, 100)
    labels = torch.full((5,), digit, dtype=torch.long)
    with torch.no_grad():
        generated_imgs = generator(z, labels)

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axes[i].imshow(generated_imgs[i][0].cpu(), cmap="gray")
        axes[i].axis("off")
    st.pyplot(fig)