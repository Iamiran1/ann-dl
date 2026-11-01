"""Bônus do item 1D: onde o encoder coloca cada dígito no espaço latente de 16 dimensões.

Os μ das 10 000 imagens de teste (16 números cada) são projetados em 2D por t-SNE, só para
poder desenhar, e cada ponto é colorido pelo dígito. O t-SNE preserva quem é vizinho de quem
no espaço de 16 dimensões; as distâncias entre grupos no desenho não têm significado.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from ex1_vae import train_vae, X_test, y_test, FIGURES

model, _ = train_vae(d=16)  # mesma semente 42: é o modelo do Exercise 1
model.eval()

with torch.no_grad():
    mu, _ = model.encode(X_test)  # (10000, 16): o código de cada imagem, sem sortear z
labels = y_test.numpy()

pts = TSNE(n_components=2, random_state=0, init="pca").fit_transform(mu.numpy())  # (10000, 2)

cores = plt.get_cmap("tab10")
fig, ax = plt.subplots(figsize=(8, 7))
for digit in range(10):
    p = pts[labels == digit]
    ax.scatter(p[:, 0], p[:, 1], s=3, alpha=0.5, color=cores(digit), label=str(digit))
for digit in range(10):  # rótulo no centro de cada nuvem
    c = np.median(pts[labels == digit], axis=0)
    ax.text(c[0], c[1], str(digit), fontsize=14, weight="bold", ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.2", facecolor="white", edgecolor=cores(digit), lw=2))
ax.set_title("μ das 10 000 imagens de teste no latente de 16 dimensões, projetado por t-SNE")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.legend(title="Dígito", markerscale=4, loc="center left", bbox_to_anchor=(1, 0.5))
fig.savefig(FIGURES / "bonus-latente-tsne.png", dpi=150, bbox_inches="tight")
plt.close(fig)
