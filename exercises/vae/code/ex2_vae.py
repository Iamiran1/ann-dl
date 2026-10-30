import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm

# reaproveita do Exercise 1: dados já achatados, a classe VAE, o treino e a avaliação
# (o ex1_vae.py só treina quando é rodado diretamente, então importar não retreina nada)
from ex1_vae import VAE, train_vae, evaluate, X_test, y_test, FIGURES

# ---------------------------------------------------------------------------
# A — VAE 2-D: mesmo modelo, dados, semente, β e orçamento; só muda d
# ---------------------------------------------------------------------------
model2, hist2 = train_vae(d=2)
# o modelo d = 16 do Exercise 1 é treinado de novo aqui: train_vae fixa
# torch.manual_seed(42) antes de tudo, então ele sai idêntico ao do ex1_vae.py
model16, hist16 = train_vae(d=16)

recon2, kl2 = evaluate(model2, X_test)
recon16, kl16 = evaluate(model16, X_test)
print(f"teste d=16: recon {recon16:.2f}  KL {kl16:.2f}  -ELBO {recon16 + kl16:.2f}")
print(f"teste d=2 : recon {recon2:.2f}  KL {kl2:.2f}  -ELBO {recon2 + kl2:.2f}")

model2.eval()
model16.eval()

# ---------------------------------------------------------------------------
# B — Figura 4: μ das 10 000 imagens de teste, colorido por dígito
# ---------------------------------------------------------------------------
with torch.no_grad():
    mu_test, _ = model2.encode(X_test)  # (10000, 2): usa μ, sem sortear z
mu_test = mu_test.numpy()
labels = y_test.numpy()

fig, ax = plt.subplots(figsize=(8, 7))
cmap = plt.get_cmap("tab10")
for digit in range(10):
    pts = mu_test[labels == digit]
    ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.6, color=cmap(digit), label=str(digit))
ax.set_title("Figura 4 — μ das 10 000 imagens de teste no espaço latente (d = 2)")
ax.set_xlabel("$\\mu_1$")
ax.set_ylabel("$\\mu_2$")
ax.legend(title="Dígito", markerscale=4, loc="center left", bbox_to_anchor=(1, 0.5))
fig.savefig(FIGURES / "fig04-latent-scatter.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# centro de cada dígito no plano, para localizar as sobreposições no texto
for digit in range(10):
    c = mu_test[labels == digit].mean(axis=0)
    print(f"dígito {digit}: centro de μ = ({c[0]:+.2f}, {c[1]:+.2f})")

# ---------------------------------------------------------------------------
# C — Figura 5: plano decodificado numa grade 15×15 em quantis do prior
# ---------------------------------------------------------------------------
n = 15
q = np.linspace(0.05, 0.95, n)  # 15 probabilidades igualmente espaçadas
grid = norm.ppf(q)              # z = Φ⁻¹(q): inversa da CDF da N(0, 1)
tick_labels = [f"{abs(v) if abs(v) < 1e-9 else v:.2f}" for v in grid]  # evita "-0.00" no centro
print("valores de z na grade:", np.round(grid, 2))

canvas = np.zeros((n * 28, n * 28))
with torch.no_grad():
    for row, z2 in enumerate(grid[::-1]):  # z2 cresce de baixo para cima, como num gráfico
        for col, z1 in enumerate(grid):
            z = torch.tensor([[z1, z2]], dtype=torch.float32)
            img = model2.decoder(z).view(28, 28).numpy()
            canvas[row * 28:(row + 1) * 28, col * 28:(col + 1) * 28] = img

fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(canvas, cmap="gray")
centers = np.arange(n) * 28 + 14
ax.set_xticks(centers)
ax.set_xticklabels(tick_labels, rotation=90)
ax.set_yticks(centers)
ax.set_yticklabels(tick_labels[::-1])
ax.set_xlabel("$z_1 = \\Phi^{-1}(q)$")
ax.set_ylabel("$z_2 = \\Phi^{-1}(q)$")
ax.set_title("Figura 5 — Plano latente decodificado, grade 15×15 em quantis da N(0, 1) (d = 2)")
fig.savefig(FIGURES / "fig05-latent-plane.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# D — Figura 6: interpolação em pixels × no latente (com o modelo d = 16)
# ---------------------------------------------------------------------------
IDX_A, IDX_B = 3, 2  # índices de teste escolhidos (dígitos diferentes)
x_a, x_b = X_test[IDX_A], X_test[IDX_B]
print(f"interpolação: índice {IDX_A} (dígito {y_test[IDX_A].item()}) -> "
      f"índice {IDX_B} (dígito {y_test[IDX_B].item()})")

ts = torch.linspace(0, 1, 10)
with torch.no_grad():
    pixel_row = [(1 - t) * x_a + t * x_b for t in ts]           # mistura direta dos pixels
    mu_ab, _ = model16.encode(torch.stack([x_a, x_b]))
    mu_a, mu_b = mu_ab[0], mu_ab[1]
    z_path = torch.stack([(1 - t) * mu_a + t * mu_b for t in ts])  # caminho entre os μ
    latent_row = model16.decoder(z_path)

fig, axes = plt.subplots(2, 10, figsize=(13, 2.9))
for col, t in enumerate(ts):
    axes[0, col].imshow(pixel_row[col].view(28, 28), cmap="gray")
    axes[1, col].imshow(latent_row[col].view(28, 28), cmap="gray")
    axes[0, col].set_title(f"t = {t:.2f}", fontsize=9)
    for row in range(2):
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
axes[0, 0].set_ylabel("Pixels")
axes[1, 0].set_ylabel("Latente\n(d = 16)")
fig.suptitle(f"Figura 6 — Interpolação entre as imagens de teste {IDX_A} e {IDX_B}: "
             "em pixels (cima) e no espaço latente (baixo)")
fig.savefig(FIGURES / "fig06-interpolation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
