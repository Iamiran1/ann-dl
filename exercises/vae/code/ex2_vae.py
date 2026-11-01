import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm

# reaproveita do Exercise 1: dados já achatados, a classe VAE, o treino e a avaliação
# (o ex1_vae.py só treina quando é rodado diretamente, então importar não retreina nada)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from ex1_vae import VAE, train_vae, evaluate, X_val, y_val, X_test, y_test, FIGURES

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

# quanto os dígitos se misturam no plano: um k-NN (k = 15) "ajustado" nos μ da validação
# classifica os μ de teste pelos vizinhos. Os pares que ele mais confunde são os pares
# que se sobrepõem na Figura 4. O mesmo teste no latente de 16 dimensões serve de comparação.
with torch.no_grad():
    mu_val2, _ = model2.encode(X_val)
    mu_val16, _ = model16.encode(X_val)
    mu_test16, logvar_test16 = model16.encode(X_test)
    _, logvar_test2 = model2.encode(X_test)

knn2 = KNeighborsClassifier(n_neighbors=15).fit(mu_val2.numpy(), y_val.numpy())
pred2 = knn2.predict(mu_test)
knn16 = KNeighborsClassifier(n_neighbors=15).fit(mu_val16.numpy(), y_val.numpy())
pred16 = knn16.predict(mu_test16.numpy())
print(f"k-NN nos μ: acerto {np.mean(pred2 == labels):.1%} (d=2) × {np.mean(pred16 == labels):.1%} (d=16)")

conf = confusion_matrix(labels, pred2)
np.fill_diagonal(conf, 0)
mix = conf + conf.T  # confusões nos dois sentidos (a→b e b→a)
pares = sorted(((mix[a, b], a, b) for a in range(10) for b in range(a + 1, 10)), reverse=True)
print("pares mais misturados no plano (d=2):",
      ", ".join(f"{a}/{b}: {n}" for n, a, b in pares[:5]))

# KL média por dimensão no teste: mostra quantas dimensões cada modelo de fato usa
def kl_por_dim(mu, logvar):
    return (0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)).mean(dim=0)

kl_dim2 = kl_por_dim(torch.from_numpy(mu_test), logvar_test2)
kl_dim16 = kl_por_dim(mu_test16, logvar_test16)
print("KL por dimensão (d=2): ", np.round(kl_dim2.numpy(), 2))
print("KL por dimensão (d=16):", np.round(kl_dim16.numpy(), 2),
      f"-> {int((kl_dim16 > 0.1).sum())} de 16 dimensões com KL > 0.1 nat")

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
IDX_A, IDX_B = 4, 0  # índices de teste escolhidos: um 4 e um 7 (dígitos diferentes)
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
             "em pixels (cima) e no espaço latente (baixo)", y=1.06)
fig.savefig(FIGURES / "fig06-interpolation.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# E — o que d = 2 joga fora: medidas de estilo, original × reconstrução (decoder(μ))
# ---------------------------------------------------------------------------
def medidas_estilo(X):
    """Tinta (espessura), largura e inclinação de cada imagem, pelos momentos dos pixels."""
    X = X.reshape(-1, 28, 28)
    yy, xx = np.mgrid[0:28, 0:28]
    m = X.sum(axis=(1, 2)) + 1e-8                       # tinta: soma dos pixels
    cx = (X * xx).sum(axis=(1, 2)) / m
    cy = (X * yy).sum(axis=(1, 2)) / m
    dx, dy = xx[None] - cx[:, None, None], yy[None] - cy[:, None, None]
    vx = (X * dx**2).sum(axis=(1, 2)) / m
    vy = (X * dy**2).sum(axis=(1, 2)) / m
    cxy = (X * dx * dy).sum(axis=(1, 2)) / m
    return {"tinta": m, "largura": np.sqrt(vx), "inclinação": cxy / vy}


def pct_cinza(X):
    """Porcentagem de pixels entre 0.2 e 0.8: quanto maior, mais borrada a imagem."""
    return np.mean((X > 0.2) & (X < 0.8))


orig = X_test.numpy()
med_orig = medidas_estilo(orig)
print(f"pixels cinza: original {pct_cinza(orig):.1%}")
for nome, modelo in [("d=16", model16), ("d=2", model2)]:
    with torch.no_grad():
        m_, _ = modelo.encode(X_test)
        rec = modelo.decoder(m_).numpy()
    med_rec = medidas_estilo(rec)
    corr = "  ".join(f"{k} r={np.corrcoef(med_orig[k], med_rec[k])[0, 1]:.2f}" for k in med_orig)
    print(f"{nome}: {corr}  pixels cinza {pct_cinza(rec):.1%}")
