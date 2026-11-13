"""Passo a passo de uma imagem de teste pelo VAE treinado (d = 16).

Gera uma figura por etapa do forward pass, usadas no item B do relatório.
Reaproveita os dados, a classe VAE e o treino do Exercise 1.
"""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ex1_vae import train_vae, vae_loss, X_test, y_test, FIGURES

model, _ = train_vae(d=16)  # mesma semente 42: é o modelo do Exercise 1
model.eval()

IDX = 0  # primeira imagem de teste (um 7)
x = X_test[IDX:IDX + 1]  # (1, 784)
OUT = FIGURES / "passos"
OUT.mkdir(exist_ok=True)


def salvar(fig, nome):
    fig.savefig(OUT / nome, dpi=150, bbox_inches="tight")
    plt.close(fig)


def mapa(ax, valores, forma, titulo):
    """Mostra um vetor de ativações como um mapa de calor (só para visualizar)."""
    im = ax.imshow(valores.reshape(forma), cmap="magma", aspect="auto")
    ax.set_title(titulo, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


with torch.no_grad():
    # forward pass camada por camada, guardando cada saída
    h1 = model.encoder[1](model.encoder[0](x))      # 784 -> 512, ReLU
    h2 = model.encoder[3](model.encoder[2](h1))     # 512 -> 256, ReLU
    mu = model.head_mu(h2)                          # 256 -> 16
    logvar = model.head_logvar(h2)                  # 256 -> 16
    std = torch.exp(0.5 * logvar)
    torch.manual_seed(1)
    eps = torch.randn_like(std)
    z = mu + std * eps                              # reparametrização
    g1 = model.decoder[1](model.decoder[0](z))      # 16 -> 256, ReLU
    g2 = model.decoder[3](model.decoder[2](g1))     # 256 -> 512, ReLU
    x_hat = model.decoder[5](model.decoder[4](g2))  # 512 -> 784, sigmoide
    recon, kl = vae_loss(x_hat, x, mu, logvar)

print(f"imagem de teste {IDX} (dígito {y_test[IDX].item()})")
print(f"ReLU zerou {np.mean(h1.numpy() == 0):.0%} da camada 512 e {np.mean(h2.numpy() == 0):.0%} da 256 (encoder)")
print(f"recon {recon.item():.2f}  KL {kl.item():.2f} nats")

# 1 — entrada: imagem 28x28 achatada em 784 números
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 2.4), gridspec_kw={"width_ratios": [1, 6]})
a.imshow(x.view(28, 28), cmap="gray")
a.set_title(f"Imagem de teste {IDX} (28×28)", fontsize=10)
a.axis("off")
b.imshow(x.numpy(), cmap="gray", aspect="auto", interpolation="nearest")  # (1, 784): uma linha
b.set_title("O vetor de entrada: os 784 pixels em fila, linha após linha da imagem", fontsize=10)
b.set_yticks([])
b.set_xticks([0, 196, 392, 588, 783])
b.set_xlabel("posição no vetor (pixel 0 a 783)")
fig.suptitle("Entrada: a imagem vira um vetor de 784 números em [0, 1]", y=1.05)
salvar(fig, "passo1-entrada.png")

# 2 — encoder: 784 -> 512 -> 256
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.2))
mapa(a, h1.numpy(), (16, 32), f"Camada 1: 512 saídas após ReLU ({np.mean(h1.numpy() == 0):.0%} zeradas)")
im = mapa(b, h2.numpy(), (16, 16), f"Camada 2: 256 saídas após ReLU ({np.mean(h2.numpy() == 0):.0%} zeradas)")
fig.colorbar(im, ax=[a, b], shrink=0.8, label="ativação")
fig.suptitle("Encoder: 784 → 512 → 256 (cada quadrado é a saída de um neurônio)", y=1.03)
salvar(fig, "passo2-encoder.png")

# 3 — cabeças: 256 -> μ (16) e 256 -> log σ² (16)
dims = np.arange(1, 17)
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.2))
a.bar(dims, mu[0].numpy(), color="tab:blue")
a.axhline(0, color="gray", lw=0.8)
a.set_title("Cabeça μ: 16 médias", fontsize=10)
a.set_xlabel("Dimensão latente")
a.set_ylabel("μ")
b.bar(dims, logvar[0].numpy(), color="tab:orange")
b.axhline(0, color="gray", lw=0.8)
b.set_title("Cabeça log σ²: 16 log-variâncias", fontsize=10)
b.set_xlabel("Dimensão latente")
b.set_ylabel("log σ²")
for ax in (a, b):
    ax.set_xticks(dims)
fig.suptitle("Cabeças: os mesmos 256 valores viram μ e log σ² (podem ser negativos)", y=1.03)
fig.tight_layout()
salvar(fig, "passo3-cabecas.png")

# 4 — reparametrização: z = μ + σ·ε
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.errorbar(dims, mu[0].numpy(), yerr=std[0].numpy(), fmt="o", color="tab:blue",
            capsize=4, label="μ ± σ (onde z deve cair)")
ax.scatter(dims, z[0].numpy(), marker="x", s=60, color="tab:red", zorder=3, label="z sorteado = μ + σ·ε")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xticks(dims)
ax.set_xlabel("Dimensão latente")
ax.set_ylabel("valor")
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
ax.set_title("Reparametrização: sorteia z em torno de μ, com espalhamento σ (sem pesos)")
salvar(fig, "passo4-reparametrizacao.png")

# 5 — decoder: 16 -> 256 -> 512
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.2))
mapa(a, g1.numpy(), (16, 16), f"Camada 1: 256 saídas após ReLU ({np.mean(g1.numpy() == 0):.0%} zeradas)")
im = mapa(b, g2.numpy(), (16, 32), f"Camada 2: 512 saídas após ReLU ({np.mean(g2.numpy() == 0):.0%} zeradas)")
fig.colorbar(im, ax=[a, b], shrink=0.8, label="ativação")
fig.suptitle("Decoder: 16 → 256 → 512 (o caminho inverso do encoder)", y=1.03)
salvar(fig, "passo5-decoder.png")

# 6 — saída: 512 -> 784 com sigmoide, desachatada em 28x28
fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 3.2))
a.imshow(x.view(28, 28), cmap="gray")
a.set_title("Original x", fontsize=10)
im = b.imshow(x_hat.view(28, 28), cmap="gray", vmin=0, vmax=1)
b.set_title("Reconstrução x̂ (sigmoide)", fontsize=10)
for ax in (a, b):
    ax.axis("off")
fig.colorbar(im, ax=[a, b], shrink=0.8, label="probabilidade do pixel aceso")
fig.suptitle("Saída: 784 valores entre 0 e 1, de volta para 28×28", y=1.03)
salvar(fig, "passo6-saida.png")

# 7 — loss: BCE pixel a pixel e KL por dimensão
bce_px = F.binary_cross_entropy(x_hat, x, reduction="none").view(28, 28)
kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)[0]
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 2]})
im = a.imshow(bce_px, cmap="inferno")
a.set_title(f"Reconstrução (BCE) por pixel\nsoma = {recon.item():.1f} nats", fontsize=10)
a.axis("off")
fig.colorbar(im, ax=a, shrink=0.8, label="nats")
b.bar(dims, kl_dim.numpy(), color="tab:green")
b.set_xticks(dims)
b.set_xlabel("Dimensão latente")
b.set_ylabel("nats")
b.set_title(f"KL por dimensão: ½(μ² + σ² − 1 − log σ²)\nsoma = {kl.item():.1f} nats", fontsize=10)
fig.suptitle("Loss desta imagem: reconstrução + KL", y=1.06)
fig.tight_layout()
salvar(fig, "passo7-loss.png")
