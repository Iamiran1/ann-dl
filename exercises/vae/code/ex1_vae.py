import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).resolve().parents[1] / "figures"

train_full = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
test = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
train, val = random_split(train_full, [50_000, 10_000],
                          generator=torch.Generator().manual_seed(42))


def to_tensors(dataset):
    """Empilha um dataset de imagens (1, 28, 28) em X (N, 784) e y (N,)."""
    X = torch.stack([img for img, _ in dataset]).view(-1, 784)  # achata 28x28 -> 784
    y = torch.tensor([label for _, label in dataset])
    return X, y


X_train, y_train = to_tensors(train)
X_val, y_val = to_tensors(val)
X_test, y_test = to_tensors(test)  # só usado nos números finais

# as 10 primeiras imagens de treino, desachatadas de volta para 28x28
fig, axes = plt.subplots(1, 10, figsize=(12, 2))
for i, ax in enumerate(axes):
    ax.imshow(X_train[i].view(28, 28), cmap="gray")
    ax.set_title(y_train[i].item())
    ax.axis("off")
fig.suptitle("Primeiras 10 imagens de treino do MNIST (rótulo acima)")
fig.savefig(FIGURES / "amostras-treino.png", dpi=150, bbox_inches="tight")
plt.close(fig)

import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        # encoder: 784 -> 512 -> 256, ReLU depois de cada
        self.encoder = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        # duas cabeças lineares 256 -> d (sem ativação!)
        self.head_mu = nn.Linear(256, d)
        self.head_logvar = nn.Linear(256, d)
        # decoder: d -> 256 -> 512 -> 784, sigmoid na saída
        self.decoder = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512,784), nn.Sigmoid(),
        )
    def encode(self, x):
        h = self.encoder(x)
        mu = self.head_mu(h)
        logvar = self.head_logvar(h)
        return mu, logvar
    def reparameterize(self,mu,logvar ):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar


def vae_loss(x_hat, x, mu, logvar):
    """Devolve os dois termos da loss separados, em nats por imagem."""
    batch = x.shape[0]
    # reconstrução: BCE somada nos 784 pixels e dividida pelo nº de imagens
    # (reduction="mean" dividiria também por 784 e encolheria esse termo)
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / batch
    # KL na forma fechada: ½ Σ (μ² + σ² − 1 − log σ²), somada nas d dimensões
    kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar) / batch
    return recon, kl


@torch.no_grad()  # só avalia: não calcula gradiente nem mexe nos pesos
def evaluate(model, X, batch_size=1000):
    """Reconstrução e KL médias (nats por imagem) de um conjunto inteiro."""
    model.eval()
    recon_total, kl_total = 0.0, 0.0
    for i in range(0, len(X), batch_size):
        x = X[i:i + batch_size]
        x_hat, mu, logvar = model(x)
        recon, kl = vae_loss(x_hat, x, mu, logvar)
        recon_total += recon.item() * len(x)  # volta à soma do batch, para fazer a média no fim
        kl_total += kl.item() * len(x)
    return recon_total / len(X), kl_total / len(X)


def train_vae(d, epochs=20, batch_size=128, lr=1e-3):
    """Treina um VAE com latente de dimensão d e devolve o modelo e o histórico por época."""
    torch.manual_seed(42)  # semente antes de construir o modelo (regra técnica)
    model = VAE(d)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"train_recon": [], "train_kl": [], "val_recon": [], "val_kl": []}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))  # embaralha o treino a cada época
        recon_total, kl_total = 0.0, 0.0
        for i in range(0, len(X_train), batch_size):
            x = X_train[perm[i:i + batch_size]]
            x_hat, mu, logvar = model(x)              # forward
            recon, kl = vae_loss(x_hat, x, mu, logvar)
            loss = recon + kl                         # o que o otimizador minimiza
            opt.zero_grad()
            loss.backward()                           # gradiente de todos os pesos
            opt.step()                                # ajusta os pesos
            recon_total += recon.item() * len(x)
            kl_total += kl.item() * len(x)

        # treino: média da época; validação: avaliada no fim da época
        hist["train_recon"].append(recon_total / len(X_train))
        hist["train_kl"].append(kl_total / len(X_train))
        val_recon, val_kl = evaluate(model, X_val)
        hist["val_recon"].append(val_recon)
        hist["val_kl"].append(val_kl)
        print(f"d={d} época {epoch + 1:2d}: treino recon {hist['train_recon'][-1]:.2f} "
              f"KL {hist['train_kl'][-1]:.2f} | val recon {val_recon:.2f} KL {val_kl:.2f}")

    return model, hist


# só treina e gera as figuras quando o script é rodado diretamente;
# assim o ex2_vae.py pode importar VAE, train_vae, evaluate e os dados sem retreinar tudo
if __name__ == "__main__":
    model, hist = train_vae(d=16)

    # C — números finais no teste (a única vez em que o teste é usado)
    test_recon, test_kl = evaluate(model, X_test)
    print(f"teste (d=16): recon {test_recon:.2f}  KL {test_kl:.2f}  "
          f"-ELBO {test_recon + test_kl:.2f} nats/imagem")
    print(f"validação KL: 1ª época {hist['val_kl'][0]:.2f}, última época {hist['val_kl'][-1]:.2f}")

    # Figura 1 — os dois termos da loss por época, treino e validação
    epochs = range(1, len(hist["train_recon"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, hist["train_recon"], marker="o", label="Treino")
    ax1.plot(epochs, hist["val_recon"], marker="o", label="Validação")
    ax1.set_title("Reconstrução (BCE) por época")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Reconstrução (nats/imagem)")
    ax1.legend()
    ax2.plot(epochs, hist["train_kl"], marker="o", label="Treino")
    ax2.plot(epochs, hist["val_kl"], marker="o", label="Validação")
    ax2.set_title("KL por época")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("KL (nats/imagem)")
    ax2.legend()
    for ax in (ax1, ax2):
        ax.set_xticks(range(2, len(epochs) + 1, 2))  # épocas inteiras no eixo x
    fig.suptitle("Figura 1 — Termos da loss do VAE (d = 16)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01-loss-terms.png", dpi=150)
    plt.close(fig)

    # D — Figura 2: primeira imagem de teste de cada dígito e sua reconstrução
    model.eval()
    first_idx = [int((y_test == digit).nonzero()[0]) for digit in range(10)]
    print("índices de teste da Figura 2:", first_idx)
    with torch.no_grad():
        mu, _ = model.encode(X_test[first_idx])
        recon_imgs = model.decoder(mu)  # decodifica μ, sem sortear z

    fig, axes = plt.subplots(2, 10, figsize=(12, 3))
    for col in range(10):
        axes[0, col].imshow(X_test[first_idx[col]].view(28, 28), cmap="gray")
        axes[1, col].imshow(recon_imgs[col].view(28, 28), cmap="gray")
        axes[0, col].set_title(str(col))
        for row in range(2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstrução")
    fig.suptitle("Figura 2 — Primeira imagem de teste de cada dígito e sua reconstrução (decoder(μ), d = 16)")
    fig.savefig(FIGURES / "fig02-reconstructions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # D — Figura 3: 8x8 imagens geradas a partir de z ~ N(0, I)
    torch.manual_seed(0)
    with torch.no_grad():
        z = torch.randn(64, 16)          # sorteio do prior, escrito à mão
        samples = model.decoder(z)

    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for ax, img in zip(axes.flat, samples):
        ax.imshow(img.view(28, 28), cmap="gray")
        ax.axis("off")
    fig.suptitle("Figura 3 — 64 amostras do prior z ~ N(0, I) decodificadas (d = 16)")
    fig.savefig(FIGURES / "fig03-prior-samples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

