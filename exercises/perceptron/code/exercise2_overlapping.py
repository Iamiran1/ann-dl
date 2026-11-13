"""Exercise 2 — Perceptron em dados sobrepostos, com o algoritmo pocket."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from perceptron import Perceptron

np.set_printoptions(precision=6, suppress=True)

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Mesmo RNG durante todo o script — criado uma unica vez
rng = np.random.default_rng(42)

MEAN_0 = np.array([3.0, 3.0])
COV_0 = np.array([[1.5, 0.0], [0.0, 1.5]])
MEAN_1 = np.array([4.0, 4.0])
COV_1 = np.array([[1.5, 0.0], [0.0, 1.5]])

ETA = 0.01
MAX_EPOCHS = 100


def bayes_optimal_accuracy():
    """Melhor acuracia possivel para QUALQUER fronteira linear neste problema.

    Como as duas classes tem a mesma covariancia isotropica, a fronteira
    otima de Bayes e a mediatriz do segmento que une as medias -- ou seja,
    e linear, e coincide com "a melhor reta possivel" citada no enunciado.
    So usa math.erf (biblioteca padrao), nao e um modelo ajustado a dados.
    """
    d = np.linalg.norm(MEAN_1 - MEAN_0)
    sigma = np.sqrt(COV_0[0, 0])
    z = -d / (2 * sigma)
    bayes_error = 0.5 * (1 + math.erf(z / np.sqrt(2)))
    return 1 - bayes_error


# ============================================================
# A — Generate the data
# ============================================================

classA = rng.multivariate_normal(MEAN_0, COV_0, size=1000)
classB = rng.multivariate_normal(MEAN_1, COV_1, size=1000)

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(classA[:, 0], classA[:, 1], label="Classe 0", alpha=0.6)
ax.scatter(classB[:, 0], classB[:, 1], label="Classe 1", alpha=0.6)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_title("Figura 4 — Dados sobrepostos")
ax.legend()
fig.savefig(FIGURES / "fig04-overlapping-data.png", dpi=150)
plt.close(fig)

X = np.vstack([classA, classB])
y = np.concatenate([np.zeros(1000), np.ones(1000)]).astype(int)

bayes_acc = bayes_optimal_accuracy()
mean_norm = float(np.mean(np.linalg.norm(X, axis=1)))
print(f"Acuracia da melhor reta possivel (Bayes, covariancias iguais) = {bayes_acc:.4f}")
print(f"||x|| medio no dataset = {mean_norm:.4f}")

# ============================================================
# B — treino com pocket (mesma classe do Exercise 1, so liga track_pocket)
# ============================================================

w_init = rng.normal(0, 0.01, size=2)
print(f"\nw_init sorteado = {w_init}")

perceptron = Perceptron(eta=ETA, w_init=w_init, b_init=0.0, max_epochs=MAX_EPOCHS)
hist = perceptron.fit(X, y, track_pocket=True)

print("\n=== Exercise 2B — pesos finais vs. pocket ===")
print(f"w final  = [{perceptron.w[0]:.6f}, {perceptron.w[1]:.6f}], "
      f"b final  = {perceptron.b:.6f}, acc final  = {hist['acc_final']:.4f}")
print(f"w pocket = [{hist['pocket_w'][0]:.6f}, {hist['pocket_w'][1]:.6f}], "
      f"b pocket = {hist['pocket_b']:.6f}, acc pocket = {hist['pocket_acc']:.4f}")
print(f"epoca do recorde pocket = {hist['pocket_epoch']}")
print(f"epochs rodados = {hist['epochs']} (bateu no teto de {MAX_EPOCHS}: {hist['epochs'] == MAX_EPOCHS})")
w0, w1, b = perceptron.w[0], perceptron.w[1], perceptron.b
print(f"||w final|| = {np.linalg.norm([w0, w1]):.6f}, |b final| = {abs(b):.6f}")
print(f"passo de w por update: eta*||x|| ~ {ETA*mean_norm:.4f}  |  passo de b por update: eta = {ETA:.4f}")

# ============================================================
# C — Figure 5: fronteiras final e pocket
# ============================================================

def predict_com_pesos(w0_, w1_, b_, X):
    """Mesma formula do Perceptron.predict, so que para um w/b escolhido
    na mao -- usada aqui so para desenhar as duas fronteiras (final e
    pocket) lado a lado, sem precisar de dois objetos Perceptron."""
    y_pred = []
    for x0, x1 in X:
        z = w0_*x0 + w1_*x1 + b_
        y_pred.append(1 if z >= 0 else 0)
    return np.array(y_pred)


fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1

configs = [
    (axes[0], perceptron.w[0], perceptron.w[1], perceptron.b, "Pesos finais"),
    (axes[1], hist["pocket_w"][0], hist["pocket_w"][1], hist["pocket_b"], "Pesos pocket"),
]

for ax, w0_, w1_, b_, title in configs:
    y_pred = predict_com_pesos(w0_, w1_, b_, X)
    misclassified = y_pred != y

    for c in (0, 1):
        mask = (y == c) & (~misclassified)
        ax.scatter(*X[mask].T, s=12, alpha=0.6, label=f"Classe {c}")
    if misclassified.any():
        ax.scatter(*X[misclassified].T, s=30, facecolors="none",
                   edgecolors="red", linewidths=1.0, label="Mal classificado")

    if abs(w1_) > 1e-12:
        x1_line = np.array([x1_min, x1_max])
        x2_line = -(w0_*x1_line + b_) / w1_
        ax.plot(x1_line, x2_line, "k--", linewidth=1.5, label="Fronteira")

    ax.set_xlim(x1_min, x1_max)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    acc = float(np.mean(y_pred == y))
    ax.set_title(f"{title} (acurácia = {acc:.1%})")
    ax.legend(loc="best", fontsize=8)

fig.suptitle("Figura 5 — Fronteiras final e pocket")
fig.tight_layout()
fig.savefig(FIGURES / "fig05-final-vs-pocket.png", dpi=150)
plt.close(fig)

# ============================================================
# C — Figure 6: acuracia atual vs. pocket, por epoca
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))
epochs_axis = np.arange(1, len(hist["acc_history"]) + 1)
ax.plot(epochs_axis, hist["acc_history"], label="Acurácia dos pesos atuais", alpha=0.8)
ax.plot(epochs_axis, hist["pocket_acc_history"], label="Melhor até agora (pocket)", linewidth=2)
ax.set_xlabel("Época")
ax.set_ylabel("Acurácia")
ax.set_title("Figura 6 — Acurácia por época: atual vs. pocket")
ax.set_ylim(0, 1.02)
ax.legend(loc="best")
ax.grid(alpha=0.3)
fig.savefig(FIGURES / "fig06-accuracy-final-vs-pocket.png", dpi=150)
plt.close(fig)
