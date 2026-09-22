"""Exercise 2 — Perceptron on overlapping data, tracked with the pocket algorithm."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from perceptron import Perceptron

np.set_printoptions(precision=6, suppress=True)

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# Mesmo RNG durante todo o script — nunca recriado
RNG = np.random.default_rng(42)

MEAN_0 = np.array([3.0, 3.0])
COV_0 = np.array([[1.5, 0.0], [0.0, 1.5]])
MEAN_1 = np.array([4.0, 4.0])
COV_1 = np.array([[1.5, 0.0], [0.0, 1.5]])

N_PER_CLASS = 1000
ETA = 0.01
MAX_EPOCHS = 100


def generate_data():
    class0 = RNG.multivariate_normal(MEAN_0, COV_0, size=N_PER_CLASS)
    class1 = RNG.multivariate_normal(MEAN_1, COV_1, size=N_PER_CLASS)
    X = np.vstack([class0, class1])
    y = np.concatenate([np.zeros(N_PER_CLASS), np.ones(N_PER_CLASS)]).astype(int)
    return X, y


def bayes_optimal_accuracy():
    """Melhor acurácia possível para QUALQUER fronteira linear neste problema.

    As duas classes têm a mesma covariância isotrópica, então a fronteira
    ótima de Bayes é a mediatriz do segmento que une as médias — ou seja, é
    linear, e coincide com "a melhor reta possível" citada no enunciado.
    Usa só a função erro da biblioteca padrão (math.erf), não é um modelo
    ajustado a dados: é a fórmula fechada do erro de Bayes para duas
    gaussianas de covariância igual.
    """
    d = np.linalg.norm(MEAN_1 - MEAN_0)
    sigma = np.sqrt(COV_0[0, 0])  # covariância isotrópica: sigma é igual em qualquer direção
    z = -d / (2 * sigma)
    bayes_error = 0.5 * (1 + math.erf(z / np.sqrt(2)))
    return 1 - bayes_error


def plot_scatter(X, y, path, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(*X[y == 0].T, s=14, alpha=0.6, label="Classe 0")
    ax.scatter(*X[y == 1].T, s=14, alpha=0.6, label="Classe 1")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_two_boundaries(X, y, w_final, b_final, w_pocket, b_pocket, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    for ax, w, b, title in [
        (axes[0], w_final, b_final, "Pesos finais"),
        (axes[1], w_pocket, b_pocket, "Pesos pocket"),
    ]:
        y_pred = (X @ w + b >= 0).astype(int)
        misclassified = y_pred != y

        for c in (0, 1):
            mask = (y == c) & (~misclassified)
            ax.scatter(*X[mask].T, s=12, alpha=0.6, label=f"Classe {c}")
        if misclassified.any():
            ax.scatter(
                *X[misclassified].T, s=30, facecolors="none",
                edgecolors="red", linewidths=1.0, label="Mal classificado",
            )

        if abs(w[1]) > 1e-12:
            x1_line = np.array([x1_min, x1_max])
            x2_line = -(w[0] * x1_line + b) / w[1]
            ax.plot(x1_line, x2_line, "k--", linewidth=1.5, label="Fronteira")

        ax.set_xlim(x1_min, x1_max)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        acc = float(np.mean(y_pred == y))
        ax.set_title(f"{title} (acurácia = {acc:.1%})")
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Figura 5 — Fronteiras final e pocket (Exercise 2C)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_accuracy_curves(acc_history, pocket_acc_history, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    epochs = np.arange(1, len(acc_history) + 1)
    ax.plot(epochs, acc_history, label="Acurácia dos pesos atuais", alpha=0.8)
    ax.plot(epochs, pocket_acc_history, label="Melhor até agora (pocket)", linewidth=2)
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia")
    ax.set_title("Figura 6 — Acurácia por época: atual vs. pocket (Exercise 2C)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)

    X, y = generate_data()

    w_init = RNG.normal(0, 0.01, size=2)
    print(f"w_init sorteado = {w_init}")

    plot_scatter(
        X, y, FIGURES / "fig04-overlapping-data.png",
        "Figura 4 — Dados sobrepostos (Exercise 2A)",
    )

    bayes_acc = bayes_optimal_accuracy()
    mean_norm = float(np.mean(np.linalg.norm(X, axis=1)))
    print(f"\nAcuracia da melhor reta possivel (Bayes, covariancias iguais) = {bayes_acc:.4f}")
    print(f"||x|| medio no dataset = {mean_norm:.4f}")

    model = Perceptron(eta=ETA, w_init=w_init, max_epochs=MAX_EPOCHS)
    hist = model.fit(X, y, track_pocket=True)

    print("\n=== Exercise 2B — pesos finais vs. pocket ===")
    print(f"w final  = {hist['w']}, b final  = {hist['b']:.6f}, acc final  = {hist['acc_final']:.4f}")
    print(f"w pocket = {hist['pocket_w']}, b pocket = {hist['pocket_b']:.6f}, acc pocket = {hist['pocket_acc']:.4f}")
    print(f"epoca do recorde pocket = {hist['pocket_epoch']}")
    print(f"epochs rodados = {hist['epochs']} (atingiu o teto de {MAX_EPOCHS}: {hist['epochs'] == MAX_EPOCHS})")
    print(f"||w final|| = {np.linalg.norm(hist['w']):.6f}, |b final| = {abs(hist['b']):.6f}")
    print(f"passo de w por update: eta*||x|| ~ {ETA * mean_norm:.4f}  |  passo de b por update: eta = {ETA:.4f}")

    plot_two_boundaries(
        X, y, hist["w"], hist["b"], hist["pocket_w"], hist["pocket_b"],
        FIGURES / "fig05-final-vs-pocket.png",
    )

    plot_accuracy_curves(
        hist["acc_history"], hist["pocket_acc_history"],
        FIGURES / "fig06-accuracy-final-vs-pocket.png",
    )


if __name__ == "__main__":
    main()
