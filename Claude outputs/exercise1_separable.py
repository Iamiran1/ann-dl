"""Exercise 1 — Perceptron on separable data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from perceptron import Perceptron

np.set_printoptions(precision=6, suppress=True)

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# Mesmo RNG durante todo o script — nunca recriado
RNG = np.random.default_rng(42)

MEAN_0 = np.array([1.5, 1.5])
COV_0 = np.array([[0.5, 0.0], [0.0, 0.5]])
MEAN_1 = np.array([5.0, 5.0])
COV_1 = np.array([[0.5, 0.0], [0.0, 0.5]])

N_PER_CLASS = 1000
ETA = 0.01
ETA_FAST = 1.0
MAX_EPOCHS = 100


def generate_data():
    class0 = RNG.multivariate_normal(MEAN_0, COV_0, size=N_PER_CLASS)
    class1 = RNG.multivariate_normal(MEAN_1, COV_1, size=N_PER_CLASS)
    X = np.vstack([class0, class1])
    y = np.concatenate([np.zeros(N_PER_CLASS), np.ones(N_PER_CLASS)]).astype(int)
    return X, y


def plot_scatter(X, y, path, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(*X[y == 0].T, s=14, alpha=0.75, label="Classe 0")
    ax.scatter(*X[y == 1].T, s=14, alpha=0.75, label="Classe 1")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_boundary(X, y, w, b, path, title):
    y_pred = (X @ w + b >= 0).astype(int)
    misclassified = y_pred != y

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in (0, 1):
        mask = (y == c) & (~misclassified)
        ax.scatter(*X[mask].T, s=14, alpha=0.75, label=f"Classe {c}")
    if misclassified.any():
        ax.scatter(
            *X[misclassified].T, s=45, facecolors="none",
            edgecolors="red", linewidths=1.4, label="Mal classificado",
        )

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    if abs(w[1]) > 1e-12:
        x1_line = np.array([x1_min, x1_max])
        x2_line = -(w[0] * x1_line + b) / w[1]
        ax.plot(x1_line, x2_line, "k--", linewidth=1.5, label="Fronteira de decisão")

    ax.set_xlim(x1_min, x1_max)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_accuracy(acc_history, path, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    epochs = np.arange(1, len(acc_history) + 1)
    ax.plot(epochs, acc_history, marker="o", markersize=3)
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)

    X, y = generate_data()

    # Mesma inicialização não nula, compartilhada pelas duas rodadas do item D
    w_init = RNG.normal(0, 0.01, size=2)
    print(f"w_init sorteado = {w_init}  (||w_init|| = {np.linalg.norm(w_init):.6f})")

    # ---- Figure 1 ----
    plot_scatter(
        X, y, FIGURES / "fig01-separable-data.png",
        "Figura 1 — Dados linearmente separáveis (Exercise 1A)",
    )

    # ---- C: treino com eta = 0.01 ----
    model = Perceptron(eta=ETA, w_init=w_init, max_epochs=MAX_EPOCHS)
    hist = model.fit(X, y)

    print("\n=== Exercise 1C — eta = 0.01 ===")
    print(f"w final = {hist['w']}")
    print(f"b final = {hist['b']:.6f}")
    print(f"epochs = {hist['epochs']}")
    print(f"accuracy final = {hist['acc_final']:.4f}")
    print(f"updates por epoca (primeiras 10) = {hist['updates_per_epoch'][:10]}")
    print(f"updates por epoca (ultimas 5) = {hist['updates_per_epoch'][-5:]}")

    # ---- Figure 2 ----
    plot_boundary(
        X, y, hist["w"], hist["b"], FIGURES / "fig02-decision-boundary.png",
        "Figura 2 — Fronteira de decisão, eta = 0.01 (Exercise 1C)",
    )

    # ---- Figure 3 ----
    plot_accuracy(
        hist["acc_history"], FIGURES / "fig03-accuracy-curve.png",
        "Figura 3 — Acurácia por época, eta = 0.01 (Exercise 1C)",
    )

    # ---- D2: mesma execução com eta = 1.0, mesmo w_init ----
    model_fast = Perceptron(eta=ETA_FAST, w_init=w_init, max_epochs=MAX_EPOCHS)
    hist_fast = model_fast.fit(X, y)

    cos_theta = float(
        np.dot(hist["w"], hist_fast["w"])
        / (np.linalg.norm(hist["w"]) * np.linalg.norm(hist_fast["w"]))
    )
    angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    print("\n=== Exercise 1D2 — eta = 1.0, mesmo w_init ===")
    print(f"w final = {hist_fast['w']}")
    print(f"b final = {hist_fast['b']:.6f}")
    print(f"epochs = {hist_fast['epochs']}")
    print(f"accuracy final = {hist_fast['acc_final']:.4f}")
    print(f"cos(angulo entre w_0.01 e w_1.0) = {cos_theta:.6f}")
    print(f"angulo entre as direcoes = {angle_deg:.3f} graus")

    # ---- D3: demonstração empírica do argumento w0 = 0, b0 = 0 ----
    zero_init = np.zeros(2)
    hist_zero_slow = Perceptron(eta=ETA, w_init=zero_init, max_epochs=MAX_EPOCHS).fit(X, y)
    hist_zero_fast = Perceptron(eta=ETA_FAST, w_init=zero_init, max_epochs=MAX_EPOCHS).fit(X, y)

    ratio_w = hist_zero_fast["w"] / hist_zero_slow["w"]
    ratio_b = hist_zero_fast["b"] / hist_zero_slow["b"] if hist_zero_slow["b"] != 0 else float("nan")

    print("\n=== Exercise 1D3 — a partir de w0 = 0, b0 = 0 ===")
    print(f"razao nominal eta_1.0 / eta_0.01 = {ETA_FAST / ETA:.1f}")
    print(f"epochs(eta=0.01) = {hist_zero_slow['epochs']}, epochs(eta=1.0) = {hist_zero_fast['epochs']}")
    print(f"w(eta=0.01) = {hist_zero_slow['w']}")
    print(f"w(eta=1.0)  = {hist_zero_fast['w']}")
    print(f"w(eta=1.0) / w(eta=0.01) elemento a elemento = {ratio_w}")
    print(f"b(eta=1.0) / b(eta=0.01) = {ratio_b:.6f}")
    print(f"acuracia final identica? {hist_zero_slow['acc_final'] == hist_zero_fast['acc_final']}")


if __name__ == "__main__":
    main()
