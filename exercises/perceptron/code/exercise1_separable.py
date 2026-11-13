"""Exercise 1 — Exploring class separability in 2D."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from perceptron import Perceptron

np.set_printoptions(precision=6, suppress=True)

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Mesmo RNG durante todo o script — criado uma unica vez
rng = np.random.default_rng(42)

ETA = 0.01
ETA_FAST = 1.0
MAX_EPOCHS = 100

# ============================================================
# A — Generate the data
# ============================================================

classA = rng.multivariate_normal([1.5, 1.5], [[0.5, 0.0], [0.0, 0.5]], size=1000)
classB = rng.multivariate_normal([5, 5], [[0.5, 0.0], [0.0, 0.5]], size=1000)

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(classA[:, 0], classA[:, 1], label="Classe 0")
ax.scatter(classB[:, 0], classB[:, 1], label="Classe 1")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_title("Figura 1 — Dados linearmente separáveis")
ax.legend()
fig.savefig(FIGURES / "fig01-separable-data.png", dpi=150)
plt.close(fig)

X = np.vstack([classA, classB])
y = np.concatenate([np.zeros(1000), np.ones(1000)]).astype(int)

# ============================================================
# B/C — treinar o perceptron, eta = 0.01
# ============================================================

# w_init sorteado UMA vez e reaproveitado nas duas rodadas do item D
w_init = rng.normal(0, 0.01, size=2)
print(f"w_init sorteado = {w_init}")

perceptron = Perceptron(eta=ETA, w_init=w_init, b_init=0.0, max_epochs=MAX_EPOCHS)
hist = perceptron.fit(X, y)

print("\n=== Exercise 1C — eta = 0.01 ===")
print(f"w final = [{perceptron.w[0]:.6f}, {perceptron.w[1]:.6f}]")
print(f"b final = {perceptron.b:.6f}")
print(f"epochs = {hist['epochs']}")
print(f"accuracy final = {hist['acc_final']:.4f}")
print(f"updates por epoca (primeiras 10) = {hist['updates_per_epoch'][:10]}")
print(f"updates por epoca (ultimas 5) = {hist['updates_per_epoch'][-5:]}")

# ---- Figure 2: fronteira de decisao ----
y_pred = np.array(perceptron.predict(X))
misclassified = y_pred != y

fig, ax = plt.subplots(figsize=(7, 5))
for c in (0, 1):
    mask = (y == c) & (~misclassified)
    ax.scatter(*X[mask].T, s=14, alpha=0.75, label=f"Classe {c}")
if misclassified.any():
    ax.scatter(*X[misclassified].T, s=45, facecolors="none",
               edgecolors="red", linewidths=1.4, label="Mal classificado")

w0, w1, b = perceptron.w[0], perceptron.w[1], perceptron.b
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
if abs(w1) > 1e-12:
    x1_line = np.array([x1_min, x1_max])
    x2_line = -(w0*x1_line + b) / w1
    ax.plot(x1_line, x2_line, "k--", linewidth=1.5, label="Fronteira de decisão")
ax.set_xlim(x1_min, x1_max)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_title("Figura 2 — Fronteira de decisão, eta = 0.01")
ax.legend(loc="best")
fig.savefig(FIGURES / "fig02-decision-boundary.png", dpi=150)
plt.close(fig)

# ---- Figure 3: acuracia por epoca ----
fig, ax = plt.subplots(figsize=(7, 5))
epochs_axis = np.arange(1, len(hist["acc_history"]) + 1)
ax.plot(epochs_axis, hist["acc_history"], marker="o", markersize=3)
ax.set_xlabel("Época")
ax.set_ylabel("Acurácia")
ax.set_title("Figura 3 — Acurácia por época, eta = 0.01")
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3)
fig.savefig(FIGURES / "fig03-accuracy-curve.png", dpi=150)
plt.close(fig)

# ============================================================
# D2 — mesma rodada com eta = 1.0, mesmo w_init
# ============================================================

perceptron_fast = Perceptron(eta=ETA_FAST, w_init=w_init, b_init=0.0, max_epochs=MAX_EPOCHS)
hist_fast = perceptron_fast.fit(X, y)

w_slow = np.array(perceptron.w, dtype=float)
w_fast = np.array(perceptron_fast.w, dtype=float)
cos_theta = float(np.dot(w_slow, w_fast) / (np.linalg.norm(w_slow) * np.linalg.norm(w_fast)))
angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

print("\n=== Exercise 1D2 — eta = 1.0, mesmo w_init ===")
print(f"w final = [{perceptron_fast.w[0]:.6f}, {perceptron_fast.w[1]:.6f}]")
print(f"b final = {perceptron_fast.b:.6f}")
print(f"epochs = {hist_fast['epochs']}")
print(f"accuracy final = {hist_fast['acc_final']:.4f}")
print(f"cos(angulo entre w_0.01 e w_1.0) = {cos_theta:.6f}")
print(f"angulo entre as direcoes = {angle_deg:.3f} graus")

# ============================================================
# D3 — demonstracao empirica: w0 = 0, b0 = 0
# ============================================================

zero_init = np.zeros(2)
p_zero_slow = Perceptron(eta=ETA, w_init=zero_init, b_init=0.0, max_epochs=MAX_EPOCHS)
hist_zero_slow = p_zero_slow.fit(X, y)

p_zero_fast = Perceptron(eta=ETA_FAST, w_init=zero_init, b_init=0.0, max_epochs=MAX_EPOCHS)
hist_zero_fast = p_zero_fast.fit(X, y)

w_zero_slow = np.array(p_zero_slow.w, dtype=float)
w_zero_fast = np.array(p_zero_fast.w, dtype=float)
ratio_w = w_zero_fast / w_zero_slow
ratio_b = p_zero_fast.b / p_zero_slow.b if p_zero_slow.b != 0 else float("nan")

print("\n=== Exercise 1D3 — a partir de w0 = 0, b0 = 0 ===")
print(f"razao nominal eta_1.0 / eta_0.01 = {ETA_FAST/ETA:.1f}")
print(f"epochs(eta=0.01) = {hist_zero_slow['epochs']}, epochs(eta=1.0) = {hist_zero_fast['epochs']}")
print(f"w(eta=0.01) = {w_zero_slow}")
print(f"w(eta=1.0)  = {w_zero_fast}")
print(f"w(eta=1.0) / w(eta=0.01) elemento a elemento = {ratio_w}")
print(f"b(eta=1.0) / b(eta=0.01) = {ratio_b:.6f}")
print(f"acuracia final identica? {hist_zero_slow['acc_final'] == hist_zero_fast['acc_final']}")
