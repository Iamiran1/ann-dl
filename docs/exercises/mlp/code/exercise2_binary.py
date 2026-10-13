"""Exercise 2 — Classificação binária que uma reta não resolve."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mlp import MLP

FIGURES = Path(__file__).resolve().parents[1] / "figures"

rng = np.random.default_rng(42)

# A — dados: 3 blobs; c diz de qual blob veio cada ponto
X, c = make_blobs(n_samples=[500, 250, 250], centers=[[0, 0], [3, 3], [-3, -3]],
                  cluster_std=1.2, random_state=42)
y = (c > 0).astype(int)  # classe 0 = blob central, classe 1 = os dois blobs das pontas

# 80% treino / 20% teste, mantendo a proporção das classes (stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

plt.figure(figsize=(6, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], s=10, label="Classe 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1], s=10, label="Classe 1")
plt.title("Figura 1 — Dados do Exercise 2")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend()
plt.savefig(FIGURES / "fig01-data.png", dpi=150)
plt.close()

# B — baseline linear
baseline = LogisticRegression().fit(X_train, y_train)
print(f"baseline: acurácia de teste = {baseline.score(X_test, y_test):.4f}")

# C — MLP: 2 entradas, 8 neurônios ocultos, 1 saída sigmoide
model = MLP([2, 8, 1], output="sigmoid", rng=rng)
# y vira coluna (800, 1) para ter o mesmo formato da saída da rede
losses = model.fit(X_train, y_train.reshape(-1, 1), epochs=1000, lr=0.5)
print(f"MLP: perda final = {losses[-1]:.4f}")
print(f"MLP: acurácia treino = {np.mean(model.predict(X_train) == y_train):.4f}, "
      f"teste = {np.mean(model.predict(X_test) == y_test):.4f}")

plt.figure(figsize=(7, 4))
plt.plot(losses, label="Perda de treino")
plt.title("Figura 2 — Perda de treino por época (Exercise 2)")
plt.xlabel("Época")
plt.ylabel("Binary cross-entropy")
plt.legend()
plt.savefig(FIGURES / "fig02-loss-ex2.png", dpi=150)
plt.close()

# fronteira de decisão: avalia a rede em uma grade de pontos
g1, g2 = np.meshgrid(np.linspace(-8, 8, 300), np.linspace(-8, 8, 300))
Z = model.predict(np.column_stack([g1.ravel(), g2.ravel()])).reshape(g1.shape)

plt.figure(figsize=(6, 6))
plt.contourf(g1, g2, Z, levels=[-0.5, 0.5, 1.5], colors=["tab:blue", "tab:orange"], alpha=0.2)
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], s=15, label="Classe 0 (teste)")
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], s=15, label="Classe 1 (teste)")
plt.title("Figura 3 — Fronteira de decisão da MLP")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend()
plt.savefig(FIGURES / "fig03-decision-boundary-ex2.png", dpi=150)
plt.close()
