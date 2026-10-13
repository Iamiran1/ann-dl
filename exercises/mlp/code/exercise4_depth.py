"""Exercise 4 — Uma segunda camada oculta ajuda?

Mesmos dados, épocas e learning rate do Exercise 3; só muda a profundidade.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exercise3_multiclass import EPOCHS, LR, load_data
from mlp import MLP

FIGURES = Path(__file__).resolve().parents[1] / "figures"

SEEDS = [42, 43, 44]
NETWORKS = {
    "1 camada oculta (Ex. 3)": [4, 16, 3],
    "2 camadas ocultas (Ex. 4)": [4, 16, 16, 3],
}

X_train, X_test, y_train, y_test = load_data()
Y_train = np.eye(3)[y_train]

plt.figure(figsize=(7, 4))

for name, sizes in NETWORKS.items():
    test_accs, final_losses, all_losses = [], [], []
    for seed in SEEDS:
        model = MLP(sizes, output="softmax", rng=np.random.default_rng(seed))
        losses = model.fit(X_train, Y_train, epochs=EPOCHS, lr=LR)
        acc = np.mean(model.predict(X_test) == y_test)
        print(f"{name}, seed {seed}: perda final = {losses[-1]:.4f}, teste = {acc:.4f}")
        test_accs.append(acc)
        final_losses.append(losses[-1])
        all_losses.append(losses)

    print(f"{name}: {model.n_parameters()} parâmetros, "
          f"perda final média = {np.mean(final_losses):.4f}, "
          f"teste médio = {np.mean(test_accs):.4f}\n")
    plt.plot(np.mean(all_losses, axis=0), label=f"{name} — média de 3 seeds")

plt.title("Figura 6 — Perda de treino: 1 vs. 2 camadas ocultas")
plt.xlabel("Época")
plt.ylabel("Categorical cross-entropy")
plt.legend()
plt.savefig(FIGURES / "fig06-loss-depth.png", dpi=150)
plt.close()
