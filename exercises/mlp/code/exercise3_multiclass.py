"""Exercise 3 — Multiclasse com a mesma rede (mlp.py sem alteração)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

from mlp import MLP

FIGURES = Path(__file__).resolve().parents[1] / "figures"

EPOCHS = 1000
LR = 0.5


def load_data():
    # 1500 amostras, 4 features informativas, 3 classes com 2 clusters cada
    X, y = make_classification(n_samples=1500, n_features=4, n_informative=4,
                               n_redundant=0, n_repeated=0, n_classes=3,
                               n_clusters_per_class=2, class_sep=1.0, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # A — dados
    X_train, X_test, y_train, y_test = load_data()
    Y_train = np.eye(3)[y_train]  # one-hot

    # C — treino e avaliação
    # mesma classe MLP do Exercise 2; só mudam os argumentos (3 saídas e softmax)
    model = MLP([4, 16, 3], output="softmax", rng=rng)
    losses = model.fit(X_train, Y_train, epochs=EPOCHS, lr=LR)
    print(f"MLP: perda final = {losses[-1]:.4f}")
    print(f"MLP: acurácia treino = {np.mean(model.predict(X_train) == y_train):.4f}, "
          f"teste = {np.mean(model.predict(X_test) == y_test):.4f}")

    plt.figure(figsize=(7, 4))
    plt.plot(losses, label="Perda de treino")
    plt.title("Figura 4 — Perda de treino por época (Exercise 3)")
    plt.xlabel("Época")
    plt.ylabel("Categorical cross-entropy")
    plt.legend()
    plt.savefig(FIGURES / "fig04-loss-ex3.png", dpi=150)
    plt.close()

    cm = confusion_matrix(y_test, model.predict(X_test))
    print("matriz de confusão (linhas = classe verdadeira):")
    print(cm)
    ConfusionMatrixDisplay(cm).plot(cmap="Blues", colorbar=False)
    plt.title("Figura 5 — Matriz de confusão (teste)")
    plt.xlabel("Classe prevista")
    plt.ylabel("Classe verdadeira")
    plt.savefig(FIGURES / "fig05-confusion-ex3.png", dpi=150)
    plt.close()

    # D — baseline linear
    baseline = LogisticRegression().fit(X_train, y_train)
    print(f"baseline: acurácia de teste = {baseline.score(X_test, y_test):.4f}")
