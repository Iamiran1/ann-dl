"""Exercise 1 — Exploring class separability in 2D."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIGURES = Path(__file__).resolve().parents[1] / "figures"

# Mesmo RNG durante toda a atividade
RNG = np.random.default_rng(42)


CLASSES = {
    0: {"mean": [2.0, 3.0], "std": [0.8, 2.5]},
    1: {"mean": [5.0, 6.0], "std": [1.2, 1.9]},
    2: {"mean": [8.0, 1.0], "std": [0.9, 0.9]},
    3: {"mean": [15.0, 4.0], "std": [0.5, 2.0]},
}

N_PER_CLASS = 100



def generate(scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Gera 100 pontos por classe."""

    xs = []
    ys = []

    for label, params in CLASSES.items():

        mean = np.asarray(params["mean"])

        std = np.asarray(params["std"]) * scale

        xs.append(
            RNG.normal(
                mean,
                std,
                size=(N_PER_CLASS, 2)
            )
        )

        ys.append(
            np.full(N_PER_CLASS, label)
        )

    return np.vstack(xs), np.concatenate(ys)




def separation_ratios(scale: float = 1.0):

    ratios = {}

    for c1 in CLASSES:
        for c2 in CLASSES:

            # Evita pares repetidos:
            # (0,1) existe, então não precisamos de (1,0)
            if c1 < c2:

                mean1 = np.array(CLASSES[c1]["mean"])
                mean2 = np.array(CLASSES[c2]["mean"])

                std1 = np.array(CLASSES[c1]["std"]) * scale
                std2 = np.array(CLASSES[c2]["std"]) * scale

                # Distância euclidiana entre os centros
                distancia = np.linalg.norm(mean1 - mean2)

                # sigma médio de cada classe
                std_medio1 = np.mean(std1)
                std_medio2 = np.mean(std2)

                # separation ratio
                rij = distancia / (std_medio1 + std_medio2)

                ratios[(c1, c2)] = rij

    return ratios


def mixing_rate(X, y):

    centers = np.array([
        CLASSES[c]["mean"]
        for c in CLASSES
    ])

    erros = 0

    for ponto, classe_real in zip(X, y):

        # distância do ponto para cada um dos 4 centros
        distancias = np.linalg.norm(
            centers - ponto,
            axis=1
        )

        # índice do centro mais próximo
        centro_mais_proximo = np.argmin(distancias)

        if centro_mais_proximo != classe_real:
            erros += 1

    return erros / len(X)



def main():

    FIGURES.mkdir(parents=True, exist_ok=True)

    scales = [0.5, 1.0, 2.0, 4.0]


    # ========================================================
    # Exercise 1A
    # ========================================================

    X, y = generate(scale=1.0)


    # ========================================================
    # Gera datasets do 1B
    # ========================================================

    datasets = {}

    for scale in scales:

        # Para s = 1 usamos exatamente o dataset do exercício A
        if scale == 1.0:
            datasets[scale] = (X, y)

        else:
            datasets[scale] = generate(scale)





    # ========================================================
    # FIGURE 1
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    for c in CLASSES:

        points = ax.scatter(
            *X[y == c].T,
            s=14,
            alpha=0.75,
            label=f"Classe {c}"
        )

        # mesma cor dos pontos
        color = points.get_facecolor()[0]

        mean = CLASSES[c]["mean"]

        # centro da classe
        ax.scatter(
            mean[0],
            mean[1],
            marker="X",
            s=150,
            edgecolors="black",
            color=color
        )


    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    ax.set_title(
        "Nuvens de pontos gaussianas (scale = 1.0)"
    )

    ax.legend(loc="upper left")

    fig.tight_layout()

    fig.savefig(
        FIGURES / "fig01-point-clouds.png",
        dpi=150
    )

    plt.close(fig)


    # ========================================================
    # Exercise 1B — FIGURE 2
    # ========================================================

    fig2, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        sharex=True,
        sharey=True
    )

    for ax, scale in zip(axes.flatten(), scales):
        Xs, ys = datasets[scale]
        for c in CLASSES:
            points = ax.scatter(
                *Xs[ys == c].T,
                s=14,
                alpha=0.75,
                label=f"Classe {c}"
            )

            color = points.get_facecolor()[0]

            mean = CLASSES[c]["mean"]

            ax.scatter(
                mean[0],
                mean[1],
                marker="X",
                s=120,
                edgecolors="black",
                color=color
            )

        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

        ax.set_title(
            f"scale = {scale}"
        )

        ax.legend(loc="upper left")


    fig2.suptitle(
        "Figure 2 — Effect of scale on class spread"
    )

    fig2.tight_layout()

    fig2.savefig(
        FIGURES / "fig02-point-clouds-scales.png",
        dpi=150
    )

    plt.close(fig2)


    # ========================================================
    # Separation ratios — s = 1
    # ========================================================

    ratios = separation_ratios(scale=1.0)

    print("\nSEPARATION RATIOS — s = 1")
    print("--------------------------------")

    for pair, value in ratios.items():

        print(
            f"Classes {pair[0]}-{pair[1]}: "
            f"r_ij = {value:.3f}"
        )


    # ========================================================
    # Menor separation ratio
    # ========================================================

    menor_par = min(
        ratios,
        key=ratios.get
    )

    menor_rij = ratios[menor_par]

    print("\nMenor separation ratio:")

    print(
        f"Classes {menor_par[0]}-{menor_par[1]} "
        f"-> r_ij = {menor_rij:.3f}"
    )


    # ========================================================
    # Como r_ij varia com 1/s:
    # para s=2 basta dividir por 2
    # ========================================================

    menor_rij_s2 = menor_rij / 2

    print(
        f"Mesmo par em s=2: "
        f"r_ij = {menor_rij_s2:.3f}"
    )


    # ========================================================
    # Mixing rates
    # ========================================================

    mixing_rates = {}

    print("\nMIXING RATES")
    print("--------------------------------")

    for scale in scales:

        Xs, ys = datasets[scale]

        rate = mixing_rate(
            Xs,
            ys
        )

        mixing_rates[scale] = rate

        print(
            f"scale={scale}: "
            f"mixing rate={rate:.3f} "
            f"({rate * 100:.1f}%)"
        )


    # ========================================================
    # FIGURE 3 — Mixing rate x scale
    # ========================================================

    fig3, ax3 = plt.subplots(
        figsize=(7, 5)
    )

    rates = [
        mixing_rates[s]
        for s in scales
    ]

    ax3.plot(
        scales,
        rates,
        marker="o",
        label="Mixing rate"
    )

    ax3.set_xlabel(
        "Scale factor $s$"
    )

    ax3.set_ylabel(
        "Mixing rate"
    )

    ax3.set_title(
        "Figure 3 — Mixing rate vs. scale"
    )

    ax3.set_xticks(scales)

    ax3.legend()

    ax3.grid(
        alpha=0.3
    )

    fig3.tight_layout()

    fig3.savefig(
        FIGURES / "fig03-mixing-rate.png",
        dpi=150
    )

    plt.close(fig3)


if __name__ == "__main__":
    main()