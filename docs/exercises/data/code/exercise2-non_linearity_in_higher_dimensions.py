"""Exercise 2 — Non-Linearity in Higher Dimensions."""

from pathlib import Path
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np
FIGURES = Path(__file__).resolve().parents[1] / "figures"

def main():

    CLASSES = {
        "A": {
            "mean": [0, 0, 0, 0, 0],
            "cov": [
                [1.0, 0.8, 0.1, 0.0, 0.0],
                [0.8, 1.0, 0.3, 0.0, 0.0],
                [0.1, 0.3, 1.0, 0.5, 0.0],
                [0.0, 0.0, 0.5, 1.0, 0.2],
                [0.0, 0.0, 0.0, 0.2, 1.0],
            ],
        },
        "B": {
            "mean": [1.5, 1.5, 1.5, 1.5, 1.5],
            "cov": [
                [1.5, -0.7, 0.2, 0.0, 0.0],
                [-0.7, 1.5, 0.4, 0.0, 0.0],
                [0.2, 0.4, 1.5, 0.6, 0.0],
                [0.0, 0.0, 0.6, 1.5, 0.3],
                [0.0, 0.0, 0.0, 0.3, 1.5],
            ],
        },
    }

    np.random.seed(42)
    n_samples = 500

    X_A = np.random.multivariate_normal(
        mean=CLASSES["A"]["mean"], cov=CLASSES["A"]["cov"], size=n_samples
    )

    X_B = np.random.multivariate_normal(
        mean=CLASSES["B"]["mean"], cov=CLASSES["B"]["cov"], size=n_samples
    )
    # Classe C
    v_C = np.random.normal(0, 1, size=(n_samples, 5))
    u_C = v_C / np.linalg.norm(v_C, axis=1, keepdims=True)

    rho_C = np.random.normal(
        loc=2.0,
        scale=0.4,
        size=n_samples
    )

    X_C = u_C * rho_C[:, None]


    # Classe D
    v_D = np.random.normal(0, 1, size=(n_samples, 5))
    u_D = v_D / np.linalg.norm(v_D, axis=1, keepdims=True)

    rho_D = np.random.normal(
        loc=5.0,
        scale=0.4,
        size=n_samples
    )

    X_D = u_D * rho_D[:, None]
    
    X_AB = np.vstack([X_A, X_B]) # dataset I
    X_CD = np.vstack([X_C, X_D]) # dataset II
    
    
    
    pca_AB = PCA(n_components=2)
    X_AB_2D = pca_AB.fit_transform(X_AB)

    pca_CD = PCA(n_components=2)
    X_CD_2D = pca_CD.fit_transform(X_CD)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Dataset I — A e B
    axes[0].scatter(
        X_AB_2D[:n_samples, 0],
        X_AB_2D[:n_samples, 1],
        label="Class A",
        alpha=0.6
    )

    axes[0].scatter(
        X_AB_2D[n_samples:, 0],
        X_AB_2D[n_samples:, 1],
        label="Class B",
        alpha=0.6
    )

    axes[0].set_title("Dataset I — Shifted Gaussians")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend()


    # Dataset II — C e D
    axes[1].scatter(
        X_CD_2D[:n_samples, 0],
        X_CD_2D[:n_samples, 1],
        label="Class C",
        alpha=0.6
    )

    axes[1].scatter(
        X_CD_2D[n_samples:, 0],
        X_CD_2D[n_samples:, 1],
        label="Class D",
        alpha=0.6
    )

    axes[1].set_title("Dataset II — Concentric Shells")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend()
    fig.tight_layout()

    fig.savefig(
        FIGURES / "fig05-pca-comparison.png",
        dpi=150
    )

    plt.close(fig)
    # ========================================================
    # Explained variance
    # ========================================================

    print("\nEXPLAINED VARIANCE")
    print("--------------------------------")

    print(
        f"Dataset I — PC1: "
        f"{pca_AB.explained_variance_ratio_[0]:.3f}"
    )

    print(
        f"Dataset I — PC2: "
        f"{pca_AB.explained_variance_ratio_[1]:.3f}"
    )

    print(
        f"Dataset I — Total: "
        f"{pca_AB.explained_variance_ratio_.sum():.3f}"
    )

    print(
        f"Dataset II — PC1: "
        f"{pca_CD.explained_variance_ratio_[0]:.3f}"
    )

    print(
        f"Dataset II — PC2: "
        f"{pca_CD.explained_variance_ratio_[1]:.3f}"
    )

    print(
        f"Dataset II — Total: "
        f"{pca_CD.explained_variance_ratio_.sum():.3f}"
    )


    # ========================================================
    # Distance between class centers in 5D
    # ========================================================

    center_A = np.mean(X_A, axis=0)
    center_B = np.mean(X_B, axis=0)

    center_C = np.mean(X_C, axis=0)
    center_D = np.mean(X_D, axis=0)

    distance_AB = np.linalg.norm(
        center_A - center_B
    )

    distance_CD = np.linalg.norm(
        center_C - center_D
    )

    print("\nDISTANCE BETWEEN CLASS CENTERS")
    print("--------------------------------")

    print(
        f"Dataset I — A/B: {distance_AB:.3f}"
    )

    print(
        f"Dataset II — C/D: {distance_CD:.3f}"
    )


    # ========================================================
    # Figure 5 — Radius histograms
    # ========================================================

    radius_A = np.linalg.norm(X_A, axis=1)
    radius_B = np.linalg.norm(X_B, axis=1)

    radius_C = np.linalg.norm(X_C, axis=1)
    radius_D = np.linalg.norm(X_D, axis=1)


    fig5, axes5 = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    axes5[0].hist(
        radius_A,
        bins=30,
        alpha=0.6,
        label="Class A"
    )

    axes5[0].hist(
        radius_B,
        bins=30,
        alpha=0.6,
        label="Class B"
    )

    axes5[0].set_title(
        "Dataset I — Shifted Gaussians"
    )

    axes5[0].set_xlabel("Radius ||x||")
    axes5[0].set_ylabel("Frequency")
    axes5[0].legend()


    axes5[1].hist(
        radius_C,
        bins=30,
        alpha=0.6,
        label="Class C"
    )

    axes5[1].hist(
        radius_D,
        bins=30,
        alpha=0.6,
        label="Class D"
    )

    axes5[1].set_title(
        "Dataset II — Concentric Shells"
    )

    axes5[1].set_xlabel("Radius ||x||")
    axes5[1].set_ylabel("Frequency")
    axes5[1].legend()


    fig5.tight_layout()

    fig5.savefig(
        FIGURES / "fig06-radius-histograms.png",
        dpi=150
    )

    plt.close(fig5)
if __name__ == "__main__":
    main()
