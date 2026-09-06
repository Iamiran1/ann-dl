"""Exercise 3 — Preparing Real-World Data for a Neural Network."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ========================================================
# Paths
# ========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET = BASE_DIR / "train.csv"
FIGURES = BASE_DIR / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)


# ========================================================
# Columns
# ========================================================

SPENDING_COLUMNS = [
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
]

NUMERICAL_COLUMNS = [
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
]

CATEGORICAL_COLUMNS = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
]

DROP_COLUMNS = [
    "Cabin",
    "Name",
    "PassengerId",
]


def main():

    # ========================================================
    # Exercise 3A — Get to know the data
    # ========================================================

    df = pd.read_csv(DATASET)

    print("\nCOLUMN NAMES")
    print(df.columns.tolist())

    # --------------------------------------------------------
    # Class balance
    # --------------------------------------------------------

    print("\nCLASS BALANCE")
    print(df["Transported"].value_counts())

    print("\nCLASS BALANCE (%)")
    print(df["Transported"].value_counts(normalize=True).mul(100).round(2))

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_table = pd.DataFrame(
        {
            "Missing Count": df.isna().sum(),
            "Missing Percentage": (df.isna().mean() * 100).round(2),
        }
    )

    print("\nMISSING VALUES")
    print(missing_table)

    # --------------------------------------------------------
    # Spending statistics
    # --------------------------------------------------------

    spending_stats = df[SPENDING_COLUMNS].agg(
        [
            "mean",
            "median",
            "max",
        ]
    )

    print("\nSPENDING STATISTICS")
    print(spending_stats)

    # ========================================================
    # Exercise 3B — Split before transforming
    # ========================================================

    X = df.drop(columns=["Transported"])
    y = df["Transported"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nTRAIN / TEST")
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)

    print("\nTrain balance:")
    print(y_train.value_counts(normalize=True))

    print("\nTest balance:")
    print(y_test.value_counts(normalize=True))

    # ========================================================
    # Exercise 3C — Preprocess
    # ========================================================

    # Trabalhamos em cópias
    X_train = X_train.copy()
    X_test = X_test.copy()

    # --------------------------------------------------------
    # 1. Drop unused columns
    # --------------------------------------------------------

    X_train = X_train.drop(columns=DROP_COLUMNS)
    X_test = X_test.drop(columns=DROP_COLUMNS)

    # --------------------------------------------------------
    # 2. Missing numerical data
    #
    # Mediana:
    # adequada para distribuições com forte assimetria
    # e valores extremos.
    # --------------------------------------------------------

    numerical_imputer = SimpleImputer(strategy="median")

    X_train[NUMERICAL_COLUMNS] = numerical_imputer.fit_transform(
        X_train[NUMERICAL_COLUMNS]
    )

    X_test[NUMERICAL_COLUMNS] = numerical_imputer.transform(X_test[NUMERICAL_COLUMNS])

    # --------------------------------------------------------
    # 3. Missing categorical data
    #
    # Usa a categoria mais frequente do conjunto de treino.
    # --------------------------------------------------------

    categorical_imputer = SimpleImputer(strategy="most_frequent")

    X_train[CATEGORICAL_COLUMNS] = categorical_imputer.fit_transform(
        X_train[CATEGORICAL_COLUMNS]
    )

    X_test[CATEGORICAL_COLUMNS] = categorical_imputer.transform(
        X_test[CATEGORICAL_COLUMNS]
    )

    # --------------------------------------------------------
    # 4. Feature engineering — TotalSpend
    # --------------------------------------------------------

    X_train["TotalSpend"] = X_train[SPENDING_COLUMNS].sum(axis=1)

    X_test["TotalSpend"] = X_test[SPENDING_COLUMNS].sum(axis=1)

    # --------------------------------------------------------
    # 5. Save FoodCourt before log transformation
    #
    # Só usamos o TRAIN para analisar a transformação.
    # --------------------------------------------------------

    foodcourt_before = X_train["FoodCourt"].copy()

    # --------------------------------------------------------
    # 6. Heavy tails — log(1 + x)
    #
    # TotalSpend entra junto: é a soma das cinco colunas de
    # gasto e herda a mesma cauda longa. Sem o log, ela chega
    # ao StandardScaler com máximo em torno de 12 desvios.
    # --------------------------------------------------------

    log_columns = SPENDING_COLUMNS + ["TotalSpend"]

    X_train[log_columns] = np.log1p(X_train[log_columns])

    X_test[log_columns] = np.log1p(X_test[log_columns])

    foodcourt_after = X_train["FoodCourt"].copy()

    # --------------------------------------------------------
    # Figure 6 — FoodCourt before / after log
    # --------------------------------------------------------

    fig6, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    axes[0].hist(
        foodcourt_before,
        bins=40,
        alpha=0.75,
    )

    axes[0].set_title("FoodCourt — Before log transformation")

    axes[0].set_xlabel("FoodCourt")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(
        foodcourt_after,
        bins=40,
        alpha=0.75,
    )

    axes[1].set_title("FoodCourt — After log(1 + x)")

    axes[1].set_xlabel("log(1 + FoodCourt)")
    axes[1].set_ylabel("Frequency")

    fig6.tight_layout()

    fig6.savefig(
        FIGURES / "fig07-foodcourt-log.png",
        dpi=150,
    )

    plt.close(fig6)

    # --------------------------------------------------------
    # 7. One-Hot Encoding
    # --------------------------------------------------------

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    train_encoded = encoder.fit_transform(X_train[CATEGORICAL_COLUMNS])

    test_encoded = encoder.transform(X_test[CATEGORICAL_COLUMNS])

    encoded_columns = encoder.get_feature_names_out(CATEGORICAL_COLUMNS)

    train_encoded = pd.DataFrame(
        train_encoded,
        columns=encoded_columns,
        index=X_train.index,
    )

    test_encoded = pd.DataFrame(
        test_encoded,
        columns=encoded_columns,
        index=X_test.index,
    )

    # Remove as colunas categóricas originais
    X_train = X_train.drop(columns=CATEGORICAL_COLUMNS)

    X_test = X_test.drop(columns=CATEGORICAL_COLUMNS)

    # Adiciona as versões one-hot
    X_train = pd.concat(
        [X_train, train_encoded],
        axis=1,
    )

    X_test = pd.concat(
        [X_test, test_encoded],
        axis=1,
    )

    # ========================================================
    # 8. Scaling
    # ========================================================

    numerical_to_scale = [
        "Age",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "TotalSpend",
    ]

    scaler = StandardScaler()

    X_train[numerical_to_scale] = scaler.fit_transform(X_train[numerical_to_scale])

    X_test[numerical_to_scale] = scaler.transform(X_test[numerical_to_scale])

    # ========================================================
    # Final checks
    # ========================================================

    print("\nFINAL CHECKS")
    print("--------------------------------")

    print(
        "NaN train:",
        X_train.isna().sum().sum(),
    )

    print(
        "NaN test:",
        X_test.isna().sum().sum(),
    )

    print(
        "Train shape:",
        X_train.shape,
    )

    print(
        "Test shape:",
        X_test.shape,
    )

    print(
        "Train min:",
        X_train.to_numpy().min(),
    )

    print(
        "Train max:",
        X_train.to_numpy().max(),
    )

    print(
        "Test min:",
        X_test.to_numpy().min(),
    )

    print(
        "Test max:",
        X_test.to_numpy().max(),
    )


if __name__ == "__main__":
    main()
