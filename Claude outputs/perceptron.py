"""Single-layer perceptron, written from scratch.

Reused, unchanged, by Exercise 1 (separable data) and Exercise 2 (overlapping
data). The only thing Exercise 2 adds is the pocket bookkeeping, controlled
by the ``track_pocket`` flag — the prediction rule, the update rule and the
stopping condition below are never touched.
"""

import numpy as np


class Perceptron:
    """Binary perceptron with labels in {0, 1}.

    Prediction
        y_hat = 1 if w . x + b >= 0 else 0

    Update (applied only when the prediction is wrong)
        error = y - y_hat            # 0 on a correct prediction, else +-1
        w    += eta * error * x
        b    += eta * error

    Stopping
        a full epoch with zero updates, or ``max_epochs`` epochs, whichever
        comes first.
    """

    def __init__(self, eta: float, w_init: np.ndarray, b_init: float = 0.0,
                 max_epochs: int = 100):
        self.eta = eta
        self.w_init = np.asarray(w_init, dtype=float)
        self.b_init = float(b_init)
        self.max_epochs = max_epochs

        self.w = None
        self.b = None

    def net_input(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.net_input(X) >= 0).astype(int)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    def fit(self, X: np.ndarray, y: np.ndarray, track_pocket: bool = False) -> dict:
        """Train and return a history dict.

        ``track_pocket=True`` additionally tracks the best-so-far (pocket)
        weights: every time an update produces a higher full-dataset
        accuracy than any previously seen, the current weights are copied
        into the pocket. This is the only extra bookkeeping added to the
        loop; prediction, update rule and stopping condition are identical
        to the ``track_pocket=False`` case.
        """
        self.w = self.w_init.copy()
        self.b = self.b_init
        n_samples = X.shape[0]

        acc_history = []
        updates_per_epoch = []

        pocket_w = self.w.copy()
        pocket_b = self.b
        pocket_acc = self.accuracy(X, y)
        pocket_epoch = 0
        pocket_acc_history = []

        epochs_run = 0

        for epoch in range(1, self.max_epochs + 1):
            n_updates = 0

            for i in range(n_samples):
                xi = X[i]
                yi = y[i]

                y_hat = 1 if (np.dot(self.w, xi) + self.b) >= 0 else 0
                error = yi - y_hat

                if error != 0:
                    self.w += self.eta * error * xi
                    self.b += self.eta * error
                    n_updates += 1

                    if track_pocket:
                        acc_now = self.accuracy(X, y)
                        if acc_now > pocket_acc:
                            pocket_acc = acc_now
                            pocket_w = self.w.copy()
                            pocket_b = self.b
                            pocket_epoch = epoch

            epochs_run = epoch
            updates_per_epoch.append(n_updates)
            acc_history.append(self.accuracy(X, y))
            if track_pocket:
                pocket_acc_history.append(pocket_acc)

            if n_updates == 0:
                break

        history = {
            "w": self.w.copy(),
            "b": self.b,
            "epochs": epochs_run,
            "acc_final": acc_history[-1],
            "acc_history": acc_history,
            "updates_per_epoch": updates_per_epoch,
        }

        if track_pocket:
            history.update({
                "pocket_w": pocket_w,
                "pocket_b": pocket_b,
                "pocket_acc": pocket_acc,
                "pocket_epoch": pocket_epoch,
                "pocket_acc_history": pocket_acc_history,
            })

        return history
