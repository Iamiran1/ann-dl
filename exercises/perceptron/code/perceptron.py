"""Perceptron implementation, escrita do zero e reutilizada sem alteração no
Exercise 2 (só o `track_pocket` do `fit` é ligado lá -- a única coisa que o
enunciado permite adicionar ao laço de treino)."""

import numpy as np


class Perceptron:
    def __init__(self, eta, w_init, b_init=0.0, max_epochs=100):
        self.eta = eta  # learning rate
        self.w = w_init.copy()
        self.b = b_init
        self.max_epochs = max_epochs

    def predict(self, X):
        y_pred = []
        for x0, x1 in X:
            z = self.w[0]*x0 + self.w[1]*x1 + self.b
            if (z >= 0):
                z = 1
            else:
                z = 0
            y_pred.append(z)
        return y_pred

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        right = 0
        for y_p, y_r in zip(y_pred, y):
            if (y_p == y_r):
                right += 1
        return right / len(y_pred)

    def fit(self, X, y, track_pocket=False):
        acc_history = []
        updates_per_epoch = []

        # pocket: melhor conjunto de pesos ja visto, comparado por acuracia
        pocket_w = self.w.copy()
        pocket_b = self.b
        pocket_acc = self.accuracy(X, y)
        pocket_epoch = 0
        pocket_acc_history = []

        epochs_run = 0

        for epoch in range(1, self.max_epochs + 1):
            n_updates = 0

            for i in range(len(X)):
                xi = X[i]
                yi = y[i]

                z = self.w[0]*xi[0] + self.w[1]*xi[1] + self.b
                y_hat = 1 if z >= 0 else 0
                error = yi - y_hat

                if error != 0:
                    self.w[0] = self.w[0] + self.eta*error*xi[0]
                    self.w[1] = self.w[1] + self.eta*error*xi[1]
                    self.b = self.b + self.eta*error
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
