"""MLP feita do zero com numpy, usada sem alteração nos Exercises 2, 3 e 4."""

import numpy as np


def sigmoid(u):
    return 1 / (1 + np.exp(-u))


def softmax(u):
    e = np.exp(u - u.max(axis=1, keepdims=True))  # tira o máximo para evitar overflow
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, layer_sizes, output, rng):
        # layer_sizes = [entradas, oculta_1, ..., oculta_k, saidas]
        # output = "sigmoid" (binário, BCE) ou "softmax" (multiclasse, CE)
        self.output = output
        self.W = []
        self.b = []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.W.append(rng.normal(0, 0.5, size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))

    def n_parameters(self):
        return sum(W.size + b.size for W, b in zip(self.W, self.b))

    def forward(self, X):
        self.h = [X]  # guarda a saída de cada camada para o backward
        for i in range(len(self.W)):
            u = self.h[-1] @ self.W[i] + self.b[i]
            if i < len(self.W) - 1:
                self.h.append(np.tanh(u))
            elif self.output == "sigmoid":
                self.h.append(sigmoid(u))
            else:
                self.h.append(softmax(u))
        return self.h[-1]

    def loss(self, y_hat, y):
        y_hat = np.clip(y_hat, 1e-12, 1 - 1e-12)  # evita log(0)
        if self.output == "sigmoid":
            return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return -np.mean(np.sum(y * np.log(y_hat), axis=1))

    def backward(self, y):
        # sigmoid + BCE e softmax + CE dão o mesmo gradiente na saída: y_hat - y
        delta = (self.h[-1] - y) / len(y)
        grads_W = []
        grads_b = []
        for i in reversed(range(len(self.W))):  # da última camada para a primeira
            grads_W.insert(0, self.h[i].T @ delta)  # dL/dW = entrada da camada x delta
            grads_b.insert(0, delta.sum(axis=0))    # dL/db = soma de delta no batch
            if i > 0:
                delta = (delta @ self.W[i].T) * (1 - self.h[i] ** 2)  # derivada da tanh
        return grads_W, grads_b

    def update(self, grads_W, grads_b, lr):
        # descida do gradiente: theta <- theta - lr * dL/dtheta
        for i in range(len(self.W)):
            self.W[i] -= lr * grads_W[i]
            self.b[i] -= lr * grads_b[i]

    def fit(self, X, y, epochs, lr):
        # gradiente descendente usando o conjunto de treino inteiro em cada época
        losses = []
        for _ in range(epochs):
            y_hat = self.forward(X)
            losses.append(self.loss(y_hat, y))
            grads_W, grads_b = self.backward(y)
            self.update(grads_W, grads_b, lr)
        return losses

    def predict(self, X):
        y_hat = self.forward(X)
        if self.output == "sigmoid":
            return (y_hat[:, 0] >= 0.5).astype(int)
        return np.argmax(y_hat, axis=1)
