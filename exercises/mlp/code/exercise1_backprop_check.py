"""Exercise 1 — Conferência das contas feitas à mão (só usa o módulo math)."""

import math

x1, x2, y = 0.5, -0.2, 1.0
w11, w12, w21, w22 = 0.3, -0.1, 0.2, 0.4  # W1
b1_1, b1_2 = 0.1, -0.2                    # b1
v1, v2 = 0.5, -0.3                        # W2
b2 = 0.2
eta = 0.3

# A — forward
z1 = w11 * x1 + w12 * x2 + b1_1
z2 = w21 * x1 + w22 * x2 + b1_2
h1, h2 = math.tanh(z1), math.tanh(z2)
u = v1 * h1 + v2 * h2 + b2
y_hat = math.tanh(u)
L = (y - y_hat) ** 2
print(f"z = [{z1:.4f}, {z2:.4f}]  h = [{h1:.4f}, {h2:.4f}]")
print(f"u = {u:.4f}  y_hat = {y_hat:.4f}  L = {L:.4f}")

# B — backward
dL_dyhat = -2 * (y - y_hat)
dL_du = dL_dyhat * (1 - y_hat ** 2)
dv1, dv2, db2 = dL_du * h1, dL_du * h2, dL_du
dh1, dh2 = dL_du * v1, dL_du * v2
dz1, dz2 = dh1 * (1 - h1 ** 2), dh2 * (1 - h2 ** 2)
dw11, dw12, dw21, dw22 = dz1 * x1, dz1 * x2, dz2 * x1, dz2 * x2
print(f"dL/dy_hat = {dL_dyhat:.4f}  dL/du = {dL_du:.4f}")
print(f"dL/dW2 = [{dv1:.4f}, {dv2:.4f}]  dL/db2 = {db2:.4f}")
print(f"dL/dh = [{dh1:.4f}, {dh2:.4f}]  dL/dz = [{dz1:.4f}, {dz2:.4f}]")
print(f"dL/dW1 = [[{dw11:.4f}, {dw12:.4f}], [{dw21:.4f}, {dw22:.4f}]]")

# C — atualização
w11, w12, w21, w22 = w11 - eta * dw11, w12 - eta * dw12, w21 - eta * dw21, w22 - eta * dw22
b1_1, b1_2 = b1_1 - eta * dz1, b1_2 - eta * dz2
v1, v2, b2 = v1 - eta * dv1, v2 - eta * dv2, b2 - eta * db2
print(f"W1 = [[{w11:.4f}, {w12:.4f}], [{w21:.4f}, {w22:.4f}]]  b1 = [{b1_1:.4f}, {b1_2:.4f}]")
print(f"W2 = [{v1:.4f}, {v2:.4f}]  b2 = {b2:.4f}")

# segundo forward, com os parâmetros novos
h1 = math.tanh(w11 * x1 + w12 * x2 + b1_1)
h2 = math.tanh(w21 * x1 + w22 * x2 + b1_2)
y_hat = math.tanh(v1 * h1 + v2 * h2 + b2)
print(f"segundo forward: y_hat = {y_hat:.4f}  L = {(y - y_hat) ** 2:.4f}")
