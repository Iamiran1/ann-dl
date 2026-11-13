"""Diagrama da arquitetura do VAE (d = 16) no estilo de grafo de MLP.

Cada coluna é uma camada; só alguns neurônios são desenhados (com "⋮" no meio) e o tamanho
real aparece embaixo. Linhas cinza contínuas são pesos treináveis (camadas nn.Linear);
linhas tracejadas são a reparametrização, que não tem pesos.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

FIGURES = Path(__file__).resolve().parents[1] / "figures"

COR = {
    "entrada": "#7cb3f0",
    "encoder": "#f4a261",
    "mu": "#a77be0",
    "logvar": "#c9a7f0",
    "eps": "#d9d9d9",
    "z": "#f6d55c",
    "decoder": "#5cc8b0",
    "saida": "#8fe0a0",
}
R = 0.3  # raio dos neurônios


def coluna(x, n, y0, y1, gap=True):
    """Posições y de n neurônios entre y0 e y1, com um espaço para o '⋮' no meio."""
    if n == 1:
        return [(y0 + y1) / 2]
    passo = (y1 - y0) / (n - 1 + (1 if gap else 0))
    ys = []
    for i in range(n):
        extra = 1 if gap and i >= n // 2 else 0
        ys.append(y1 - (i + extra) * passo)
    return ys


def desenhar(ax, x, ys, cor, rotulo="", gap=True, alpha=1.0):
    for y in ys:
        ax.add_patch(Circle((x, y), R, facecolor=cor, edgecolor="#444", lw=1.2, zorder=3,
                            alpha=alpha))
        if rotulo:
            ax.text(x, y, rotulo, ha="center", va="center", zorder=4, alpha=alpha,
                    fontsize=6.5 if len(rotulo) > 2 else 9)
    if gap and len(ys) > 1:
        meio = (ys[len(ys) // 2 - 1] + ys[len(ys) // 2]) / 2
        ax.text(x, meio, "⋮", ha="center", va="center", fontsize=16, color="#555", alpha=alpha)


def ligar_todos(ax, xa, ysa, xb, ysb, aceso=None):
    """Camada totalmente conectada: cada neurônio de b recebe todos os de a."""
    cor, lw, alpha = "#b5b5b5", 0.6, 1.0     # sem destaque: todas iguais
    if aceso is True:
        cor, lw, alpha = "#555555", 0.9, 1.0  # etapa em destaque
    elif aceso is False:
        alpha = 0.15                          # etapa apagada
    for ya in ysa:
        for yb in ysb:
            ax.plot([xa + R, xb - R], [ya, yb], color=cor, lw=lw, alpha=alpha, zorder=1)


TOP, BOT = 7.0, 0.6
xs = {"in": 0, "e1": 2.2, "e2": 4.4, "head": 6.8, "z": 9.6, "d1": 11.8, "d2": 14.0, "out": 16.2}
ys = {
    "in": coluna(xs["in"], 6, BOT, TOP),
    "e1": coluna(xs["e1"], 6, BOT, TOP),
    "e2": coluna(xs["e2"], 5, BOT + 0.5, TOP - 0.5),
    "mu": coluna(xs["head"], 3, 4.5, TOP - 0.2),
    "lv": coluna(xs["head"], 3, BOT + 0.2, 3.1),
    "z": coluna(xs["z"], 3, 2.6, 5.2),
    "d1": coluna(xs["d1"], 5, BOT + 0.5, TOP - 0.5),
    "d2": coluna(xs["d2"], 6, BOT, TOP),
    "out": coluna(xs["out"], 6, BOT, TOP),
}
x_eps = (xs["head"] + xs["z"]) / 2
ys["eps"] = coluna(x_eps, 3, BOT, 3.2)

# o que fica aceso em cada passo dos slides do relatório (nós e grupos de ligações);
# o título de cada passo fica no próprio slide
PASSOS = {
    1: {"in"},
    2: {"in", "e1", "e2", "in-e1", "e1-e2"},
    3: {"e2", "mu", "lv", "e2-cab"},
    4: {"mu", "lv", "eps", "z", "reparam"},
    5: {"z", "d1", "d2", "z-d1", "d1-d2"},
    6: {"d2", "out", "d2-out"},
    7: {"in", "out", "mu", "lv"},  # a loss usa x, x̂, μ e log σ²
}


def figura(acesas=None, titulo=None, arquivo="arquitetura-vae.png", legenda=True):
    """Desenha o grafo; com `acesas`, destaca só esses elementos e apaga o resto."""
    def a(k):  # alpha de um nó
        return 1.0 if acesas is None or k in acesas else 0.15

    def e(k):  # estado de um grupo de ligações
        return None if acesas is None else (k in acesas)

    fig, ax = plt.subplots(figsize=(15, 9.2 if legenda else 7.6))
    ax.set_xlim(-0.8, 17.3)
    ax.set_ylim(-1.9 if legenda else -0.7, 8.6)
    ax.axis("off")
    ax.set_aspect("equal")

    # pesos treináveis (nn.Linear): todas as ligações
    ligar_todos(ax, xs["in"], ys["in"], xs["e1"], ys["e1"], e("in-e1"))
    ligar_todos(ax, xs["e1"], ys["e1"], xs["e2"], ys["e2"], e("e1-e2"))
    ligar_todos(ax, xs["e2"], ys["e2"], xs["head"], ys["mu"], e("e2-cab"))
    ligar_todos(ax, xs["e2"], ys["e2"], xs["head"], ys["lv"], e("e2-cab"))
    ligar_todos(ax, xs["z"], ys["z"], xs["d1"], ys["d1"], e("z-d1"))
    ligar_todos(ax, xs["d1"], ys["d1"], xs["d2"], ys["d2"], e("d1-d2"))
    ligar_todos(ax, xs["d2"], ys["d2"], xs["out"], ys["out"], e("d2-out"))

    # reparametrização: z_i depende só de μ_i, log σ²_i e ε_i (um-para-um, sem pesos)
    ar = 1.0 if acesas is None or "reparam" in acesas else 0.15
    for i in range(3):
        for (xa, ya, cor) in [(xs["head"], ys["mu"][i], COR["mu"]),
                              (xs["head"], ys["lv"][i], COR["logvar"]),
                              (x_eps, ys["eps"][i], "#999")]:
            ax.plot([xa + R, xs["z"] - R], [ya, ys["z"][i]], color=cor, lw=1.6, ls="--",
                    zorder=2, alpha=ar)

    desenhar(ax, xs["in"], ys["in"], COR["entrada"], "x", alpha=a("in"))
    desenhar(ax, xs["e1"], ys["e1"], COR["encoder"], "ReLU", alpha=a("e1"))
    desenhar(ax, xs["e2"], ys["e2"], COR["encoder"], "ReLU", alpha=a("e2"))
    desenhar(ax, xs["head"], ys["mu"], COR["mu"], "μ", alpha=a("mu"))
    desenhar(ax, xs["head"], ys["lv"], COR["logvar"], "log σ²", alpha=a("lv"))
    desenhar(ax, x_eps, ys["eps"], COR["eps"], "ε", alpha=a("eps"))
    desenhar(ax, xs["z"], ys["z"], COR["z"], "z", alpha=a("z"))
    desenhar(ax, xs["d1"], ys["d1"], COR["decoder"], "ReLU", alpha=a("d1"))
    desenhar(ax, xs["d2"], ys["d2"], COR["decoder"], "ReLU", alpha=a("d2"))
    desenhar(ax, xs["out"], ys["out"], COR["saida"], "sig", alpha=a("out"))

    # nomes das camadas (em cima) e tamanhos (embaixo)
    cab = [("in", "in", "Entrada", "784 pixels"),
           ("e1", "e1", "Encoder 1", "512 neurônios"),
           ("e2", "e2", "Encoder 2", "256 neurônios"),
           ("head", "mu", "Cabeças", "16 + 16"),
           ("z", "z", "Latente z", "16 valores"),
           ("d1", "d1", "Decoder 1", "256 neurônios"),
           ("d2", "d2", "Decoder 2", "512 neurônios"),
           ("out", "out", "Saída x̂", "784 pixels")]
    for k, kn, nome, tam in cab:
        al = max(a(kn), a("lv")) if k == "head" else a(kn)
        ax.text(xs[k], 8.0, nome, ha="center", fontsize=11, weight="bold", alpha=max(al, 0.35))
        ax.text(xs[k], -0.35, tam, ha="center", fontsize=9.5, color="#333", alpha=max(al, 0.35))
    ax.text(x_eps, -0.35, "ε ~ N(0, I)", ha="center", fontsize=9.5, color="#333",
            alpha=max(a("eps"), 0.35))
    ax.text(xs["z"], ys["z"][0] + 0.55, "z = μ + σ·ε", ha="center", fontsize=9.5, color="#333",
            alpha=max(a("z"), 0.35))

    # faixas de encoder / decoder
    for x0, x1, nome, cor in [(xs["in"] + 1.2, xs["head"] + 0.4, "ENCODER  q(z | x)", COR["encoder"]),
                              (xs["z"] + 1.2, xs["out"] + 0.4, "DECODER  p(x | z)", COR["decoder"])]:
        ax.add_patch(FancyBboxPatch((x0, 7.55), x1 - x0, 0.0, boxstyle="round,pad=0",
                                    edgecolor=cor, lw=3))
        ax.text((x0 + x1) / 2, 7.65, nome, ha="center", va="bottom", fontsize=9, color="#555")

    if legenda:
        itens = [(COR["entrada"], "Entrada"), (COR["encoder"], "Encoder (ReLU)"),
                 (COR["mu"], "Cabeça μ"), (COR["logvar"], "Cabeça log σ²"), (COR["eps"], "Ruído ε"),
                 (COR["z"], "Latente z"), (COR["decoder"], "Decoder (ReLU)"),
                 (COR["saida"], "Saída (sigmoide)")]
        for i, (cor, txt) in enumerate(itens):
            x = -0.4 + i * 2.05
            ax.add_patch(Circle((x, -1.35), 0.17, facecolor=cor, edgecolor="#444", lw=1))
            ax.text(x + 0.25, -1.35, txt, va="center", fontsize=9)
        ax.plot([-0.55, -0.1], [-1.85, -1.85], color="#b5b5b5", lw=1.5)
        ax.text(0.0, -1.85, "peso treinável (nn.Linear)", va="center", fontsize=9)
        ax.plot([4.0, 4.45], [-1.85, -1.85], color="#888", lw=1.6, ls="--")
        ax.text(4.55, -1.85, "reparametrização: ligação um-para-um, sem peso", va="center",
                fontsize=9)

    ax.set_title(titulo or "Arquitetura do VAE (d = 16): os dados fluem da esquerda para a direita",
                 fontsize=14, weight="bold", pad=4)
    fig.savefig(FIGURES / arquivo, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    figura()  # visão geral, com legenda
    (FIGURES / "passos").mkdir(exist_ok=True)
    for n, acesas in PASSOS.items():
        figura(acesas, titulo=" ", arquivo=f"passos/arq-passo{n}.png", legenda=False)
