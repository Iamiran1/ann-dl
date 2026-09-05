---
exercise: data
ai_use: "Utilizei ChatGPT e Claude como apoio para compreender o enunciado, revisar código e discutir as análises. Todo o código foi lido e testado por mim."
---

# 1. Data

!!! abstract "Enunciado"

    [Exercises → Data](https://insper.github.io/ann-dl/){:target='_blank'}

---

## Exercise 1

Quatro nuvens gaussianas bidimensionais, geradas com médias e desvios-padrão diferentes,
servem para medir **quão separáveis** essas classes são — primeiro na configuração original
(item A) e depois sob diferentes fatores de dispersão (item B).

Cada classe contribui com **100 pontos** — 400 no total — amostrados de uma normal
bivariada com os parâmetros abaixo. Todas as amostragens partem do mesmo gerador,
`np.random.default_rng(42)`, o que torna os números deste relatório reprodutíveis.

| Classe | $\mu$ | $\sigma$ | Característica |
|:---:|:---:|:---:|---|
| **0** | `[2.0, 3.0]` | `[0.8, 2.5]` | estreita em $x_1$, muito alongada em $x_2$ |
| **1** | `[5.0, 6.0]` | `[1.2, 1.9]` | a mais dispersa nas duas direções |
| **2** | `[8.0, 1.0]` | `[0.9, 0.9]` | isotrópica e compacta |
| **3** | `[15.0, 4.0]` | `[0.5, 2.0]` | afastada das demais em $x_1$ |


### Código

Um único script gera os dados e todas as figuras dos itens A e B:
[`code/exercise1_point_clouds.py`](https://github.com/Iamiran1/ann-dl/blob/main/docs/exercises/data/code/exercise1_point_clouds.py).

??? example "Ver o código completo"

    ```{ .python .copy .select linenums='1' title="docs/exercises/data/code/exercise1_point_clouds.py" }
    --8<-- "docs/exercises/data/code/exercise1_point_clouds.py"
    ```

### A — Generate the clouds

#### Abordagem

Cada classe é amostrada de uma normal bivariada com as médias e desvios-padrão da tabela
acima, 100 pontos por classe. Os desvios são multiplicados por um fator `scale`, fixado em
`1.0` neste item, e o gerador `np.random.default_rng(42)` é criado uma única vez no módulo,
de modo que a mesma sequência de números é reproduzida a cada execução do script.

#### Resultado

![Nuvens de pontos das quatro classes gaussianas](figures/fig01-point-clouds.png)

/// caption
**Figura 1** — Distribuição das quatro classes no plano $(x_1, x_2)$ com `scale = 1.0`.
O marcador **✕** indica o centro $\mu$ de cada classe.
///

A classe 3 fica isolada à direita ($\mu_1 = 15$) e é visualmente separável das demais por uma
reta. As classes 0 e 1 se sobrepõem: seus centros distam 4.24 unidades, mas o desvio vertical
da classe 0 ($\sigma_2 = 2.5$) é da mesma ordem dessa distância.

### B — More or less spread out

#### Abordagem

Os mesmos quatro centros foram reutilizados, variando apenas a dispersão pelos fatores
`scale ∈ {0.5, 1.0, 2.0, 4.0}` — as médias permanecem fixas, então qualquer mudança na
sobreposição vem só do alargamento das gaussianas. Duas métricas foram calculadas:

- **Separation ratio** — razão entre a distância dos centros e a soma dos desvios médios:

    $$
    r_{ij} = \frac{\lVert \mu_i - \mu_j \rVert}
                  {\bar{\sigma}_i + \bar{\sigma}_j}
    $$

    Como o numerador não depende de $s$ e o denominador é proporcional a $s$, vale
    $r_{ij}(s) = r_{ij}(1)/s$ — a separação cai com o inverso da escala.

- **Taxa de mistura** — fração de pontos cujo centro mais próximo não é o da própria classe,
  isto é, o erro de um classificador de distância mínima aos centros verdadeiros.

#### Resultado

![Comparação das classes para diferentes escalas](figures/fig02-point-clouds-scales.png)

/// caption
**Figura 2** — As mesmas quatro classes para `scale = 0.5`, `1.0`, `2.0` e `4.0`,
com eixos compartilhados para permitir comparação direta.
///


=== "Separation ratio"

    | Par de classes | $s=0.5$ | $s=1.0$ | $s=2.0$ | $s=4.0$ |
    |---|---:|---:|---:|---:|
    | 0–1 | 2.652 | **1.326** | **0.663** | **0.331** |
    | 0–2 | 4.960 | 2.480 | 1.240 | 0.620 |
    | 0–3 | 8.992 | 4.496 | 2.248 | 1.124 |
    | 1–2 | 4.760 | 2.380 | 1.190 | 0.595 |
    | 1–3 | 7.284 | 3.642 | 1.821 | 0.911 |
    | 2–3 | 7.084 | 3.542 | 1.771 | 0.886 |

    Em negrito, o **par mais crítico** em cada escala. O par 0–1 é sempre o menor, e é o
    primeiro a cruzar $r_{ij} < 1$ (centros mais próximos que a soma das dispersões),
    o que acontece já em `scale = 2.0`.

=== "Taxa de mistura"

    | `scale` | Taxa de mistura | Pontos mal atribuídos |
    |---:|---:|---:|
    | 0.5 | 0.0 % | 0 / 400 |
    | 1.0 | 5.0 % | 20 / 400 |
    | 2.0 | 19.25 % | 77 / 400 |
    | 4.0 | 48.25 % | 193 / 400 |

    ![Taxa de mistura em função da escala](figures/fig03-mixing-rate.png)

    /// caption
    **Figura 3** — Taxa de mistura em função do fator de escala $s$.
    ///


### C — Overlap and decision boundaries

#### Pergunta 1
Descreva a sobreposição das quatro classes no dataset original. Uma única fronteira linear conseguiria separar todas as classes? E um conjunto de fronteiras lineares?


#### Resposta 1
No dataset original, as quatro classes formam grupos relativamente distintos. Não seria possível utilizar uma única fronteira linear para separar todas as classes. No entanto, um conjunto de fronteiras lineares poderia dividir o espaço em diferentes regiões, permitindo uma melhor separação entre as quatro classes.

---
#### Pergunta 2
Desenhe sobre a Figura 1 as possíveis fronteiras de decisão que uma rede neural treinada aprenderia para separar as classes.


#### Resposta 2
![Esboço da Fronteiras de decisão](figures/fig04-decision-boundaries.png)


### Pergunta 3
Compare o desenho das fronteiras com o item B: quando as classes ficam mais dispersas, como muda a região em que a rede inevitavelmente irá errar?


### Resposta 3
À medida que o scale aumenta, as classes se tornam mais dispersas e as regiões de sobreposição entre elas crescem. Com isso, a delimitação por fronteiras de decisão se torna mais difícil, pois amostras de classes diferentes passam a ocupar regiões próximas ou coincidentes. Esse comportamento está diretamente relacionado ao mixing rate: quanto maior a escala, maior tende a ser a taxa de mistura entre as classes e, consequentemente, maior a região em que a rede inevitavelmente cometerá erros.

---
## Results summary

Esta tabela não substitui nenhuma análise — é um índice dos números já calculados, reunidos
em um só lugar para que a correção confira cada valor sem ter que procurá-lo no texto.

!!! note "Preenchimento parcial"

    As linhas 1–5 vêm do Exercise 1. As linhas 6–13 serão preenchidas com os Exercises 2 e 3.

| # | Item | Your value |
|---|------|------------|
| 1 | Mixing rate at $s = 0.5$ | 0.0 % (0/400) |
| 2 | Mixing rate at $s = 1$ | 5.0 % (20/400) |
| 3 | Mixing rate at $s = 2$ | 19.25 % (77/400) |
| 4 | Mixing rate at $s = 4$ | 48.25 % (193/400) |
| 5 | Smallest $r_{ij}$ at $s = 1$, and which pair | 1.326 — par 0–1 |
| 6 | Distance between centers — Dataset I | |
| 7 | Distance between centers — Dataset II | |
| 8 | Explained variance PC1 + PC2 — Dataset I | |
| 9 | Explained variance PC1 + PC2 — Dataset II | |
| 10 | Share of the positive class in `Transported` | |
| 11 | Mean and median of `FoodCourt` on the training set, before transforming | |
| 12 | Final shape of the training feature matrix | |
| 13 | Minimum and maximum of the training and test sets after scaling | |

## Discussão

O que foi difícil? Onde a intuição falhou? Que decisão você tomaria diferente?

## Conclusão

O que este exercício mostrou sobre a relação entre distribuição dos dados e a complexidade
da fronteira de decisão que a rede precisa aprender?
