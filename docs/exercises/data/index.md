---
exercise: data
ai_use: "Utilizei ChatGPT e Claude como apoio para compreender o enunciado, revisar código e discutir as análises, estruturação do documento e desenvolvimento do conteúdo produzido por mim. Todo o código foi lido e testado por mim."
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
![Esboço das fronteiras de decisão](figures/fig04-decision-boundaries.png)

/// caption
**Figura 4** — Esboço das fronteiras de decisão sobre as nuvens da Figura 1.
///


#### Pergunta 3
Compare o desenho das fronteiras com o item B: quando as classes ficam mais dispersas, como muda a região em que a rede inevitavelmente irá errar?


#### Resposta 3
À medida que o scale aumenta, as classes se tornam mais dispersas e as regiões de sobreposição entre elas crescem. Com isso, a delimitação por fronteiras de decisão se torna mais difícil, pois amostras de classes diferentes passam a ocupar regiões próximas ou coincidentes. Esse comportamento está diretamente relacionado ao mixing rate: quanto maior a escala, maior tende a ser a taxa de mistura entre as classes e, consequentemente, maior a região em que a rede inevitavelmente cometerá erros.

---
## Exercise 2

Dois problemas de classificação em **5 dimensões**, com estruturas geométricas opostas,
servem para separar duas perguntas que o Exercise 1 mantinha juntas: **quão longe estão as
classes** e **de que forma elas se distinguem**. Cada dataset tem duas classes com
**500 amostras** cada — 1000 pontos no total — e cada amostra é um vetor

$$
x = [x_1, x_2, x_3, x_4, x_5].
$$

| Dataset | Classes | Estrutura | O que distingue as classes |
|:---:|:---:|---|---|
| **I** | A / B | duas gaussianas deslocadas, com covariâncias diferentes | deslocamento entre os centros |
| **II** | C / D | duas cascas esféricas concêntricas | distância radial até a origem |

A semente é fixada com `np.random.seed(42)` antes da primeira amostragem, de modo que todos
os números abaixo são reprodutíveis.

### Código

Um único script gera os dois datasets, aplica PCA e produz as figuras dos itens A, B e C:
[`code/exercise2-non_linearity_in_higher_dimensions.py`](https://github.com/Iamiran1/ann-dl/blob/main/docs/exercises/data/code/exercise2-non_linearity_in_higher_dimensions.py).

??? example "Ver o código completo"

    ```{ .python .copy .select linenums='1' title="docs/exercises/data/code/exercise2-non_linearity_in_higher_dimensions.py" }
    --8<-- "docs/exercises/data/code/exercise2-non_linearity_in_higher_dimensions.py"
    ```

### A — Dataset I: shifted Gaussians

#### Abordagem

As classes **A** e **B** são amostradas de normais multivariadas em 5D com
`np.random.multivariate_normal`, 500 pontos cada — uma matriz `(500, 5)` por classe.

**Classe A** — média $\mu_A = [0,\ 0,\ 0,\ 0,\ 0]$ e covariância $\Sigma_A$:

|  | $x_1$ | $x_2$ | $x_3$ | $x_4$ | $x_5$ |
|:---:|---:|---:|---:|---:|---:|
| **$x_1$** | 1.0 | 0.8 | 0.1 | 0.0 | 0.0 |
| **$x_2$** | 0.8 | 1.0 | 0.3 | 0.0 | 0.0 |
| **$x_3$** | 0.1 | 0.3 | 1.0 | 0.5 | 0.0 |
| **$x_4$** | 0.0 | 0.0 | 0.5 | 1.0 | 0.2 |
| **$x_5$** | 0.0 | 0.0 | 0.0 | 0.2 | 1.0 |

**Classe B** — média $\mu_B = [1.5,\ 1.5,\ 1.5,\ 1.5,\ 1.5]$ e covariância $\Sigma_B$:

|  | $x_1$ | $x_2$ | $x_3$ | $x_4$ | $x_5$ |
|:---:|---:|---:|---:|---:|---:|
| **$x_1$** | 1.5 | −0.7 | 0.2 | 0.0 | 0.0 |
| **$x_2$** | −0.7 | 1.5 | 0.4 | 0.0 | 0.0 |
| **$x_3$** | 0.2 | 0.4 | 1.5 | 0.6 | 0.0 |
| **$x_4$** | 0.0 | 0.0 | 0.6 | 1.5 | 0.3 |
| **$x_5$** | 0.0 | 0.0 | 0.0 | 0.3 | 1.5 |

As duas classes não diferem apenas pela média. A classe B tem variâncias maiores (1.5 contra
1.0 na diagonal) e, sobretudo, uma **estrutura de correlação diferente**: as duas primeiras
dimensões são positivamente correlacionadas em A ($\Sigma_{12} = 0.8$) e negativamente
correlacionadas em B ($\Sigma_{12} = -0.7$). 
#### Resultado

A distância entre os centros empíricos das duas classes é

$$
\lVert \bar{x}_A - \bar{x}_B \rVert = 3.390,
$$

próxima do valor teórico $\lVert \mu_A - \mu_B \rVert = 1.5\sqrt{5} \approx 3.354$. Como o
desvio-padrão por dimensão é da ordem de 1.0–1.22, os centros estão a cerca de três desvios
um do outro: há sobreposição, mas o deslocamento domina.

### B — Dataset II: concentric shells

#### Abordagem

O segundo dataset também tem 500 amostras por classe em 5D, mas é construído em coordenadas
radiais. Para cada amostra, sorteia-se primeiro uma **direção** — um vetor gaussiano
normalizado:

$$
v \sim \mathcal{N}(0, I_5),
\qquad
u = \frac{v}{\lVert v \rVert}
\quad \Longrightarrow \quad
\lVert u \rVert = 1.
$$

Como $\mathcal{N}(0, I_5)$ é isotrópica, $u$ é uniforme sobre a esfera unitária: carrega
apenas direção, nenhuma informação de escala. Em seguida cada classe recebe um **raio** de
uma distribuição própria, e o ponto final é $x = \rho\,u$:

| Classe | Raio $\rho$ | Interpretação |
|:---:|:---:|---|
| **C** (*core*) | $\rho_C \sim \mathcal{N}(2.0,\ 0.4)$ | casca interna, a $\approx 2$ da origem |
| **D** (*shell*) | $\rho_D \sim \mathcal{N}(5.0,\ 0.4)$ | casca externa, a $\approx 5$ da origem |

#### Resultado

Diferentemente do Dataset I, o que separa C e D **não é um deslocamento entre centros**. Os
raios médios amostrados são 2.013 e 4.970, mas as direções são uniformes nos dois casos, de
modo que ambas as classes têm o centro praticamente na origem:

$$
\lVert \bar{x}_C - \bar{x}_D \rVert = 0.199.
$$

Esse número é o ponto central do exercício. Comparado aos 3.390 do Dataset I, ele sugeriria
classes quase coincidentes — e ainda assim C e D são **perfeitamente separáveis**, porque a
informação discriminante está inteiramente em $\lVert x \rVert$, não na posição do centro.
Distância entre centros é uma métrica que só faz sentido para estruturas do tipo do
Dataset I.

### C — Visualize and compare

#### Abordagem

Com cinco dimensões não é possível visualizar o espaço original diretamente. Para comparar os
dois datasets, cada um foi projetado de 5D para 2D com **PCA**, ajustado de forma
independente em cada dataset:

```python
pca_AB = PCA(n_components=2)
X_AB_2D = pca_AB.fit_transform(X_AB)

pca_CD = PCA(n_components=2)
X_CD_2D = pca_CD.fit_transform(X_CD)
```

Cada ponto $[x_1, x_2, x_3, x_4, x_5]$ passa a ser representado por $[PC_1, PC_2]$.

#### Resultado

![Projeção PCA em 2D dos datasets I e II](figures/fig05-pca-comparison.png)

/// caption
**Figura 5** — Projeção dos datasets I e II sobre as duas primeiras componentes principais.
///

No **Dataset I**, A e B se deslocam visivelmente ao longo de $PC_1$. Há sobreposição, mas a
projeção preserva a maior parte da diferença entre os grupos — uma reta em 2D já classifica
razoavelmente bem.

No **Dataset II**, a projeção destrói a estrutura: as duas cascas viram duas nuvens
concêntricas mal distinguíveis, com C no miolo e D espalhada ao redor. Uma casca esférica em
5D projetada em um plano vira um **disco cheio**, não um anel — todos os pontos cuja direção
é quase ortogonal ao plano do PCA caem perto do centro da projeção. A separação existe no
espaço original, mas não sobrevive a esta projeção linear.

##### Explained variance

A razão de variância explicada mede quanto da variabilidade das cinco dimensões originais
cada componente preserva:

| Dataset | $PC_1$ | $PC_2$ | $PC_1 + PC_2$ |
|---|---:|---:|---:|
| **I** — A/B | 52.07 % | 15.85 % | **67.91 %** |
| **II** — C/D | 22.14 % | 20.48 % | **42.63 %** |

A diferença vem da geometria. No Dataset I o deslocamento entre as classes acontece na
direção $[1,1,1,1,1]$, que é uma **única** direção do espaço — o PCA a encontra e a coloca em
$PC_1$, que sozinha responde por mais de metade da variância. No Dataset II não existe
direção privilegiada: por construção, as cascas são isotrópicas, então a variância se
distribui quase igualmente pelas cinco dimensões ($\approx 20\%$ em cada) e duas componentes
nunca poderiam capturar muito mais do que os 40 % observados.


##### A feature certa resolve o Dataset II

Para confirmar que a informação não se perdeu — apenas ficou invisível ao PCA — basta olhar
para o histograma de $\lVert x \rVert$:

![Histogramas do raio para os dois datasets](figures/fig06-radius-histograms.png)

/// caption
**Figura 6** — Distribuição do raio $\lVert x \rVert$ em cada classe dos dois datasets.
///

No Dataset II as duas faixas de raio nem se tocam: a classe C vai de 0.525 a **3.146** e a
classe D, de **3.707** a 6.121 — uma folga de 0.561 entre elas. No Dataset I, ao contrário,
os raios se sobrepõem bastante (médias 2.06 e 4.17), porque ali o raio não é a característica
relevante.

#### Pergunta 1

No Dataset II, a distância entre os centros das classes é próxima de zero, mas os histogramas
dos raios são bem separados. O que essa combinação indica sobre a possibilidade de separar as
classes usando um hiperplano?

#### Resposta 1

Como os centros das duas classes estão praticamente no mesmo ponto ($0.199$ de distância), a
separação não vem de um deslocamento no espaço — não existe direção ao longo da qual as
classes estejam de lados opostos. O que os separa é a distância à origem, e os histogramas
mostram que essa separação é limpa: as classes ocupam faixas radiais disjuntas.

Um único hiperplano não dá conta dessa combinação. Ele divide o espaço em dois semiespaços,
mas o Dataset II tem estrutura **concêntrica** — uma classe no miolo e a outra em volta, em
todas as direções. Qualquer hiperplano que se tente traçar corta as duas cascas ao mesmo
tempo, deixando pontos de C e de D dos dois lados.

---

#### Pergunta 2

Explique por que a estrutura do Dataset II não pode ser separada por uma fronteira linear,
independentemente da quantidade de dados coletados.

#### Resposta 2

Uma fronteira linear não consegue separar corretamente duas classes com estrutura concêntrica, pois ela apenas divide o espaço em dois lados. No Dataset II, uma classe ocupa uma região mais interna e a outra forma uma região externa ao seu redor, o que exige uma fronteira não linear. Esse problema não depende da quantidade de amostras disponíveis, mas da própria geometria das classes; portanto, mesmo com mais dados, uma fronteira linear continuaria incapaz de separá-las completamente.

---

#### Pergunta 3

O PCA é uma transformação linear. Uma projeção em 2D na qual as classes parecem misturadas
prova que elas são inseparáveis no espaço original? Justifique com os seus próprios
resultados e escreva uma função simples das entradas que consiga separar o Dataset II.

#### Resposta 3

Não. O fato de as classes parecerem misturadas após o PCA não prova que sejam inseparáveis no espaço original, pois a projeção pode descartar informações relevantes. No Dataset II, a separação é melhor observada pelo raio dos pontos. Uma regra simples como classificar pontos com \(\|x\| < 3.5\) como classe C e os demais como classe D consegue separar bem as duas classes.

---
## Exercise 3

!!! info "Fonte dos dados"

    [Spaceship Titanic — Kaggle](https://www.kaggle.com/competitions/spaceship-titanic/overview){:target='_blank'}

Os dois primeiros exercícios usaram dados sintéticos, onde a geometria era conhecida de
antemão. Este usa um dataset real — o **Spaceship Titanic**, da competição homônima do
Kaggle — em que a estrutura precisa ser descoberta e os dados vêm com os problemas de sempre:
valores faltantes, variáveis categóricas e features em escalas muito diferentes.

O arquivo `train.csv` — o mesmo da competição — tem **8693 linhas e 14 colunas**:
identificação do passageiro (`PassengerId`, `Name`, `Cabin`), atributos categóricos
(`HomePlanet`, `CryoSleep`, `Destination`, `VIP`), a idade (`Age`), cinco colunas de gastos
a bordo (`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`) e a coluna-alvo
`Transported`.

### Código

Um único script carrega o CSV, produz as tabelas desta seção e faz o split:
[`code/exercise3_spaceship_tatanic.py`](https://github.com/Iamiran1/ann-dl/blob/main/docs/exercises/data/code/exercise3_spaceship_tatanic.py).

??? example "Ver o código completo"

    ```{ .python .copy .select linenums='1' title="docs/exercises/data/code/exercise3_spaceship_tatanic.py" }
    --8<-- "docs/exercises/data/code/exercise3_spaceship_tatanic.py"
    ```

### Pergunta 1

Descreva o objetivo do dataset: o que a coluna `Transported` representa? Qual é o balanço de
classes entre os dois rótulos?

### Resposta 1

O objetivo é prever quais passageiros da Spaceship Titanic foram **transportados para uma
dimensão alternativa** depois que a nave colidiu com uma anomalia espaço-temporal. A coluna
`Transported` registra esse desfecho e é o nosso **alvo**: assume `True` para o passageiro
que foi transportado e `False` para quem não foi. Como só há dois valores possíveis, trata-se
de um problema de **classificação binária**.

Quanto ao balanço, as duas classes estão praticamente empatadas:

| `Transported` | Passageiros | Proporção |
|:---:|---:|---:|
| `True` | 4378 | **50.36 %** |
| `False` | 4315 | 49.64 % |
| **Total** | 8693 | 100 % |

A diferença entre os rótulos é de apenas 63 passageiros — cerca de 0.7 ponto percentual — e
não há valores faltantes em `Transported`.

### Pergunta 2

Liste as features, separando as numéricas (ex.: `Age`, `RoomService`) das categóricas
(ex.: `HomePlanet`, `Destination`).

### Resposta 2

Tirando a coluna-alvo `Transported`, sobram **13 features**: 6 numéricas e 7 categóricas.

**Features numéricas** — valores quantitativos, em que a ordem e a diferença entre os
valores têm significado:

| Feature | Únicos | Faltando | Descrição |
|---|---:|---:|---|
| `Age` | 80 | 179 (2.1 %) | idade do passageiro |
| `RoomService` | 1273 | 181 (2.1 %) | gasto com serviço de quarto |
| `FoodCourt` | 1507 | 183 (2.1 %) | gasto na praça de alimentação |
| `ShoppingMall` | 1115 | 208 (2.4 %) | gasto no shopping |
| `Spa` | 1327 | 183 (2.1 %) | gasto no spa |
| `VRDeck` | 1306 | 188 (2.2 %) | gasto no deck de realidade virtual |

As cinco colunas de gasto formam um bloco à parte: são todas em dólares, fortemente
assimétricas (a maioria dos passageiros gasta 0).

**Features categóricas** — valores que nomeiam grupos, sem ordem natural:

| Feature | Únicos | Faltando | Descrição |
|---|---:|---:|---|
| `HomePlanet` | 3 | 201 (2.3 %) | planeta de origem |
| `CryoSleep` | 2 | 217 (2.5 %) | se estava em animação suspensa |
| `Destination` | 3 | 182 (2.1 %) | planeta de destino |
| `VIP` | 2 | 203 (2.3 %) | se pagou serviço VIP |
| `Cabin` | 6560 | 199 (2.3 %) | cabine, no formato `deck/num/lado` |
| `PassengerId` | 8693 | 0 | identificador, no formato `gggg_pp` |
| `Name` | 8473 | 200 (2.3 %) | nome do passageiro |



### Pergunta 3

Monte uma tabela de valores faltantes por coluna, em contagem absoluta e em porcentagem.

### Resposta 3

| Coluna | Faltando | % |
|---|---:|---:|
| `PassengerId` | 0 | 0.00 % |
| `HomePlanet` | 201 | 2.31 % |
| `CryoSleep` | 217 | 2.50 % |
| `Cabin` | 199 | 2.29 % |
| `Destination` | 182 | 2.09 % |
| `Age` | 179 | 2.06 % |
| `VIP` | 203 | 2.34 % |
| `RoomService` | 181 | 2.08 % |
| `FoodCourt` | 183 | 2.11 % |
| `ShoppingMall` | 208 | 2.39 % |
| `Spa` | 183 | 2.11 % |
| `VRDeck` | 188 | 2.16 % |
| `Name` | 200 | 2.30 % |
| `Transported` | 0 | 0.00 % |


---

### Pergunta 4

Para as colunas de gasto (`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`),
informe média, mediana e máximo. Compare média e mediana: o que essa diferença diz sobre a
dispersão e a assimetria dessas distribuições?

### Resposta 4

| | `RoomService` | `FoodCourt` | `ShoppingMall` | `Spa` | `VRDeck` |
|---|---:|---:|---:|---:|---:|
| **média** | 224.69 | 458.08 | 173.73 | 311.14 | 304.85 |
| **mediana** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **máximo** | 14 327 | 29 813 | 23 492 | 22 408 | 24 133 |

A mediana é **zero nas cinco colunas**, enquanto a média fica entre 174 e 458. Uma mediana
zero significa que **mais da metade dos passageiros não gastou nada** naquele serviço — de
fato, o percentual de zeros vai de 62.6 % (`Spa`) a 65.8 % (`ShoppingMall`). Se a média
mesmo assim é positiva e alta, é porque a minoria que gasta, gasta muito.

O tamanho do desequilíbrio fica claro comparando os extremos com o centro:

| | `RoomService` | `FoodCourt` | `ShoppingMall` | `Spa` | `VRDeck` |
|---|---:|---:|---:|---:|---:|
| % de zeros | 65.5 % | 64.1 % | 65.8 % | 62.6 % | 64.6 % |
| percentil 75 | 47 | 76 | 27 | 59 | 46 |
| percentil 99 | 3 096 | 8 033 | 2 333 | 5 390 | 5 647 |
| média entre quem gastou | 652 | 1 276 | 509 | 831 | 861 |

Em `FoodCourt`, o máximo (29 813) é **65 vezes** a média e cerca de 390 vezes o percentil 75.

Isso caracteriza distribuições fortemente **assimétricas à direita** (*right-skewed*): a
massa dos dados está empilhada no zero e uma cauda longa se estende para valores muito altos.
Como a média é sensível a valores extremos e a mediana não, ter a **média muito acima da
mediana** é a assinatura clássica desse formato. Vale notar que a média aqui é uma
péssima descrição do passageiro típico: ela não representa nem quem não gastou (a maioria)
nem quem gastou de verdade (cuja média é 2 a 3 vezes maior).

Há uma explicação estrutural para tantos zeros. Passageiros em animação suspensa não podem
consumir nada a bordo, e os dados confirmam a regra sem exceção: dos **3037** passageiros com
`CryoSleep = True`, **100 %** têm gasto total zero. Ou seja, boa parte dos zeros não é dado
faltante disfarçado nem ausência de consumo por escolha — é uma consequência determinística
de outra coluna.

Consequências para o pré-processamento:

- **Escalonar sem transformar não resolve.** Um `StandardScaler` aplicado direto move e
  reescala, mas não muda o formato: a cauda continua lá, e mais de 60 % dos pontos ficam
  amontoados no mesmo valor.
- **Uma transformação logarítmica é o caminho natural.** Como há zeros, a forma usual é
  `log(1 + x)` (`np.log1p`), que comprime a cauda e mantém o zero em zero.
- **A imputação das numéricas deve usar a mediana, não a média.** Preencher um `FoodCourt`
  ausente com 458 inventaria um gasto alto para um passageiro que muito provavelmente não
  gastou nada — a mediana (0) é a escolha coerente com a distribuição.

### B — Split before you transform

#### Abordagem

A separação usa `train_test_split` do scikit-learn com três decisões explícitas:

```python
X = df.drop(columns=["Transported"])
y = df["Transported"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)
```

- `test_size=0.20` — divisão 80/20;
- `stratify=y` — preserva a proporção das classes nas duas partes;
- `random_state=42` — semente fixa, para que a divisão seja sempre a mesma.

#### Resultado

| Conjunto | Linhas | Colunas | `True` | `False` | % da classe positiva |
|---|---:|---:|---:|---:|---:|
| **Treino** | 6954 | 13 | 3502 | 3452 | 50.36 % |
| **Teste** | 1739 | 13 | 876 | 863 | 50.37 % |
| **Total** | 8693 | 13 | 4378 | 4315 | 50.36 % |

A estratificação funcionou: a proporção da classe positiva na base completa (50.36 %) é
reproduzida no treino (50.36 %) e no teste (50.37 %), com diferença de 0.01 ponto percentual.
As 13 colunas são as features — `Transported` saiu para o vetor `y`.

#### Pergunta

Explique, em duas ou três frases, por que esse split vem antes da imputação e do
escalonamento.

#### Resposta

O split deve vir antes da imputação e da normalização para evitar **data leakage**.
Estatísticas como média, mediana, desvio-padrão e as categorias observadas precisam ser
calculadas **apenas** com os dados de treino; se informações do conjunto de teste entrarem
nessas transformações, o modelo recebe indiretamente informação que não deveria conhecer
durante o treinamento. O resultado é uma avaliação otimista: o desempenho medido no teste
deixa de estimar o desempenho em dados realmente novos, que é a única coisa que interessa.


### D — Verify and visualize

#### Resultado

![FoodCourt antes e depois da transformação logarítmica](figures/fig07-foodcourt-log.png)

/// caption
**Figura 7** — `FoodCourt` no conjunto de treino, antes e depois de `log(1 + x)`.
///

À esquerda, a distribuição original: uma única barra gigante junto ao zero e o restante do
eixo praticamente vazio, esticado até 29 813 por causa de um punhado de passageiros.

À direita, depois de `log(1 + x)`, a cauda foi comprimida para o intervalo 0–10 e a
distribuição revela uma estrutura que antes estava escondida: o pico em zero continua (são os
~64 % que não gastaram, e o log preserva o zero), mas quem gastou aparece agora como um
segundo agrupamento em torno de 6–8. De uma barra única passamos a **duas populações
distinguíveis** — exatamente o tipo de estrutura que a rede consegue aproveitar.

#### Checagens finais

| Verificação | Treino | Teste |
|---|---|---|
| `NaN` restantes | **0** | **0** |
| Shape da matriz de features | **(6954, 17)** | **(1739, 17)** |
| Valor mínimo | −2.00 | −2.00 |
| Valor máximo | **3.51** | **3.37** |

As 17 colunas se dividem em 7 numéricas (`Age`, as cinco de gasto e `TotalSpend`) e 10
geradas pelo *one-hot* de `HomePlanet` (3), `CryoSleep` (2), `Destination` (3) e `VIP` (2).

**Sobre a compatibilidade com a tanh.** A faixa está adequada, mas com uma ressalva que vale
registrar. A tanh satura fora de aproximadamente $[-2, 2]$ — `tanh(2) = 0.964`,
`tanh(3.51) = 0.9982` — e nessa região o gradiente é quase nulo, o que trava o aprendizado.
Na matriz final, **95.6 %** dos valores numéricos estão dentro de $[-2, 2]$ e apenas
**0.06 %** passam de 3. Ou seja, a grande maioria dos dados cai na região útil da ativação;
os poucos extremos vêm de `Age`, cujo máximo padronizado é 3.51.


#### Pergunta

Em um parágrafo: qual das suas decisões de pré-processamento você acha que mais afetaria o
treinamento da rede, e por quê?

#### Resposta

A decisão de maior impacto é a **transformação logarítmica das colunas de gasto**, e este
exercício produziu a evidência disso sem que fosse preciso treinar nada. Sem o `log1p`, o
`TotalSpend` chegava ao escalonamento com a cauda bruta e a matriz final ia até 12.54 desvios
padrão; como `tanh(12.54)` é 1.0 até a décima casa decimal, todos esses exemplos cairiam na
região plana da ativação, onde o gradiente é essencialmente zero — os neurônios que os
recebessem simplesmente parariam de aprender, e os passageiros de gasto alto, que são
justamente os mais informativos, seriam os primeiros a ser ignorados. Com o `log1p` o máximo
cai para 3.51 e 95.6 % dos valores passam a ocupar a faixa útil da tanh. Vale notar que o
escalonamento sozinho **não** resolveria: o `StandardScaler` desloca e reescala, mas não muda
o formato da distribuição, de modo que a razão entre o extremo e o corpo dos dados
permaneceria a mesma. A imputação pela mediana e o *one-hot* também importam, mas erram por
margens menores — trocar mediana por média em `FoodCourt` inventaria um gasto de 458 para
quem provavelmente não gastou nada, o que desloca a distribuição sem, no entanto, saturar a
ativação.

---
## Results summary

Esta tabela não substitui nenhuma análise — é um índice dos números já calculados, reunidos
em um só lugar para que a correção confira cada valor sem ter que procurá-lo no texto.

!!! note "Preenchimento parcial"

    As linhas 1–5 vêm do Exercise 1 e as linhas 6–9, do Exercise 2. A linha 10 vem da
    Pergunta 1 do Exercise 3; as linhas 11–13 serão preenchidas conforme o Exercise 3 avança.

| # | Item | Your value |
|---|------|------------|
| 1 | Mixing rate at $s = 0.5$ | 0.0 % (0/400) |
| 2 | Mixing rate at $s = 1$ | 5.0 % (20/400) |
| 3 | Mixing rate at $s = 2$ | 19.25 % (77/400) |
| 4 | Mixing rate at $s = 4$ | 48.25 % (193/400) |
| 5 | Smallest $r_{ij}$ at $s = 1$, and which pair | 1.326 — par 0–1 |
| 6 | Distance between centers — Dataset I | 3.390 |
| 7 | Distance between centers — Dataset II | 0.199 |
| 8 | Explained variance PC1 + PC2 — Dataset I | 67.91 % |
| 9 | Explained variance PC1 + PC2 — Dataset II | 42.63 % |
| 10 | Share of the positive class in `Transported` | 50.36 % (4378/8693) |
| 11 | Mean and median of `FoodCourt` on the training set, before transforming | média 458.08 — mediana 0.00 |
| 12 | Final shape of the training feature matrix | (6954, 17) |
| 13 | Minimum and maximum of the training and test sets after scaling | treino [−2.00, 3.51] — teste [−2.00, 3.37] |

## Discussão

A principal dificuldade foi perceber como diferentes distribuições e escalas afetam a separabilidade dos dados. A intuição falhou principalmente nos casos em que classes visualmente misturadas ainda podiam ser separadas por informações presentes em dimensões maiores.

## Conclusão

O exercício mostrou que quanto mais complexa é a distribuição dos dados, mais complexa precisa ser a fronteira de decisão. Estruturas não lineares, como classes concêntricas, exigem modelos capazes de aprender fronteiras também não lineares.
