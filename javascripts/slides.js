// Carrossel de slides: transforma cada <div class="slides"> em uma sequência navegável
// com setas, contador e pontos. Sem JavaScript, todos os slides aparecem um embaixo do outro.
function montarSlides() {
  document.querySelectorAll(".slides:not(.slides-pronto)").forEach(function (box) {
    var slides = Array.prototype.slice.call(box.querySelectorAll(":scope > .slide"));
    if (slides.length === 0) return;
    var atual = 0;

    // a mesma barra de navegação em cima e embaixo dos slides
    var barras = [criarBarra(), criarBarra()];
    box.insertBefore(barras[0].nav, box.firstChild);
    box.appendChild(barras[1].nav);

    function criarBarra() {
      var nav = document.createElement("div");
      nav.className = "slides-nav";
      var ant = criarSeta("◀", "Slide anterior", -1);
      var prox = criarSeta("▶", "Próximo slide", +1);
      var contador = document.createElement("span");
      contador.className = "slides-contador";
      var pontos = document.createElement("div");
      pontos.className = "slides-pontos";
      slides.forEach(function (_, i) {
        var p = document.createElement("button");
        p.type = "button";
        p.className = "slides-ponto";
        p.setAttribute("aria-label", "Ir para o slide " + (i + 1));
        p.addEventListener("click", function () { mostrar(i); });
        pontos.appendChild(p);
      });
      nav.appendChild(ant);
      nav.appendChild(pontos);
      nav.appendChild(contador);
      nav.appendChild(prox);
      return { nav: nav, ant: ant, prox: prox, contador: contador, pontos: pontos };
    }

    function criarSeta(texto, rotulo, passo) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "slides-seta";
      b.setAttribute("aria-label", rotulo);
      b.textContent = texto;
      b.addEventListener("click", function () { mostrar(atual + passo); });
      return b;
    }

    function mostrar(i) {
      atual = Math.max(0, Math.min(slides.length - 1, i));
      slides.forEach(function (s, k) { s.classList.toggle("ativo", k === atual); });
      barras.forEach(function (b) {
        b.pontos.childNodes.forEach(function (p, k) { p.classList.toggle("ativo", k === atual); });
        b.contador.textContent = (atual + 1) + " / " + slides.length;
        b.ant.disabled = atual === 0;
        b.prox.disabled = atual === slides.length - 1;
      });
    }

    box.tabIndex = 0;
    box.addEventListener("keydown", function (ev) {
      if (ev.key === "ArrowLeft") { mostrar(atual - 1); ev.preventDefault(); }
      if (ev.key === "ArrowRight") { mostrar(atual + 1); ev.preventDefault(); }
    });

    box.classList.add("slides-pronto");
    mostrar(0);
  });
}

// com navigation.instant, o Material troca a página sem recarregar: document$ avisa a cada troca
if (typeof document$ !== "undefined") {
  document$.subscribe(montarSlides);
} else {
  document.addEventListener("DOMContentLoaded", montarSlides);
}
