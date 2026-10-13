window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre"]
  }
};

// Re-typeset after each page load (Material SPA).
// Com navigation.instant a página troca sem recarregar: sem limpar o cache, o
// MathJax reaproveita estilos de caracteres da página anterior que não existem
// mais, e símbolos somem das fórmulas.
document$.subscribe(() => {
  if (window.MathJax?.typesetPromise) {
    window.MathJax.startup.output.clearCache();
    window.MathJax.typesetClear();
    window.MathJax.texReset();
    window.MathJax.typesetPromise();
  }
});
