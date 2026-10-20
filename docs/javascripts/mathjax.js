window.MathJax = {
  // \boldsymbol은 자동 로드에 맡기면 texReset과 순서가 엇갈려 정의되지 않은 명령으로 표시됩니다.
  // 시작할 때 미리 불러옵니다.
  loader: {
    load: ["[tex]/boldsymbol"]
  },
  tex: {
    packages: { "[+]": ["boldsymbol"] },
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
