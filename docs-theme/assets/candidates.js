/* The filter on a candidates page: a card stays visible when its text contains every typed word (case-insensitive),
   and an origin section, its H2 included, is hidden when none of its cards is visible. */
(function () {
  function init() {
    var input = document.querySelector("input.candidate-filter");
    if (!input || input.__sbInit) return;
    input.__sbInit = true;
    var lists = Array.prototype.slice.call(document.querySelectorAll(".md-typeset h2 + ul"));
    function apply() {
      var words = input.value.toLowerCase().split(/\s+/).filter(Boolean);
      lists.forEach(function (list) {
        var shown = 0;
        Array.prototype.forEach.call(list.children, function (card) {
          var text = card.textContent.toLowerCase();
          var match = words.every(function (w) { return text.indexOf(w) !== -1; });
          card.hidden = !match;
          if (match) shown++;
        });
        list.hidden = shown === 0;
        list.previousElementSibling.hidden = shown === 0;
      });
    }
    input.addEventListener("input", apply);
  }
  if (window.document$ && window.document$.subscribe) window.document$.subscribe(init);
  else init();
})();
