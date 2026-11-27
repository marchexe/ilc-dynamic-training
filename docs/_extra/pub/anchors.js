// Anchor links for overview.html and log.html.
// Every element with data-anchor gets a small "#" button that copies the full URL of that anchor.
// On log.html, opening a link to a trial (#t18) expands its details and scrolls to it.
(function () {
  function url(id) { return location.href.split('#')[0] + '#' + id; }

  function copy(text, button) {
    function done(ok) {
      button.textContent = ok ? '✓' : '#';
      button.title = ok ? 'Link copied' : 'Copy failed: ' + text;
      setTimeout(function () { button.textContent = '#'; button.title = 'Copy link'; }, 1400);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () { done(true); }, function () { done(fallback(text)); });
    } else {
      done(fallback(text));
    }
  }

  function fallback(text) {
    var area = document.createElement('textarea');
    area.value = text;
    area.setAttribute('readonly', '');
    area.style.position = 'fixed';
    area.style.opacity = '0';
    document.body.appendChild(area);
    area.select();
    var ok = false;
    try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
    document.body.removeChild(area);
    return ok;
  }

  function addButtons() {
    var targets = document.querySelectorAll('[data-anchor]');
    for (var i = 0; i < targets.length; i++) {
      (function (el) {
        var id = el.getAttribute('data-anchor');
        var b = document.createElement('a');
        b.className = 'anchor';
        b.href = '#' + id;
        b.textContent = '#';
        b.title = 'Copy link';
        b.setAttribute('aria-label', 'Copy link to this section');
        b.addEventListener('click', function (ev) {
          ev.preventDefault();
          ev.stopPropagation();  // do not toggle an enclosing <details>
          history.replaceState(null, '', '#' + id);
          copy(url(id), b);
        });
        el.appendChild(b);
      })(targets[i]);
    }
  }

  function openTarget() {
    var id = decodeURIComponent(location.hash.slice(1));
    if (!id) return;
    var el = document.getElementById(id);
    if (!el) return;
    if (el.tagName === 'DETAILS') el.open = true;
    var parent = el.closest && el.closest('details');
    if (parent) parent.open = true;
    el.scrollIntoView({ block: 'start' });
  }

  document.addEventListener('DOMContentLoaded', function () {
    addButtons();
    openTarget();
  });
  window.addEventListener('hashchange', openTarget);
  // figures and web fonts load after DOMContentLoaded and move the target; scroll again once the page is complete
  window.addEventListener('load', openTarget);
})();
