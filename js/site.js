(function () {
  var root = document.documentElement;
  var KEY = "zmcp-theme";

  function systemDark() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  function current() {
    var t = root.getAttribute("data-theme");
    return t ? t : (systemDark() ? "dark" : "light");
  }
  function label(btn) {
    var next = current() === "dark" ? "light" : "dark";
    btn.setAttribute("aria-label", "Switch to " + next + " theme");
    btn.setAttribute("title", "Switch to " + next + " theme");
  }

  // Theme toggle: flips away from whatever is showing, remembers the choice.
  document.querySelectorAll(".theme-toggle").forEach(function (btn) {
    label(btn);
    btn.addEventListener("click", function () {
      var next = current() === "dark" ? "light" : "dark";
      if ((next === "dark") === systemDark()) {
        root.removeAttribute("data-theme");
        try { localStorage.removeItem(KEY); } catch (e) {}
      } else {
        root.setAttribute("data-theme", next);
        try { localStorage.setItem(KEY, next); } catch (e) {}
      }
      label(btn);
    });
  });

  // Copy buttons on code blocks.
  document.querySelectorAll(".code").forEach(function (block) {
    var pre = block.querySelector("pre");
    if (!pre || block.hasAttribute("data-nocopy")) return;
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy";
    btn.textContent = "Copy";
    btn.addEventListener("click", function () {
      var text = Array.prototype.map.call(pre.querySelectorAll(".line"), function (l) {
        return l.getAttribute("data-copy") || l.textContent;
      }).join("\n") || pre.textContent;
      text = text.replace(/^\$ /gm, "");
      var done = function () {
        btn.textContent = "Copied";
        setTimeout(function () { btn.textContent = "Copy"; }, 1600);
      };
      if (navigator.clipboard) navigator.clipboard.writeText(text.trim()).then(done, function () {});
    });
    block.appendChild(btn);
  });

  // Tabs.
  document.querySelectorAll("[data-tabs]").forEach(function (group) {
    var tabs = group.querySelectorAll('[role="tab"]');
    function select(tab) {
      tabs.forEach(function (t) {
        var on = t === tab;
        t.setAttribute("aria-selected", on ? "true" : "false");
        t.tabIndex = on ? 0 : -1;
        document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
      });
    }
    tabs.forEach(function (tab, i) {
      tab.addEventListener("click", function () { select(tab); });
      tab.addEventListener("keydown", function (e) {
        var d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
        if (!d) return;
        var n = tabs[(i + d + tabs.length) % tabs.length];
        select(n);
        n.focus();
      });
    });
  });

  // Table of contents: mark the section in view.
  var toc = document.querySelector(".toc");
  if (toc && "IntersectionObserver" in window) {
    var links = {};
    toc.querySelectorAll("a[href^='#']").forEach(function (a) { links[a.getAttribute("href").slice(1)] = a; });
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting && links[en.target.id]) {
          Object.keys(links).forEach(function (k) { links[k].classList.remove("active"); });
          links[en.target.id].classList.add("active");
        }
      });
    }, { rootMargin: "-20% 0px -70% 0px" });
    Object.keys(links).forEach(function (id) {
      var el = document.getElementById(id);
      if (el) obs.observe(el);
    });
  }

  // Motion: header shadow, scroll reveals, chart bars, counters, highlighter.
  var header = document.querySelector(".site-header");
  if (header) {
    var onScroll = function () { header.classList.toggle("scrolled", window.scrollY > 8); };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }

  document.querySelectorAll(".theme-toggle").forEach(function (btn) {
    btn.addEventListener("click", function () {
      btn.classList.toggle("spin");
    });
  });

  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduce || !("IntersectionObserver" in window)) return;
  root.classList.add("motion");

  var groups = [
    ".section-head", ".releases .release", ".bench div", ".routes .route", ".cost",
    ".modes .mode", ".caps .cap", ".step", ".group-head", ".table-wrap", ".tools .tool",
    ".section .code", ".section p.fine"
  ];
  var targets = [];
  groups.forEach(function (sel) {
    document.querySelectorAll(sel).forEach(function (el) {
      if (el.closest(".hero") || el.hasAttribute("data-reveal")) return;
      var siblings = el.parentElement ? Array.prototype.filter.call(el.parentElement.children, function (c) {
        return c.matches(sel);
      }) : [el];
      var i = Math.max(0, siblings.indexOf(el));
      el.setAttribute("data-reveal", "");
      el.style.setProperty("--d", Math.min(i, 6) * 0.08 + "s");
      targets.push(el);
    });
  });

  function countUp(el) {
    var text = el.textContent;
    var m = text.match(/^([\d,]*\.?\d+)(.*)$/);
    if (!m) return;
    var raw = m[1], suffix = m[2];
    var end = parseFloat(raw.replace(/,/g, ""));
    var decimals = (raw.split(".")[1] || "").length;
    var comma = raw.indexOf(",") >= 0;
    var start = performance.now(), dur = 1000;
    function fmt(v) {
      var s = v.toFixed(decimals);
      return comma ? Number(s).toLocaleString("en-US", { minimumFractionDigits: decimals }) : s;
    }
    function tick(now) {
      var t = Math.min(1, (now - start) / dur);
      var eased = 1 - Math.pow(1 - t, 3);
      el.textContent = fmt(end * eased) + suffix;
      if (t < 1) requestAnimationFrame(tick); else el.textContent = text;
    }
    requestAnimationFrame(tick);
  }

  var seen = new IntersectionObserver(function (entries) {
    entries.forEach(function (en) {
      if (!en.isIntersecting) return;
      var el = en.target;
      seen.unobserve(el);
      el.classList.add("in");
      if (el.hasAttribute("data-reveal")) {
        setTimeout(function () { el.classList.add("settled"); }, 900 + parseFloat(el.style.getPropertyValue("--d") || 0) * 1000);
      }
      if (el.classList.contains("hl")) el.classList.remove("pending");
      if (el.matches(".bench div")) el.querySelectorAll(".fast").forEach(countUp);
      if (el.classList.contains("bars")) el.querySelectorAll(".bar-val").forEach(countUp);
    });
  }, { rootMargin: "0px 0px -12% 0px", threshold: 0.08 });

  targets.forEach(function (el) { seen.observe(el); });
  document.querySelectorAll(".bars").forEach(function (el) { seen.observe(el); });
  document.querySelectorAll(".hl").forEach(function (el, i) {
    el.classList.add("pending");
    if (el.closest(".hero")) {
      setTimeout(function () { el.classList.remove("pending"); }, 450);
    } else {
      seen.observe(el);
    }
  });
})();
