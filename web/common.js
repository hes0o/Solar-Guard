// Shared helpers for all pages.
const NAV = [["/", "Manager"], ["/staff", "Staff"], ["/dataset", "Dataset Mode"], ["/try", "Try an image"]];

function renderNav(extra = "") {
  const nav = document.createElement("nav");
  nav.innerHTML = `<span class="brand">☀ SolarGuard</span>` +
    NAV.map(([h, t]) => `<a href="${h}" class="${location.pathname === h ? "active" : ""}">${t}</a>`).join("") +
    `<span class="spacer"></span>${extra}`;
  document.body.prepend(nav);
}

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}

const post = (path, body) => api(path, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined,
});

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function chip(label) {
  if (!label) return `<span class="chip c-none">night / no reading</span>`;
  return `<span class="chip c-${esc(label.split(" ")[0])}">${esc(label)}</span>`;
}

// Polling re-renders often; only touch the DOM when content changed, so buttons aren't swapped mid-click.
function setHTML(el, html) {
  if (typeof el === "string") el = document.getElementById(el);
  if (el._html !== html) { el.innerHTML = html; el._html = html; }
}

const pct =v => v == null ? "–" : (v * 100).toFixed(0) + "%";

function toast(msg) {
  let el = document.querySelector(".toast");
  if (!el) { el = document.createElement("div"); el.className = "toast"; document.body.append(el); }
  el.textContent = msg; el.style.display = "block";
  clearTimeout(el._t); el._t = setTimeout(() => el.style.display = "none", 3000);
}

async function act(fn) {
  try { await fn(); } catch (e) { toast("⚠ " + e.message); }
}
