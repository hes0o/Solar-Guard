// Shared helpers for all pages.
const NAV = [["/", "Manager"], ["/staff", "Staff"], ["/dataset", "Dataset Mode"], ["/try", "Try an image"]];

async function renderNav(extra = "") {
  // Build synchronously so pages can find the elements in `extra` right away; links arrive once the role is known.
  const nav = document.createElement("nav");
  nav.innerHTML = `<span class="brand">☀ SolarGuard</span><span id="links" style="display:contents"></span>` +
    `<span class="spacer"></span>${extra}<span class="muted" id="role"></span><button id="logout">Log out</button>`;
  document.body.prepend(nav);
  document.getElementById("logout").onclick = async () => { await post("/api/logout"); location.href = "/login"; };
  const { role } = await api("/api/me");
  document.getElementById("role").textContent = role;
  document.getElementById("links").innerHTML = NAV.filter(([h]) => role === "manager" || h !== "/")
    .map(([h, t]) => `<a href="${h}" class="${location.pathname === h ? "active" : ""}">${t}</a>`).join("");
}

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (r.status === 401) { location.href = "/login?next=" + encodeURIComponent(location.pathname); throw new Error("please log in"); }
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
