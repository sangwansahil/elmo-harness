// elmo single-page UI. vanilla JS, no build step.

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const view = $("#view");
const STAGES = ["spec", "plan", "data", "train", "eval", "diagnose", "export"];

const fmt = {
  ts: (sec) => {
    const d = new Date(sec * 1000);
    const pad = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  },
  hms: (sec) => {
    const d = new Date(sec * 1000);
    const pad = (n) => String(n).padStart(2, "0");
    return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  },
  num: (x, n = 3) => (x == null ? "—" : Number(x).toFixed(n)),
  delta: (x) => {
    if (x == null) return "—";
    const s = x >= 0 ? "+" : "";
    return `${s}${Number(x).toFixed(3)}`;
  },
};

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json();
}

function el(tag, props = {}, children = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else if (k === "ds") for (const [dk, dv] of Object.entries(v)) e.dataset[dk] = dv;
    else if (k === "style") for (const [sk, sv] of Object.entries(v)) e.style[sk] = sv;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v != null) e.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c == null) continue;
    if (typeof c === "string") e.appendChild(document.createTextNode(c));
    else e.appendChild(c);
  }
  return e;
}

// --- runs list -----------------------------------------------------------
async function renderRunsList() {
  setActiveNav("nav-runs");
  view.innerHTML = "";
  view.appendChild(el("h1", {}, "runs"));
  view.appendChild(el("p", { class: "micro" }, "every fine-tune is a row. click for the receipt."));
  let runs = [];
  try { runs = await api("/api/runs"); } catch (e) { runs = []; }
  if (!runs.length) {
    const t = $("#t-empty").content.cloneNode(true);
    view.appendChild(t);
    return;
  }
  const grid = el("div", { class: "card", style: { padding: "0" } });
  for (const r of runs) {
    const delta = r.final_score != null && r.baseline_score != null ? r.final_score - r.baseline_score : null;
    const dCls = delta == null ? "" : (delta >= 0 ? "up" : "down");
    const a = el("a", { class: "row", href: `#/runs/${r.id}` }, [
      el("span", { class: "mo id" }, r.id.slice(0, 24)),
      el("span", { class: "task" }, r.task_name),
      el("span", { class: "mo model" }, r.base_model.split("/").pop()),
      el("span", { class: "mo base" }, fmt.num(r.baseline_score)),
      el("span", { class: `mo delta ${dCls}` }, fmt.delta(delta)),
      el("span", { class: "mo when" }, fmt.ts(r.created_at)),
      el("span", { class: "status", ds: { s: r.status } }, r.status),
    ]);
    grid.appendChild(a);
  }
  view.appendChild(grid);
}

// --- run detail ----------------------------------------------------------
async function renderRun(runId) {
  setActiveNav("nav-runs");
  view.innerHTML = "";
  let run;
  try { run = await api(`/api/runs/${runId}`); } catch (e) {
    view.appendChild(el("p", { class: "micro" }, `not found: ${runId}`));
    return;
  }
  const report = run.report || {};
  const baseline = report.baseline || {};
  const best = report.best || {};
  const iters = report.iterations || [];

  // header
  view.appendChild(el("h1", {}, `run · ${run.task_name}`));
  view.appendChild(el("p", { class: "micro" }, run.id));

  // prompt + meta receipt
  const promptCard = el("div", { class: "card" });
  promptCard.appendChild(el("div", { class: "lab" }, "prompt"));
  promptCard.appendChild(el("div", { class: "mo", style: { fontSize: "14px" } }, run.spec?.prompt || "—"));
  const dl = el("dl", { class: "receipt" });
  for (const [k, v] of [
    ["base", run.base_model],
    ["backend", run.backend],
    ["iterations", `${iters.length} of ${run.spec?.budget?.max_iterations ?? "?"}`],
    ["status", run.status],
    ["target", run.spec?.eval?.target_score ?? "—"],
  ]) {
    dl.appendChild(el("dt", {}, k));
    dl.appendChild(el("dd", { class: "mo" }, String(v)));
  }
  promptCard.appendChild(dl);
  view.appendChild(promptCard);

  // stage rail (best-effort from last event)
  const stagesWrap = el("div", { class: "stages-wrap" });
  const stages = el("div", { class: "stages" });
  const events = await api(`/api/runs/${runId}/events?limit=500`);
  const lastStage = events.length ? events[events.length - 1].stage : null;
  const stageSet = new Set(events.map((e) => e.stage));
  STAGES.forEach((name) => {
    const isLive = run.status === "running" && name === lastStage;
    const isDone = stageSet.has(name) && !isLive;
    const cls = "stage" + (isLive ? " live" : isDone ? " done" : "");
    stages.appendChild(el("div", { class: cls }, name));
  });
  stagesWrap.appendChild(stages);
  view.appendChild(stagesWrap);

  // capabilities + track
  view.appendChild(el("h3", {}, "capabilities · best vs baseline"));
  const capCard = el("div", { class: "card" });
  const capGrid = el("div", { class: "cap-grid" });
  for (const k of ["tool_selection", "arguments", "parallel_calls", "overall"]) {
    const b = baseline[k] ?? 0;
    const v = best[k] ?? b;
    const d = v - b;
    capGrid.appendChild(el("span", { class: "name" }, k.replace(/_/g, " ")));
    const bar = el("div", { class: "bar" });
    bar.appendChild(el("div", { style: { width: `${Math.max(0, Math.min(1, v)) * 100}%` } }));
    capGrid.appendChild(bar);
    capGrid.appendChild(el("span", { class: "v" }, fmt.num(v)));
    capGrid.appendChild(el("span", { class: `d ${d > 0 ? "up" : d < 0 ? "down" : ""}` }, fmt.delta(d)));
  }
  capCard.appendChild(capGrid);
  // bfcl track
  const target = run.spec?.eval?.target_score;
  const base = baseline.overall ?? 0;
  const cur = best.overall ?? base;
  if (target) {
    const wrap = el("div", { class: "track-wrap" });
    wrap.appendChild(el("div", { class: "lab" }, "overall"));
    const track = el("div", { class: "track" }, [
      el("div", { class: "fill", style: { width: `${Math.max(0, Math.min(1, cur)) * 100}%` } }),
      el("div", { class: "tick", style: { left: `${Math.max(0, Math.min(1, base)) * 100}%` } }),
      el("div", { class: "tick", style: { left: `${Math.max(0, Math.min(1, target)) * 100}%`, background: "var(--ink)" } }),
    ]);
    wrap.appendChild(track);
    wrap.appendChild(el("div", { class: "track-labels" }, [
      el("span", {}, `baseline ${fmt.num(base)}`),
      el("span", { style: { color: "var(--accent)" } }, `now ${fmt.num(cur)}`),
      el("span", {}, `target ${fmt.num(target)}`),
    ]));
    capCard.appendChild(wrap);
  }
  view.appendChild(capCard);

  // iteration chips
  if (iters.length || (run.spec?.budget?.max_iterations || 0) > 0) {
    view.appendChild(el("h3", {}, "iterations"));
    const max = run.spec?.budget?.max_iterations || iters.length;
    const strip = el("div", { class: "iters" });
    for (let n = 1; n <= max; n++) {
      const it = iters.find((x) => x.n === n);
      let chip;
      if (it) {
        const sc = it.score?.overall ?? 0;
        const dv = it.score?.overall != null && baseline.overall != null ? sc - baseline.overall : 0;
        chip = el("span", { class: "chip" }, [
          `${String(n).padStart(2, "0")} `,
          el("span", { class: dv >= 0 ? "up" : "down" }, fmt.delta(dv)),
        ]);
      } else if (n === iters.length + 1 && run.status === "running") {
        chip = el("span", { class: "chip", ds: { status: "running" } }, `${String(n).padStart(2, "0")} running`);
      } else {
        chip = el("span", { class: "chip", ds: { status: "future" } }, String(n).padStart(2, "0"));
      }
      strip.appendChild(chip);
    }
    view.appendChild(strip);
  }

  // log feed
  view.appendChild(el("h3", {}, "live"));
  const logCard = el("div", { class: "card" });
  const log = el("div", { class: "log", id: "log" });
  for (const e of events.slice(-200)) {
    log.appendChild(eventLine(e));
  }
  logCard.appendChild(log);
  view.appendChild(logCard);

  // live updates via websocket if still running
  if (run.status === "running") attachLive(runId, log);
}

function eventLine(e) {
  const line = el("div", {});
  line.appendChild(el("span", { class: "ts" }, fmt.hms(e.created_at)));
  line.appendChild(el("span", { class: "stg" }, e.stage));
  line.appendChild(document.createTextNode(e.message));
  return line;
}

function attachLive(runId, logEl) {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/api/runs/${runId}/live`);
  ws.onmessage = (m) => {
    const e = JSON.parse(m.data);
    if (e._status) {
      $("#meta").textContent = `run finished · ${e._status}`;
      ws.close();
      // refresh the whole view to show final scores
      setTimeout(() => renderRun(runId), 800);
      return;
    }
    logEl.appendChild(eventLine(e));
    logEl.scrollTop = logEl.scrollHeight;
  };
}

// --- regression suite ----------------------------------------------------
async function renderRegression(task) {
  setActiveNav("nav-reg");
  view.innerHTML = "";
  view.appendChild(el("h1", {}, `regression · ${task}`));
  view.appendChild(el("p", { class: "micro" }, "every failure becomes a permanent case. the model ships with this list."));
  let data;
  try { data = await api(`/api/regression/${task}`); } catch { data = { cases: [], by_capability: {} }; }
  if (!data.cases.length) {
    view.appendChild(el("p", { class: "micro" }, "no cases yet — run a few iterations first."));
    return;
  }
  for (const [cap, cases] of Object.entries(data.by_capability)) {
    const fixed = cases.filter((c) => c.fixed_in_iter != null).length;
    const wrap = el("div", { class: "reg-cap" });
    const head = el("h2");
    head.appendChild(document.createTextNode(cap.replace(/_/g, " ")));
    head.appendChild(el("span", { class: "micro" }, `${fixed}/${cases.length} fixed`));
    wrap.appendChild(head);
    const ul = el("ul", { class: "reg-list" });
    for (const c of cases) {
      ul.appendChild(el("li", {}, [
        el("span", { class: "id" }, c.id.slice(0, 12)),
        el("span", { class: "q" }, c.query),
        el("span", { class: "seen" }, `seen iter ${c.first_seen_iter}`),
        el("span", { class: c.fixed_in_iter != null ? "fixed" : "open" }, c.fixed_in_iter != null ? `fixed iter ${c.fixed_in_iter}` : "open"),
      ]));
    }
    wrap.appendChild(ul);
    view.appendChild(wrap);
  }
}

// --- router --------------------------------------------------------------
function setActiveNav(id) {
  $$(".nav a").forEach((a) => a.classList.remove("on"));
  const n = document.getElementById(id);
  if (n) n.classList.add("on");
}

async function route() {
  const h = location.hash || "#/";
  let m;
  if (h === "#/") return renderRunsList();
  if ((m = h.match(/^#\/runs\/([^/]+)$/))) return renderRun(m[1]);
  if ((m = h.match(/^#\/regression\/([^/]+)$/))) return renderRegression(m[1]);
  view.innerHTML = `<p class="micro">unknown route: ${h}</p>`;
}

async function loadMeta() {
  try {
    const h = await api("/api/health");
    const provs = h.providers_configured.length;
    $("#meta").textContent = `backend ${h.backend} · ${provs} provider${provs === 1 ? "" : "s"} configured`;
  } catch {}
}

window.addEventListener("hashchange", route);
window.addEventListener("DOMContentLoaded", () => { loadMeta(); route(); });
