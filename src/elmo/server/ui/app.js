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

// --- onboarding wizard ---------------------------------------------------
const WIZARD_STEPS = ["diagnose", "choose", "download", "probe", "task", "review", "train"];
const wizardState = {
  step: 0,
  system: null,
  models: [],
  picked: null,
  download_id: null,
  download_path: null,
  probe_done: false,
  task_text: "",
  discovered: null,     // { spec, gates, guess, dataset_pretty }
  run_id: null,         // actual run id once started
  handle_id: null,      // wizard handle (for the runner)
  run_done: false,
  run_result: null,     // { baseline_overall, best_overall, adapter_path }
  saved_to_hub: false,
};

function pip(filled, n = 5, accent = false) {
  const wrap = el("div", { class: "pips" });
  for (let i = 0; i < n; i++) {
    wrap.appendChild(el("div", {
      class: "pip" + (i < filled ? (accent ? " accent" : " on") : ""),
    }));
  }
  return wrap;
}

function fmtBytes(n) {
  if (!n) return "0 B";
  const u = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i === 0 ? 0 : 1)} ${u[i]}`;
}

function wizardRail() {
  const r = el("div", { class: "wiz-rail" });
  WIZARD_STEPS.forEach((name, i) => {
    const cls = "step" + (i === wizardState.step ? " on" : i < wizardState.step ? " done" : "");
    r.appendChild(el("div", { class: cls }, `${String(i + 1).padStart(2, "0")} ${name}`));
    if (i < WIZARD_STEPS.length - 1) r.appendChild(el("span", { class: "sep" }, "·"));
  });
  return r;
}

async function renderOnboard() {
  setActiveNav("nav-onboard");
  view.innerHTML = "";
  view.appendChild(el("h1", {}, "onboard"));
  view.appendChild(el("p", { class: "micro" }, "we'll size your hardware, pick a base model, download it, and kick off a fine-tune."));
  view.appendChild(wizardRail());

  if (wizardState.step === 0) return renderStepDiagnose();
  if (wizardState.step === 1) return renderStepChoose();
  if (wizardState.step === 2) return renderStepDownload();
  if (wizardState.step === 3) return renderStepProbe();
  if (wizardState.step === 4) return renderStepTask();
  if (wizardState.step === 5) return renderStepReview();
  if (wizardState.step === 6) return renderStepTrain();
}

async function renderStepDiagnose() {
  const card = el("div", { class: "card" });
  card.appendChild(el("div", { class: "lab" }, "system probe"));
  card.appendChild(el("p", { class: "micro" }, "reading your hardware…"));
  view.appendChild(card);
  try {
    const data = await api("/api/catalog");
    wizardState.system = data.system;
    wizardState.models = data.models;
    card.innerHTML = "";
    card.appendChild(el("div", { class: "lab" }, "system probe"));
    const dl = el("dl", { class: "kv" });
    for (const [k, v] of [
      ["chip", data.system.chip],
      ["arch", data.system.arch],
      ["ram", `${data.system.ram_gb} GB`],
      ["free disk", `${data.system.free_disk_gb} GB`],
      ["gpu", data.system.gpu_name || "(none)"],
      ["backend", data.system.suggested_backend],
    ]) {
      dl.appendChild(el("dt", {}, k));
      dl.appendChild(el("dd", { class: "mo" }, String(v)));
    }
    card.appendChild(dl);
    const ok = data.system.suggested_backend !== "none";
    const msg = ok
      ? "looks good. we'll pick base models that fit comfortably."
      : "no local training backend detected. fine-tuning needs apple silicon or nvidia. you can still proceed to read-only mode.";
    card.appendChild(el("p", { class: "micro", style: { marginTop: "12px" } }, msg));
    const next = el("button", { class: "btn primary", onclick: () => { wizardState.step = 1; route(); } }, "next →");
    card.appendChild(next);
  } catch (e) {
    card.innerHTML = `<p class="micro">probe failed: ${e}</p>`;
  }
}

function renderStepChoose() {
  const card = el("div", { class: "card" });
  card.appendChild(el("div", { class: "lab" }, "pick a base model"));
  card.appendChild(el("p", { class: "micro" }, "speed = approx tokens / second for this machine. intelligence = relative capability."));
  const grid = el("div", { class: "model-grid" });
  for (const r of wizardState.models) {
    const m = r.model;
    const disabled = !r.fits;
    const picked = wizardState.picked && wizardState.picked.id === m.id;
    const cls = "mod" + (picked ? " on" : "") + (r.elmo_choice ? " choice" : "") + (disabled ? " disabled" : "");
    const cardEl = el("div", { class: cls });
    const head = el("div", { class: "mod-head" });
    head.appendChild(el("h3", {}, m.display_name));
    if (r.elmo_choice) head.appendChild(el("span", { class: "pill" }, "elmo's choice"));
    cardEl.appendChild(head);
    cardEl.appendChild(el("div", { class: "mod-meta" }, `${m.params_b}B · 4-bit · ${m.disk_4bit_gb}GB · ${m.specialty}`));
    cardEl.appendChild(el("div", { class: "mod-note" }, m.note));
    const bars = el("div", { class: "bars" });
    bars.appendChild(el("span", { class: "lab" }, "speed"));
    bars.appendChild(pip(m.speed, 5, r.elmo_choice));
    bars.appendChild(el("span", { class: "val" }, `${r.tokps_estimate} tok/s`));
    bars.appendChild(el("span", { class: "lab" }, "smarts"));
    bars.appendChild(pip(m.intelligence, 5, r.elmo_choice));
    bars.appendChild(el("span", { class: "val" }, `${m.intelligence}/5`));
    cardEl.appendChild(bars);
    if (!r.fits) cardEl.appendChild(el("p", { class: "micro", style: { marginTop: "10px" } }, r.reason));
    cardEl.addEventListener("click", () => {
      if (disabled) return;
      wizardState.picked = m;
      renderStepChoose();  // re-render in place
    });
    grid.appendChild(cardEl);
  }
  card.appendChild(grid);
  view.querySelectorAll(".card").forEach((c) => c.remove());
  view.appendChild(card);

  const actions = el("div", { style: { display: "flex", gap: "10px", marginTop: "16px" } });
  actions.appendChild(el("button", { class: "btn ghost", onclick: () => { wizardState.step = 0; route(); } }, "← back"));
  const next = el("button", {
    class: "btn primary",
    onclick: () => { if (wizardState.picked) { wizardState.step = 2; route(); } },
  }, wizardState.picked ? `download ${wizardState.picked.display_name} →` : "pick a model");
  if (!wizardState.picked) next.setAttribute("disabled", "true");
  actions.appendChild(next);
  view.appendChild(actions);
}

async function renderStepDownload() {
  const m = wizardState.picked;
  if (!m) { wizardState.step = 1; return route(); }
  const card = el("div", { class: "card dl-card" });
  card.appendChild(el("div", { class: "lab" }, `download ${m.display_name}`));
  card.appendChild(el("div", { class: "mo", style: { fontSize: "12px", color: "var(--ink-muted)" } }, m.hf_id_mlx));
  const barWrap = el("div", { class: "dl-bar" });
  const fill = el("div", { class: "fill", style: { width: "0%" } });
  barWrap.appendChild(fill);
  card.appendChild(barWrap);
  const meta = el("div", { class: "dl-meta" });
  meta.appendChild(el("span", { id: "dl-progress" }, "queueing…"));
  meta.appendChild(el("span", { id: "dl-rate" }, ""));
  card.appendChild(meta);
  const msg = el("div", { class: "dl-msg", id: "dl-msg" }, "");
  card.appendChild(msg);
  view.appendChild(card);

  if (!wizardState.download_id) {
    try {
      const start = await fetch(`/api/models/${m.id}/download`, { method: "POST" });
      const body = await start.json();
      wizardState.download_id = body.download_id;
    } catch (e) {
      msg.textContent = `start failed: ${e}`;
      return;
    }
  }

  const totalBytes = m.disk_4bit_gb * (1024 ** 3);
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/api/downloads/${wizardState.download_id}/live`);
  ws.onmessage = (ev) => {
    const st = JSON.parse(ev.data);
    const total = st.bytes_total || totalBytes;
    const pct = Math.min(100, total ? (st.bytes_downloaded / total) * 100 : 0);
    fill.style.width = `${pct.toFixed(1)}%`;
    document.getElementById("dl-progress").textContent =
      `${fmtBytes(st.bytes_downloaded)} / ${fmtBytes(total)}  ·  ${pct.toFixed(1)}%`;
    document.getElementById("dl-msg").textContent = st.status;
    if (st.status === "done") {
      wizardState.download_path = st.path;
      ws.close();
      const next = el("button", { class: "btn primary", onclick: () => { wizardState.step = 3; route(); } }, "model ready · probe it →");
      next.style.marginTop = "14px";
      card.appendChild(next);
    } else if (st.status === "error") {
      ws.close();
      document.getElementById("dl-msg").textContent = `error: ${st.error}`;
    }
  };
  ws.onerror = () => { document.getElementById("dl-msg").textContent = "websocket error"; };
}

async function renderStepProbe() {
  const m = wizardState.picked;
  const card = el("div", { class: "card" });
  card.appendChild(el("div", { class: "lab" }, "say hi"));
  card.appendChild(el("p", { class: "micro" }, "send one prompt to confirm the model loads. try a simple greeting first."));
  const f = el("div", { class: "field" });
  f.appendChild(el("label", {}, "prompt"));
  const inp = el("input", { type: "text", value: "tell me one short fact about owls.", id: "probe-input" });
  f.appendChild(inp);
  card.appendChild(f);
  const out = el("div", { class: "chat-out", id: "probe-out" }, "(no response yet)");
  card.appendChild(out);
  const actions = el("div", { style: { display: "flex", gap: "10px", marginTop: "12px" } });
  const sendBtn = el("button", { class: "btn primary" }, "send →");
  sendBtn.addEventListener("click", async () => {
    sendBtn.setAttribute("disabled", "true");
    out.textContent = "thinking…";
    try {
      const r = await fetch("/api/probe", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ hf_id: m.hf_id_mlx, prompt: inp.value }),
      });
      const body = await r.json();
      out.textContent = body.completion || body.detail || "(no output)";
      if (r.ok) wizardState.probe_done = true;
    } catch (e) {
      out.textContent = `probe failed: ${e}`;
    }
    sendBtn.removeAttribute("disabled");
    if (wizardState.probe_done) {
      next.removeAttribute("disabled");
    }
  });
  actions.appendChild(sendBtn);
  const next = el("button", {
    class: "btn ghost",
    onclick: () => { wizardState.step = 4; route(); },
    disabled: wizardState.probe_done ? null : "true",
  }, "next: describe the task →");
  actions.appendChild(next);
  card.appendChild(actions);
  view.appendChild(card);
}

function renderStepTask() {
  const card = el("div", { class: "card" });
  card.appendChild(el("div", { class: "lab" }, "what should this model be expert at?"));
  card.appendChild(el("p", { class: "micro" }, "one sentence is enough. elmo figures out the dataset, capabilities, and acceptance gates."));
  const f = el("div", { class: "field" });
  f.appendChild(el("label", {}, "task"));
  const ta = el("textarea", { id: "task-input" });
  ta.value = wizardState.task_text || "build a function-calling expert that handles parallel tool calls.";
  f.appendChild(ta);
  card.appendChild(f);
  const ex = el("p", { class: "micro" }, "examples: \"convert dates and times across timezones\" · \"extract structured json from invoices\" · \"solve grade-school math word problems\"");
  card.appendChild(ex);
  const actions = el("div", { style: { display: "flex", gap: "10px", marginTop: "10px" } });
  actions.appendChild(el("button", { class: "btn ghost", onclick: () => { wizardState.step = 3; route(); } }, "← back"));
  const discoverBtn = el("button", { class: "btn primary" }, "discover data →");
  discoverBtn.addEventListener("click", async () => {
    wizardState.task_text = ta.value.trim();
    if (!wizardState.task_text) return;
    discoverBtn.setAttribute("disabled", "true");
    discoverBtn.textContent = "discovering…";
    try {
      const r = await fetch("/api/wizard/discover", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({
          prompt: wizardState.task_text,
          base_model: wizardState.picked.hf_id_mlx,
        }),
      });
      if (!r.ok) throw new Error(await r.text());
      wizardState.discovered = await r.json();
      wizardState.step = 5;
      route();
    } catch (e) {
      discoverBtn.removeAttribute("disabled");
      discoverBtn.textContent = "discover data →";
      const err = el("p", { class: "micro", style: { color: "var(--fail)" } }, `discovery failed: ${e}`);
      card.appendChild(err);
    }
  });
  actions.appendChild(discoverBtn);
  card.appendChild(actions);
  view.appendChild(card);
}

function renderStepReview() {
  const d = wizardState.discovered;
  if (!d) { wizardState.step = 4; return route(); }

  // Header: what elmo figured out
  const head = el("div", { class: "card" });
  head.appendChild(el("div", { class: "lab" }, "discovery"));
  const dl = el("dl", { class: "kv" });
  const kind = d.guess.kind.replace(/-/g, " ");
  const conf = d.guess.confidence;
  const confLabel = conf >= 0.66 ? "high" : conf >= 0.33 ? "medium" : conf > 0 ? "low" : "fallback";
  for (const [k, v] of [
    ["task kind", kind],
    ["confidence", `${confLabel}${d.guess.matched_keywords.length ? " · " + d.guess.matched_keywords.join(", ") : ""}`],
    ["dataset", `${d.dataset_pretty.source}  ·  up to ${d.dataset_pretty.max_rows.toLocaleString()} rows`],
    ["base", wizardState.picked.display_name],
  ]) {
    dl.appendChild(el("dt", {}, k));
    dl.appendChild(el("dd", { class: "mo" }, v));
  }
  head.appendChild(dl);
  view.appendChild(head);

  // Capabilities + acceptance gates
  view.appendChild(el("h3", {}, "acceptance gates"));
  const gateCard = el("div", { class: "card" });
  const gateGrid = el("div", {
    style: { display: "grid", gridTemplateColumns: "max-content 1fr max-content", gap: "8px 14px", fontSize: "13px" },
  });
  for (const g of d.gates) {
    gateGrid.appendChild(el("span", { class: "lab" }, g.capability.replace(/_/g, " ")));
    gateGrid.appendChild(el("span", { style: { color: "var(--ink-2)" } }, g.rule));
    gateGrid.appendChild(el("span", { class: "mo", style: { color: "var(--ink-muted)" } }, g.verifier));
  }
  gateCard.appendChild(gateGrid);
  view.appendChild(gateCard);

  // Spec preview (collapsed-ish)
  view.appendChild(el("h3", {}, "training plan"));
  const planCard = el("div", { class: "card" });
  const plan = el("div", {
    style: { display: "grid", gridTemplateColumns: "max-content 1fr", gap: "6px 16px", fontSize: "13px" },
  });
  for (const [k, v] of [
    ["objective", d.spec.train.objective || d.spec.train.method],
    ["lora rank", d.spec.train.lora_rank],
    ["max steps", d.spec.train.max_steps || "auto"],
    ["benchmark", d.spec.eval.benchmark],
    ["eval examples", d.spec.eval.max_examples],
    ["target score", d.spec.eval.target_score],
    ["foundry", d.spec.foundry.enabled ? `enabled · ${d.spec.foundry.scenarios_per_brief} scenarios` : "off"],
  ]) {
    plan.appendChild(el("span", { class: "lab" }, k));
    plan.appendChild(el("span", { class: "mo" }, String(v)));
  }
  planCard.appendChild(plan);
  view.appendChild(planCard);

  const actions = el("div", { style: { display: "flex", gap: "10px", marginTop: "16px" } });
  actions.appendChild(el("button", { class: "btn ghost", onclick: () => { wizardState.step = 4; route(); } }, "← edit task"));
  const startBtn = el("button", { class: "btn primary" }, "start training →");
  startBtn.addEventListener("click", async () => {
    startBtn.setAttribute("disabled", "true");
    startBtn.textContent = "starting…";
    try {
      const r = await fetch("/api/wizard/start", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ spec: d.spec }),
      });
      if (!r.ok) throw new Error(await r.text());
      const body = await r.json();
      wizardState.handle_id = body.run_id;
      wizardState.run_id = body.run_id;
      wizardState.step = 6;
      route();
    } catch (e) {
      startBtn.removeAttribute("disabled");
      startBtn.textContent = "start training →";
      const err = el("p", { class: "micro", style: { color: "var(--fail)" } }, `start failed: ${e}`);
      view.appendChild(err);
    }
  });
  actions.appendChild(startBtn);
  view.appendChild(actions);
}

function renderStepTrain() {
  const card = el("div", { class: "card" });
  card.appendChild(el("div", { class: "lab" }, "training"));
  card.appendChild(el("p", { class: "micro" }, "live stream from the run loop. accepted vs rejected rows from the foundry, then sft steps, then eval."));

  const counters = el("div", {
    id: "wizard-counters",
    style: { display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "12px", margin: "14px 0" },
  });
  for (const [lab, id] of [
    ["accepted", "ctr-accepted"],
    ["rejected", "ctr-rejected"],
    ["train step", "ctr-step"],
    ["eval score", "ctr-score"],
  ]) {
    const cell = el("div", { style: { background: "var(--surface-2)", borderRadius: "var(--radius)", padding: "12px 14px" } });
    cell.appendChild(el("div", { class: "lab" }, lab));
    cell.appendChild(el("div", { class: "mo", style: { fontSize: "20px", color: "var(--ink)" } }, [el("span", { id }, "—")]));
    counters.appendChild(cell);
  }
  card.appendChild(counters);

  const stageEl = el("div", { class: "lab", id: "wizard-stage" }, "queued");
  card.appendChild(stageEl);

  const log = el("div", { class: "log", id: "wizard-log", style: { maxHeight: "320px", marginTop: "10px" } });
  card.appendChild(log);

  view.appendChild(card);

  if (!wizardState.run_done) attachWizardLive();
  else renderWizardDone(card);
}

function attachWizardLive() {
  const run_id = wizardState.run_id;
  if (!run_id) return;
  const stageEl = document.getElementById("wizard-stage");
  const logEl = document.getElementById("wizard-log");

  let accepted = 0, rejected = 0;
  let bestStep = 0;
  let bestScore = null;

  const proto = location.protocol === "https:" ? "wss:" : "ws:";

  // The actual run id may differ from the handle once execute() begins.
  // Poll the handle once a second until it reports `done`, then refresh
  // and pull the actual run id out of `extra`.
  let pollHandle;
  const pollOnce = async () => {
    try {
      const h = await api(`/api/wizard/runs/${wizardState.handle_id}`);
      if (h.status === "done") {
        wizardState.run_done = true;
        wizardState.run_result = h.extra;
        wizardState.run_id = h.extra.actual_run_id || wizardState.run_id;
        clearInterval(pollHandle);
        renderWizardDone(document.querySelector(".card"));
      } else if (h.status === "error") {
        stageEl.textContent = `error: ${h.error}`;
        stageEl.style.color = "var(--fail)";
        clearInterval(pollHandle);
      }
    } catch {}
  };
  pollHandle = setInterval(pollOnce, 1500);

  // Tail events directly from the SQLite events table.
  const ws = new WebSocket(`${proto}//${location.host}/api/runs/${run_id}/live`);
  ws.onmessage = (m) => {
    const e = JSON.parse(m.data);
    if (e._status) return;
    const line = el("div", {});
    line.appendChild(el("span", { class: "ts" }, fmt.hms(e.created_at)));
    line.appendChild(el("span", { class: "stg" }, e.stage));
    line.appendChild(document.createTextNode(e.message));
    logEl.appendChild(line);
    logEl.scrollTop = logEl.scrollHeight;
    stageEl.textContent = `stage: ${e.stage}`;

    // Lift counters from the message strings the run loop emits.
    const acc = /accepted (\d+)/.exec(e.message);
    if (acc) { accepted = +acc[1]; document.getElementById("ctr-accepted").textContent = accepted; }
    const rej = /rejected (\d+)/.exec(e.message);
    if (rej) { rejected = +rej[1]; document.getElementById("ctr-rejected").textContent = rejected; }
    const stp = /step (\d[\d,]*)/.exec(e.message);
    if (stp) { bestStep = +stp[1].replace(/,/g, ""); document.getElementById("ctr-step").textContent = bestStep.toLocaleString(); }
    const sc = /overall\s+([0-9.]+)/.exec(e.message);
    if (sc) { bestScore = +sc[1]; document.getElementById("ctr-score").textContent = bestScore.toFixed(3); }
  };
  ws.onerror = () => { stageEl.textContent = "websocket error"; };
}

function renderWizardDone(card) {
  const r = wizardState.run_result || {};
  const summary = el("div", { class: "card", style: { marginTop: "12px" } });
  summary.appendChild(el("div", { class: "lab" }, "done"));
  const dl = el("dl", { class: "kv", style: { marginTop: "6px" } });
  for (const [k, v] of [
    ["baseline", r.baseline_overall != null ? r.baseline_overall.toFixed(3) : "—"],
    ["best", r.best_overall != null ? r.best_overall.toFixed(3) : "—"],
    ["Δ", r.best_overall != null && r.baseline_overall != null ?
      fmt.delta(r.best_overall - r.baseline_overall) : "—"],
    ["best iter", r.best_iteration ?? "—"],
    ["adapter", r.adapter_path || "—"],
  ]) {
    dl.appendChild(el("dt", {}, k));
    dl.appendChild(el("dd", { class: "mo" }, v));
  }
  summary.appendChild(dl);

  const actions = el("div", { style: { display: "flex", gap: "10px", marginTop: "12px" } });
  const saveBtn = el("button", { class: "btn primary" }, wizardState.saved_to_hub ? "saved ✓" : "save to elmo model hub");
  if (wizardState.saved_to_hub) saveBtn.setAttribute("disabled", "true");
  saveBtn.addEventListener("click", async () => {
    saveBtn.setAttribute("disabled", "true");
    saveBtn.textContent = "saving…";
    try {
      const display = `${wizardState.picked.display_name} → ${wizardState.discovered.spec.name}`;
      const resp = await fetch("/api/hub/save", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({
          display_name: display,
          base_hf_id: wizardState.picked.hf_id_mlx,
          adapter_path: r.adapter_path || "",
          task_name: wizardState.discovered.spec.name,
          run_id: wizardState.run_id,
          baseline_score: r.baseline_overall,
          final_score: r.best_overall,
        }),
      });
      if (!resp.ok) throw new Error(await resp.text());
      wizardState.saved_to_hub = true;
      saveBtn.textContent = "saved ✓";
    } catch (e) {
      saveBtn.textContent = "save failed — retry";
      saveBtn.removeAttribute("disabled");
    }
  });
  actions.appendChild(saveBtn);
  actions.appendChild(el("button", {
    class: "btn ghost",
    onclick: () => { location.hash = `#/runs/${wizardState.run_id}`; },
  }, "open run →"));
  actions.appendChild(el("button", {
    class: "btn ghost",
    onclick: () => { location.hash = "#/hub"; },
  }, "model hub →"));
  summary.appendChild(actions);
  card.appendChild(summary);
}

// --- model hub -----------------------------------------------------------
async function renderHub() {
  setActiveNav("nav-hub");
  view.innerHTML = "";
  view.appendChild(el("h1", {}, "model hub"));
  view.appendChild(el("p", { class: "micro" }, "base models you've downloaded and fine-tunes you've saved."));
  let data;
  try { data = await api("/api/hub"); } catch { data = { entries: [] }; }
  const bases = data.entries.filter((e) => e.kind === "base");
  const tunes = data.entries.filter((e) => e.kind === "fine-tuned");

  view.appendChild(el("h3", {}, `base models · ${bases.length}`));
  if (!bases.length) {
    view.appendChild(el("p", { class: "micro" }, "none yet. start the onboarding wizard to download one."));
  } else {
    const grid = el("div", { class: "card", style: { padding: "0" } });
    for (const b of bases) {
      grid.appendChild(hubRow(b));
    }
    view.appendChild(grid);
  }

  view.appendChild(el("h3", {}, `fine-tuned · ${tunes.length}`));
  if (!tunes.length) {
    view.appendChild(el("p", { class: "micro" }, "none yet. finish a run and save the result to the hub."));
  } else {
    const grid = el("div", { class: "card", style: { padding: "0" } });
    for (const t of tunes) {
      grid.appendChild(hubRow(t));
    }
    view.appendChild(grid);
  }
}

function hubRow(entry) {
  const a = el("div", { class: "row", style: { gridTemplateColumns: "1fr 200px 80px 80px 110px" } });
  const left = el("div", {}, [
    el("div", { class: "task" }, entry.display_name),
    el("div", { class: "micro" }, entry.kind === "base" ? entry.hf_id : `${entry.task_name} · base ${entry.hf_id}`),
  ]);
  a.appendChild(left);
  a.appendChild(el("span", { class: "mo when" }, entry.kind));
  a.appendChild(el("span", { class: "mo base" }, `${entry.size_gb || 0} GB`));
  const delta = entry.final_score != null && entry.baseline_score != null ? (entry.final_score - entry.baseline_score) : null;
  a.appendChild(el("span", { class: `mo delta ${delta != null && delta >= 0 ? "up" : delta != null ? "down" : ""}` },
    delta != null ? fmt.delta(delta) : "—"));
  a.appendChild(el("span", { class: "mo when" }, fmt.ts(entry.added_at)));
  return a;
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
  if (h === "#/onboard") return renderOnboard();
  if (h === "#/hub") return renderHub();
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
