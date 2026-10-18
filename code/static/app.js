const state = { requests: [], filtered: [], activeId: null, sessionId: "sess-" + Math.random().toString(36).slice(2) };

function fmtAmount(amount, currency) {
  if (amount === null || amount === undefined || isNaN(amount)) return "—";
  const decimals = currency === "IDR" ? 0 : 2;
  const n = Number(amount).toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  return `${currency || ""} ${n}`.trim();
}

function statusStampClass(status) {
  return { affordable_now: "now", affordable_with_plan: "plan", affordable_later: "later", not_affordable: "not" }[status] || "";
}
function statusLabel(status) {
  return { affordable_now: "Affordable now", affordable_with_plan: "Affordable with a plan", affordable_later: "Affordable later", not_affordable: "Not affordable" }[status] || status || "";
}

/* --------------------------------------------------------------- requests */
async function loadRequests() {
  const res = await fetch("/api/requests");
  state.requests = await res.json();
  state.filtered = state.requests;
  renderRail();
  loadSummary();
}

async function loadSummary() {
  try {
    const res = await fetch("/api/output-summary");
    if (!res.ok) return;
    const data = await res.json();
    document.getElementById("stat-total").textContent = data.row_count ?? "—";
    const s = data.affordability_status || {};
    document.getElementById("stat-now").textContent = s.affordable_now || 0;
    document.getElementById("stat-plan").textContent = (s.affordable_with_plan || 0) + (s.affordable_later || 0);
    document.getElementById("stat-not").textContent = s.not_affordable || 0;
  } catch (e) { /* output.csv not generated yet -- fine */ }
}

function renderRail() {
  const list = document.getElementById("rail-list");
  if (!state.filtered.length) {
    list.innerHTML = '<div class="rail-empty">No matching requests.</div>';
    return;
  }
  list.innerHTML = state.filtered.map(r => `
    <div class="rail-item ${r.request_id === state.activeId ? "active" : ""}" data-id="${r.request_id}">
      <div>
        <div class="rail-item-type">${r.request_type.replace(/_/g, " ")}</div>
        <div class="rail-item-id">${r.request_id} · ${r.user_id}</div>
      </div>
      <div class="rail-item-amt">${fmtAmount(r.requested_amount, r.currency)}</div>
    </div>
  `).join("");
  list.querySelectorAll(".rail-item").forEach(el => {
    el.addEventListener("click", () => openRequest(el.dataset.id));
  });
}

document.getElementById("search").addEventListener("input", (e) => {
  const q = e.target.value.toLowerCase().trim();
  state.filtered = !q ? state.requests : state.requests.filter(r =>
    r.request_id.toLowerCase().includes(q) ||
    r.user_id.toLowerCase().includes(q) ||
    r.request_type.toLowerCase().includes(q)
  );
  renderRail();
});

/* --------------------------------------------------------------- statement */
async function openRequest(requestId) {
  state.activeId = requestId;
  renderRail();
  const statement = document.getElementById("statement");
  statement.innerHTML = '<div class="loading">Running the decision engine…</div>';

  const req = state.requests.find(r => r.request_id === requestId);
  const [decisionRes, forecastRes] = await Promise.all([
    fetch("/api/affordability", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ request_id: requestId }) }),
    fetch(`/api/forecast/${requestId}`),
  ]);
  const decision = await decisionRes.json();
  const forecast = await forecastRes.json();
  renderStatement(req, decision, forecast);

  document.getElementById("chat-context").textContent = `${requestId} · ${req.user_id} · ${req.request_type.replace(/_/g, " ")}`;
  resetChat();
  renderChatSuggestions();
}

function renderStatement(req, d, f) {
  const currency = req.currency;
  const statement = document.getElementById("statement");

  const planRows = (d.payment_plan && d.payment_plan !== "none")
    ? d.payment_plan.split("|").map(p => {
        const [date, amt] = p.split(":");
        return `<tr><td>${date}</td><td class="num">${fmtAmount(parseFloat(amt), currency)}</td></tr>`;
      }).join("")
    : `<tr><td colspan="2" class="changes-none">No payment recommended</td></tr>`;

  const changesHtml = (d.spending_changes_needed && d.spending_changes_needed !== "none")
    ? `<ul class="changes-list">${d.spending_changes_needed.split("|").map(c => {
        if (c.startsWith("stop:")) return `<li>Stop <code>${c.slice(5)}</code></li>`;
        const [, eventId, amt] = c.split(":");
        return `<li>Reduce <code>${eventId}</code> to ${fmtAmount(parseFloat(amt), currency)}</li>`;
      }).join("")}</ul>`
    : `<p class="changes-none">No spending changes required.</p>`;

  statement.innerHTML = `
    <div class="st-header">
      <div class="st-eyebrow">${req.request_id} · ${req.user_id} · ${req.request_type.replace(/_/g, " ")}</div>
      <div class="st-question">${req.request_text || "(no request text provided)"}</div>
      <div class="st-meta">
        <div class="st-meta-item">Requested <b>${fmtAmount(req.requested_amount, currency)}</b></div>
        <div class="st-meta-item">Request date <b>${req.request_date}</b></div>
        <div class="st-meta-item">Wanted by <b>${req.desired_completion_date}</b></div>
        <div class="st-meta-item">Partial payment allowed <b>${req.allows_partial_payment ? "yes" : "no"}</b></div>
      </div>
    </div>

    <div class="st-grid">
      <div class="panel">
        <span class="stamp ${statusStampClass(d.affordability_status)}">${statusLabel(d.affordability_status).toUpperCase()}</span>
        <div class="decision-headline">${(d.recommended_payment_method || "").replace(/_/g, " ")}</div>
        <div class="decision-sub">Safe to pay today: ${fmtAmount(d.amount_safe_to_pay, currency)}
          ${d.earliest_date_for_full_payment ? ` · Earliest full payment: ${d.earliest_date_for_full_payment}` : ""}</div>

        <h3 style="margin-top:20px">Payment plan</h3>
        <table class="ledger-table"><thead><tr><th>Date</th><th style="text-align:right">Amount</th></tr></thead>
          <tbody>${planRows}</tbody></table>

        <h3 style="margin-top:20px">Spending changes</h3>
        ${changesHtml}

        <div class="decision-explanation">${d.decision_explanation || ""}</div>
      </div>

      <div class="panel">
        <h3>90-day balance forecast</h3>
        <div class="figures">
          <div class="figure"><span class="figure-val">${fmtAmount(f.min_balance, currency)}</span><span class="figure-label">minimum to keep</span></div>
          <div class="figure"><span class="figure-val">${fmtAmount(f.amount_safe0, currency)}</span><span class="figure-label">baseline safe today</span></div>
        </div>
        <div class="chart-wrap">
          <canvas id="forecast-canvas" width="640" height="170"></canvas>
        </div>
        <div class="chart-legend">
          <span><span class="legend-swatch" style="background:#c6a15b"></span>projected balance</span>
          <span><span class="legend-swatch" style="background:#bd5c50"></span>minimum to keep</span>
        </div>
      </div>
    </div>
  `;

  drawForecast(f);
}

function drawForecast(f) {
  const canvas = document.getElementById("forecast-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 640, h = 170;
  canvas.width = w * dpr; canvas.height = h * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const series = f.daily_balance || [];
  if (!series.length) return;
  const minB = f.min_balance || 0;
  const allVals = series.concat([minB]);
  const lo = Math.min(...allVals), hi = Math.max(...allVals);
  const pad = (hi - lo) * 0.1 || 1;
  const yMin = lo - pad, yMax = hi + pad;

  const x = i => (i / (series.length - 1)) * (w - 20) + 10;
  const y = v => h - 14 - ((v - yMin) / (yMax - yMin)) * (h - 28);

  ctx.strokeStyle = "#bd5c50";
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(10, y(minB));
  ctx.lineTo(w - 10, y(minB));
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = "#c6a15b";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  series.forEach((v, i) => { const px = x(i), py = y(v); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); });
  ctx.stroke();

  ctx.fillStyle = "#ece7db";
  ctx.beginPath();
  ctx.arc(x(0), y(series[0]), 2.5, 0, Math.PI * 2);
  ctx.fill();
}

document.getElementById("run-batch-btn").addEventListener("click", async () => {
  const btn = document.getElementById("run-batch-btn");
  btn.disabled = true;
  btn.textContent = "Running batch…";
  await fetch("/api/run-batch", { method: "POST" });
  const poll = setInterval(async () => {
    const res = await fetch("/api/run-batch/status");
    const status = await res.json();
    if (!status.running) {
      clearInterval(poll);
      btn.disabled = false;
      btn.textContent = "Run full 250-request batch";
      loadSummary();
    }
  }, 800);
});

/* --------------------------------------------------------------- chat */
const SUGGESTIONS = [
  "Why did you recommend this?",
  "Can I pay this today?",
  "When can I safely pay in full?",
  "Can I use installments?",
  "What expenses are affecting this?",
  "Can I reduce my flexible expenses?",
  "Which payment option is safest?",
];

function resetChat() {
  document.getElementById("chat-log").innerHTML = `
    <div class="chat-msg chat-msg-bot">Ask me anything about this request — I'll answer using the
      same decision engine shown on the left.</div>`;
}

function renderChatSuggestions() {
  const box = document.getElementById("chat-suggestions");
  box.innerHTML = SUGGESTIONS.map(s => `<span class="chip">${s}</span>`).join("");
  box.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => sendChat(chip.textContent));
  });
}

function appendChatMsg(role, html) {
  const log = document.getElementById("chat-log");
  const div = document.createElement("div");
  div.className = `chat-msg chat-msg-${role}`;
  div.innerHTML = html;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function factsToHtml(facts) {
  if (!facts || !Object.keys(facts).length) return "";
  const skip = new Set(["top_expenses", "reducible_expenses", "installment_options", "safe_candidates"]);
  const rows = Object.entries(facts).filter(([k, v]) => !skip.has(k) && v !== null && v !== undefined);
  if (!rows.length) return "";
  return `<div class="chat-msg-facts">${rows.map(([k, v]) =>
    `<div><span>${k.replace(/_/g, " ")}</span><span>${v}</span></div>`).join("")}</div>`;
}

async function sendChat(message) {
  if (!message) return;
  appendChatMsg("user", message);
  const input = document.getElementById("chat-input");
  input.value = "";
  const btn = document.querySelector("#chat-form button");
  btn.disabled = true;

  try {
    const res = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, request_id: state.activeId, session_id: state.sessionId }),
    });
    const data = await res.json();
    let html = data.answer || "(no answer)";
    if (data.payment_plan) html += `<div class="chat-msg-plan">Plan: ${data.payment_plan}</div>`;
    html += factsToHtml(data.facts);
    appendChatMsg("bot", html);
  } catch (e) {
    appendChatMsg("bot", "Something went wrong reaching the chat endpoint.");
  } finally {
    btn.disabled = false;
  }
}

document.getElementById("chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const input = document.getElementById("chat-input");
  sendChat(input.value.trim());
});

loadRequests();
