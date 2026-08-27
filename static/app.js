/**
 * AI Risk Manager — Frontend Logic
 * Single-order scoring, batch CSV scoring, metric loading, animations.
 */

const REQUIRED_COLUMNS = [
  "order_id",
  "category",
  "customer_type",
  "payment_method",
  "delivery_pincode_risk_tier",
  "day_of_week",
  "order_value",
  "customer_order_count",
  "discount_percent",
  "order_hour",
];

// Batch state (kept in module scope so pagination + download can access it)
let BATCH_RESULTS = [];
let BATCH_SORT = { key: "risk_score", dir: "desc" };
const PAGE_SIZE = 25;
let PAGE_INDEX = 0;

// ===== DOM Ready =====
document.addEventListener("DOMContentLoaded", () => {
  loadMetrics();
  animateFeatureBars();
  setupSmoothScroll();
  setupBatchUpload();
  setupTableSort();
  setupPager();
  setupSampleDownload();
});

// ===== Load Metrics from API =====
async function loadMetrics() {
  try {
    const res = await fetch("/metrics");
    if (!res.ok) throw new Error("Metrics not available");
    const data = await res.json();
    populateMetrics(data);
  } catch (err) {
    console.warn("Could not load metrics:", err.message);
  }
}

function populateMetrics(m) {
  setIfExists("metric-roc-auc", m.roc_auc?.toFixed(4));
  setIfExists("metric-pr-auc", m.pr_auc?.toFixed(4));
  setIfExists("metric-precision", (m.precision_at_threshold * 100).toFixed(1) + "%");
  setIfExists("metric-recall", (m.recall_at_threshold * 100).toFixed(1) + "%");
  setIfExists("metric-f1", m.f1_at_threshold?.toFixed(4));
  setIfExists("metric-threshold", m.chosen_threshold);
  setIfExists("metric-flag-rate", (m.flag_rate_at_threshold * 100).toFixed(1) + "%");
  setIfExists("metric-cost-savings", "₹" + formatNumber(m.cost_savings_vs_default));

  if (m.confusion_matrix_at_threshold) {
    const cm = m.confusion_matrix_at_threshold;
    setIfExists("cm-tn", formatNumber(cm.true_negative));
    setIfExists("cm-fp", formatNumber(cm.false_positive));
    setIfExists("cm-fn", formatNumber(cm.false_negative));
    setIfExists("cm-tp", formatNumber(cm.true_positive));
  }
}

function setIfExists(id, value) {
  const el = document.getElementById(id);
  if (el && value !== undefined) el.textContent = value;
}

function formatNumber(n) {
  if (n === undefined || n === null) return "—";
  return Number(n).toLocaleString("en-IN");
}

// ===== Single-Order Scoring =====
async function scoreOrder(event) {
  event.preventDefault();

  const form = event.target;
  const submitBtn = form.querySelector(".form-submit");
  const spinner = document.getElementById("spinner");
  const resultContent = document.getElementById("result-content");
  const resultPlaceholder = document.getElementById("result-placeholder");

  const payload = {
    order_id: form.order_id.value || "ORD-" + Date.now(),
    order_value: parseFloat(form.order_value.value),
    category: form.category.value,
    customer_type: form.customer_type.value,
    customer_order_count: parseInt(form.customer_order_count.value),
    payment_method: form.payment_method.value,
    delivery_pincode_risk_tier: form.delivery_pincode_risk_tier.value,
    discount_percent: parseFloat(form.discount_percent.value),
    order_hour: parseInt(form.order_hour.value),
    day_of_week: form.day_of_week.value,
  };

  if (isNaN(payload.order_value) || payload.order_value <= 0) {
    showToast("Please enter a valid order value.");
    return;
  }

  submitBtn.disabled = true;
  submitBtn.textContent = "Scoring…";
  spinner.style.display = "block";
  resultContent.classList.remove("visible");

  try {
    const res = await fetch("/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Server error" }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    displayResult(data);
  } catch (err) {
    showToast(err.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "⚡ Score This Order";
    spinner.style.display = "none";
  }
}
window.scoreOrder = scoreOrder;

function displayResult(data) {
  const resultContent = document.getElementById("result-content");
  const resultPlaceholder = document.getElementById("result-placeholder");

  resultPlaceholder.style.display = "none";
  resultContent.classList.add("visible");

  const score = data.risk_score;
  const scoreEl = document.getElementById("gauge-score");
  const gaugeEl = document.getElementById("gauge-circle");
  const badgeEl = document.getElementById("risk-badge");

  scoreEl.textContent = (score * 100).toFixed(1) + "%";

  gaugeEl.className = "gauge-circle";
  gaugeEl.style.setProperty("--score", score);
  if (data.flagged) {
    gaugeEl.classList.add(score >= 0.7 ? "high" : "medium");
    badgeEl.className = "risk-badge flagged";
    badgeEl.textContent = "⚠ FLAGGED — High Risk";
  } else {
    gaugeEl.classList.add("low");
    badgeEl.className = "risk-badge safe";
    badgeEl.textContent = "✓ CLEAR — Low Risk";
  }

  setIfExists("res-order-id", data.order_id);
  setIfExists("res-risk-score", score.toFixed(4));
  setIfExists("res-threshold", data.threshold_used);
  setIfExists("res-flagged", data.flagged ? "Yes" : "No");

  renderReasons(data.top_reasons || []);
}

// ===== Reasons rendering =====
function renderReasons(reasons) {
  const list = document.getElementById("reasons-list");
  if (!list) return;
  list.innerHTML = "";
  if (!reasons.length) {
    list.innerHTML = '<div class="reasons-empty">No explanations available.</div>';
    return;
  }
  const maxAbs = Math.max(...reasons.map((r) => Math.abs(r.impact)), 0.0001);
  reasons.forEach((r) => {
    const pct = (Math.abs(r.impact) / maxAbs) * 100;
    const dir = r.impact >= 0 ? "up" : "down";
    const row = document.createElement("div");
    row.className = "reason-row " + dir;
    row.innerHTML = `
      <div class="reason-label">
        <span class="reason-feature">${escapeHtml(r.feature)}</span>
        <span class="reason-value">${escapeHtml(String(r.value))}</span>
      </div>
      <div class="reason-bar-wrap">
        <div class="reason-bar ${dir}" style="width:${pct.toFixed(1)}%"></div>
      </div>
      <div class="reason-impact">${r.impact >= 0 ? "+" : ""}${r.impact.toFixed(3)}</div>
    `;
    list.appendChild(row);
  });
}

// ===== Batch CSV upload =====
function setupBatchUpload() {
  const dropZone = document.getElementById("drop-zone");
  const input = document.getElementById("csv-input");
  if (!dropZone || !input) return;

  input.addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (file) handleCsvFile(file);
  });

  ["dragenter", "dragover"].forEach((ev) =>
    dropZone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropZone.classList.add("drag-over");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dropZone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");
    })
  );
  dropZone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files?.[0];
    if (file) handleCsvFile(file);
  });
}

async function handleCsvFile(file) {
  const status = document.getElementById("batch-status");
  const results = document.getElementById("batch-results");
  const downloadBtn = document.getElementById("download-scored");
  status.textContent = `Uploading ${file.name} (${(file.size / 1024).toFixed(1)} KB)…`;
  results.hidden = true;
  downloadBtn.disabled = true;

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch("/score/csv", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Server error" }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    BATCH_RESULTS = data.results || [];
    PAGE_INDEX = 0;
    BATCH_SORT = { key: "risk_score", dir: "desc" };
    sortBatch();
    renderBatch();
    downloadBtn.disabled = BATCH_RESULTS.length === 0;
    const flagged = BATCH_RESULTS.filter((r) => r.flagged).length;
    status.innerHTML = `Scored <strong>${BATCH_RESULTS.length}</strong> orders — <strong>${flagged}</strong> flagged
      (${((flagged / BATCH_RESULTS.length) * 100).toFixed(1)}%).`;
    results.hidden = false;
  } catch (err) {
    status.textContent = "";
    showToast(err.message);
  }
}

function sortBatch() {
  const { key, dir } = BATCH_SORT;
  const sign = dir === "asc" ? 1 : -1;
  BATCH_RESULTS.sort((a, b) => {
    let av = a[key];
    let bv = b[key];
    if (typeof av === "string") av = av.toLowerCase();
    if (typeof bv === "string") bv = bv.toLowerCase();
    if (av < bv) return -1 * sign;
    if (av > bv) return 1 * sign;
    return 0;
  });
}

function renderBatch() {
  const tbody = document.getElementById("results-tbody");
  tbody.innerHTML = "";
  const start = PAGE_INDEX * PAGE_SIZE;
  const page = BATCH_RESULTS.slice(start, start + PAGE_SIZE);

  for (const r of page) {
    const tr = document.createElement("tr");
    tr.className = r.flagged ? "row-flagged" : "row-clear";
    tr.innerHTML = `
      <td class="mono">${escapeHtml(r.order_id)}</td>
      <td class="mono"><span class="score-pill ${riskClass(r.risk_score)}">${(r.risk_score * 100).toFixed(1)}%</span></td>
      <td>${r.flagged
        ? '<span class="pill flagged">Flagged</span>'
        : '<span class="pill clear">Clear</span>'}</td>
      <td class="reasons-cell">${renderReasonsCell(r.top_reasons || [])}</td>
    `;
    tbody.appendChild(tr);
  }

  const pages = Math.max(1, Math.ceil(BATCH_RESULTS.length / PAGE_SIZE));
  document.getElementById("page-info").textContent =
    `Page ${PAGE_INDEX + 1} of ${pages} · ${BATCH_RESULTS.length} orders`;
  document.getElementById("page-prev").disabled = PAGE_INDEX === 0;
  document.getElementById("page-next").disabled = PAGE_INDEX >= pages - 1;
}

function riskClass(score) {
  if (score >= 0.7) return "high";
  if (score >= 0.4) return "medium";
  return "low";
}

function renderReasonsCell(reasons) {
  if (!reasons.length) return '<span class="muted">—</span>';
  return reasons
    .map(
      (r) =>
        `<span class="chip ${r.impact >= 0 ? "up" : "down"}" title="impact ${r.impact.toFixed(3)}">
           ${r.impact >= 0 ? "↑" : "↓"} ${escapeHtml(r.feature)}
         </span>`
    )
    .join("");
}

function setupTableSort() {
  const table = document.getElementById("results-table");
  if (!table) return;
  table.querySelectorAll("th[data-sort]").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.sort;
      if (BATCH_SORT.key === key) {
        BATCH_SORT.dir = BATCH_SORT.dir === "asc" ? "desc" : "asc";
      } else {
        BATCH_SORT = { key, dir: "desc" };
      }
      sortBatch();
      renderBatch();
    });
  });
}

function setupPager() {
  const prev = document.getElementById("page-prev");
  const next = document.getElementById("page-next");
  if (prev)
    prev.addEventListener("click", () => {
      if (PAGE_INDEX > 0) {
        PAGE_INDEX--;
        renderBatch();
      }
    });
  if (next)
    next.addEventListener("click", () => {
      const pages = Math.ceil(BATCH_RESULTS.length / PAGE_SIZE);
      if (PAGE_INDEX < pages - 1) {
        PAGE_INDEX++;
        renderBatch();
      }
    });

  const dl = document.getElementById("download-scored");
  if (dl) dl.addEventListener("click", downloadScoredCsv);
}

function downloadScoredCsv() {
  if (!BATCH_RESULTS.length) return;
  const header = ["order_id", "risk_score", "flagged", "threshold_used", "top_reasons"];
  const lines = [header.join(",")];
  for (const r of BATCH_RESULTS) {
    const reasons = (r.top_reasons || [])
      .map((x) => `${x.feature}=${x.value}(${x.impact >= 0 ? "+" : ""}${x.impact})`)
      .join(" | ");
    lines.push(
      [
        csvCell(r.order_id),
        r.risk_score,
        r.flagged,
        r.threshold_used,
        csvCell(reasons),
      ].join(",")
    );
  }
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  triggerDownload(blob, "scored_orders.csv");
}

function csvCell(v) {
  const s = String(v ?? "");
  if (s.includes(",") || s.includes('"') || s.includes("\n")) {
    return '"' + s.replace(/"/g, '""') + '"';
  }
  return s;
}

function setupSampleDownload() {
  const link = document.getElementById("download-sample");
  if (!link) return;
  link.addEventListener("click", (e) => {
    e.preventDefault();
    const rows = [
      REQUIRED_COLUMNS.join(","),
      "ORD100001,electronics,new,COD,high,Fri,4999,1,25,22",
      "ORD100002,apparel,returning,UPI,low,Mon,1299,12,10,14",
      "ORD100003,home,new,card,medium,Sat,7499,2,35,20",
      "ORD100004,beauty,returning,wallet,low,Sun,899,8,5,11",
      "ORD100005,grocery,new,COD,high,Tue,349,1,0,9",
    ];
    const blob = new Blob([rows.join("\n")], { type: "text/csv" });
    triggerDownload(blob, "sample_orders.csv");
  });
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// ===== Utilities =====
function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function showToast(message) {
  let toast = document.getElementById("toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "toast";
    toast.className = "toast";
    document.body.appendChild(toast);
  }
  toast.textContent = "⚠ " + message;
  toast.classList.add("visible");
  setTimeout(() => toast.classList.remove("visible"), 4000);
}

function animateFeatureBars() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const bars = entry.target.querySelectorAll(".feature-bar");
          bars.forEach((bar) => {
            bar.style.width = bar.dataset.width;
          });
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.2 }
  );

  const featureList = document.querySelector(".feature-list");
  if (featureList) observer.observe(featureList);
}

function setupSmoothScroll() {
  document.querySelectorAll('.nav-link[href^="#"]').forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const target = document.querySelector(link.getAttribute("href"));
      if (target) {
        target.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  });
}
