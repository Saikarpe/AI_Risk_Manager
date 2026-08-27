/**
 * AI Risk Manager — Frontend Logic
 * Handles the order scoring form, result display, and metric animations.
 */

// ===== DOM Ready =====
document.addEventListener("DOMContentLoaded", () => {
  loadMetrics();
  animateFeatureBars();
  setupSmoothScroll();
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

  // Confusion matrix
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

// ===== Order Scoring =====
async function scoreOrder(event) {
  event.preventDefault();

  const form = event.target;
  const submitBtn = form.querySelector(".form-submit");
  const spinner = document.getElementById("spinner");
  const resultContent = document.getElementById("result-content");
  const resultPlaceholder = document.getElementById("result-placeholder");

  // Build payload
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

  // Validate
  if (isNaN(payload.order_value) || payload.order_value <= 0) {
    showToast("Please enter a valid order value.");
    return;
  }

  // Show loading
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

function displayResult(data) {
  const resultContent = document.getElementById("result-content");
  const resultPlaceholder = document.getElementById("result-placeholder");

  resultPlaceholder.style.display = "none";
  resultContent.classList.add("visible");

  // Score
  const score = data.risk_score;
  const scoreEl = document.getElementById("gauge-score");
  const gaugeEl = document.getElementById("gauge-circle");
  const badgeEl = document.getElementById("risk-badge");

  scoreEl.textContent = (score * 100).toFixed(1) + "%";

  // Determine risk level and colors
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

  // Detail values
  setIfExists("res-order-id", data.order_id);
  setIfExists("res-risk-score", score.toFixed(4));
  setIfExists("res-threshold", data.threshold_used);
  setIfExists("res-flagged", data.flagged ? "Yes" : "No");
}

// ===== Toast Notifications =====
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

// ===== Feature Bar Animations =====
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

// ===== Smooth Scroll for Nav Links =====
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
