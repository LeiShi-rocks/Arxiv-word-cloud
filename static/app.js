/* ── State ────────────────────────────────────────────────── */
let selectedCategories = new Set();
let currentMode = "both";

/* ── DOM refs ─────────────────────────────────────────────── */
const form             = document.getElementById("search-form");
const submitBtn        = document.getElementById("submit-btn");
const btnText          = submitBtn.querySelector(".btn-text");
const btnSpinner       = submitBtn.querySelector(".btn-spinner");
const formError        = document.getElementById("form-error");
const maxInput         = document.getElementById("max-results");
const maxDisplay       = document.getElementById("max-results-display");
const modeInput        = document.getElementById("mode");
const modeTabs         = document.querySelectorAll(".mode-tab");
const emptyState       = document.getElementById("empty-state");
const output           = document.getElementById("output");
const outputMeta       = document.getElementById("output-meta");
const wcWordsSec       = document.getElementById("wc-words-section");
const wcPhrasesSec     = document.getElementById("wc-phrases-section");
const wcTrendingSec    = document.getElementById("wc-trending-section");
const wcWordsImg       = document.getElementById("wc-words-img");
const wcPhrasesImg     = document.getElementById("wc-phrases-img");
const wcTrendingImg    = document.getElementById("wc-trending-img");
const trendingWindows  = document.getElementById("trending-windows");
const trendingTableWrap= document.getElementById("trending-table-wrap");
const paperList        = document.getElementById("paper-list");
const paperCount       = document.getElementById("paper-count-label");
const clearCats        = document.getElementById("clear-cats");
const catsContainer    = document.getElementById("categories-container");
const trendingHint     = document.getElementById("trending-date-hint");

/* ── Helpers ──────────────────────────────────────────────── */
function setLoading(on) {
  submitBtn.disabled = on;
  btnText.hidden = on;
  btnSpinner.hidden = !on;
}

function showError(msg) {
  formError.textContent = msg;
  formError.hidden = false;
}

function hideError() {
  formError.hidden = true;
  formError.textContent = "";
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatDate(isoStr) {
  if (!isoStr) return "";
  const d = new Date(isoStr + "T00:00:00Z");
  return d.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" });
}

/* ── Categories ───────────────────────────────────────────── */
async function loadCategories() {
  try {
    const res = await fetch("/api/categories");
    const data = await res.json();
    renderCategories(data);
  } catch {
    catsContainer.innerHTML = '<div class="category-loading">Failed to load categories.</div>';
  }
}

function renderCategories(data) {
  catsContainer.innerHTML = "";
  for (const [group, cats] of Object.entries(data)) {
    const groupEl = document.createElement("div");
    groupEl.className = "cat-group";
    groupEl.innerHTML = `<div class="cat-group-title">${escapeHtml(group)}</div><div class="cat-chips"></div>`;
    const chips = groupEl.querySelector(".cat-chips");
    cats.forEach(cat => {
      const chip = document.createElement("span");
      chip.className = "cat-chip";
      chip.textContent = cat;
      chip.dataset.cat = cat;
      chip.addEventListener("click", () => toggleCategory(cat, chip));
      chips.appendChild(chip);
    });
    catsContainer.appendChild(groupEl);
  }
}

function toggleCategory(cat, el) {
  if (selectedCategories.has(cat)) {
    selectedCategories.delete(cat);
    el.classList.remove("selected");
  } else {
    selectedCategories.add(cat);
    el.classList.add("selected");
  }
}

clearCats.addEventListener("click", () => {
  selectedCategories.clear();
  document.querySelectorAll(".cat-chip.selected").forEach(el => el.classList.remove("selected"));
});

/* ── Range slider ─────────────────────────────────────────── */
maxInput.addEventListener("input", () => {
  maxDisplay.textContent = maxInput.value;
});

/* ── Mode tabs ────────────────────────────────────────────── */
modeTabs.forEach(tab => {
  tab.addEventListener("click", () => {
    modeTabs.forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    currentMode = tab.dataset.mode;
    modeInput.value = currentMode;
    trendingHint.hidden = currentMode !== "trending";
  });
});

/* ── Form Submit ──────────────────────────────────────────── */
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();

  const query = document.getElementById("query").value.trim();
  const startDate = document.getElementById("start-date").value;
  const endDate = document.getElementById("end-date").value;
  const maxResults = parseInt(maxInput.value, 10);

  if (!query && selectedCategories.size === 0) {
    showError("Please enter a search topic or select at least one category.");
    return;
  }
  if (startDate && endDate && startDate > endDate) {
    showError("Start date must be before end date.");
    return;
  }

  setLoading(true);
  emptyState.hidden = true;
  output.hidden = true;

  const payload = {
    query,
    start_date: startDate || null,
    end_date: endDate || null,
    categories: [...selectedCategories],
    max_results: maxResults,
    mode: currentMode,
  };

  try {
    const res = await fetch("/api/wordcloud", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Unknown error" }));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const data = await res.json();
    renderOutput(data, payload);
  } catch (err) {
    showError(err.message || "Failed to fetch data. Please try again.");
    emptyState.hidden = false;
  } finally {
    setLoading(false);
  }
});

/* ── Render Output ────────────────────────────────────────── */
function renderOutput(data, payload) {
  const isTrending = payload.mode === "trending";

  // Meta bar
  const catStr = payload.categories.length > 0
    ? payload.categories.slice(0, 4).join(", ") + (payload.categories.length > 4 ? "…" : "")
    : "All categories";

  let metaHtml = `<span>Found</span>
    <span class="meta-pill green">${data.total} paper${data.total !== 1 ? "s" : ""}</span>
    <span>for</span>
    <span class="meta-pill blue">${escapeHtml(payload.query || "(no query)")}</span>
    <span>${escapeHtml(catStr)}</span>`;

  if (isTrending && data.n_new !== undefined) {
    metaHtml += `
      <span>·</span>
      <span class="meta-pill orange">${data.n_new} recent</span>
      <span class="meta-pill gray">${data.n_old} baseline</span>`;
  } else {
    const dateStr = payload.start_date || payload.end_date
      ? `${payload.start_date || "—"} → ${payload.end_date || "—"}`
      : "All dates";
    metaHtml += `<span>·</span><span>${escapeHtml(dateStr)}</span>`;
  }

  outputMeta.innerHTML = metaHtml;

  // Show/hide sections based on mode
  wcWordsSec.hidden    = !data.wordcloud_words;
  wcPhrasesSec.hidden  = !data.wordcloud_phrases;
  wcTrendingSec.hidden = !data.wordcloud_trending;

  if (data.wordcloud_words) {
    wcWordsImg.src = `data:image/png;base64,${data.wordcloud_words}`;
  }
  if (data.wordcloud_phrases) {
    wcPhrasesImg.src = `data:image/png;base64,${data.wordcloud_phrases}`;
  }
  if (data.wordcloud_trending) {
    wcTrendingImg.src = `data:image/png;base64,${data.wordcloud_trending}`;
    renderTrendingWindows(data);
    renderTrendingTable(data.trending_terms || []);
  }

  // Paper list
  paperCount.textContent = `${data.total} Paper${data.total !== 1 ? "s" : ""}`;
  if (isTrending && data.n_new) {
    paperCount.textContent += ` (${data.n_new} recent · ${data.n_old} baseline)`;
  }
  paperList.innerHTML = "";
  data.papers.forEach(paper => {
    const card = document.createElement("div");
    card.className = "paper-card";
    const cats = paper.categories.slice(0, 3).map(c =>
      `<span class="paper-cat">${escapeHtml(c)}</span>`
    ).join("");
    const authorsStr = paper.authors.length > 0
      ? escapeHtml(paper.authors.slice(0, 3).join(", ")) + (paper.authors.length > 3 ? " et al." : "")
      : "Unknown authors";
    card.innerHTML = `
      <div class="paper-title">
        <a href="${escapeHtml(paper.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(paper.title)}</a>
      </div>
      <div class="paper-meta">
        <span>${authorsStr}</span>
        <span>·</span>
        <span class="paper-date">${formatDate(paper.published)}</span>
        <span>·</span>
        <div class="paper-cats">${cats}</div>
      </div>
      <div class="paper-abstract">${escapeHtml(paper.abstract)}</div>
    `;
    paperList.appendChild(card);
  });

  output.hidden = false;
}

function renderTrendingWindows(data) {
  if (!data.trending_window_new) { trendingWindows.innerHTML = ""; return; }
  trendingWindows.innerHTML = `
    <span class="tw-label tw-new">Recent</span>
    <span class="tw-range">${escapeHtml(data.trending_window_new)}</span>
    <span class="tw-vs">vs.</span>
    <span class="tw-label tw-old">Baseline</span>
    <span class="tw-range">${escapeHtml(data.trending_window_old)}</span>
  `;
}

function renderTrendingTable(terms) {
  if (!terms || terms.length === 0) { trendingTableWrap.innerHTML = ""; return; }

  const maxScore = terms[0]?.score || 1;
  const rows = terms.slice(0, 30).map((t, i) => {
    const pct = Math.round((t.score / maxScore) * 100);
    return `
      <tr>
        <td class="tt-rank">${i + 1}</td>
        <td class="tt-term">${escapeHtml(t.term)}</td>
        <td class="tt-bar-cell">
          <div class="tt-bar-bg">
            <div class="tt-bar-fill" style="width:${pct}%"></div>
          </div>
        </td>
        <td class="tt-score">${t.score.toFixed(1)}</td>
      </tr>`;
  }).join("");

  trendingTableWrap.innerHTML = `
    <details class="trending-table-details" open>
      <summary class="trending-table-summary">Top Trending Terms (ranked by enrichment score)</summary>
      <div class="tt-scroll">
        <table class="trending-table">
          <thead><tr><th>#</th><th>Term</th><th>Trend strength</th><th>Score</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </details>
  `;
}

/* ── Init ─────────────────────────────────────────────────── */
loadCategories();

(function setDefaultDates() {
  const today = new Date();
  const sixMonthsAgo = new Date();
  sixMonthsAgo.setMonth(today.getMonth() - 6);
  const fmt = d => d.toISOString().split("T")[0];
  document.getElementById("end-date").value = fmt(today);
  document.getElementById("start-date").value = fmt(sixMonthsAgo);
})();
