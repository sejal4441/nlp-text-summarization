let benchmarkSamples = [];
let currentResults = null;

// DOM Elements
const inputText = document.getElementById("inputText");
const referenceText = document.getElementById("referenceText");
const docStats = document.getElementById("docStats");
const refStats = document.getElementById("refStats");
const modelChoice = document.getElementById("modelChoice");
const extractiveMethod = document.getElementById("extractiveMethod");
const maxLengthSlider = document.getElementById("maxLengthSlider");
const maxLengthVal = document.getElementById("maxLengthVal");
const sentenceSlider = document.getElementById("sentenceSlider");
const sentenceVal = document.getElementById("sentenceVal");
const summarizeBtn = document.getElementById("summarizeBtn");
const emptyState = document.getElementById("emptyState");
const loadingState = document.getElementById("loadingState");
const summariesContainer = document.getElementById("summariesContainer");
const rougeSection = document.getElementById("rougeSection");
const rougeTableBody = document.getElementById("rougeTableBody");
const presetChips = document.getElementById("presetChips");
const viewToggles = document.getElementById("viewToggles");

// Update word counts
function updateStats() {
  const words = inputText.value.trim() ? inputText.value.trim().split(/\s+/).length : 0;
  const chars = inputText.value.length;
  docStats.textContent = `${words} words • ${chars} chars`;

  const refWords = referenceText.value.trim() ? referenceText.value.trim().split(/\s+/).length : 0;
  refStats.textContent = `${refWords} words`;
}

inputText.addEventListener("input", updateStats);
referenceText.addEventListener("input", updateStats);

// Slider events
maxLengthSlider.addEventListener("input", (e) => {
  maxLengthVal.textContent = `${e.target.value} tokens`;
});

sentenceSlider.addEventListener("input", (e) => {
  sentenceVal.textContent = `${e.target.value} sentences`;
});

// Load Benchmark Samples
async function loadBenchmarkSamples() {
  try {
    const res = await fetch("/api/benchmark-samples");
    if (res.ok) {
      benchmarkSamples = await res.json();
      renderPresetChips();
      if (benchmarkSamples.length > 0) {
        selectSample(0);
      }
    }
  } catch (err) {
    console.warn("Could not load samples from API, fallback to default", err);
  }
}

function renderPresetChips() {
  presetChips.innerHTML = "";
  benchmarkSamples.forEach((sample, idx) => {
    const btn = document.createElement("button");
    btn.className = `chip ${idx === 0 ? "active" : ""}`;
    btn.textContent = `${sample.dataset.includes("CNN") ? "CNN/DM" : "News"}: ${sample.title.slice(0, 24)}...`;
    btn.addEventListener("click", () => {
      document.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
      btn.classList.add("active");
      selectSample(idx);
    });
    presetChips.appendChild(btn);
  });
}

function selectSample(idx) {
  if (benchmarkSamples[idx]) {
    const s = benchmarkSamples[idx];
    inputText.value = s.article;
    referenceText.value = s.highlights;
    updateStats();
  }
}

// View Toggles
viewToggles.querySelectorAll(".toggle-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    viewToggles.querySelectorAll(".toggle-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    const view = btn.dataset.view;
    if (view === "cards") {
      summariesContainer.classList.remove("hidden");
      rougeSection.classList.add("hidden");
    } else {
      summariesContainer.classList.add("hidden");
      rougeSection.classList.remove("hidden");
    }
  });
});

// Summarize action
summarizeBtn.addEventListener("click", async () => {
  const text = inputText.value.trim();
  if (!text) {
    alert("Please provide an article or document text to summarize.");
    return;
  }

  emptyState.classList.add("hidden");
  summariesContainer.classList.add("hidden");
  rougeSection.classList.add("hidden");
  loadingState.classList.remove("hidden");
  summarizeBtn.disabled = true;

  try {
    const payload = {
      text: text,
      reference: referenceText.value.trim() || null,
      model_choice: modelChoice.value,
      extractive_method: extractiveMethod.value,
      max_length: parseInt(maxLengthSlider.value),
      min_length: 30,
      extractive_sentences: parseInt(sentenceSlider.value)
    };

    const res = await fetch("/api/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Summarization failed");
    }

    const data = await res.json();
    currentResults = data;
    renderResults(data);
  } catch (err) {
    alert("Error generating summary: " + err.message);
  } finally {
    loadingState.classList.add("hidden");
    summarizeBtn.disabled = false;
  }
});

function renderResults(data) {
  summariesContainer.innerHTML = "";
  rougeTableBody.innerHTML = "";

  const summaries = data.summaries || {};
  const evaluation = data.evaluation || {};

  // Render cards
  for (const [name, info] of Object.entries(summaries)) {
    const card = document.createElement("div");
    card.className = "summary-card";

    const isExtractive = info.type === "extractive";
    const typeClass = isExtractive ? "type-extractive" : "type-abstractive";
    const typeLabel = isExtractive ? "Extractive" : "Abstractive";

    card.innerHTML = `
      <div class="summary-header">
        <div class="summary-title-wrapper">
          <span class="summary-title">${name}</span>
          <span class="type-pill ${typeClass}">${typeLabel}</span>
        </div>
        <div class="summary-meta">
          <span class="meta-badge">${info.words} words</span>
          <span class="meta-badge">${(info.compression * 100).toFixed(1)}% size</span>
          <span class="meta-badge">${info.latency.toFixed(2)}s</span>
        </div>
      </div>
      <div class="summary-body">${info.text}</div>
    `;
    summariesContainer.appendChild(card);
  }

  // Render ROUGE table if evaluation available
  const hasEval = Object.keys(evaluation).length > 0;
  if (hasEval) {
    for (const [name, scores] of Object.entries(evaluation)) {
      const info = summaries[name] || {};
      const r1 = scores.rouge1 || {};
      const r2 = scores.rouge2 || {};
      const rl = scores.rougeL || {};

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-weight: 600;">${name}</td>
        <td><span class="type-pill ${info.type === 'extractive' ? 'type-extractive' : 'type-abstractive'}">${info.type}</span></td>
        <td class="score-highlight">${r1.fmeasure ? r1.fmeasure.toFixed(4) : '-'}</td>
        <td class="score-highlight">${r2.fmeasure ? r2.fmeasure.toFixed(4) : '-'}</td>
        <td class="score-highlight">${rl.fmeasure ? rl.fmeasure.toFixed(4) : '-'}</td>
        <td style="color: #94a3b8;">${r1.recall ? r1.recall.toFixed(3) : '-'} / ${r2.recall ? r2.recall.toFixed(3) : '-'} / ${rl.recall ? rl.recall.toFixed(3) : '-'}</td>
        <td>${info.compression ? (info.compression * 100).toFixed(1) + '%' : '-'}</td>
      `;
      rougeTableBody.appendChild(tr);
    }
  }

  // Show active view
  const activeToggle = viewToggles.querySelector(".toggle-btn.active");
  if (activeToggle && activeToggle.dataset.view === "rouge" && hasEval) {
    rougeSection.classList.remove("hidden");
  } else {
    summariesContainer.classList.remove("hidden");
  }
}

// Initial load
loadBenchmarkSamples();
