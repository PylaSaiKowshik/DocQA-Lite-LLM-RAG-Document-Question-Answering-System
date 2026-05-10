let uploadedFileName = "";
let uploadedFilePath = "";
let uploadedFileHash = "";

function formatTime(seconds) {
  seconds = parseFloat(seconds);
  if (isNaN(seconds)) return "—";
  if (seconds < 60) return seconds.toFixed(2) + " sec";
  const mins = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(2);
  return mins + " min " + secs + " sec";
}

function generateId() {
  return Math.random().toString(36).substr(2, 9);
}

const uploadArea = document.getElementById("uploadArea");
uploadArea.addEventListener("dragover", e => { e.preventDefault(); uploadArea.classList.add("dragover"); });
uploadArea.addEventListener("dragleave", () => uploadArea.classList.remove("dragover"));
uploadArea.addEventListener("drop", e => {
  e.preventDefault();
  uploadArea.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) { document.getElementById("pdfUpload").files = e.dataTransfer.files; doUpload(file); }
});
document.getElementById("pdfUpload").addEventListener("change", e => {
  if (e.target.files[0]) doUpload(e.target.files[0]);
});

function doUpload(file) {
  const status = document.getElementById("uploadStatus");
  status.className = "upload-status";
  status.textContent = "⏳ Uploading...";
  status.style.display = "block";

  const formData = new FormData();
  formData.append("file", file);

  fetch("../upload.php", { method: "POST", body: formData })
    .then(r => r.json())
    .then(data => {
      if (data.status === "success") {
        uploadedFileName = data.filename;
        uploadedFilePath = data.path;
        uploadedFileHash = data.hash;
        status.className = "upload-status success";
        status.textContent = "✅ " + uploadedFileName;
        document.querySelector('.upload-icon').textContent = "📋";
      } else {
        status.className = "upload-status error";
        status.textContent = "❌ " + data.message;
      }
    })
    .catch(() => {
      status.className = "upload-status error";
      status.textContent = "❌ Upload failed";
    });
}

const EVENT_CONFIG = {
  extract_start:  { icon: "📄", label: "EXTRACTION" },
  workflow_start: { icon: "⚙️", label: "WORKFLOW" },
  agent_start:    { icon: "🤖", label: "AGENT INIT" },
  agent_thought:  { icon: "💭", label: "REASONING" },
  agent_search:   { icon: "🔍", label: "SEARCHING" },
  agent_action:   { icon: "⚡", label: "ACTION" },
  agent_done:     { icon: "✅", label: "COMPLETE" },
  stream_start:   { icon: "✍️", label: "STREAMING" },
  heartbeat:      null,
  done:           null,
  token:          null,
};

function addTimelineEvent(type, message) {
  const cfg = EVENT_CONFIG[type];
  if (!cfg) return;
  const timeline = document.getElementById("timeline");
  const item = document.createElement("div");
  item.className = `timeline-item timeline-line event-${type}`;
  item.innerHTML = `
    <div class="timeline-dot">${cfg.icon}</div>
    <div class="timeline-content">
      <div class="timeline-label">${cfg.label}</div>
      <div class="timeline-message">${message}</div>
    </div>`;
  timeline.appendChild(item);
  item.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function askQuestion() {
  const question = document.getElementById("questionInput").value.trim();
  if (!question) { alert("Please enter a question."); return; }
  if (!uploadedFileName) { alert("Please upload a PDF first."); return; }

  const requestId = generateId();
  const summaryKeywords = ["summary", "summarize", "summarise", "tell me about this pdf"];
  const isSummary = summaryKeywords.some(k => question.toLowerCase().includes(k));

  document.getElementById("agentPanel").style.display = isSummary ? "none" : "block";
  document.getElementById("answerPanel").style.display = "none";
  document.getElementById("metricsPanel").style.display = "none";
  document.getElementById("timeline").innerHTML = "";
  document.getElementById("agentSpinner").style.display = "block";
  document.getElementById("submitBtn").disabled = true;
  document.getElementById("submitBtn").textContent = "Processing...";

  let evtSource = null;

  if (!isSummary) {
    evtSource = new EventSource(`../ask.php?stream=1&request_id=${requestId}`);
    document.getElementById("answerPanel").style.display = "block";
    document.getElementById("answerBox").textContent = "";
    document.getElementById("answerBox").setAttribute("data-streaming", "true");

    evtSource.onmessage = function(e) {
      const data = JSON.parse(e.data);
      if (data.type === "heartbeat") return;
      if (data.type === "done") {
        evtSource.close();
        document.getElementById("agentSpinner").style.display = "none";
        document.getElementById("answerBox").removeAttribute("data-streaming");
        return;
      }
      if (data.type === "stream_start") {
        document.getElementById("answerPanel").style.display = "block";
        document.getElementById("answerBox").setAttribute("data-streaming", "true");
        return;
      }
      if (data.type === "token") {
        const box = document.getElementById("answerBox");
        document.getElementById("answerPanel").style.display = "block";
        box.textContent += data.message;
        box.scrollTop = box.scrollHeight;
        return;
      }
      addTimelineEvent(data.type, data.message);
    };

    evtSource.onerror = function() { evtSource.close(); };

  } else {
    document.getElementById("agentPanel").style.display = "block";
    document.getElementById("timeline").innerHTML = "";
    addTimelineEvent("extract_start", "📄 Extracting PDF text...");
    addTimelineEvent("workflow_start", "⚙️ Running RAPTOR hierarchical summarization...");
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 3600000);

  fetch("../ask.php", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      filename:  uploadedFileName,
      file_hash: uploadedFileHash,
      request_id: requestId
    }),
    signal: controller.signal
  })
  .then(r => r.json())
  .then(data => {
    clearTimeout(timeout);
    if (evtSource) evtSource.close();
    document.getElementById("agentSpinner").style.display = "none";
    document.getElementById("submitBtn").disabled = false;
    document.getElementById("submitBtn").textContent = "Run Agent →";

    const box = document.getElementById("answerBox");
    box.removeAttribute("data-streaming");
    document.getElementById("answerPanel").style.display = "block";
    box.textContent = data.answer || "No answer returned.";

    if (data.metrics) showMetrics(data.metrics);
  })
  .catch(err => {
    clearTimeout(timeout);
    if (evtSource) evtSource.close();
    document.getElementById("submitBtn").disabled = false;
    document.getElementById("submitBtn").textContent = "Run Agent →";
    document.getElementById("answerPanel").style.display = "block";
    document.getElementById("answerBox").textContent =
      err.name === "AbortError" ? "Request timed out." : "Error: " + err.message;
  });
}

function scoreClass(val, goodThresh, warnThresh) {
  if (val >= goodThresh) return "good";
  if (val >= warnThresh) return "warn";
  return "bad";
}

function showMetrics(m) {
  const grid = document.getElementById("metricsGrid");
  grid.innerHTML = "";

  // ── Core Performance ──
  grid.innerHTML += `<div class="metrics-section-title">📊 Core Performance</div>`;
  const core = [
    { label: "Type",       value: m.type === "summary" ? "Summary" : "Q&A", unit: "" },
    { label: "Total Time", value: formatTime(m.response_time_sec),           unit: "" },
    { label: "Extraction", value: formatTime(m.extraction_time_sec),         unit: "" },
  ];
  if (m.type === "summary") {
    core.push({ label: "Summary Time",   value: formatTime(m.summary_time_sec),               unit: "" });
    core.push({ label: "Summary Length", value: (m.summary_length_words||0).toLocaleString(), unit: "words" });
    core.push({ label: "LLM Calls",      value: m.llm_calls || "—",                           unit: "calls" });
  } else {
    core.push({ label: "QA Time",   value: formatTime(m.qa_time_sec), unit: "" });
    core.push({ label: "Model",     value: m.model_used || "—",       unit: "" });
    core.push({ label: "LLM Calls", value: m.llm_calls ?? "—",        unit: "calls" });
  }
  core.push({ label: "Pages",      value: m.pages_processed,                            unit: "pages" });
  core.push({ label: "Characters", value: (m.characters_processed||0).toLocaleString(), unit: "chars" });
  core.push({ label: "Words",      value: (m.words_processed||0).toLocaleString(),      unit: "words" });
  core.forEach(item => {
    grid.innerHTML += `<div class="metric-item"><div class="metric-label">${item.label}</div><div class="metric-value">${item.value}<span class="metric-unit"> ${item.unit}</span></div></div>`;
  });

  // ── Latency Metrics ──
  grid.innerHTML += `<div class="metrics-section-title">⚡ Latency Metrics</div>`;
  const ttft = parseFloat(m.ttft_sec || m.response_time_sec || 0);
  const e2e  = parseFloat(m.e2e_latency_sec || m.response_time_sec || 0);
  const tps  = parseFloat(m.tps || 0);
  [
    { label: "TTFT", sub: "Time To First Token", value: formatTime(ttft),
      note: ttft < e2e ? "✅ Streaming active!" : "= E2E (no streaming yet)",
      cls: ttft < e2e ? "good" : "warn" },
    { label: "E2E Latency", sub: "End To End", value: formatTime(e2e),
      note: "total response time", cls: "warn" },
    { label: "TPS", sub: "Tokens/Second", value: tps.toFixed(1),
      note: tps >= 5 ? "✅ Fast" : tps >= 2 ? "⚠️ Medium" : "❌ Slow",
      cls: tps >= 5 ? "good" : tps >= 2 ? "warn" : "bad" },
  ].forEach(item => {
    grid.innerHTML += `<div class="metric-item ${item.cls}"><div class="metric-label">${item.label} <span style="color:var(--muted);font-weight:400">· ${item.sub}</span></div><div class="metric-value">${item.value}</div><div style="font-size:10px;color:var(--muted);margin-top:4px;font-family:'JetBrains Mono',monospace;">${item.note}</div></div>`;
  });

  // ── RAG Quality Metrics ──
  if (m.type !== "summary") {
    grid.innerHTML += `<div class="metrics-section-title">🎯 RAG Quality Metrics</div>`;
    const retrieval = parseFloat(m.retrieval_score || 0);
    const conf      = parseFloat(m.confidence_score || 0);
    const recallAtK = parseFloat(m.recall_at_k || 0);

    [
      {
        label: "Retrieval Score", sub: "Chunk Relevance",
        value: retrieval.toFixed(1) + "%",
        note:  retrieval >= 40 ? "✅ Good" : "⚠️ Low",
        cls:   scoreClass(retrieval, 40, 25)
      },
      {
        label: "Confidence", sub: "Retrieval + Verified",
        value: conf.toFixed(1) + "%",
        note:  conf >= 60 ? "✅ High" : conf >= 30 ? "⚠️ Medium" : "❌ Low",
        cls:   scoreClass(conf, 60, 30)
      },
      {
        label: "Recall@K", sub: "Oracle Chunk Hit",
        value: recallAtK.toFixed(1) + "%",
        note:  recallAtK === 100  ? "✅ Oracle retrieved" :
               recallAtK >= 70   ? "✅ Very close" :
               recallAtK >= 40   ? "⚠️ Partial" : "❌ Missed",
        cls:   scoreClass(recallAtK, 70, 40)
      },
    ].forEach(item => {
      grid.innerHTML += `<div class="metric-item ${item.cls}">
        <div class="metric-label">${item.label} <span style="color:var(--muted);font-weight:400">· ${item.sub}</span></div>
        <div class="metric-value">${item.value}</div>
        <div style="font-size:10px;color:var(--muted);margin-top:4px;font-family:'JetBrains Mono',monospace;">${item.note}</div>
      </div>`;
    });

    // ── Decision Intelligence ──
    grid.innerHTML += `<div class="metrics-section-title">🧠 Decision Intelligence</div>`;
    const dt = m.decision_type || "accepted";
    const dtLabel = dt === "accepted"                  ? "✅ Accepted"
                  : dt === "rejected_verification"     ? "❌ Rejected · Verification"
                  : dt === "rejected_low_retrieval"    ? "🚫 Rejected · Low Retrieval"
                  : dt === "hard_reject"               ? "🚫 Hard Reject"
                  : dt;
    const dtCls = dt === "accepted" ? "decision-accepted" : "decision-rejected";
    grid.innerHTML += `<div class="metric-item" style="grid-column: 1/-1; background: var(--surface);">
      <div class="metric-label">Decision Type</div>
      <div style="margin-top:6px;"><span class="decision-badge ${dtCls}">${dtLabel}</span></div>
    </div>`;

    const vm       = m.verification_mode || "none";
    const kwScore  = parseFloat(m.keyword_score || 0);
    const verified = m.verified === true;

    [
      {
        label: "Verification Mode", sub: "Strictness",
        value: vm === "strict" ? "STRICT" : vm === "normal" ? "NORMAL" : "NONE",
        note:  vm === "strict" ? "⚠️ Weak retrieval" : vm === "normal" ? "✅ Standard" : "— Not applied",
        cls:   vm === "strict" ? "warn" : vm === "normal" ? "good" : ""
      },
      {
        label: "Keyword Score", sub: "Answer Grounding",
        value: kwScore.toFixed(1) + "%",
        note:  kwScore >= 80 ? "✅ Strong" : kwScore >= 60 ? "✅ Good" : kwScore >= 30 ? "⚠️ Weak" : "❌ Low",
        cls:   scoreClass(kwScore, 60, 30)
      },
      {
        label: "Verified", sub: "Answer Confirmed",
        value: verified ? "YES" : "NO",
        note:  verified ? "✅ Answer verified" : "❌ Not verified",
        cls:   verified ? "good" : "bad"
      },
    ].forEach(item => {
      grid.innerHTML += `<div class="metric-item ${item.cls}">
        <div class="metric-label">${item.label} <span style="color:var(--muted);font-weight:400">· ${item.sub}</span></div>
        <div class="metric-value">${item.value}</div>
        <div style="font-size:10px;color:var(--muted);margin-top:4px;font-family:'JetBrains Mono',monospace;">${item.note}</div>
      </div>`;
    });

    // ── Debug Metrics ──
    grid.innerHTML += `<div class="metrics-section-title">🔬 Debug Metrics · For Analysis Only</div>`;
    const precision = parseFloat(m.context_precision || 0);
    const grounding = parseFloat(m.answer_grounding  || 0);
    [
      {
        label: "Context Precision", sub: "Relevant/Retrieved",
        value: precision.toFixed(1) + "%",
        note:  "Debug only — not used in decisions",
        cls:   ""
      },
      {
        label: "Answer Grounding", sub: "Word Overlap",
        value: grounding.toFixed(1) + "%",
        note:  "Debug only — not used in decisions",
        cls:   ""
      },
    ].forEach(item => {
      grid.innerHTML += `<div class="metric-item" style="opacity:0.6;">
        <div class="metric-label">${item.label} <span style="color:var(--muted);font-weight:400">· ${item.sub}</span></div>
        <div class="metric-value" style="font-size:16px;">${item.value}</div>
        <div style="font-size:10px;color:var(--muted);margin-top:4px;font-family:'JetBrains Mono',monospace;">${item.note}</div>
      </div>`;
    });
  }

  document.getElementById("metricsPanel").style.display = "block";
}

document.getElementById("questionInput").addEventListener("keydown", e => {
  if (e.key === "Enter") askQuestion();
});
document.getElementById("submitBtn").addEventListener("click", askQuestion);