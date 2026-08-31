const API_BASE = "http://localhost:8000";
document.getElementById("apiBase").textContent = API_BASE;

const syncBtn = document.getElementById("syncBtn");
const syncStatus = document.getElementById("syncStatus");
const sourceList = document.getElementById("sourceList");
const resetBtn = document.getElementById("resetBtn");

const messagesEl = document.getElementById("messages");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");

let polling = false;

// ---------- Ingestion (no upload - the backend re-scans its own data folder) ----------
syncBtn.addEventListener("click", () => runJob("/ingest", "Scanning the data folder..."));
resetBtn.addEventListener("click", () => runJob("/reset", "Rebuilding the store from the data folder..."));

async function runJob(path, startMessage) {
  setBusy(true);
  setStatus(startMessage, false);
  try {
    const res = await fetch(`${API_BASE}${path}`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");
    await pollUntilDone();
  } catch (err) {
    setStatus(err.message, true);
    setBusy(false);
  }
}

// Ingestion runs as a background job on the backend, so poll it rather than
// hanging on one long request - a big PDF takes minutes to embed.
async function pollUntilDone() {
  polling = true;
  while (polling) {
    let job;
    try {
      job = await (await fetch(`${API_BASE}/ingest/status`)).json();
    } catch {
      setStatus("Lost contact with the backend.", true);
      break;
    }

    if (job.state === "running") {
      const done = job.files_done ?? 0;
      const total = job.files_total ?? 0;
      const current = job.current_file ? ` - ${job.current_file}` : "";
      setStatus(`Ingesting${current} (${done}/${total || "?"})...`, false);
      await sleep(1000);
      continue;
    }

    if (job.state === "error") {
      setStatus(`Ingestion failed: ${job.error}`, true);
    } else {
      const results = job.results || [];
      const ingested = results.filter(r => r.status === "ingested").length;
      const skipped = results.filter(r => r.status === "skipped").length;
      setStatus(
        ingested > 0
          ? `Ingested ${ingested} document(s).${skipped ? ` ${skipped} skipped (no extractable text).` : ""}`
          : "Nothing new - the data folder is already fully synced.",
        false
      );
    }
    await refreshSources();
    break;
  }
  polling = false;
  setBusy(false);
}

function setBusy(busy) {
  syncBtn.disabled = busy;
  resetBtn.disabled = busy;
}

function setStatus(text, isError) {
  syncStatus.textContent = text;
  syncStatus.className = "status" + (isError ? " error" : "");
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function refreshSources() {
  try {
    const res = await fetch(`${API_BASE}/stats`);
    const data = await res.json();

    if (!data.documents || data.documents.length === 0) {
      sourceList.innerHTML = `<li class="empty">Nothing ingested yet</li>`;
    } else {
      sourceList.innerHTML = data.documents.map(d => `
        <li>
          <span class="doc-name">${escapeHtml(d.filename)}</span>
          <span class="doc-meta">${d.pages ?? "?"} pages · ${d.chunks ?? "?"} chunks</span>
        </li>`).join("");
    }

    // A job may already be running when the page loads (e.g. server just started).
    if (data.ingesting && !polling) pollUntilDone();
  } catch {
    // backend not reachable yet - fine on first load, ignore
  }
}

// ---------- Chat ----------
chatForm.addEventListener("submit", async e => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage(question, "user");
  questionInput.value = "";
  sendBtn.disabled = true;

  const thinkingEl = addMessage("Thinking...", "assistant");

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail?.[0]?.msg || data.detail || "Request failed");

    thinkingEl.textContent = data.answer;
    if (data.sources && data.sources.length > 0) {
      const src = document.createElement("div");
      src.className = "sources";
      src.textContent = "Sources: " + data.sources
        .map(s => `${s.source} ${s.pages} (${s.similarity})`)
        .join(" · ");
      thinkingEl.appendChild(src);
    }
  } catch (err) {
    thinkingEl.textContent = err.message;
    thinkingEl.className = "msg error";
  } finally {
    sendBtn.disabled = false;
  }
});

function addMessage(text, role) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// Load current state on page open
refreshSources();
