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

// ---------- Sync (no upload - just tells the backend to re-scan its own data folder) ----------
syncBtn.addEventListener("click", async () => {
  syncBtn.disabled = true;
  setStatus("Scanning the data folder...", false);
  try {
    const res = await fetch(`${API_BASE}/ingest`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Sync failed");

    const ingested = data.results.filter(r => r.status === "ingested").length;
    const skipped = data.results.filter(r => r.status === "skipped").length;
    setStatus(
      ingested > 0
        ? `Ingested ${ingested} new document(s).${skipped ? ` ${skipped} skipped (no extractable text).` : ""}`
        : "Nothing new to ingest - data folder already fully synced.",
      false
    );
    await refreshSources();
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    syncBtn.disabled = false;
  }
});

function setStatus(text, isError) {
  syncStatus.textContent = text;
  syncStatus.className = "status" + (isError ? " error" : "");
}

async function refreshSources() {
  try {
    const res = await fetch(`${API_BASE}/stats`);
    const data = await res.json();
    if (!data.sources || data.sources.length === 0) {
      sourceList.innerHTML = `<li class="empty">Nothing ingested yet</li>`;
      return;
    }
    sourceList.innerHTML = data.sources.map(s => `<li>${escapeHtml(s)}</li>`).join("");
  } catch {
    // backend not reachable yet - fine on first load, ignore
  }
}

resetBtn.addEventListener("click", async () => {
  try {
    setStatus("Rebuilding from the data folder...", false);
    await fetch(`${API_BASE}/reset`, { method: "POST" });
    await refreshSources();
    setStatus("Vector store cleared and rebuilt from the data folder.", false);
  } catch (err) {
    setStatus("Couldn't reach the backend.", true);
  }
});

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
    if (!res.ok) throw new Error(data.detail || "Request failed");

    thinkingEl.textContent = data.answer;
    if (data.sources && data.sources.length > 0) {
      const src = document.createElement("div");
      src.className = "sources";
      src.textContent = "Sources: " + data.sources
        .map(s => `${s.source} p.${s.page} (${s.similarity})`)
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
