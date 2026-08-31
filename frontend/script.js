const API_BASE = "http://localhost:8000";
document.getElementById("apiBase").textContent = API_BASE;

const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const uploadStatus = document.getElementById("uploadStatus");
const sourceList = document.getElementById("sourceList");
const resetBtn = document.getElementById("resetBtn");

const messagesEl = document.getElementById("messages");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");

// ---------- Upload ----------
browseBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) uploadFile(fileInput.files[0]);
});

["dragover", "dragleave", "drop"].forEach(evt => {
  dropZone.addEventListener(evt, e => e.preventDefault());
});
dropZone.addEventListener("dragover", () => dropZone.classList.add("drag"));
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag"));
dropZone.addEventListener("drop", e => {
  dropZone.classList.remove("drag");
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});

async function uploadFile(file) {
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    setStatus("Only PDF files are supported.", true);
    return;
  }
  setStatus(`Uploading ${file.name}...`, false);

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");

    setStatus(`Stored ${data.chunks_stored} chunks from ${data.pages} pages.`, false);
    await refreshSources();
  } catch (err) {
    setStatus(err.message, true);
  }
}

function setStatus(text, isError) {
  uploadStatus.textContent = text;
  uploadStatus.className = "status" + (isError ? " error" : "");
}

async function refreshSources() {
  try {
    const res = await fetch(`${API_BASE}/stats`);
    const data = await res.json();
    if (!data.sources || data.sources.length === 0) {
      sourceList.innerHTML = `<li class="empty">Nothing uploaded yet</li>`;
      return;
    }
    sourceList.innerHTML = data.sources.map(s => `<li>${escapeHtml(s)}</li>`).join("");
  } catch {
    // backend not reachable yet - fine on first load, ignore
  }
}

resetBtn.addEventListener("click", async () => {
  try {
    await fetch(`${API_BASE}/reset`, { method: "POST" });
    await refreshSources();
    setStatus("Vector store cleared.", false);
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
