/* ============================================================================
   Document Q&A — front end
   The page is served by FastAPI itself (src/main.py mounts src/static at "/"), so every
   call is same-origin and no host needs hard-coding. Set an absolute URL here only if you
   deliberately serve this page from somewhere other than the API.
   ========================================================================== */
const API_BASE = "";

const $ = (id) => document.getElementById(id);

const el = {
  menuBtn: $("menuBtn"),
  sidebar: $("sidebar"),
  scrim: $("scrim"),
  serverStatus: $("serverStatus"),
  syncBtn: $("syncBtn"),
  resetBtn: $("resetBtn"),
  syncStatus: $("syncStatus"),
  syncProgress: $("syncProgress"),
  syncProgressBar: $("syncProgressBar"),
  sourceList: $("sourceList"),
  docCount: $("docCount"),
  chunkCount: $("chunkCount"),
  railCount: $("railCount"),
  scopeSelect: $("scopeSelect"),
  app: document.querySelector(".app"),
  addBtn: $("addBtn"),
  uploadModal: $("uploadModal"),
  uploadNote: $("uploadNote"),
  dropzone: $("dropzone"),
  fileInput: $("fileInput"),
  uploadList: $("uploadList"),
  maxUploadMb: $("maxUploadMb"),
  messages: $("messages"),
  welcome: $("welcome"),
  chatForm: $("chatForm"),
  questionInput: $("questionInput"),
  sendBtn: $("sendBtn"),
  clearBtn: $("clearBtn"),
  apiBase: $("apiBase"),
};

el.apiBase.textContent = window.location.origin;

let polling = false;
// Conversation history lives in the page and is sent with each question; the backend is
// stateless. Follow-ups ("what about the second one?") are rewritten server-side before
// retrieval, which is why the raw text is enough here.
let history = [];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* ─────────────────────────── Text helpers ───────────────────────────── */

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

/**
 * Minimal, dependency-free Markdown renderer for answers.
 *
 * The model replies in Markdown, so rendering it as plain text showed literal `**bold**`
 * and `* bullet` markers. Everything is HTML-escaped FIRST and only a fixed set of
 * formatting is re-introduced afterwards, so model output can never inject markup.
 */
function renderMarkdown(text) {
  const inline = (s) => s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,;:!?]|$)/g, "$1<em>$2</em>")
    .replace(/(^|[\s(])_([^_\n]+)_(?=[\s).,;:!?]|$)/g, "$1<em>$2</em>");

  const lines = escapeHtml(text).split("\n");
  const out = [];
  let list = null;          // "ul" | "ol" | null
  let paragraph = [];
  let code = null;          // buffered fenced code block

  const closeParagraph = () => {
    if (paragraph.length) {
      out.push(`<p>${inline(paragraph.join(" "))}</p>`);
      paragraph = [];
    }
  };
  const closeList = () => {
    if (list) { out.push(`</${list}>`); list = null; }
  };

  for (const line of lines) {
    const fence = line.match(/^\s*```/);
    if (fence) {
      if (code === null) { closeParagraph(); closeList(); code = []; }
      else { out.push(`<pre><code>${code.join("\n")}</code></pre>`); code = null; }
      continue;
    }
    if (code !== null) { code.push(line); continue; }

    if (!line.trim()) { closeParagraph(); closeList(); continue; }

    const heading = line.match(/^\s*#{1,6}\s+(.*)$/);
    if (heading) {
      closeParagraph(); closeList();
      out.push(`<h3>${inline(heading[1])}</h3>`);
      continue;
    }

    const bullet = line.match(/^\s*[-*+]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (bullet || numbered) {
      closeParagraph();
      const wanted = bullet ? "ul" : "ol";
      if (list !== wanted) { closeList(); out.push(`<${wanted}>`); list = wanted; }
      out.push(`<li><span>${inline((bullet || numbered)[1])}</span></li>`);
      continue;
    }

    closeList();
    paragraph.push(line.trim());
  }

  if (code !== null) out.push(`<pre><code>${code.join("\n")}</code></pre>`);
  closeParagraph();
  closeList();
  return out.join("");
}

/** "textbooks/Some Very Long Book Title.pdf" -> "Some Very Long Book Title" */
function shortDocName(name) {
  return String(name || "document").split("/").pop().replace(/\.pdf$/i, "");
}

/* ─────────────────────────── Sidebar ─────────────────────────────────
   One button, two behaviours, because the sidebar is two different things: a grid column
   on desktop (toggling collapses it and the chat takes the width) and an off-canvas drawer
   on narrow screens (toggling slides it over the chat, with a scrim).
   ==================================================================== */

const isNarrow = () => window.matchMedia("(max-width: 860px)").matches;

function setDrawer(open) {
  el.sidebar.classList.toggle("is-open", open);
  el.scrim.hidden = !open;
  syncMenuButton();
}

function setCollapsed(collapsed) {
  el.app.classList.toggle("is-collapsed", collapsed);
  syncMenuButton();
}

function sidebarVisible() {
  return isNarrow()
    ? el.sidebar.classList.contains("is-open")
    : !el.app.classList.contains("is-collapsed");
}

function syncMenuButton() {
  const shown = sidebarVisible();
  el.menuBtn.setAttribute("aria-expanded", String(shown));
  el.menuBtn.setAttribute("aria-label", shown ? "Hide the sidebar" : "Show the sidebar");
}

el.menuBtn.addEventListener("click", () => {
  if (isNarrow()) setDrawer(!el.sidebar.classList.contains("is-open"));
  else setCollapsed(!el.app.classList.contains("is-collapsed"));
});

el.scrim.addEventListener("click", () => setDrawer(false));
document.addEventListener("keydown", (e) => {
  // The dialog handles its own Escape; closing the drawer as well would do two things at once.
  if (e.key === "Escape" && !el.uploadModal.open) setDrawer(false);
});

// Crossing the breakpoint leaves the other mode's state behind: a drawer left open becomes
// a permanently "open" class on a grid column, and vice versa.
window.matchMedia("(max-width: 860px)").addEventListener("change", () => {
  el.sidebar.classList.remove("is-open");
  el.scrim.hidden = true;
  syncMenuButton();
});
syncMenuButton();

/* ─────────────────────────── Add-documents dialog ───────────────────── */

function openUploadModal() {
  clearFinishedUploads();
  if (!el.uploadModal.open) el.uploadModal.showModal();
  setDrawer(false);        // on a phone the drawer would sit over the dialog's backdrop
}

el.addBtn.addEventListener("click", openUploadModal);

// Clicking the backdrop closes it. The dialog element itself covers the whole viewport, so
// "outside" means the click landed on the dialog box but not on its content.
el.uploadModal.addEventListener("click", (e) => {
  if (e.target !== el.uploadModal) return;
  const box = el.uploadModal.getBoundingClientRect();
  const outside =
    e.clientX < box.left || e.clientX > box.right ||
    e.clientY < box.top || e.clientY > box.bottom;
  if (outside) el.uploadModal.close();
});

/* ─────────────────────────── Server status ──────────────────────────── */

function setServerStatus(state, label, title) {
  el.serverStatus.dataset.state = state;
  el.serverStatus.querySelector(".status-dot__label").textContent = label;
  el.serverStatus.title = title || label;
}

/* ─────────────────────────── Upload ─────────────────────────────────
   The browser sends PDFs to POST /upload, which writes them into the server's data folder
   and starts the same background ingestion job that the sync button uses. Two progress
   phases, because they fail differently and take wildly different amounts of time:
   the transfer (XHR, has real byte counts) and the indexing (polled from /ingest/status).
   ==================================================================== */

let MAX_UPLOAD_MB = 100;   // replaced with the server's real limit by refreshSources()
const uploadCards = new Map();

function humanSize(bytes) {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1e3))} KB`;
}

/** One row per file, so a batch shows which specific file failed rather than "upload failed". */
function addUploadCard(name, bytes) {
  const li = document.createElement("li");
  li.className = "upload";
  li.innerHTML = `
    <div class="upload__top">
      <span class="upload__name"></span>
      <span class="upload__size"></span>
    </div>
    <div class="upload__bar"><div class="upload__bar-fill"></div></div>
    <span class="upload__state">Waiting…</span>`;
  li.querySelector(".upload__name").textContent = shortDocName(name);
  li.querySelector(".upload__name").title = name;
  li.querySelector(".upload__size").textContent = humanSize(bytes);
  el.uploadList.appendChild(li);

  const card = {
    li,
    bar: li.querySelector(".upload__bar-fill"),
    state: li.querySelector(".upload__state"),
    set(percent, label, tone) {
      if (percent !== null) this.bar.style.width = `${Math.max(2, Math.min(100, percent))}%`;
      if (label) this.state.textContent = label;
      li.dataset.tone = tone || "";
    },
  };
  uploadCards.set(name, card);
  return card;
}

function clearFinishedUploads() {
  for (const [name, card] of uploadCards) {
    if (card.li.dataset.tone === "done") {
      card.li.remove();
      uploadCards.delete(name);
    }
  }
}

/** XHR rather than fetch(): fetch has no upload-progress events, and a 40MB book needs one. */
function sendFiles(files, cards) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    for (const file of files) form.append("files", file, file.name);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/upload`);

    xhr.upload.addEventListener("progress", (e) => {
      if (!e.lengthComputable) return;
      const percent = Math.round((e.loaded / e.total) * 100);
      // One request carries the whole batch, so every card in it shares the transfer bar.
      for (const card of cards) card.set(percent, `Uploading… ${percent}%`);
    });

    xhr.addEventListener("load", () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch { /* non-JSON error page */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new Error(body.detail || `Upload failed (${xhr.status})`));
    });
    xhr.addEventListener("error", () => reject(new Error("Upload failed — the server is not reachable.")));
    xhr.addEventListener("abort", () => reject(new Error("Upload cancelled.")));
    xhr.send(form);
  });
}

async function uploadFiles(fileList) {
  const files = [...fileList];
  if (!files.length) return;

  clearFinishedUploads();

  // Checked here as well as on the server: no point spending two minutes uploading 300MB
  // only to be told at the end.
  const tooBig = files.filter((f) => f.size > MAX_UPLOAD_MB * 1024 * 1024);
  const notPdf = files.filter((f) => !/\.pdf$/i.test(f.name));
  const good = files.filter((f) => !tooBig.includes(f) && !notPdf.includes(f));

  for (const f of tooBig) addUploadCard(f.name, f.size).set(100, `Too large (limit ${MAX_UPLOAD_MB} MB)`, "error");
  for (const f of notPdf) addUploadCard(f.name, f.size).set(100, "Not a PDF", "error");
  if (!good.length) return;

  const cards = good.map((f) => addUploadCard(f.name, f.size));
  cards.forEach((c) => c.set(2, "Uploading…"));
  setBusy(true);

  try {
    const result = await sendFiles(good, cards);

    for (const bad of result.rejected || []) {
      const card = uploadCards.get(bad.filename);
      if (card) card.set(100, bad.error, "error");
    }
    // The server renames on collision ("notes (2).pdf"), so track the name it actually used.
    (result.accepted || []).forEach((ok, i) => {
      const card = cards[i];
      if (!card) return;
      card.set(100, "Uploaded — indexing…", "working");
      card.serverName = ok.filename;
      uploadCards.set(ok.filename, card);
    });

    await pollUntilDone();

    for (const ok of result.accepted || []) {
      const card = uploadCards.get(ok.filename);
      if (card && card.li.dataset.tone !== "error") card.set(100, "Ready", "done");
    }
  } catch (err) {
    cards.forEach((c) => c.set(100, err.message, "error"));
    setStatus(err.message, "error");
    setBusy(false);
  }
}

el.fileInput.addEventListener("change", () => {
  uploadFiles(el.fileInput.files);
  el.fileInput.value = "";   // so re-picking the same file fires 'change' again
});

// Drag and drop. The counter exists because dragleave fires when the pointer crosses onto
// a child element, which made the highlight flicker.
let dragDepth = 0;
["dragenter", "dragover"].forEach((type) => {
  el.dropzone.addEventListener(type, (e) => {
    e.preventDefault();
    if (type === "dragenter") dragDepth++;
    el.dropzone.classList.add("is-over");
  });
});
["dragleave", "drop"].forEach((type) => {
  el.dropzone.addEventListener(type, (e) => {
    e.preventDefault();
    if (type === "dragleave") dragDepth = Math.max(0, dragDepth - 1);
    else dragDepth = 0;
    if (dragDepth === 0) el.dropzone.classList.remove("is-over");
  });
});
el.dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer?.files?.length) uploadFiles(e.dataTransfer.files);
});
// Dropping a PDF anywhere else in the window would otherwise navigate away from the app
// and open the file in the browser.
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());

/* ───────────── Ingestion (the backend re-scans its own folder) ───────────── */

el.syncBtn.addEventListener("click", () => runJob("/ingest", "Scanning the data folder…"));
el.resetBtn.addEventListener("click", () => {
  if (!confirm("Rebuild the index from scratch?\n\nEvery document in the data folder is re-read and re-embedded, which can take several minutes for large PDFs.")) return;
  runJob("/reset", "Rebuilding the index…");
});

async function runJob(path, startMessage) {
  setBusy(true);
  setStatus(startMessage);
  el.syncProgress.hidden = false;
  el.syncProgress.classList.add("progress--indeterminate");
  try {
    const res = await fetch(`${API_BASE}${path}`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");
    await pollUntilDone();
  } catch (err) {
    setStatus(err.message, "error");
    setBusy(false);
    el.syncProgress.hidden = true;
  }
}

// Ingestion runs as a background job on the backend, so poll it rather than hanging on one
// long request — a big PDF takes minutes to embed.
async function pollUntilDone() {
  polling = true;
  while (polling) {
    let job;
    try {
      job = await (await fetch(`${API_BASE}/ingest/status`)).json();
    } catch {
      setStatus("Lost contact with the backend.", "error");
      setServerStatus("offline", "Offline", "The backend is not responding");
      break;
    }

    if (job.state === "running") {
      setServerStatus("busy", "Indexing", "An ingestion job is running");
      const done = job.files_done ?? 0;
      const total = job.files_total ?? 0;
      const chunksDone = job.chunks_done ?? 0;
      const chunksTotal = job.chunks_total ?? 0;

      // Progress within the current document, so one 900-page book is not a bar frozen at
      // 0% for five minutes: whole files finished, plus the fraction of the current one.
      let percent = total > 0 ? (done / total) * 100 : 0;
      if (total > 0 && chunksTotal > 0) percent += (chunksDone / chunksTotal) * (100 / total);
      if (total > 0) {
        el.syncProgress.classList.remove("progress--indeterminate");
        el.syncProgressBar.style.width = `${Math.min(100, Math.round(percent))}%`;
      }

      const current = job.current_file ? ` — ${shortDocName(job.current_file)}` : "";
      const detail = job.stage === "extracting"
        ? " · reading pages"
        : (chunksTotal ? ` · ${chunksDone}/${chunksTotal} passages` : "");
      setStatus(`Indexing ${Math.min(done + 1, total || 1)}/${total || "?"}${current}${detail}`);

      // Mirror it onto the upload row for the file currently being processed.
      const card = job.current_file && uploadCards.get(job.current_file);
      if (card && card.li.dataset.tone !== "error") {
        const filePercent = chunksTotal ? Math.round((chunksDone / chunksTotal) * 100) : null;
        card.set(
          filePercent,
          job.stage === "extracting"
            ? "Reading pages…"
            : `Indexing… ${filePercent === null ? "" : filePercent + "%"}`.trim(),
          "working",
        );
      }
      await sleep(1000);
      continue;
    }

    if (job.state === "error") {
      setStatus(`Ingestion failed: ${job.error}`, "error");
    } else {
      const results = job.results || [];
      const count = (status) => results.filter((r) => r.status === status).length;
      const ingested = count("ingested");
      const removed = count("removed");
      const failed = count("failed");
      const skipped = count("skipped");

      const parts = [];
      if (ingested) parts.push(`${ingested} indexed`);
      if (removed) parts.push(`${removed} removed`);
      if (skipped) parts.push(`${skipped} skipped`);
      if (failed) parts.push(`${failed} failed`);

      setStatus(
        parts.length ? parts.join(" · ") : "Everything is already up to date.",
        failed ? "error" : "ok"
      );
    }
    await refreshSources();
    break;
  }
  polling = false;
  setBusy(false);
  el.syncProgress.hidden = true;
  el.syncProgressBar.style.width = "";
  // refreshSources() above ran while `polling` was still true, so it deliberately left the
  // status dot alone; the job is over now, so put it back to "Online".
  setServerStatus("online", "Online", `Connected to ${window.location.origin}`);
}

function setBusy(busy) {
  el.syncBtn.disabled = busy;
  el.resetBtn.disabled = busy;
}

function setStatus(text, tone) {
  el.syncStatus.textContent = text || "";
  el.syncStatus.className = "status" + (tone ? ` status--${tone}` : "");
}

async function refreshSources() {
  try {
    const res = await fetch(`${API_BASE}/stats`);
    const data = await res.json();
    const docs = data.documents || [];

    el.docCount.textContent = docs.length;
    if (el.railCount) el.railCount.textContent = docs.length;
    if (data.max_upload_mb) {
      MAX_UPLOAD_MB = data.max_upload_mb;
      if (el.maxUploadMb) el.maxUploadMb.textContent = data.max_upload_mb;
    }
    el.chunkCount.textContent = data.total_chunks
      ? `${data.total_chunks.toLocaleString()} passages`
      : "empty";

    if (!polling) {
      setServerStatus("online", "Online", `Connected to ${window.location.origin}`);
    }

    // Only fully-indexed documents appear here: the list comes from the manifest, and a
    // manifest entry is written after the last chunk of that file is stored.
    el.sourceList.innerHTML = docs.length === 0
      ? `<li class="doc-empty">Nothing ingested yet</li>`
      : docs.map((d) => `
        <li title="${escapeHtml(d.filename)}">
          <div class="doc-text">
            <span class="doc-name">${escapeHtml(shortDocName(d.filename))}</span>
            <span class="doc-meta">${d.pages ?? "?"} pages · ${(d.chunks ?? 0).toLocaleString()} chunks</span>
          </div>
          <button type="button" class="icon-btn doc-remove" data-doc="${escapeHtml(d.filename)}"
                  title="Remove this document" aria-label="Remove ${escapeHtml(shortDocName(d.filename))}">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/></svg>
          </button>
        </li>`).join("");

    // Keep the scope dropdown in sync with what is actually stored.
    const previous = el.scopeSelect.value;
    el.scopeSelect.innerHTML = `<option value="">All documents</option>` +
      docs.map((d) => `<option value="${escapeHtml(d.filename)}">${escapeHtml(shortDocName(d.filename))}</option>`).join("");
    if (docs.some((d) => d.filename === previous)) el.scopeSelect.value = previous;

    // A job may already be running when the page loads (e.g. the server just started).
    if (data.ingesting && !polling) pollUntilDone();
  } catch {
    setServerStatus("offline", "Offline", "The backend is not responding");
  }
}

/* ─────────────────────────── Remove a document ───────────────────────── */

el.sourceList.addEventListener("click", async (e) => {
  const btn = e.target.closest(".doc-remove");
  if (!btn) return;

  const filename = btn.dataset.doc;
  if (!confirm(
    `Remove "${shortDocName(filename)}"?\n\n` +
    "Its passages are deleted from the index and the PDF is deleted from the server's " +
    "data folder. This cannot be undone."
  )) return;

  btn.disabled = true;
  try {
    const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(filename)}`, {
      method: "DELETE",
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || `Could not remove it (${res.status})`);
    setStatus(`Removed ${shortDocName(filename)}.`, "ok");
    await refreshSources();
  } catch (err) {
    btn.disabled = false;
    setStatus(err.message, "error");
  }
});

/* ─────────────────────────── Chat ───────────────────────────── */

el.clearBtn.addEventListener("click", () => {
  history = [];
  el.messages.replaceChildren(el.welcome);
  el.welcome.hidden = false;
  el.questionInput.focus();
});

// Enter sends, Shift+Enter makes a new line.
el.questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    el.chatForm.requestSubmit();
  }
});

/**
 * Grow the textarea with its content, up to the CSS max-height (4 lines).
 *
 * overflow-y stays 'hidden' in CSS and is only switched to 'auto' once the content really
 * exceeds that height. Leaving it on the browser default ('auto') showed a scrollbar
 * gutter on a single line, because scrollHeight rounds up past clientHeight at fractional
 * line heights.
 */
function autoGrow() {
  const ta = el.questionInput;
  ta.style.height = "auto";
  const max = parseFloat(getComputedStyle(ta).maxHeight) || Infinity;
  const needed = ta.scrollHeight;
  ta.style.height = `${Math.min(needed, max)}px`;
  // 1px of slack absorbs sub-pixel rounding, so the bar appears only on a real overflow.
  ta.style.overflowY = needed > max + 1 ? "auto" : "hidden";
}

el.questionInput.addEventListener("input", autoGrow);
autoGrow();

el.chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = el.questionInput.value.trim();
  if (!question || el.sendBtn.disabled) return;

  el.welcome.hidden = true;
  addMessage("user", question);

  el.questionInput.value = "";
  autoGrow();
  el.sendBtn.disabled = true;

  const answerMsg = addMessage("assistant", "");
  // Retrieval (embed -> hybrid search -> re-rank) runs before the first token arrives, so
  // the bubble would otherwise sit empty and blank for a second or two. Show what the
  // server is actually doing instead.
  const thinking = document.createElement("div");
  thinking.className = "thinking";
  thinking.innerHTML =
    '<span class="thinking__dots" aria-hidden="true"><i></i><i></i><i></i></span>' +
    '<span class="thinking__label">Searching your documents…</span>';
  answerMsg.body.appendChild(thinking);

  const setThinking = (label) => {
    const el_ = thinking.querySelector(".thinking__label");
    if (el_) el_.textContent = label;
  };
  const clearThinking = () => thinking.remove();

  let answer = "";
  let frame = null;
  const paint = () => {
    frame = null;
    clearThinking();
    answerMsg.body.innerHTML = renderMarkdown(answer);
    scrollToBottom();
  };

  try {
    const res = await fetch(`${API_BASE}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        source: el.scopeSelect.value || null,
        history,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail?.[0]?.msg || err.detail || `Request failed (${res.status})`);
    }

    // Server-Sent Events: `sources` first, then a stream of `token`s, then `done`.
    for await (const { event, data } of readSSE(res)) {
      if (event === "sources") {
        // Retrieval is done; the model is now writing.
        setThinking("Writing the answer…");
        renderSources(answerMsg.wrapper, data);
      } else if (event === "token") {
        if (!answer) {
          clearThinking();
          answerMsg.row.classList.add("msg--streaming");
        }
        answer += data.text;
        // Repainting on every token would re-parse the whole answer per chunk; one paint
        // per animation frame keeps it smooth on long replies.
        if (!frame) frame = requestAnimationFrame(paint);
      } else if (event === "done") {
        break;
      }
    }

    if (frame) cancelAnimationFrame(frame);
    paint();

    history.push({ question, answer });
    if (history.length > 8) history = history.slice(-8);
  } catch (err) {
    clearThinking();
    answerMsg.row.className = "msg msg--error";
    answerMsg.avatar.textContent = "!";
    answerMsg.role.textContent = "Error";
    answerMsg.body.textContent = err.message;
  } finally {
    answerMsg.row.classList.remove("msg--streaming");
    el.sendBtn.disabled = false;
    el.questionInput.focus();
    scrollToBottom();
  }
});

async function* readSSE(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let split;
    while ((split = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);

      let event = "message";
      let data = "";
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) yield { event, data: JSON.parse(data) };
    }
  }
}

function addMessage(role, text) {
  const row = document.createElement("div");
  row.className = `msg msg--${role}`;

  const avatar = document.createElement("div");
  avatar.className = "msg__avatar";
  if (role === "user") {
    avatar.textContent = "You";
  } else {
    avatar.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/></svg>`;
  }

  const wrapper = document.createElement("div");
  wrapper.className = "msg__body";

  const roleLabel = document.createElement("div");
  roleLabel.className = "msg__role";
  roleLabel.textContent = role === "user" ? "You" : "Answer";

  const bubble = document.createElement("div");
  bubble.className = "msg__bubble";

  const body = document.createElement("div");
  body.className = role === "user" ? "md" : "md";
  if (role === "user") body.textContent = text;

  bubble.appendChild(body);
  wrapper.append(roleLabel, bubble);
  row.append(avatar, wrapper);
  el.messages.appendChild(row);
  scrollToBottom();

  return { row, avatar, role: roleLabel, wrapper, bubble, body };
}

/**
 * Sources, grouped by document.
 *
 * The raw list repeats the same long filename once per chunk. Grouping collapses that to
 * one chip per document with its page list. Re-rank scores are deliberately NOT shown as
 * confidences — they are unbounded cross-encoder logits — so they live in the tooltip.
 */
function renderSources(wrapper, data) {
  const sources = data.sources || [];
  if (sources.length === 0) return;

  const primary = sources.filter((s) => !s.neighbor);
  const neighbours = sources.length - primary.length;
  if (primary.length === 0) return;

  const groups = new Map();
  for (const s of primary) {
    if (!groups.has(s.source)) groups.set(s.source, { pages: new Set(), scores: [] });
    const group = groups.get(s.source);
    // The API labels each chunk "page 7" / "pages 7-8"; keep just the numbers so a
    // grouped chip reads "pp. 2, 7-8" rather than "page 2, pages 7-8".
    group.pages.add(String(s.pages).replace(/^pages?\s+/i, ""));
    if (s.rerank_score != null) group.scores.push(s.rerank_score);
  }

  const box = document.createElement("div");
  box.className = "sources";

  const label = document.createElement("div");
  label.className = "sources__label";
  label.textContent = `Sources · ${primary.length} passage${primary.length === 1 ? "" : "s"}`;

  const chips = document.createElement("div");
  chips.className = "chips";

  for (const [source, group] of groups) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.title = group.scores.length
      ? `${source}\nRelevance scores: ${group.scores.join(", ")}`
      : source;

    const name = document.createElement("span");
    name.className = "chip__doc";
    name.textContent = shortDocName(source);

    const pages = document.createElement("span");
    pages.className = "chip__pages";
    const list = [...group.pages];
    pages.textContent = (list.length === 1 && !list[0].includes("-") ? "p. " : "pp. ") + list.join(", ");

    chip.append(name, pages);
    chips.appendChild(chip);
  }

  if (neighbours > 0) {
    const chip = document.createElement("span");
    chip.className = "chip chip--muted";
    chip.title = "Adjacent passages included so the model reads continuous prose";
    chip.textContent = `+${neighbours} neighbouring`;
    chips.appendChild(chip);
  }

  box.append(label, chips);

  if (data.search_query) {
    const rewritten = document.createElement("div");
    rewritten.className = "rewritten";
    rewritten.textContent = `Searched for: “${data.search_query}”`;
    box.appendChild(rewritten);
  }

  wrapper.querySelector(".msg__bubble").appendChild(box);
}

function scrollToBottom() {
  el.messages.scrollTop = el.messages.scrollHeight;
}

/* ─────────────────────────── Boot ───────────────────────────── */

/**
 * Hold a loading screen up until the app is actually usable.
 *
 * Two things used to look like breakage on a cold start: opening the page while uvicorn
 * was still importing torch (the browser shows a connection error, so the overlay retries
 * instead of giving up), and opening it during the first ingest (the index is empty, so
 * every question would answer "not in these documents"). The overlay reports which of the
 * two is happening and lets you in anyway.
 */
async function boot() {
  const box = $("boot");
  const title = $("bootTitle");
  const text = $("bootText");
  const bar = $("bootBar");
  const skip = $("bootSkip");

  let dismissed = false;
  const close = () => {
    if (dismissed) return;
    dismissed = true;
    if (!box.hidden) {
      box.classList.add("boot--closing");
      setTimeout(() => { box.hidden = true; }, 300);
    }
    el.questionInput.focus();
  };
  skip.addEventListener("click", close);

  // The overlay starts hidden and is only revealed if the app is NOT immediately usable.
  // On a warm start /stats answers in a few milliseconds, so nothing flashes over the UI
  // at all; the loading screen exists for the cold start, not for every page load.
  const reveal = () => {
    if (!dismissed) box.hidden = false;
  };

  const indeterminate = (on) =>
    bar.classList.toggle("boot__bar-fill--indeterminate", on);
  indeterminate(true);

  const startedAt = Date.now();

  while (!dismissed) {
    let stats = null;
    try {
      stats = await (await fetch(`${API_BASE}/stats`)).json();
    } catch {
      // Server not listening yet — normal for the first ~20s while the models load.
      reveal();
      const secs = Math.round((Date.now() - startedAt) / 1000);
      title.textContent = "Starting up…";
      text.textContent = secs > 5
        ? `Waiting for the server (${secs}s). It loads the embedding model on startup.`
        : "Waiting for the server.";
      await sleep(1000);
      continue;
    }

    if (stats.embedding_model_ready === false && !stats.ingesting) {
      // Port is open but the embedding model is still loading in its warm-up thread.
      reveal();
      title.textContent = "Loading the search model…";
      text.textContent = "About 15 seconds. It only happens once per server start.";
      indeterminate(true);
      skip.hidden = false;
      await sleep(1000);
      continue;
    }

    if (stats.ingesting) {
      let job = {};
      try {
        job = await (await fetch(`${API_BASE}/ingest/status`)).json();
      } catch { /* transient; the next loop retries */ }
      const done = job.files_done ?? 0;
      const total = job.files_total ?? 0;
      reveal();
      title.textContent = "Indexing your documents…";
      const current = job.current_file ? ` — ${shortDocName(job.current_file)}` : "";
      text.textContent = total
        ? `Document ${Math.min(done + 1, total)} of ${total}${current}. Only happens once per file; later starts reuse the index.`
        : "Reading the data folder…";
      if (total) {
        indeterminate(false);
        bar.style.width = `${Math.round((done / total) * 100) || 4}%`;
      }
      // Nothing to answer from yet, but a partially built index is already queryable.
      skip.hidden = (stats.total_chunks || 0) === 0;
      await sleep(1500);
      continue;
    }

    // An empty index is not a reason to hold the app back any more - "Add documents" is
    // right there in the sidebar, and blocking the UI would hide the very button that
    // fixes it.
    break;
  }

  close();
}

refreshSources();
boot();
