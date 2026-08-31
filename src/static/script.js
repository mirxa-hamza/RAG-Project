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
  scopeSelect: $("scopeSelect"),
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

/* ─────────────────────────── Sidebar drawer ─────────────────────────── */

function setDrawer(open) {
  el.sidebar.classList.toggle("is-open", open);
  el.scrim.hidden = !open;
  el.menuBtn.setAttribute("aria-expanded", String(open));
}

el.menuBtn.addEventListener("click", () => setDrawer(!el.sidebar.classList.contains("is-open")));
el.scrim.addEventListener("click", () => setDrawer(false));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") setDrawer(false);
});

/* ─────────────────────────── Server status ──────────────────────────── */

function setServerStatus(state, label, title) {
  el.serverStatus.dataset.state = state;
  el.serverStatus.querySelector(".status-dot__label").textContent = label;
  el.serverStatus.title = title || label;
}

/* ───────────── Ingestion (no upload — the backend re-scans its own folder) ───────────── */

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
      if (total > 0) {
        el.syncProgress.classList.remove("progress--indeterminate");
        el.syncProgressBar.style.width = `${Math.round((done / total) * 100)}%`;
      }
      const current = job.current_file ? ` — ${shortDocName(job.current_file)}` : "";
      setStatus(`Indexing ${done}/${total || "?"}${current}`);
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
    el.chunkCount.textContent = data.total_chunks
      ? `${data.total_chunks.toLocaleString()} passages`
      : "empty";

    if (!polling) {
      setServerStatus("online", "Online", `Connected to ${window.location.origin}`);
    }

    el.sourceList.innerHTML = docs.length === 0
      ? `<li class="doc-empty">Nothing ingested yet</li>`
      : docs.map((d) => `
        <li title="${escapeHtml(d.filename)}">
          <span class="doc-name">${escapeHtml(shortDocName(d.filename))}</span>
          <span class="doc-meta">${d.pages ?? "?"} pages · ${(d.chunks ?? 0).toLocaleString()} chunks</span>
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

// Grow the textarea with its content, up to the CSS max-height.
el.questionInput.addEventListener("input", () => {
  el.questionInput.style.height = "auto";
  el.questionInput.style.height = `${el.questionInput.scrollHeight}px`;
});

el.chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = el.questionInput.value.trim();
  if (!question || el.sendBtn.disabled) return;

  el.welcome.hidden = true;
  addMessage("user", question);

  el.questionInput.value = "";
  el.questionInput.style.height = "auto";
  el.sendBtn.disabled = true;

  const answerMsg = addMessage("assistant", "");
  answerMsg.row.classList.add("msg--streaming");

  let answer = "";
  let frame = null;
  const paint = () => {
    frame = null;
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
        renderSources(answerMsg.wrapper, data);
      } else if (event === "token") {
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
refreshSources();
