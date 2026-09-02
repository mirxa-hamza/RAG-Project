/* ============================================================================
   Document Q&A — front end
   The page is served by FastAPI itself (src/main.py mounts src/static at "/"), so every
   call is same-origin and no host needs hard-coding. Set an absolute URL here only if you
   deliberately serve this page from somewhere other than the API.
   ========================================================================== */
const API_BASE = "";

const $ = (id) => document.getElementById(id);

const el = {
  app: null,          // replaced below; declared first so the auth block can use el.*
  auth: $("auth"),
  authForm: $("authForm"),
  authTitle: $("authTitle"),
  authText: $("authText"),
  authUser: $("authUser"),
  authPass: $("authPass"),
  authError: $("authError"),
  authSubmit: $("authSubmit"),
  authSubmitText: $("authSubmitText"),
  authSwitch: $("authSwitch"),
  authSwitchText: $("authSwitchText"),
  who: $("who"),
  whoName: $("whoName"),
  whoAvatar: $("whoAvatar"),
  logoutBtn: $("logoutBtn"),
  accountModal: $("accountModal"),
  accountName: $("accountName"),
  accountError: $("accountError"),
  passwordForm: $("passwordForm"),
  passwordSubmit: $("passwordSubmit"),
  currentPassword: $("currentPassword"),
  newPassword: $("newPassword"),
  signoutAllBtn: $("signoutAllBtn"),
  deleteAccountBtn: $("deleteAccountBtn"),
  quotaText: $("quotaText"),
  quotaFill: $("quotaFill"),
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
  scopeBtn: $("scopeBtn"),
  scopeLabel: $("scopeLabel"),
  selectAllBtn: $("selectAllBtn"),
  appEl: $("app"),
  docsBtn: $("docsBtn"),
  docsPanel: $("docsPanel"),
  docsScrim: $("docsScrim"),
  docsClose: $("docsClose"),
  newChatBtn: $("newChatBtn"),
  attachBtn: $("attachBtn"),
  activityBtn: $("activityBtn"),
  activityDot: $("activityDot"),
  activityModal: $("activityModal"),
  activitySummary: $("activitySummary"),
  activityStage: $("activityStage"),
  activityPercent: $("activityPercent"),
  activityBar: $("activityBar"),
  activityFile: $("activityFile"),
  activityLog: $("activityLog"),
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

el.app = el.appEl;
el.apiBase.textContent = window.location.origin;

/* ─────────────────────────── Session ─────────────────────────────────
   The JWT lives in localStorage and rides on every request through authFetch(). A 401
   from anywhere means the token is gone, expired, or was signed with a different key, so
   the only sane response is to drop it and show the sign-in screen again.
   ==================================================================== */

const TOKEN_KEY = "docqa.token";
let session = null;          // { username } once signed in
// Bumped on every sign-out. Background loops capture it and stop when it changes, so work
// started under one session can never keep running (or keep 401ing) under the next.
let sessionEpoch = 0;

function readToken() {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    // Private mode, or storage blocked. The app still works for this tab; the user just
    // signs in again next time.
    return null;
  }
}

function writeToken(token) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch { /* not fatal - see readToken */ }
}

/**
 * fetch() with the bearer token attached, and one rule: a 401 ends the session.
 *
 * Every call in this file goes through it, including the SSE stream - which works only
 * because the stream is read with fetch() rather than EventSource, and EventSource cannot
 * set headers.
 */
async function authFetch(path, options = {}) {
  const token = readToken();
  const headers = new Headers(options.headers || {});
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
  if (response.status === 401) {
    endSession("Your session expired. Sign in again.");
    // Tagged, so a background loop can tell "you are signed out, stop" apart from "that
    // request failed, try again". A loop that cannot tell them apart re-fires forever.
    const err = new Error("Signed out");
    err.signedOut = true;
    throw err;
  }
  return response;
}

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
  if (e.key !== "Escape") return;
  // One Escape closes one thing, topmost first: an open dialog handles its own, then the
  // documents panel, then the mobile drawer.
  if (el.accountModal.open || el.activityModal.open) return;
  if (docsOpen()) { closeDocs(); return; }
  setDrawer(false);
});

// Crossing the breakpoint leaves the other mode's state behind: a drawer left open becomes
// a permanently "open" class on a grid column, and vice versa.
window.matchMedia("(max-width: 860px)").addEventListener("change", () => {
  el.sidebar.classList.remove("is-open");
  el.scrim.hidden = true;
  syncMenuButton();
});
syncMenuButton();

/* ─────────────────────────── Search scope ───────────────────────────
   Which documents a question is answered from. A Set of filenames rather than one value,
   because a person can tick several - and an EMPTY set deliberately means "all of them",
   not "none". "Search nothing" is not a state worth being able to reach by accident: it
   would answer "not in these documents" to everything, and look like a broken index.
   ==================================================================== */

const scope = new Set();

function scopeList() {
  return [...scope];
}

function renderScope(available) {
  // Documents deleted since the last render must not stay in the selection - the filter
  // would keep narrowing the search to a document the server no longer has.
  if (available) {
    for (const name of [...scope]) if (!available.includes(name)) scope.delete(name);
  }

  const total = available ? available.length : null;
  el.scopeLabel.textContent =
    scope.size === 0 ? "All documents"
      : scope.size === 1 ? shortDocName(scopeList()[0])
        : `${scope.size} document${scope.size === 1 ? "" : "s"}`;
  el.scopeBtn.title = scope.size === 0
    ? "Searching every document. Tick documents in the panel to narrow it."
    : `Searching ${scope.size} of ${total ?? "?"} documents.`;
  el.scopeBtn.classList.toggle("is-active", scope.size > 0);

  if (el.selectAllBtn) {
    el.selectAllBtn.textContent = scope.size === 0 ? "Select all" : "Clear selection";
  }
  document.querySelectorAll(".doc-pick").forEach((box) => {
    box.checked = scope.has(box.dataset.doc);
  });
}

// The scope chip is a shortcut into the panel, where the ticking happens.
if (el.scopeBtn) el.scopeBtn.addEventListener("click", () => openDocs());

if (el.selectAllBtn) el.selectAllBtn.addEventListener("click", () => {
  // "Select all" and "no selection" mean the same search, so the button clears rather than
  // ticking every box - fewer boxes to untick afterwards, same result.
  scope.clear();
  renderScope();
});

/* ─────────────────────────── Documents slide-over ────────────────────
   The panel is a plain aside, not a <dialog>, because the chat has to stay visible behind
   it. Everything a dialog would have given for free is therefore explicit here: focus goes
   into the panel on open and back to the button on close, Escape closes it, and the panel
   is made non-interactive (visibility: hidden in CSS) rather than merely slid away, so its
   buttons are not still reachable by Tab.
   ==================================================================== */

let docsOpener = null;

function docsOpen() {
  return el.docsPanel.classList.contains("is-open");
}

function openDocs() {
  if (docsOpen()) return;
  docsOpener = document.activeElement;
  clearFinishedUploads();

  el.docsScrim.hidden = false;
  // One frame between "displayed" and "is-open" so the opacity transition has two states
  // to move between; without it the scrim appears fully opaque immediately.
  requestAnimationFrame(() => el.docsScrim.classList.add("is-open"));

  el.docsPanel.classList.add("is-open");
  el.docsPanel.setAttribute("aria-hidden", "false");
  el.docsBtn.setAttribute("aria-expanded", "true");
  setDrawer(false);          // on a phone the drawer would sit under the panel's scrim

  el.docsClose.focus({ preventScroll: true });
  refreshSources();          // the list is real backend data, so re-read it on every open
}

function closeDocs() {
  if (!docsOpen()) return;
  el.docsPanel.classList.remove("is-open");
  el.docsPanel.setAttribute("aria-hidden", "true");
  el.docsBtn.setAttribute("aria-expanded", "false");
  el.docsScrim.classList.remove("is-open");
  // Hide it only once the fade has finished, and only if it was not reopened meanwhile.
  window.setTimeout(() => { if (!docsOpen()) el.docsScrim.hidden = true; }, 200);

  if (docsOpener && document.contains(docsOpener)) docsOpener.focus({ preventScroll: true });
  docsOpener = null;
}

el.docsBtn.addEventListener("click", () => (docsOpen() ? closeDocs() : openDocs()));
el.docsClose.addEventListener("click", closeDocs);
el.docsScrim.addEventListener("click", closeDocs);
if (el.attachBtn) el.attachBtn.addEventListener("click", openDocs);

// Keep Tab inside the panel while it is open. Without this, tabbing past the last control
// lands in the chat behind a scrim that is meant to be blocking it.
el.docsPanel.addEventListener("keydown", (e) => {
  if (e.key !== "Tab" || !docsOpen()) return;
  const focusable = el.docsPanel.querySelectorAll(
    'a[href], button:not(:disabled), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'
  );
  if (!focusable.length) return;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
  else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
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
    // XHR is used here (not fetch) for upload-progress events, so the token has to be
    // attached by hand rather than by authFetch.
    const token = readToken();
    if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);

    xhr.upload.addEventListener("progress", (e) => {
      if (!e.lengthComputable) return;
      const percent = Math.round((e.loaded / e.total) * 100);
      // One request carries the whole batch, so every card in it shares the transfer bar.
      for (const card of cards) card.set(percent, `Uploading… ${percent}%`);
    });

    xhr.addEventListener("load", () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch { /* non-JSON error page */ }
      if (xhr.status >= 200 && xhr.status < 300) return resolve(body);
      if (xhr.status === 401) {
        endSession("Your session expired. Sign in again.");
        return reject(new Error("Signed out"));
      }
      reject(new Error(body.detail || `Upload failed (${xhr.status})`));
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
    const res = await authFetch(path, { method: "POST" });
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
      job = await (await authFetch("/ingest/status")).json();
    } catch (err) {
      // Signed out is not "lost contact" - the sign-in screen is already up, and saying
      // the backend died on top of it is a second, wrong explanation.
      if (!(err && err.signedOut)) {
        setStatus("Lost contact with the backend.", "error");
        setServerStatus("offline", "Offline", "The backend is not responding");
      }
      break;
    }

    renderActivity(job);

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
  setActivityRunning(false);
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
    const res = await authFetch("/stats");
    const data = await res.json();
    const docs = data.documents || [];

    el.docCount.textContent = docs.length;
    if (el.railCount) el.railCount.textContent = docs.length;
    if (data.max_upload_mb) {
      MAX_UPLOAD_MB = data.max_upload_mb;
      if (el.maxUploadMb) el.maxUploadMb.textContent = data.max_upload_mb;
    }
    renderQuota(data.storage_used_bytes || 0, data.storage_quota_bytes || 0);
    // The passage count is not shown in the sidebar any more; the element only exists if
    // someone puts it back.
    if (el.chunkCount) {
      el.chunkCount.textContent = data.total_chunks
        ? `${data.total_chunks.toLocaleString()} passages`
        : "empty";
    }

    if (!polling) {
      setServerStatus("online", "Online", `Connected to ${window.location.origin}`);
    }

    // Only fully-indexed documents appear here: the list comes from the manifest, and a
    // manifest entry is written after the last chunk of that file is stored.
    el.sourceList.innerHTML = docs.length === 0
      ? `<li class="doc-empty">Nothing ingested yet</li>`
      : docs.map((d) => `
        <li title="${escapeHtml(d.filename)}">
          <label class="doc-pick-wrap" title="Search this document">
            <input type="checkbox" class="doc-pick" data-doc="${escapeHtml(d.filename)}">
            <span class="sr-only">Search ${escapeHtml(shortDocName(d.filename))}</span>
          </label>
          <div class="doc-text">
            <span class="doc-name">${escapeHtml(shortDocName(d.filename))}</span>
            <span class="doc-meta">${d.size ? `${humanSize(d.size)} · ` : ""}${d.pages ?? "?"} pages · ${(d.chunks ?? 0).toLocaleString()} chunks</span>
            <span class="doc-state">ready</span>
          </div>
          <button type="button" class="icon-btn doc-remove" data-doc="${escapeHtml(d.filename)}"
                  title="Remove this document" aria-label="Remove ${escapeHtml(shortDocName(d.filename))}">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/></svg>
          </button>
        </li>`).join("");

    // Re-tick whatever is still selected, and drop anything that has been deleted.
    renderScope(docs.map((d) => d.filename));

    setActivityRunning(Boolean(data.ingesting));
    // A job may already be running when the page loads (e.g. the server just started).
    if (data.ingesting && !polling) pollUntilDone();
  } catch (err) {
    // A 401 has already put the sign-in screen up; calling the backend "offline" on top of
    // that is both wrong and alarming.
    if (err && err.signedOut) return;
    setServerStatus("offline", "Offline", "The backend is not responding");
  }
}

/* ─────────────────────────── Remove a document ───────────────────────── */

el.sourceList.addEventListener("change", (e) => {
  const box = e.target.closest(".doc-pick");
  if (!box) return;
  if (box.checked) scope.add(box.dataset.doc);
  else scope.delete(box.dataset.doc);
  renderScope();
});

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
    const res = await authFetch(`/documents/${encodeURIComponent(filename)}`, {
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

/** Start over: drop the conversation and its history, keep the documents. */
function newChat() {
  history = [];
  el.messages.replaceChildren(el.welcome);
  el.welcome.hidden = false;
  el.questionInput.focus();
}

// Two entry points, one behaviour: the sidebar button and the one inside the composer.
el.clearBtn.addEventListener("click", newChat);
if (el.newChatBtn) el.newChatBtn.addEventListener("click", () => {
  newChat();
  if (isNarrow()) setDrawer(false);   // on a phone the drawer covers what you just cleared
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

  // A hidden element has no layout, so scrollHeight is 0 - and writing that back as an
  // inline height left the box collapsed to its padding, with the placeholder sitting
  // against the top edge, once the app was revealed after sign-in. Measuring is only
  // meaningful while the element is actually rendered.
  if (!ta.offsetParent && ta.offsetHeight === 0) return;

  ta.style.height = "auto";
  const max = parseFloat(getComputedStyle(ta).maxHeight) || Infinity;
  const needed = ta.scrollHeight;
  if (!needed) return;                       // still not laid out; leave the CSS height

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
    const res = await authFetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        // A list, so a question can be scoped to several documents at once. Empty means
        // "search everything", which is what the server treats null as.
        sources: scopeList(),
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

/* ─────────────────────────── Sign in / sign up ───────────────────────
   One form, two modes. The screen is shown until /api/me confirms a stored token, and
   returned to on any 401.
   ==================================================================== */

let authMode = "login";      // "login" | "signup"

function setAuthMode(mode) {
  authMode = mode;
  const signingUp = mode === "signup";
  el.authTitle.textContent = signingUp ? "Create an account" : "Sign in";
  el.authText.textContent = signingUp
    ? "Documents you upload are private to your account."
    : "Your documents are private to your account.";
  el.authSubmitText.textContent = signingUp ? "Create account" : "Sign in";
  el.authSwitchText.textContent = signingUp ? "Already have an account?" : "New here?";
  el.authSwitch.textContent = signingUp ? "Sign in" : "Create an account";
  // Tells a password manager whether to offer a saved password or a generated one.
  el.authPass.setAttribute("autocomplete", signingUp ? "new-password" : "current-password");
  el.authError.textContent = "";
}

el.authSwitch.addEventListener("click", () => {
  setAuthMode(authMode === "login" ? "signup" : "login");
  el.authUser.focus();
});

function showAuthScreen(message) {
  // Always back to "Sign in": the message says the session expired, so a form still
  // labelled "Create an account" from earlier in the visit contradicts it - and submitting
  // it answers "Incorrect username or password" for an account that exists.
  setAuthMode("login");
  el.auth.hidden = false;
  el.app.hidden = true;
  el.authError.textContent = message || "";
  el.authPass.value = "";
  setTimeout(() => el.authUser.focus(), 50);
}

/**
 * Everything belonging to the previous account, gone.
 *
 * Without this, signing in as someone else shows their empty library alongside the
 * previous user's chat transcript and document rows until the first refresh lands - which
 * looks exactly like a data leak even though the server never sent a byte of it.
 */
function resetAppState() {
  history = [];
  polling = false;
  session = null;
  el.messages.replaceChildren(el.welcome);
  el.welcome.hidden = false;
  el.sourceList.innerHTML = `<li class="doc-empty">Nothing ingested yet</li>`;
  el.docCount.textContent = "0";
  if (el.railCount) el.railCount.textContent = "0";
  scope.clear();
  renderScope([]);
  el.uploadList.replaceChildren();
  uploadCards.clear();
  if (el.accountModal.open) el.accountModal.close();
  if (el.activityModal.open) el.activityModal.close();
  el.activityLog.innerHTML =
    `<li class="activity__empty">Nothing yet. Upload a PDF and the steps appear here.</li>`;
  setActivityRunning(false);
  el.passwordForm.reset();
  setStatus("");
  el.syncProgress.hidden = true;
  el.questionInput.value = "";
  autoGrow();
  closeDocs();
}

function startSession(username) {
  session = { username };
  el.whoName.textContent = username;
  el.whoAvatar.textContent = (username[0] || "?").toUpperCase();
  el.auth.hidden = true;
  el.app.hidden = false;
  // The composer could not be measured while the app was hidden; now it can.
  autoGrow();
  refreshSources();
  // Not awaited: the chat is usable immediately and this only updates the status dot.
  watchStartup();
  el.questionInput.focus();
}

/**
 * Ending a session, exactly once.
 *
 * Two guards, both learned the hard way. `sessionEpoch` is bumped so every background loop
 * started under the old session stops on its next tick - without it, a loop kept polling
 * with a dead token, which 401'd, which called this again, ~every second forever. And the
 * early return means a second (or fiftieth) 401 does not re-render the sign-in screen:
 * that re-render cleared the password field and pulled focus back to the username box, so
 * the form fought anyone trying to type in it.
 */
function endSession(message) {
  writeToken(null);
  sessionEpoch += 1;

  if (session === null && !el.auth.hidden) {
    // Already signed out and already showing the form; leave it alone.
    return;
  }
  session = null;
  resetAppState();
  showAuthScreen(message);
}

el.logoutBtn.addEventListener("click", () => endSession("You're signed out."));

el.authForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const username = el.authUser.value.trim();
  const password = el.authPass.value;

  el.authSubmit.disabled = true;
  el.authError.textContent = "";
  const original = el.authSubmitText.textContent;
  el.authSubmitText.textContent = authMode === "signup" ? "Creating…" : "Signing in…";

  try {
    const res = await fetch(`${API_BASE}/api/${authMode === "signup" ? "signup" : "login"}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    const body = await res.json().catch(() => ({}));

    if (!res.ok) {
      // FastAPI validation errors arrive as a list of objects, not a string.
      const detail = Array.isArray(body.detail)
        ? (body.detail[0]?.msg || "Check the username and password.")
        : body.detail;
      // 503 means the account database is down, not that the credentials are wrong. Its
      // message says how to start it, so show it verbatim rather than a generic failure.
      throw new Error(detail || (res.status === 503
        ? "The account database is unavailable."
        : `That didn't work (${res.status}).`));
    }

    writeToken(body.access_token);
    resetAppState();
    startSession(body.username || username);
  } catch (err) {
    el.authError.textContent = err.message === "Failed to fetch"
      ? "Can't reach the server. Is it running?"
      : err.message;
  } finally {
    el.authSubmit.disabled = false;
    el.authSubmitText.textContent = original;
  }
});

/* ─────────────────────────── Processing status ──────────────────────
   One place that answers "what is it doing with my PDF": the stage, the progress, and the
   trail of steps the server actually took. The data comes from /ingest/status, which
   reports only the caller's own files.
   ==================================================================== */

const STAGE_LABELS = {
  extracting: "Reading pages",
  embedding: "Embedding and storing passages",
};

function jobPercent(job) {
  const files = job.files_total || 0;
  // Idle with nothing ever run is 0%, not 100%: a full red bar under the word "Idle" reads
  // as "something finished just now" when in fact nothing has happened at all.
  if (!files) return job.state === "running" ? null : (job.results || []).length ? 100 : 0;
  let percent = (job.files_done / files) * 100;
  if (job.chunks_total) percent += (job.chunks_done / job.chunks_total) * (100 / files);
  return Math.min(100, Math.round(percent));
}

/** Keeps the sidebar button's dot in step with the job, panel open or not. */
function setActivityRunning(running) {
  el.activityDot.hidden = !running;
}

function renderActivity(job) {
  if (!job) return;
  const running = job.state === "running";
  setActivityRunning(running);
  if (!el.activityModal.open) return;

  const percent = jobPercent(job);
  el.activityStage.textContent = running
    ? (STAGE_LABELS[job.stage] || "Preparing")
    : (job.state === "error" ? "Failed" : "Idle");
  el.activityPercent.textContent = percent === null ? "—" : `${percent}%`;
  el.activityBar.style.width = `${percent === null ? 8 : percent}%`;
  el.activityBar.parentElement.classList.toggle("progress--indeterminate", percent === null);

  el.activityFile.textContent = job.current_file
    ? `${shortDocName(job.current_file)}${job.chunks_total ? ` · ${job.chunks_done}/${job.chunks_total} passages` : ""}`
    : (job.other_user_busy ? "The server is busy with someone else's document." : "");

  if (running) {
    const total = job.files_total || 0;
    el.activitySummary.textContent = total
      ? `Processing document ${Math.min(job.files_done + 1, total)} of ${total}.`
      : "Starting…";
  } else if (job.state === "error") {
    el.activitySummary.textContent = job.error || "The last run failed.";
  } else {
    const results = job.results || [];
    const count = (status) => results.filter((r) => r.status === status).length;
    const parts = [];
    if (count("ingested")) parts.push(`${count("ingested")} indexed`);
    if (count("skipped")) parts.push(`${count("skipped")} skipped`);
    if (count("failed")) parts.push(`${count("failed")} failed`);
    if (count("removed")) parts.push(`${count("removed")} removed`);
    el.activitySummary.textContent = parts.length
      ? `Last run: ${parts.join(" · ")}.`
      : "Nothing is being processed right now.";
  }

  const events = job.events || [];
  if (!events.length) {
    el.activityLog.innerHTML =
      `<li class="activity__empty">Nothing yet. Upload a PDF and the steps appear here.</li>`;
    return;
  }

  const mark = { done: "✓", warn: "!", error: "✕" };
  // Newest first: the interesting line is the one that just happened.
  el.activityLog.innerHTML = [...events].reverse().map((e) => `
    <li data-kind="${escapeHtml(e.kind || "info")}">
      <span class="activity__mark" aria-hidden="true">${mark[e.kind] || "·"}</span>
      <span>
        ${escapeHtml(e.message)}
        ${e.file ? `<span class="activity__doc">${escapeHtml(shortDocName(e.file))}</span>` : ""}
      </span>
      <span class="activity__time">${new Date(e.at * 1000).toLocaleTimeString()}</span>
    </li>`).join("");
}

async function pollActivity() {
  // Its own gentle loop, so the panel stays live even when no upload is in flight (a
  // startup scan, or another tab's upload). Stops as soon as the panel closes.
  while (el.activityModal.open) {
    try {
      renderActivity(await (await authFetch("/ingest/status")).json());
    } catch {
      break;                                  // signed out, or the server went away
    }
    await sleep(1000);
  }
}

el.activityBtn.addEventListener("click", async () => {
  if (!el.activityModal.open) el.activityModal.showModal();
  setDrawer(false);
  pollActivity();
});

el.activityModal.addEventListener("click", (e) => {
  if (e.target !== el.activityModal) return;
  const box = el.activityModal.getBoundingClientRect();
  if (e.clientX < box.left || e.clientX > box.right ||
      e.clientY < box.top || e.clientY > box.bottom) el.activityModal.close();
});

/* ─────────────────────────── Account dialog ─────────────────────────
   Password change, sign-out-everywhere, account deletion, and the storage quota. All four
   exist because a system that can create accounts and store files has to be able to undo
   both.
   ==================================================================== */

el.who.addEventListener("click", () => {
  el.accountName.textContent = session?.username || "";
  el.accountError.textContent = "";
  el.passwordForm.reset();
  refreshSources();                    // repaints the quota bar with current numbers
  if (!el.accountModal.open) el.accountModal.showModal();
});

function renderQuota(used, quota) {
  if (!quota || !el.quotaFill) return;
  const pct = Math.min(100, Math.round((used / quota) * 100));
  el.quotaFill.style.width = `${pct}%`;
  el.quotaFill.dataset.level = pct >= 95 ? "full" : pct >= 80 ? "warn" : "";
  el.quotaText.textContent = `${humanSize(used)} of ${humanSize(quota)}`;
}

el.passwordForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  el.accountError.textContent = "";
  el.passwordSubmit.disabled = true;
  try {
    const res = await authFetch("/api/me/password", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        current_password: el.currentPassword.value,
        new_password: el.newPassword.value,
      }),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || `Could not change it (${res.status}).`);

    // The server bumped this account's token version, so the old token is dead. It hands
    // back a fresh one; storing it is what keeps this session alive.
    writeToken(body.access_token);
    el.passwordForm.reset();
    el.accountError.textContent = "";
    setStatus("Password changed. Other devices have been signed out.", "ok");
    el.accountModal.close();
  } catch (err) {
    el.accountError.textContent = err.message;
  } finally {
    el.passwordSubmit.disabled = false;
  }
});

el.signoutAllBtn.addEventListener("click", async () => {
  if (!confirm("Sign out on every device, including this one?")) return;
  try {
    await authFetch("/api/me/signout-everywhere", { method: "POST" });
  } catch { /* a 401 already ended the session, which is the intended outcome */ }
  el.accountModal.close();
  endSession("Signed out everywhere.");
});

el.deleteAccountBtn.addEventListener("click", async () => {
  const password = prompt(
    "Delete your account?\n\nThis removes your PDFs from the server, their passages from " +
    "the index, and the account itself. It cannot be undone.\n\nType your password to confirm:"
  );
  if (!password) return;

  try {
    const res = await authFetch("/api/me", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || `Could not delete it (${res.status}).`);
    el.accountModal.close();
    endSession(`Account deleted (${body.documents_removed} document(s) removed).`);
  } catch (err) {
    el.accountError.textContent = err.message;
  }
});

/* ─────────────────────────── Startup watch ─────────────────────────
   There is no loading screen any more. The app is shown as soon as the session is settled,
   and the two slow things a cold start does - loading the embedding model (~15s) and
   indexing whatever is in the data folder - are reported in the top bar's status dot while
   they happen in the background.

   Blocking the UI on them was the wrong trade: you can read the page, open the Documents
   panel, and type a question during the wait, and a question asked early simply waits for
   the model rather than failing. The one thing that IS worth saying out loud is when the
   index is still being built, because an answer then may be missing passages - the dot's
   label and tooltip say so.
   ==================================================================== */

async function watchStartup() {
  const epoch = sessionEpoch;

  while (sessionEpoch === epoch) {
    let stats = null;
    try {
      stats = await (await authFetch("/stats")).json();
    } catch (err) {
      // Signed out is terminal for this loop. Anything else - the server not listening
      // yet, a dropped connection - is transient and worth another try.
      if (err && err.signedOut) return;
      setServerStatus("checking", "Starting", "Waiting for the server to answer…");
      await sleep(1500);
      continue;
    }

    if (stats.embedding_model_ready === false && !stats.ingesting) {
      setServerStatus("busy", "Warming up",
        "Loading the search model (about 15 seconds, once per server start). " +
        "You can type a question now - it will answer as soon as the model is ready.");
      await sleep(1500);
      continue;
    }

    if (stats.ingesting) {
      let job = {};
      try {
        job = await (await authFetch("/ingest/status")).json();
      } catch { /* transient; the next loop retries */ }
      const done = job.files_done ?? 0;
      const total = job.files_total ?? 0;
      const current = job.current_file ? ` — ${shortDocName(job.current_file)}` : "";
      setServerStatus("busy", "Indexing",
        total
          ? `Indexing document ${Math.min(done + 1, total)} of ${total}${current}. ` +
            "Answers may be missing passages until it finishes."
          : "Reading the data folder…");
      await sleep(1500);
      continue;
    }

    setServerStatus("online", "Online", `Connected to ${window.location.origin}`);
    return;
  }
}

/**
 * Decide which screen to show before anything else happens.
 *
 * A token in localStorage is not proof of a session: it may be expired, or signed with a
 * key that changed. /api/me is what actually settles it, so the app is never shown to
 * someone whose next request would 401.
 */
async function startup() {
  setAuthMode("login");

  if (!readToken()) {
    showAuthScreen();
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/api/me`, {
      headers: { Authorization: `Bearer ${readToken()}` },
    });
    if (!res.ok) {
      writeToken(null);
      showAuthScreen(res.status === 401 ? "" : "Please sign in again.");
      return;
    }
    const me = await res.json();
    startSession(me.username);
  } catch {
    // The server is unreachable. Sign-in will fail too, so say that rather than showing
    // an app that cannot load anything.
    showAuthScreen("Can't reach the server. Is it running?");
  }
}

startup();
