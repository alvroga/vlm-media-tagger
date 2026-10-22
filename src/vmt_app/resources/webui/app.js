/* VLM Media Tagger front end logic.
 *
 * Two-way bridge with the Python backend (vmt_app/app.py):
 *   JS -> Python : window.pywebview.api.<method>(...)  (pywebview js_api)
 *   Python -> JS : UI.dispatch({type: ..., ...})       (via evaluate_js)
 *
 * All dynamic text goes through textContent / createElement, never
 * innerHTML: file paths, share names and error messages come from outside.
 */
"use strict";

/* ---------- tiny helpers ---------- */

function $(id) { return document.getElementById(id); }

function api(method) {
  var args = Array.prototype.slice.call(arguments, 1);
  if (!window.pywebview || !window.pywebview.api || !window.pywebview.api[method]) {
    console.warn("bridge unavailable:", method, args);
    return Promise.resolve(null);
  }
  return window.pywebview.api[method].apply(window.pywebview.api, args).catch(function (e) {
    console.error("api." + method + " failed:", e);
  });
}

/* div.checkbox component: click-to-toggle with a change callback.
 * makeCheckbox($el, cb) -> {get, set(v), setDisabled(v)}; cb fires only on
 * USER clicks, never on programmatic set(), so applying a snapshot never
 * re-saves settings. */
function makeCheckbox(el, onChange) {
  el.addEventListener("click", function () {
    if (el.classList.contains("disabled")) return;
    var on = el.classList.toggle("on");
    if (onChange) onChange(on);
  });
  // The label next to the box toggles it too (the whole .check-row).
  var row = el.closest(".check-row");
  if (row) {
    row.addEventListener("click", function (ev) {
      if (ev.target === el || el.contains(ev.target)) return; // handled above
      if (el.classList.contains("disabled")) return;
      var on = el.classList.toggle("on");
      if (onChange) onChange(on);
    });
  }
  return {
    get: function () { return el.classList.contains("on"); },
    set: function (v) { el.classList.toggle("on", !!v); },
    setDisabled: function (v) { el.classList.toggle("disabled", !!v); },
  };
}

/* ---------- state ---------- */

var state = {
  engineReady: false,
  running: false,
  stopping: false,
  animTimer: null,
  animTick: 0,
  appName: "VLM Media Tagger",
};

var checkboxes = {};   // name -> makeCheckbox handle
var sourceRows = {};   // path -> {box, progressEl}

/* ---------- navigation ---------- */

function showScreen(targetId) {
  document.querySelectorAll(".nav-item").forEach(function (b) { b.classList.remove("active"); });
  document.querySelectorAll(".screen").forEach(function (s) { s.setAttribute("hidden", ""); });
  var navBtn = document.querySelector('.nav-item[data-target="' + targetId + '"]');
  if (navBtn) navBtn.classList.add("active");
  $(targetId).removeAttribute("hidden");
}

/* ---------- engine status (sidebar) ---------- */

function setEngine(ev) {
  var pill = $("engine-pill");
  var dot = pill.querySelector(".dot");
  dot.className = "dot " + (ev.state === "ok" ? "good" : ev.state === "error" ? "bad" : "busy");
  $("engine-text").textContent = ev.text || "";
  state.engineReady = ev.state === "ok";
  $("btn-analyze").disabled = state.running;
}

/* DaVinci Resolve Studio light (sidebar): the detail line lives in Settings → Integrations. */
var RESOLVE_LIGHT = {
  off: ["off", "DaVinci Resolve Studio off"],
  busy: ["busy", "DaVinci Resolve Studio starting…"],
  ok: ["good", "✓ DaVinci Resolve Studio ready"],
  error: ["bad", "✗ DaVinci Resolve Studio unavailable"],
};

function setResolveLight(light, detail) {
  var spec = RESOLVE_LIGHT[light] || RESOLVE_LIGHT.off;
  var text = spec[1];
  if (light === "busy" && /^Stopping/.test(detail || "")) text = "DaVinci Resolve Studio stopping…";
  $("resolve-pill").querySelector(".dot").className = "dot " + spec[0];
  $("resolve-text").textContent = text;
}

/* ---------- run status animation ---------- */

function setRunStatus(text, cls) {
  var el = $("run-status");
  el.textContent = text || "";
  el.className = "run-status" + (cls ? " " + cls : "");
}

// A JS-driven glyph swap, not a CSS animation: macOS Reduce Motion freezes
// CSS animations, and this must keep showing that work is happening.
var TORCH_FRAMES = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"];

function updateAnim() {
  var dots = [".", "..", "..."];
  var label = state.stopping ? "Stopping" : "Running";
  setRunStatus(label + dots[state.animTick % 3], state.stopping ? "accent" : "");
  var torch = $("torch");
  if (torch) torch.textContent = TORCH_FRAMES[state.animTick % TORCH_FRAMES.length];
  state.animTick += 1;
}

function setRunning(running, stopping) {
  state.running = running;
  state.stopping = !!stopping;
  $("btn-analyze").disabled = running;
  $("btn-stop").disabled = !running || state.stopping;
  document.body.classList.toggle("running", running);
  if (running && !state.animTimer) {
    state.animTick = 0;
    updateAnim();
    state.animTimer = setInterval(updateAnim, 500);
  } else if (!running && state.animTimer) {
    clearInterval(state.animTimer);
    state.animTimer = null;
    var torch = $("torch");
    if (torch) torch.textContent = "";
  }
  if (running) updateAnim();
}

/* ---------- run log ---------- */

var LOG_MAX_LINES = 4000;

function logClear() {
  $("runlog").textContent = "";
}

function logAppend(t, text) {
  var log = $("runlog");
  var placeholder = log.querySelector(".placeholder");
  if (placeholder) placeholder.remove();
  var ln = document.createElement("div");
  ln.className = "ln";
  if (/^\[\d+\/\d+\] \//.test(text)) ln.classList.add("lvl-head");
  else if (text.indexOf("ERROR") !== -1 || text.indexOf("Aborting:") === 0) ln.classList.add("lvl-error");
  if (t) {
    var ts = document.createElement("span");
    ts.style.color = "var(--text-faint)";
    ts.textContent = t + "  ";
    ln.appendChild(ts);
  }
  ln.appendChild(document.createTextNode(text === "" ? " " : text));
  log.appendChild(ln);
  while (log.childNodes.length > LOG_MAX_LINES) log.removeChild(log.firstChild);
  log.scrollTop = log.scrollHeight;
}

/* ---------- progress ---------- */

function setProgress(ev) {
  var fill = $("progress-fill");
  if (ev.indeterminate) {
    fill.classList.add("indeterminate");
    $("progress-count").textContent = ev.label || "";
  } else {
    fill.classList.remove("indeterminate");
    var total = Math.max(ev.total, 1);
    fill.style.width = Math.round((100 * ev.completed) / total) + "%";
    $("progress-count").textContent = ev.completed + " / " + ev.total + " file(s) processed";
  }
}

function resetProgress() {
  var fill = $("progress-fill");
  fill.classList.remove("indeterminate");
  fill.style.width = "0%";
  $("progress-count").textContent = "";
  $("progress-phase").textContent = "Idle";
  $("progress-file").textContent = "";
}

/* ---------- sources ---------- */

function isComplete(progress) {
  return !!progress && progress.total > 0 && progress.done >= progress.total;
}

function progressText(progress) {
  if (!progress) return "";
  if (progress.total === 0) return "No media files found";
  var text = progress.done.toLocaleString() + " / " + progress.total.toLocaleString() + " analyzed";
  if (progress.done > 0 && !isComplete(progress)) {
    text += " · " + (progress.total - progress.done).toLocaleString() + " to go";
  }
  return text;
}

// The "Done" tag and dimmed row for a source with nothing left to analyze.
function applyComplete(rowEl, donePill, progress) {
  var complete = isComplete(progress);
  donePill.hidden = !complete;
  rowEl.classList.toggle("complete", complete);
}

var clearConfirmTimer = null;

function resetClearConfirm() {
  clearTimeout(clearConfirmTimer);
  var btn = $("btn-clear-sources");
  btn.classList.remove("confirm");
  btn.textContent = "Clear list";
}

function smallButton(cls, text, title, onClick) {
  var btn = document.createElement("button");
  btn.type = "button";
  btn.className = cls;
  btn.textContent = text;
  if (title) btn.title = title;
  btn.addEventListener("click", function (ev) { ev.stopPropagation(); onClick(); });
  return btn;
}

function renderSources(rows) {
  var list = $("sources-list");
  list.textContent = "";
  sourceRows = {};
  (rows || []).forEach(function (row) {
    var rowEl = document.createElement("div");
    rowEl.className = "source-row";

    var box = document.createElement("div");
    box.className = "checkbox" + (row.selected ? " on" : "");
    rowEl.appendChild(box);

    var kind = document.createElement("span");
    kind.className = "source-kind";
    kind.textContent = row.kind === "folder" ? "Folder" : row.kind === "file" ? "File" : "—";
    rowEl.appendChild(kind);

    var nameEl = document.createElement("div");
    nameEl.className = "source-name";
    var t = document.createElement("div");
    t.className = "t";
    t.textContent = row.name;
    t.title = row.path;
    nameEl.appendChild(t);
    var progressEl = document.createElement("div");
    progressEl.className = "progress";
    progressEl.textContent = progressText(row.progress);
    nameEl.appendChild(progressEl);
    rowEl.appendChild(nameEl);

    var statusEl = document.createElement("div");
    statusEl.className = "source-status";
    statusEl.textContent = row.share || row.location;
    statusEl.title = row.path;
    rowEl.appendChild(statusEl);

    var pill = document.createElement("span");
    if (row.kind === "missing") {
      pill.className = "pill bad";
      pill.textContent = row.share ? "Not mounted" : "Missing";
    } else if (row.share) {
      pill.className = "pill neutral";
      pill.textContent = "Network";
    } else {
      pill.hidden = true;
    }
    rowEl.appendChild(pill);

    var donePill = document.createElement("span");
    donePill.className = "pill good";
    donePill.textContent = "✓ Done";
    donePill.title = "Every file here is analyzed. New files added later will be picked up on the next run.";
    rowEl.appendChild(donePill);
    applyComplete(rowEl, donePill, row.kind === "missing" ? null : row.progress);

    rowEl.appendChild(smallButton("source-remove-btn", "Remove", "Remove this source (its results stay on disk)",
      function () { api("remove_source", row.path); }));

    list.appendChild(rowEl);

    var handle = makeCheckbox(box, function (on) { api("set_source_selected", row.path, on); });
    t.addEventListener("click", function () {
      var on = box.classList.toggle("on");
      api("set_source_selected", row.path, on);
    });
    t.addEventListener("dblclick", function () { api("reveal_path", row.path); });
    sourceRows[row.path] = { box: handle, progressEl: progressEl, rowEl: rowEl, donePill: donePill };
  });
  if (!rows || !rows.length) {
    var empty = document.createElement("div");
    empty.className = "field-note";
    empty.style.padding = "14px 2px";
    empty.textContent = "No sources yet. Add a folder (scanned with all its subfolders), individual files, " +
      "or a folder on a network volume.";
    list.appendChild(empty);
  }
}

function applySources(ev) {
  renderSources(ev.sources);
  if (ev.sources_status) $("sources-status").textContent = ev.sources_status;
}

function updateSourceProgress(ev) {
  var entry = sourceRows[ev.path];
  if (!entry) return;
  var progress = { done: ev.done, total: ev.total };
  entry.progressEl.textContent = progressText(progress);
  applyComplete(entry.rowEl, entry.donePill, progress);
}

/* ---------- skipped files ---------- */

function renderSkipped(ev) {
  var tbody = $("skipped-tbody");
  tbody.textContent = "";
  (ev.rows || []).forEach(function (r) {
    var tr = document.createElement("tr");
    [r.source, r.path, r.reason, r.logged_at].forEach(function (val, i) {
      var td = document.createElement("td");
      if (i === 1) {
        td.className = "file-cell";
        td.title = "Double-click to show in Finder";
        td.addEventListener("dblclick", function () { api("reveal_path", r.path); });
      }
      if (i === 3) td.className = "mono";
      td.textContent = val || "";
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  $("skipped-status").textContent = ev.message || "";
  $("skipped-count").textContent = String(ev.count);
  var badge = $("nav-skipped-badge");
  badge.textContent = String(ev.count);
  badge.hidden = !ev.count;
}

/* ---------- setup overlay ---------- */

function setSetup(ev) {
  var overlay = $("setup-overlay");
  if (ev.mode === "hidden") {
    overlay.setAttribute("hidden", "");
    return;
  }
  overlay.removeAttribute("hidden");
  // "load" = the model is already downloaded: spinner, no download copy.
  var loading = ev.stage === "load";
  overlay.classList.toggle("loading", loading);
  $("setup-title").textContent = loading ? "Starting the AI model" : "First-time setup";
  $("setup-spinner").hidden = !loading || ev.mode === "error";
  var fill = $("setup-fill");
  var status = $("setup-status");
  var retry = $("btn-setup-retry");
  if (ev.mode === "error") {
    fill.classList.remove("indeterminate");
    fill.style.width = "0%";
    status.textContent = "✗ " + ev.message;
    status.className = "setup-status err";
    retry.hidden = false;
    return;
  }
  retry.hidden = true;
  status.className = "setup-status";
  if (ev.total) {
    fill.classList.remove("indeterminate");
    fill.style.width = Math.round((100 * ev.done) / ev.total) + "%";
    status.textContent =
      ev.phase + "  —  " + (ev.done / 1e9).toFixed(2) + " of " +
      (ev.total / 1e9).toFixed(2) + " GB (" + Math.round((100 * ev.done) / ev.total) + "%)";
  } else {
    fill.classList.add("indeterminate");
    status.textContent = ev.phase || "Starting…";
  }
}

/* ---------- language pack download ---------- */

var appliedLang = "en";

function setLangDownload(ev) {
  var box = $("lang-download"), fill = $("lang-download-fill"),
      status = $("lang-download-status"), sel = $("sel-language");
  if (ev.mode === "applied") {
    appliedLang = ev.lang;
    return;
  }
  if (ev.mode === "progress") {
    box.hidden = false;
    sel.disabled = true;
    var pct = ev.total ? Math.round((100 * ev.done) / ev.total) : 0;
    fill.style.width = pct + "%";
    var mb = ev.total > 1
      ? "  " + Math.round(ev.done / 1e6) + " / " + Math.round(ev.total / 1e6) + " MB"
      : "";
    status.textContent = (ev.phase || "Downloading…") + mb;
  } else if (ev.mode === "done") {
    fill.style.width = "100%";
    status.textContent = "Downloaded — ready.";
    sel.disabled = false;
    appliedLang = ev.lang;
    setTimeout(function () { $("lang-download").hidden = true; }, 2500);
  } else if (ev.mode === "error") {
    box.hidden = false;
    sel.disabled = false;
    status.textContent = "Download failed: " + (ev.message || "unknown error") + " — language not switched.";
    sel.value = appliedLang;
  }
}

/* ---------- event dispatch (Python -> JS) ---------- */

var UI = {
  dispatch: function (ev) {
    switch (ev.type) {
      case "setup": setSetup(ev); break;
      case "engine": setEngine(ev); break;
      case "lang_download": setLangDownload(ev); break;
      case "sources": applySources(ev); break;
      case "sources_status": $("sources-status").textContent = ev.message || ""; break;
      case "source_progress": updateSourceProgress(ev); break;
      case "run_state": setRunning(ev.running, ev.stopping); break;
      case "run_status": setRunStatus(ev.text, ev.cls); break;
      case "log": logAppend(ev.t, ev.text); break;
      case "log_clear": logClear(); break;
      case "phase": $("progress-phase").textContent = ev.text || "Idle"; break;
      case "progress": setProgress(ev); break;
      case "progress_file": $("progress-file").textContent = ev.text || ""; break;
      case "progress_reset": resetProgress(); break;
      case "auto": $("auto-status").textContent = ev.text || ""; break;
      case "resolve": {
        var rs = $("resolve-status");
        rs.textContent = ev.text || "";
        rs.className = "status-line" + (ev.cls ? " " + ev.cls : "");
        setResolveLight(ev.light, ev.text);
        break;
      }
      case "skipped": renderSkipped(ev); break;
      default: console.warn("unknown event", ev);
    }
  },
};
window.UI = UI;

/* ---------- boot ---------- */

function applySnapshot(s) {
  state.appName = s.app_name || state.appName;
  $("sidebar-version").textContent = state.appName + " " + s.version;
  $("data-dir-path").textContent = s.app_support_dir;
  $("logs-dir-path").textContent = s.logs_dir;
  $("engine-description").textContent = s.engine || "";
  $("sinks-list").textContent = (s.sinks || []).join(", ") || "none";
  var dest = "→ " + ((s.sinks || []).indexOf("sidecar") !== -1 ? "JSON sidecar" : (s.sinks || []).join(", "));
  $("task-dest-tags").textContent = dest;
  $("task-dest-desc").textContent = dest;

  checkboxes.tags.set(s.tasks.tags);
  checkboxes.describe.set(s.tasks.description);

  checkboxes.auto.set(s.auto.enabled);
  $("sel-interval").value = String(s.auto.minutes);
  $("sel-interval").disabled = !s.auto.enabled;
  checkboxes.pause.set(s.pause.enabled);
  checkboxes.pause.setDisabled(!s.auto.enabled);
  $("in-pause-start").value = s.pause.start;
  $("in-pause-end").value = s.pause.end;
  updatePauseInputs();
  updateAutomationNote();

  checkboxes.resolve.set(s.resolve_enabled);
  setResolveLight(s.resolve_light);
  checkboxes.reconnectShares.set(s.reconnect_shares);
  checkboxes.unmountOnQuit.set(s.unmount_on_quit);
  checkboxes.filterProfanity.set(s.filter_profanity);
  checkboxes.redoOtherModel.set(s.redo_other_model);
  $("in-log-retention-days").value = s.log_retention_days || 30;

  var sel = $("sel-language");
  sel.textContent = "";
  (s.languages || [{ code: "en", name: "English" }]).forEach(function (lang) {
    var opt = document.createElement("option");
    opt.value = lang.code;
    opt.textContent = lang.name;
    sel.appendChild(opt);
  });
  sel.value = s.language || "en";
  appliedLang = s.language || "en";

  applySources(s);
  if (s.skipped) renderSkipped(s.skipped);
  $("btn-analyze").disabled = state.running;
}

function updatePauseInputs() {
  var on = checkboxes.auto.get() && checkboxes.pause.get();
  $("in-pause-start").disabled = !on;
  $("in-pause-end").disabled = !on;
}

/* Status line on the Analyze screen (the controls live in Settings ->
 * Automation), derived from the control states. */
function updateAutomationNote() {
  var note = $("automation-note");
  if (!checkboxes.auto.get()) {
    note.textContent = "Automatic re-run is off.";
    return;
  }
  var opt = $("sel-interval").selectedOptions[0];
  var text = "Automatic re-run: " + (opt ? opt.textContent : "on");
  if (checkboxes.pause.get()) {
    text += " (paused " + $("in-pause-start").value + "–" + $("in-pause-end").value + ")";
  }
  note.textContent = text;
}

function pushPause() {
  api("set_pause", checkboxes.pause.get(), $("in-pause-start").value || "09:00",
      $("in-pause-end").value || "18:00");
}

function wireControls() {
  document.querySelectorAll(".nav-item").forEach(function (btn) {
    btn.addEventListener("click", function () { showScreen(btn.dataset.target); });
  });

  document.querySelectorAll(".tab-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      document.querySelectorAll(".tab-btn").forEach(function (b) { b.classList.remove("active"); });
      document.querySelectorAll(".tab-panel").forEach(function (p) { p.setAttribute("hidden", ""); });
      btn.classList.add("active");
      $(btn.dataset.tab).removeAttribute("hidden");
    });
  });

  // sources
  $("btn-add-folder").addEventListener("click", function () { api("add_folders"); });
  $("btn-add-files").addEventListener("click", function () { api("add_files"); });
  $("btn-refresh-sources").addEventListener("click", function () { api("refresh_sources"); });
  $("btn-edit-sources").addEventListener("click", function () {
    var on = $("panel-sources").classList.toggle("edit-mode");
    $("btn-edit-sources").textContent = on ? "Done" : "Edit list";
    resetClearConfirm();
  });
  $("btn-remove-finished").addEventListener("click", function () { api("remove_finished_sources"); });
  // Clear list asks for a second click within 4 s (no native confirm dialog
  // in the web view).
  $("btn-clear-sources").addEventListener("click", function () {
    var btn = $("btn-clear-sources");
    if (btn.classList.contains("confirm")) {
      resetClearConfirm();
      api("clear_sources");
      return;
    }
    btn.classList.add("confirm");
    btn.textContent = "Click again to clear";
    clearConfirmTimer = setTimeout(resetClearConfirm, 4000);
  });
  checkboxes.selectAll = makeCheckbox($("cb-select-all"), function (on) { api("select_all_sources", on); });
  $("btn-show-connect").addEventListener("click", function () {
    var row = $("connect-row");
    row.hidden = !row.hidden;
    if (!row.hidden) $("in-server").focus();
  });
  function doConnect() {
    var url = $("in-server").value.trim();
    if (url) api("connect_server", url);
  }
  $("btn-connect").addEventListener("click", doConnect);
  $("in-server").addEventListener("keydown", function (ev) { if (ev.key === "Enter") doConnect(); });

  // tasks
  function pushTasks() { api("set_tasks", checkboxes.tags.get(), checkboxes.describe.get()); }
  checkboxes.tags = makeCheckbox($("cb-tags"), pushTasks);
  checkboxes.describe = makeCheckbox($("cb-describe"), pushTasks);

  // automation (controls in Settings -> Automation)
  checkboxes.auto = makeCheckbox($("cb-auto"), function (on) {
    $("sel-interval").disabled = !on;
    checkboxes.pause.setDisabled(!on);
    updatePauseInputs();
    updateAutomationNote();
    api("set_auto", on);
  });
  $("sel-interval").addEventListener("change", function () {
    updateAutomationNote();
    api("set_auto_interval", parseInt($("sel-interval").value, 10));
  });
  checkboxes.pause = makeCheckbox($("cb-pause"), function () {
    updatePauseInputs();
    updateAutomationNote();
    pushPause();
  });
  $("in-pause-start").addEventListener("change", function () { updateAutomationNote(); pushPause(); });
  $("in-pause-end").addEventListener("change", function () { updateAutomationNote(); pushPause(); });
  $("btn-goto-automation").addEventListener("click", function () {
    showScreen("screen-settings");
    var tabBtn = document.querySelector('.tab-btn[data-tab="tab-automation"]');
    if (tabBtn) tabBtn.click();
  });

  // run
  $("btn-analyze").addEventListener("click", function () { api("analyze"); });
  $("btn-stop").addEventListener("click", function () { api("stop"); });
  $("link-to-log").addEventListener("click", function (ev) {
    ev.preventDefault();
    showScreen("screen-log");
  });
  $("skipped-stat").addEventListener("click", function () { showScreen("screen-skipped"); });

  // skipped / log
  $("btn-refresh-skipped").addEventListener("click", function () { api("refresh_skipped"); });
  $("btn-retry-skipped").addEventListener("click", function () { api("retry_skipped"); });
  $("btn-reveal-logs").addEventListener("click", function () { api("reveal_logs"); });

  // settings
  $("btn-reveal-data").addEventListener("click", function () { api("reveal_app_support_dir"); });
  checkboxes.resolve = makeCheckbox($("cb-resolve"), function (on) { api("set_resolve", on); });
  checkboxes.reconnectShares = makeCheckbox($("cb-reconnect-shares"), function (on) { api("set_reconnect_shares", on); });
  checkboxes.unmountOnQuit = makeCheckbox($("cb-unmount-on-quit"), function (on) { api("set_unmount_on_quit", on); });
  checkboxes.filterProfanity = makeCheckbox($("cb-filter-profanity"), function (on) { api("set_filter_profanity", on); });
  checkboxes.redoOtherModel = makeCheckbox($("cb-redo-other-model"), function (on) { api("set_redo_other_model", on); });
  $("sel-language").addEventListener("change", function () { api("set_language", $("sel-language").value); });
  $("in-log-retention-days").addEventListener("change", function () {
    var el = $("in-log-retention-days");
    var days = parseInt(el.value, 10);
    if (!days || days < 1) days = 30;
    el.value = days;
    api("set_log_retention_days", days);
  });

  // setup overlay
  $("btn-setup-retry").addEventListener("click", function () {
    $("btn-setup-retry").hidden = true;
    api("setup_retry");
  });
}

function boot() {
  api("init").then(function (snapshot) {
    if (snapshot) applySnapshot(snapshot);
  });
}

// Loaded at the end of <body>, so the DOM is parsed: wire controls now
// (checkbox handles must exist before the first snapshot, and
// pywebviewready can fire before DOMContentLoaded).
wireControls();
if (window.pywebview && window.pywebview.api) {
  boot();
} else {
  window.addEventListener("pywebviewready", boot);
}
