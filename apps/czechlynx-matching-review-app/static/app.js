"use strict";

// Timed matching review (N3). One candidate per screen; every action is logged as an event with
// performance.now() timestamps. Decision time = screen shown (images decoded) to answer, minus pauses;
// an interval that ends in an undo belongs to the step returned to. The repo's re-costing tool
// recomputes the same quantities from the raw events.

const $ = (id) => document.getElementById(id);
const QUERY_SHOWN = 6;
const MIN_DECISION_MS = 200;  // Ignore answers faster than this after a screen appears (double presses).
const BLOB_CACHE = 600;
const LABELS = {
  audit: {same: "Correct", different: "Wrong", unsure: "Unsure"},
  pair: {same: "Same lynx", different: "Different lynx", unsure: "Unsure"},
  default: {same: "Same lynx", different: "Different", unsure: "Unsure"},
};
const PAUSE_REASONS = {
  manual: "You paused the clock.", hidden: "The page was hidden, so the clock stopped.",
  blur: "The window lost focus, so the clock stopped.", idle: "No activity for a while, so the clock stopped.",
};
const PROFILE = {name: "p-name", years_identifying_individuals: "p-years", species: "p-species",
  lynx_or_felid_experience: "p-felid", czechlynx_exposure: "p-exposure", knows_system_internals: "p-internals",
  device: "p-device", pointing_device: "p-pointer", notes: "p-notes"};

const state = {
  reviewer: null, sessionId: randomHex(16), seq: 0, queue: [], sequence: [], completed: new Set(), settings: {},
  index: -1, task: null, step: 1, answers: {}, stats: {}, screenReady: false, shownAt: null, pausedSince: null,
  pausedAcc: 0, paused: false, awaitingConfirm: false, feedbackPending: null, variant: "original",
  unidentifiable: false, lastInput: performance.now(), blobs: new Map(), tasks: new Map(), mode: "start",
  busy: false, renderToken: 0, queryTiles: [], candidateTiles: [], extraTiles: [], lastBlock: null, zoomSide: null,
};
let flushChain = Promise.resolve();

function randomHex(bytes) {
  const values = new Uint8Array(bytes);
  crypto.getRandomValues(values);
  return Array.from(values, (v) => v.toString(16).padStart(2, "0")).join("");
}

function remember(key, value) { try { localStorage.setItem(`lynx-matching:${key}`, value); } catch (_) {} }
function recall(key) { try { return localStorage.getItem(`lynx-matching:${key}`); } catch (_) { return null; } }
const round = (ms) => Math.round(ms * 10) / 10;

async function api(path, body, keepalive = false) {
  const options = body === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json", "X-Review-App": "1"},
    body: JSON.stringify(body), keepalive,
  };
  const response = await fetch(path, {...options, cache: "no-store"});
  const result = await response.json();
  if (!response.ok) {
    const error = new Error(result.error || "The workstation could not handle this request.");
    error.conflict = !!result.conflict;
    throw error;
  }
  return result;
}

function message(text, error = false) {
  $("message").textContent = text;
  $("message").classList.toggle("error", error);
  $("message").hidden = !text;
}

// --- Events ---------------------------------------------------------------------------------

function emit(type, fields = {}) {
  if (!state.reviewer) return;
  state.queue.push({seq: state.seq++, type, t: performance.now(), wall: new Date().toISOString(), ...fields});
  if (state.queue.length >= 50) flush();
}

function flush(keepalive = false) {
  flushChain = flushChain.then(async () => {
    while (state.queue.length) {
      const batch = state.queue.slice(0, 200);
      await api("/api/events", {reviewer: state.reviewer, session_id: state.sessionId, events: batch}, keepalive);
      state.queue.splice(0, batch.length);
    }
  }).catch((error) => message(`Timing events are not saved yet; the app will retry. ${error.message}`, true));
  return flushChain;
}

// --- Images ---------------------------------------------------------------------------------

function blob(token, variant) {
  const key = `${token}/${variant}`;
  if (state.blobs.has(key)) {
    const cached = state.blobs.get(key);
    state.blobs.delete(key);
    state.blobs.set(key, cached);
    return cached;
  }
  const promise = fetch(`/images/${token}/${variant}`, {cache: "no-store"})
    .then((response) => { if (!response.ok) throw new Error("An image could not be loaded."); return response.blob(); })
    .then((data) => URL.createObjectURL(data));
  promise.catch(() => state.blobs.delete(key));
  state.blobs.set(key, promise);
  while (state.blobs.size > BLOB_CACHE) {
    const [oldKey, old] = state.blobs.entries().next().value;
    state.blobs.delete(oldKey);
    old.then((url) => URL.revokeObjectURL(url), () => {});
  }
  return promise;
}

function tile(photo, side) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "tile";
  const img = document.createElement("img");
  img.alt = side === "query" ? "Photo to identify" : "Catalogue photo";
  const caption = document.createElement("span");
  caption.textContent = `${photo.date} · camera ${photo.camera}`;
  button.append(img, caption);
  button.addEventListener("click", () => { button.blur(); openZoom(photo, side, caption.textContent); });
  return {button, img, photo};
}

async function load(item) {
  const url = await blob(item.photo.token, state.variant);
  item.img.src = url;
  await item.img.decode().catch(() => {});
}

function getTask(taskId) {
  if (!state.tasks.has(taskId)) {
    const promise = api(`/api/task/${encodeURIComponent(taskId)}?reviewer=${encodeURIComponent(state.reviewer)}`);
    promise.catch(() => state.tasks.delete(taskId));
    state.tasks.set(taskId, promise);
  }
  return state.tasks.get(taskId);
}

function prefetch(task, step) {
  const next = task.screens[step];
  if (next) next.exemplars.forEach((p) => blob(p.token, state.variant).catch(() => {}));
  const entry = state.sequence[state.index + 1];
  if (entry && step >= task.screens.length - 1) {
    getTask(entry.task_id).then((t) => {
      [...t.query.slice(0, QUERY_SHOWN), ...t.screens[0].exemplars].forEach((p) => blob(p.token, "original").catch(() => {}));
    }).catch(() => {});
  }
}

// --- Screens --------------------------------------------------------------------------------

function showSection(name) {
  for (const id of ["start", "interstitial", "workspace"]) $(id).hidden = id !== name;
  state.mode = name === "workspace" ? "task" : name;
}

function stats(step) {
  if (!state.stats[step]) state.stats[step] = {ms: 0, paused_ms: 0, undo_count: 0, zooms: 0, expands: 0, toggles: 0};
  return state.stats[step];
}

function updateControls() {
  const ready = state.screenReady && !state.paused && !state.awaitingConfirm && !state.feedbackPending;
  document.querySelectorAll(".answer").forEach((b) => { b.disabled = !ready; });
  $("undo").disabled = state.paused || !!state.feedbackPending
    || !(state.awaitingConfirm || (state.screenReady && state.step > 1));
  $("variant").disabled = !state.screenReady || state.paused;
  $("variant").firstChild.textContent = state.variant === "original" ? "Masked view " : "Original view ";
  $("pause").disabled = state.paused;
  $("confirm-next").disabled = state.busy;
}

function setReady(ready) {
  state.screenReady = ready;
  updateControls();
}

function updateProgress() {
  const main = state.sequence.filter((e) => e.block > 0);
  const done = main.filter((e) => state.completed.has(e.task_id)).length;
  $("progress").max = main.length || 1;
  $("progress").value = done;
  $("progress-label").textContent = `${done} of ${main.length} timed tasks complete`;
  const entry = state.sequence[state.index];
  const blocks = state.settings.blocks || 12;
  $("block-label").textContent = !entry ? "" : entry.block === 0 ? "Practice (not timed for the results)"
    : `Block ${entry.block} of ${blocks} · reviewer ${state.reviewer}`;
}

function renderQuery() {
  const task = state.task;
  $("query-label").textContent = task.query_label;
  $("query-meta").textContent = `${task.query.length} photo${task.query.length === 1 ? "" : "s"}`;
  state.queryTiles = task.query.slice(0, QUERY_SHOWN).map((p) => tile(p, "query"));
  $("query-grid").replaceChildren(...state.queryTiles.map((t) => t.button));
  $("query-more").hidden = task.query.length <= QUERY_SHOWN;
  $("query-extra").hidden = true;
  $("query-extra").replaceChildren();
  state.extraTiles = [];
}

function renderCandidate(screen) {
  $("candidate-label").textContent = state.task.candidate_label;
  const position = screen.list_length > 1 ? `Candidate ${screen.position} of ${screen.list_length}` : "";
  $("candidate-meta").textContent = [position, `${screen.members} photo${screen.members === 1 ? "" : "s"} in the catalogue`]
    .filter(Boolean).join(" · ");
  state.candidateTiles = screen.exemplars.map((p) => tile(p, "candidate"));
  $("candidate-grid").replaceChildren(...state.candidateTiles.map((t) => t.button));
  $("candidate-more").hidden = !screen.more.length;
  $("candidate-extra").hidden = true;
  $("candidate-extra").replaceChildren();
  state.extraTiles = state.extraTiles.filter((t) => t.side === "query");
  $("position").textContent = state.task.screens.length > 1 ? `Screen ${state.step} of ${state.task.screens.length}` : "";
  $("position").hidden = state.task.screens.length <= 1;
}

async function showScreen() {
  const token = ++state.renderToken;
  const task = state.task;
  const step = state.step;
  setReady(false);
  const started = performance.now();
  renderCandidate(task.screens[step - 1]);
  try {
    await Promise.all([...state.queryTiles, ...state.candidateTiles].map(load));
  } catch (error) {
    if (token === state.renderToken) message(`${error.message} Check the workstation, then reload this page.`, true);
    return;
  }
  if (token !== state.renderToken) return;
  state.shownAt = performance.now();
  state.pausedAcc = 0;
  state.pausedSince = state.paused ? state.shownAt : null;
  setReady(true);
  emit("screen_shown", {task_id: task.task_id, step, load_ms: round(state.shownAt - started)});
  prefetch(task, step);
}

async function startTask(index) {
  state.index = index;
  const entry = state.sequence[index];
  showSection("workspace");
  setReady(false);
  let task;
  try {
    task = await getTask(entry.task_id);
  } catch (error) {
    message(`${error.message} Reload this page to continue.`, true);
    return;
  }
  Object.assign(state, {task, step: 1, answers: {}, stats: {}, unidentifiable: false, variant: "original",
    awaitingConfirm: false, feedbackPending: null});
  $("unidentifiable").checked = false;
  $("confirm").hidden = true;
  $("feedback").hidden = true;
  if (entry.block !== state.lastBlock) {
    emit("block_start", {block: entry.block});
    state.lastBlock = entry.block;
  }
  emit("task_start", {task_id: task.task_id, block: entry.block});
  $("banner").textContent = task.practice ? `Practice · ${task.banner}` : task.banner;
  const labels = LABELS[task.type] || LABELS.default;
  for (const button of document.querySelectorAll(".answer")) button.firstElementChild.textContent = labels[button.dataset.answer];
  renderQuery();
  updateProgress();
  showScreen();
}

// --- Answers --------------------------------------------------------------------------------

function lastAnswered() {
  return Math.max(...Object.keys(state.answers).map(Number));
}

function answer(value) {
  if (!state.screenReady || state.paused || state.awaitingConfirm || state.feedbackPending || state.mode !== "task") return;
  const now = performance.now();
  if (now - state.shownAt < MIN_DECISION_MS) return;
  if ($("zoom").open) $("zoom").close();
  const step = state.step;
  const s = stats(step);
  s.ms += now - state.shownAt - state.pausedAcc;
  s.paused_ms += state.pausedAcc;
  state.shownAt = null;
  setReady(false);
  for (const key of Object.keys(state.answers)) if (Number(key) > step) delete state.answers[key];
  state.answers[step] = value;
  emit("answer", {task_id: state.task.task_id, step, answer: value});
  if (state.task.practice) {
    showFeedback(step, value);
    return;
  }
  advance(step, value);
}

function advance(step, value) {
  if (value === "same" || step === state.task.screens.length) {
    state.awaitingConfirm = true;
    $("confirm").hidden = false;
    updateControls();
  } else {
    state.step = step + 1;
    showScreen();
  }
}

function showFeedback(step, value) {
  const truth = state.task.feedback[step - 1];
  const labels = LABELS[state.task.type] || LABELS.default;
  state.feedbackPending = {step, value};
  $("feedback-text").textContent = value === truth ? `Correct: “${labels[truth]}”.`
    : `The reference answer is “${labels[truth]}”.${value === "unsure" ? " Unsure is always allowed." : ""}`;
  $("feedback").hidden = false;
  emit("practice_feedback", {task_id: state.task.task_id, step, answer: value});
  updateControls();
}

function continueFeedback() {
  const pending = state.feedbackPending;
  if (!pending) return;
  state.feedbackPending = null;
  $("feedback").hidden = true;
  advance(pending.step, pending.value);
}

function undo() {
  if (state.paused || state.feedbackPending || state.mode !== "task" || state.busy) return;
  if (state.awaitingConfirm) {
    const last = lastAnswered();
    state.awaitingConfirm = false;
    $("confirm").hidden = true;
    emit("undo", {task_id: state.task.task_id, to_step: last});
    state.step = last;
    showScreen();
    return;
  }
  if (!state.screenReady || state.step <= 1) return;
  const now = performance.now();
  const target = state.step - 1;
  const s = stats(target);
  s.ms += now - state.shownAt - state.pausedAcc;
  s.paused_ms += state.pausedAcc;
  s.undo_count += 1;
  state.shownAt = null;
  setReady(false);
  emit("undo", {task_id: state.task.task_id, to_step: target});
  state.step = target;
  showScreen();
}

async function confirmTask() {
  if (!state.awaitingConfirm || state.busy) return;
  state.busy = true;
  updateControls();
  const task = state.task;
  const answers = [];
  for (let step = 1; step <= lastAnswered(); step += 1) {
    const s = stats(step);
    answers.push({step, answer: state.answers[step], decision_ms: round(s.ms), paused_ms: round(s.paused_ms),
      undo_count: s.undo_count, zooms: s.zooms, expands: s.expands, toggles: s.toggles});
  }
  emit("task_complete", {task_id: task.task_id});
  try {
    await flush();
    await api("/api/complete", {reviewer: state.reviewer, session_id: state.sessionId, task_id: task.task_id,
      answers, unidentifiable: state.unidentifiable});
    state.completed.add(task.task_id);
  } catch (error) {
    state.busy = false;
    updateControls();
    if (error.conflict) {
      message(`${error.message}`, true);
      await resumeSession();
      return;
    }
    message(`Not saved yet: ${error.message} Press Enter to try again.`, true);
    return;
  }
  state.busy = false;
  state.awaitingConfirm = false;
  $("confirm").hidden = true;
  message("");
  nextTask();
}

// --- Pauses, zoom, extra photos, variants ---------------------------------------------------

function pause(reason) {
  if (state.paused || state.mode !== "task") return;
  state.paused = true;
  if (state.shownAt !== null) state.pausedSince = performance.now();
  emit("pause", {reason});
  $("pause-reason").textContent = PAUSE_REASONS[reason] || "";
  $("pause-overlay").hidden = false;
  updateControls();
}

function resume() {
  if (!state.paused) return;
  const now = performance.now();
  state.paused = false;
  if (state.pausedSince !== null) {
    state.pausedAcc += now - state.pausedSince;
    state.pausedSince = null;
  }
  emit("resume");
  $("pause-overlay").hidden = true;
  state.lastInput = now;
  updateControls();
}

async function openZoom(photo, side, caption) {
  if (state.mode !== "task" || state.paused || !state.screenReady) return;
  stats(state.step).zooms += 1;
  state.zoomSide = side;
  emit("zoom_open", {task_id: state.task.task_id, step: state.step, side});
  $("zoom-title").textContent = caption;
  $("zoom-img").src = await blob(photo.token, state.variant);
  $("zoom").showModal();
  await $("zoom-img").decode().catch(() => {});
  zoomFit();
}

function zoomTo(percent) {
  const img = $("zoom-img");
  $("zoom-range").value = String(Math.round(percent));
  $("zoom-value").textContent = `${Math.round(percent)}%`;
  img.style.width = `${Math.max(1, img.naturalWidth * percent / 100)}px`;
}

function zoomFit() {
  const img = $("zoom-img");
  const area = $("zoom-area");
  if (!img.naturalWidth) return;
  zoomTo(Math.min(400, 100 * Math.min(area.clientWidth / img.naturalWidth, area.clientHeight / img.naturalHeight)));
}

function expand(side) {
  if (state.mode !== "task" || !state.screenReady || state.paused) return;
  const extra = $(`${side}-extra`);
  if (!extra.hidden) {
    extra.hidden = true;
    return;
  }
  const photos = side === "query" ? state.task.query.slice(QUERY_SHOWN) : state.task.screens[state.step - 1].more;
  const tiles = photos.map((p) => Object.assign(tile(p, side), {side}));
  extra.replaceChildren(...tiles.map((t) => t.button));
  extra.hidden = false;
  state.extraTiles = state.extraTiles.filter((t) => t.side !== side).concat(tiles);
  tiles.forEach((t) => load(t).catch(() => {}));
  stats(state.step).expands += 1;
  emit("expand", {task_id: state.task.task_id, step: state.step, side});
}

function toggleVariant() {
  if (state.mode !== "task" || !state.screenReady || state.paused) return;
  state.variant = state.variant === "original" ? "masked" : "original";
  stats(state.step).toggles += 1;
  emit("variant", {task_id: state.task.task_id, step: state.step, variant: state.variant});
  [...state.queryTiles, ...state.candidateTiles, ...state.extraTiles].forEach((t) => load(t).catch(() => {}));
  updateControls();
}

function flagChanged() {
  if (state.mode !== "task") return;
  state.unidentifiable = $("unidentifiable").checked;
  emit("flag", {task_id: state.task.task_id, value: state.unidentifiable});
}

// --- Flow -----------------------------------------------------------------------------------

function interstitial(title, paragraphs, buttonText, onContinue, {countdown = 0, secondary = null} = {}) {
  showSection("interstitial");
  $("inter-title").textContent = title;
  $("inter-text").replaceChildren(...paragraphs.map((text) => Object.assign(document.createElement("p"), {textContent: text})));
  const button = $("inter-button");
  button.textContent = buttonText;
  button.onclick = () => { button.onclick = null; onContinue(); };
  $("inter-secondary").hidden = !secondary;
  if (secondary) {
    $("inter-secondary").textContent = secondary.text;
    $("inter-secondary").onclick = secondary.action;
  }
  $("inter-countdown").hidden = !countdown;
  button.disabled = countdown > 0;
  if (countdown > 0) {
    let left = countdown;
    $("inter-countdown").textContent = `${left} s`;
    const timer = setInterval(() => {
      left -= 1;
      $("inter-countdown").textContent = left > 0 ? `${left} s` : "Ready when you are.";
      if (left <= 0) {
        clearInterval(timer);
        button.disabled = false;
      }
    }, 1000);
  }
  updateProgress();
}

function nextIndex() {
  return state.sequence.findIndex((e) => !state.completed.has(e.task_id));
}

function nextTask() {
  const current = state.sequence[state.index];
  const index = nextIndex();
  if (index < 0) {
    finish();
    return;
  }
  const next = state.sequence[index];
  if (current && next.block !== current.block) {
    blockBoundary(current.block, next.block, index);
    return;
  }
  startTask(index);
}

function blockBoundary(previous, upcoming, index) {
  const blocks = state.settings.blocks || 12;
  const perSession = state.settings.blocks_per_session || 3;
  const earlyStop = state.settings.early_stop_blocks || 8;
  const go = () => startTask(index);
  if (previous === 0) {
    interstitial("Practice complete", [
      "The timed trial starts now. Work at your normal pace and accuracy, as you would on real photos.",
      "Each block has about 25 tasks. Breaks between blocks are not timed.",
    ], `Start block ${upcoming}`, go);
    return;
  }
  const note = previous === earlyStop ? [`Blocks 1–${earlyStop} form the pre-declared reduced trial. Stop here only if `
    + "you have decided to end the trial early; otherwise continue as planned."] : [];
  if (previous % perSession === 0) {
    emit("break_start", {block: previous});
    interstitial(`Session complete: block ${previous} of ${blocks}`, [
      "Please stop here and start the next session at least 2 hours later. Your progress is saved.",
      "When you come back, open this page again and choose the same reviewer slot.", ...note,
    ], "Start the next session now", () => { emit("session_override", {block: previous}); emit("break_end", {block: previous}); go(); });
    return;
  }
  emit("break_start", {block: previous});
  interstitial(`Block ${previous} of ${blocks} complete`, [
    "Take a short break: stand up, rest your eyes. The clock is not running.", ...note,
  ], `Start block ${upcoming}`, () => { emit("break_end", {block: previous}); go(); },
  {countdown: state.settings.break_seconds || 120});
}

function finish() {
  state.index = state.sequence.length;
  interstitial("All tasks complete", [
    "Thank you. Every answer and timing event is saved on the workstation.",
    "The study lead exports the results with the button below.",
  ], "Export results", exportResults);
}

async function resumeSession() {
  const session = await api(`/api/session?reviewer=${encodeURIComponent(state.reviewer)}`);
  state.sequence = session.sequence;
  state.completed = new Set(session.completed);
  state.settings = session.settings;
  const index = nextIndex();
  if (index < 0) {
    finish();
    return;
  }
  const entry = state.sequence[index];
  state.index = index;
  if (index === 0) {
    interstitial("Practice", [
      "Twenty practice tasks come first. After each answer you will see the reference answer.",
      "Compare the photos on the left with the catalogue animal on the right and answer Same (S), Different (D) or "
      + "Unsure (U). Click a photo to enlarge it. V switches to the masked view, P pauses, Backspace undoes.",
      "Practice answers are excluded from the results.",
    ], "Start practice", () => startTask(index));
    return;
  }
  interstitial(entry.block === 0 ? "Welcome back: practice" : `Welcome back: block ${entry.block}`, [
    "Your saved answers are kept. The task you were working on, if any, starts again from its first screen.",
  ], "Continue", () => startTask(index));
}

async function startReviewer(event) {
  event.preventDefault();
  const reviewer = $("reviewer").value;
  if (!reviewer) {
    message("Choose your reviewer slot.", true);
    return;
  }
  const profile = {};
  for (const [field, id] of Object.entries(PROFILE)) profile[field] = $(id).value.trim();
  try {
    await api("/api/reviewers", {reviewer, profile});
    state.reviewer = reviewer;
    remember("reviewer", reviewer);
    emit("session_start", {width: window.innerWidth, height: window.innerHeight, dpr: window.devicePixelRatio || 1,
      user_agent: navigator.userAgent.slice(0, 400)});
    message("");
    await resumeSession();
  } catch (error) {
    message(error.message, true);
  }
}

async function prefill(reviewer) {
  if (!reviewer) return;
  try {
    const session = await api(`/api/session?reviewer=${encodeURIComponent(reviewer)}`);
    if (session.profile) for (const [field, id] of Object.entries(PROFILE)) $(id).value = session.profile[field] || "";
    $("start-button").textContent = session.profile ? "Continue" : "Save and continue";
  } catch (error) {
    message(error.message, true);
  }
}

async function exportResults() {
  $("export").disabled = true;
  try {
    await flush();
    const result = await api("/api/export", {});
    $("export-summary").textContent = Object.entries(result.reviewers).map(([r, v]) =>
      `${r}: ${v.main_completed}/${v.main_total} timed tasks, ${v.blocks_completed} blocks`).join(" · ") || "No answers yet.";
    $("export-download").href = result.download_url;
    $("export-download").download = result.filename;
    $("export-path").value = result.path;
    $("export-result").hidden = false;
  } catch (error) {
    message(error.message, true);
  } finally {
    $("export").disabled = false;
  }
}

// --- Wiring ---------------------------------------------------------------------------------

function onKey(event) {
  state.lastInput = performance.now();
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.target.matches("input:not([type=checkbox]), textarea, select")) return;
  if (state.paused) {
    if (["Enter", " ", "p", "P"].includes(event.key)) {
      event.preventDefault();
      resume();
    }
    return;
  }
  if (state.mode === "interstitial") {
    if (event.key === "Enter" && !$("inter-button").disabled) {
      event.preventDefault();
      $("inter-button").click();
    }
    return;
  }
  if (state.mode !== "task" || event.repeat) return;
  const key = event.key.toLowerCase();
  const answers = {s: "same", d: "different", u: "unsure"};
  if ($("zoom").open && !(key in answers)) return;  // Escape closes the zoom natively.
  if (key in answers) {
    event.preventDefault();
    answer(answers[key]);
  } else if (event.key === "Backspace") {
    event.preventDefault();
    undo();
  } else if (event.key === "Enter") {
    event.preventDefault();
    if (state.feedbackPending) continueFeedback();
    else if (state.awaitingConfirm) confirmTask();
  } else if (key === "p") {
    event.preventDefault();
    pause("manual");
  } else if (key === "v") {
    event.preventDefault();
    toggleVariant();
  } else if (key === "f") {
    event.preventDefault();
    $("unidentifiable").checked = !$("unidentifiable").checked;
    flagChanged();
  }
}

async function init() {
  document.addEventListener("keydown", onKey);
  for (const type of ["pointermove", "pointerdown", "wheel"]) {
    document.addEventListener(type, () => { state.lastInput = performance.now(); }, {passive: true});
  }
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      pause("hidden");
      flush(true);
    }
  });
  window.addEventListener("blur", () => pause("blur"));
  window.addEventListener("pagehide", () => flush(true));
  setInterval(() => flush(), 2000);
  setInterval(() => {
    const idle = (state.settings.idle_pause_seconds || 120) * 1000;
    if (state.mode === "task" && state.screenReady && !state.paused && performance.now() - state.lastInput > idle) {
      pause("idle");
    }
  }, 5000);
  $("profile-form").addEventListener("submit", startReviewer);
  $("reviewer").addEventListener("change", () => prefill($("reviewer").value));
  document.querySelectorAll(".answer").forEach((button) => button.addEventListener("click", () => {
    button.blur();
    answer(button.dataset.answer);
  }));
  $("undo").addEventListener("click", (e) => { e.currentTarget.blur(); undo(); });
  $("pause").addEventListener("click", (e) => { e.currentTarget.blur(); pause("manual"); });
  $("resume").addEventListener("click", resume);
  $("variant").addEventListener("click", (e) => { e.currentTarget.blur(); toggleVariant(); });
  $("unidentifiable").addEventListener("change", flagChanged);
  $("confirm-next").addEventListener("click", (e) => { e.currentTarget.blur(); confirmTask(); });
  $("feedback-next").addEventListener("click", (e) => { e.currentTarget.blur(); continueFeedback(); });
  $("query-more").addEventListener("click", () => expand("query"));
  $("candidate-more").addEventListener("click", () => expand("candidate"));
  $("export").addEventListener("click", exportResults);
  $("zoom-close").addEventListener("click", () => $("zoom").close());
  $("zoom-fit").addEventListener("click", zoomFit);
  $("zoom-actual").addEventListener("click", () => zoomTo(100));
  $("zoom-range").addEventListener("input", () => zoomTo(Number($("zoom-range").value)));
  $("zoom").addEventListener("close", () => {
    if (state.task) emit("zoom_close", {task_id: state.task.task_id, step: state.step, side: state.zoomSide || "candidate"});
  });
  try {
    const slots = await api("/api/slots");
    for (const slot of slots.slots) $("reviewer").append(new Option(slot, slot));
    $("progress-label").textContent = `Task set ${slots.task_set}`;
    const saved = recall("reviewer");
    if (saved && slots.slots.includes(saved)) {
      $("reviewer").value = saved;
      await prefill(saved);
    }
    showSection("start");
  } catch (error) {
    message(error.message, true);
  }
}

init();
