"use strict";

const $ = (id) => document.getElementById(id);
const FIELD_INFO = {
  animal_visibility: ["Animal visibility", "Can you clearly see an animal in this photograph?"],
  usable_markings: ["Usable individual markings", "Are coat markings visible enough to help distinguish an individual?"],
  crop_mask: ["Crop and mask", "Does the retained area correspond to the visible animal?"],
  blur_occlusion: ["Blur or occlusion", "Minor includes no or negligible problems."],
  flank: ["Visible flank", "Use the animal’s anatomical left or right side. Keep uncertain or frontal views unknown."],
  illumination: ["Illumination", "Colour and monochrome describe appearance. Verified labels need supporting evidence."],
};
const LABELS = {
  clear: "Clear", partial: "Partial", no_clear_animal: "No clear animal", unknown: "Unknown",
  yes: "Yes", no: "No", correct: "Correct", issue: "Issue",
  minor: "Minor / none", moderate: "Moderate", severe: "Severe", left: "Left", right: "Right",
  colour: "Colour", monochrome: "Monochrome", verified_day: "Verified day",
  verified_night: "Verified night", verified_ir: "Verified IR",
};
const COHORTS = {
  fitting_uniform: "Fitting · uniform sample", calibration_uniform: "Calibration · uniform sample",
  calibration_identity_coverage: "Calibration · identity coverage",
  calibration_diagnostic_extremes: "Calibration · diagnostic examples",
};
const VIEW_TITLES = {original: "Original photograph", crop: "Segmented crop", overlay: "Mask overlay"};
const state = {
  items: [], reviews: {}, current: null, values: {}, schema: null, dirty: false, editVersion: 0,
  busy: false, pendingSave: null, timer: null, blocked: false, imageController: null,
  imageGeneration: 0, loadedId: null,
  currentRevision: 0,
};

function remember(key, value) { try { localStorage.setItem(`lynx-review:${key}`, value); } catch (_) {} }
function recall(key) { try { return localStorage.getItem(`lynx-review:${key}`); } catch (_) { return null; } }

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json", "X-Review-App": "1"},
    body: JSON.stringify(body),
  };
  const response = await fetch(path, {...options, cache: "no-store"});
  const result = await response.json();
  if (!response.ok) {
    const error = new Error(result.error || "The workstation could not save this request.");
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

function reportError(error) {
  message(error.message || String(error), true);
  if (error.conflict) {
    state.blocked = true;
    $("reload-review").hidden = false;
  }
  updateStatus();
}

function filtered() {
  const status = $("status-filter").value;
  const cohort = $("cohort-filter").value;
  const search = $("search").value.trim().toLowerCase();
  return state.items.filter((item) => {
    const record = state.reviews[item.image_id];
    if (search && !item.image_id.includes(search)) return false;
    if (cohort !== "all" && !item.cohorts.includes(cohort)) return false;
    if (status === "reviewed" && !record.completed) return false;
    if (status === "unreviewed" && record.completed) return false;
    if (status === "draft" && (record.completed || !record.revision)) return false;
    if (status === "issues" && !(record.review.crop_mask === "issue"
        || record.review.animal_visibility === "no_clear_animal"
        || record.review.usable_markings === "no" || record.review.blur_occlusion === "severe")) return false;
    return true;
  });
}

function updateProgress() {
  const records = Object.values(state.reviews);
  const done = records.filter((r) => r.completed).length;
  const drafts = records.filter((r) => !r.completed && r.revision).length;
  $("progress").max = records.length || 1;
  $("progress").value = done;
  $("progress-label").textContent = `${done} of ${records.length} reviewed`;
  $("remaining-label").textContent = `${records.length - done} remaining${drafts ? ` · ${drafts} drafts` : ""}`;
  $("export-final").disabled = done !== records.length || !records.length || state.busy || !!state.pendingSave || state.blocked;
  $("export-draft").disabled = !records.length || state.busy || !!state.pendingSave || state.blocked;
}

function updateStatus() {
  updateProgress();
  const record = state.current && state.reviews[state.current];
  const saving = !!state.pendingSave;
  const disabled = state.busy || saving || state.blocked;
  const available = filtered();
  const index = available.findIndex((item) => item.image_id === state.current);
  $("previous").disabled = disabled || index <= 0;
  $("next").disabled = disabled || index < 0 || index >= available.length - 1;
  $("next-unreviewed").disabled = disabled || !available.some((i) => !state.reviews[i.image_id].completed);
  $("save-next").disabled = disabled || !record || state.loadedId !== state.current || !$("reviewer").value.trim();
  $("save-only").disabled = $("save-next").disabled;
  ["status-filter", "cohort-filter", "search", "reviewer", "notes", "illumination-evidence"].forEach((id) => {
    $(id).disabled = state.busy;
  });
  document.querySelectorAll(".choice").forEach((button) => { button.disabled = state.busy; });
  document.querySelectorAll(".image-panel").forEach((button) => { button.disabled = state.loadedId !== state.current || !state.current; });
  if (!record) return;
  $("position").textContent = index >= 0 ? `Photograph ${index + 1} of ${available.length} in this view` : "Outside the current filter";
  const label = state.dirty ? "Unsaved draft" : record.completed ? "Reviewed" : record.revision ? "Draft" : "Needs review";
  $("record-status").textContent = label;
  $("record-status").className = `badge ${state.dirty ? "draft" : record.completed ? "reviewed" : record.revision ? "draft" : ""}`;
  $("save-status").textContent = state.blocked ? "Reload the latest review to continue"
    : saving ? "Saving to the workstation…" : state.dirty ? "Draft changes pending…"
    : record.completed ? "Review complete" : record.revision ? "Draft saved · finish with Save review" : "Not reviewed yet";
  $("review-credit").textContent = record.completed
    ? `${record.review.reviewer} · ${new Date(record.review.reviewed_at).toLocaleString()}`
    : record.updated_at ? `Last saved ${new Date(record.updated_at).toLocaleTimeString()}` : "Choose your answers, then save this review.";
}

function buildFields() {
  const container = $("review-fields");
  for (const [key, [label, help]] of Object.entries(FIELD_INFO)) {
    const fieldset = document.createElement("fieldset");
    const legend = document.createElement("legend");
    legend.textContent = label;
    const choices = document.createElement("div");
    choices.className = "choices";
    choices.setAttribute("role", "group");
    choices.setAttribute("aria-label", label);
    for (const value of state.schema.choices[key]) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "choice";
      button.dataset.field = key;
      button.dataset.value = value;
      button.textContent = LABELS[value] || value;
      button.setAttribute("aria-pressed", "false");
      button.addEventListener("click", () => {
        if (state.values[key] === value) return;
        state.values[key] = value;
        paintChoices();
        edited();
      });
      choices.append(button);
    }
    const hint = document.createElement("span");
    hint.className = "field-help";
    hint.textContent = help;
    fieldset.append(legend, choices, hint);
    container.append(fieldset);
  }
}

function paintChoices() {
  document.querySelectorAll(".choice").forEach((button) => {
    button.setAttribute("aria-pressed", String(state.values[button.dataset.field] === button.dataset.value));
  });
  $("evidence-label").hidden = !state.values.illumination?.startsWith("verified_") && !$("illumination-evidence").value.trim();
}

function edited() {
  if (!state.current) return;
  state.dirty = true;
  state.editVersion += 1;
  paintChoices();
  clearTimeout(state.timer);
  if (!state.blocked) state.timer = setTimeout(() => persist(false).catch(reportError), 900);
  updateStatus();
}

function formValues() {
  return {...state.values, reviewer: $("reviewer").value.trim(),
    notes: $("notes").value, illumination_evidence: $("illumination-evidence").value};
}

async function persist(complete) {
  clearTimeout(state.timer);
  while (state.pendingSave) await state.pendingSave;
  if (state.blocked) throw new Error("Reload the latest review before continuing.");
  if (!state.current || (!complete && !state.dirty)) return;
  const rid = state.current;
  const version = state.editVersion;
  const values = formValues();
  if (complete && state.loadedId !== rid) throw new Error("Wait for all three image views to load before completing the review.");
  if (complete && !values.reviewer) throw new Error("Enter your reviewer name first.");
  if (complete && values.illumination.startsWith("verified_") && !values.illumination_evidence.trim()) {
    $("illumination-evidence").focus();
    throw new Error("Add evidence for verified illumination, or choose an appearance label / unknown.");
  }
  const request = api(`/api/reviews/${rid}`, {
    review: values, revision: state.currentRevision, complete,
  });
  state.pendingSave = request;
  updateStatus();
  try {
    const result = await request;
    state.reviews[rid] = result;
    if (state.current === rid) state.currentRevision = result.revision;
    if (state.current === rid && state.editVersion === version) state.dirty = false;
    if (state.dirty && !state.busy) state.timer = setTimeout(() => persist(false).catch(reportError), 900);
  } finally {
    state.pendingSave = null;
    updateStatus();
  }
}

async function flushDraft() {
  clearTimeout(state.timer);
  if (state.pendingSave) await state.pendingSave;
  while (state.dirty) await persist(false);
}

async function exclusive(action) {
  if (state.busy) return;
  state.busy = true;
  updateStatus();
  try { await action(); } catch (error) { reportError(error); }
  finally { state.busy = false; updateStatus(); }
}

async function showRecord(rid) {
  // Load state before changing the active ID: a failed request must never leave
  // the previous photograph's form attached to a different photograph.
  const record = rid ? await api(`/api/reviews/${rid}`) : null;
  state.imageController?.abort();
  state.loadedId = null;
  state.imageGeneration += 1;
  state.current = rid || null;
  state.dirty = false;
  state.blocked = false;
  $("reload-review").hidden = true;
  $("workspace").hidden = !rid;
  $("empty-state").hidden = !!rid;
  if (!rid) { updateStatus(); return; }
  state.reviews[rid] = record;
  state.currentRevision = record.revision;
  state.values = Object.fromEntries(Object.keys(FIELD_INFO).map((key) => [key, record.review[key]]));
  $("notes").value = record.review.notes;
  $("illumination-evidence").value = record.review.illumination_evidence;
  if (!$("reviewer").value && record.review.reviewer) $("reviewer").value = record.review.reviewer;
  const item = state.items.find((i) => i.image_id === rid);
  $("image-id").textContent = rid;
  $("cohort-label").textContent = item.cohorts.map((c) => COHORTS[c] || c).join(" · ");
  $("image-details").textContent = `${item.width} × ${item.height} px · ${(item.foreground_fraction * 100).toFixed(1)}% of crop retained`;
  remember("last-image", rid);
  paintChoices();
  updateStatus();
  loadImages(item, state.imageGeneration); // Keep navigation responsive while full images load.
}

async function move(direction) {
  await exclusive(async () => {
    await flushDraft();
    const items = filtered();
    const index = items.findIndex((i) => i.image_id === state.current);
    const next = items[index + direction];
    if (next) { message(""); await showRecord(next.image_id); }
  });
}

async function filterChanged() {
  await exclusive(async () => {
    await flushDraft();
    const items = filtered();
    const rid = items.some((i) => i.image_id === state.current) ? state.current : items[0]?.image_id;
    message("");
    await showRecord(rid);
  });
}

async function saveReview(advance) {
  await exclusive(async () => {
    const before = filtered();
    const index = before.findIndex((i) => i.image_id === state.current);
    const rid = state.current;
    await persist(true);
    message("Review saved.");
    if (advance) {
      const available = filtered();
      const next = before.slice(index + 1).find((i) => available.some((a) => a.image_id === i.image_id));
      const remaining = available.find((i) => !state.reviews[i.image_id].completed && i.image_id !== rid);
      const target = next?.image_id || remaining?.image_id || (available.some((i) => i.image_id === rid) ? rid : available[0]?.image_id);
      await showRecord(target);
      if (!next && !remaining) message("All reviews in this view are complete.");
    }
  });
}

async function loadBitmap(rid, kind, signal) {
  const response = await fetch(`/images/${rid}/${kind}`, {signal, cache: "no-store"});
  if (!response.ok) {
    const result = await response.json();
    throw new Error(result.error || "Image could not be loaded.");
  }
  return createImageBitmap(await response.blob(), {imageOrientation: "none"});
}

function drawBitmap(canvas, bitmap) {
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  canvas.getContext("2d").drawImage(bitmap, 0, 0);
}

async function loadImages(item, generation) {
  const controller = new AbortController();
  state.imageController = controller;
  document.querySelectorAll(".image-loading").forEach((el) => { el.hidden = false; el.textContent = "Loading full resolution image…"; });
  ["original", "crop", "overlay"].forEach((id) => { $(id).width = 1; $(id).height = 1; });
  const settled = await Promise.allSettled(["original", "crop", "mask"].map((kind) => loadBitmap(item.image_id, kind, controller.signal)));
  const bitmaps = settled.filter((r) => r.status === "fulfilled").map((r) => r.value);
  try {
    if (generation !== state.imageGeneration) return;
    const failure = settled.find((r) => r.status === "rejected");
    if (failure) throw failure.reason;
    const [original, crop, mask] = settled.map((r) => r.value);
    const [x0, y0, x1, y1] = item.bbox;
    if (original.width !== item.width || original.height !== item.height
        || crop.width !== x1 - x0 || crop.height !== y1 - y0
        || mask.width !== crop.width || mask.height !== crop.height) {
      throw new Error("Image dimensions differ from the audit. Please check the source before reviewing.");
    }
    drawBitmap($("original"), original);
    drawBitmap($("crop"), crop);
    drawBitmap($("overlay"), original);
    const tint = document.createElement("canvas");
    tint.width = mask.width;
    tint.height = mask.height;
    const tintContext = tint.getContext("2d");
    tintContext.drawImage(mask, 0, 0);
    const pixels = tintContext.getImageData(0, 0, tint.width, tint.height);
    for (let i = 0; i < pixels.data.length; i += 4) {
      const retained = pixels.data[i] > 127;
      pixels.data[i] = 0; pixels.data[i + 1] = 255; pixels.data[i + 2] = 80;
      pixels.data[i + 3] = retained ? 89 : 0;
    }
    tintContext.putImageData(pixels, 0, 0);
    const context = $("overlay").getContext("2d");
    context.drawImage(tint, x0, y0);
    context.strokeStyle = "#ff5656";
    context.lineWidth = Math.max(2, item.width / 400);
    context.strokeRect(x0, y0, x1 - x0, y1 - y0);
    state.loadedId = item.image_id;
    document.querySelectorAll(".image-loading").forEach((el) => { el.hidden = true; });
  } catch (error) {
    if (error.name !== "AbortError" && generation === state.imageGeneration) {
      document.querySelectorAll(".image-loading").forEach((el) => { el.textContent = "Image unavailable · reload to retry"; });
      reportError(error);
    }
  } finally {
    bitmaps.forEach((bitmap) => bitmap.close());
    updateStatus();
  }
}

function zoom(percent) {
  const canvas = $("zoom-canvas");
  percent = Math.max(10, Math.min(400, Math.round(percent)));
  canvas.style.width = `${canvas.width * percent / 100}px`;
  canvas.style.height = `${canvas.height * percent / 100}px`;
  $("zoom-range").value = percent;
  $("zoom-value").value = `${percent}%`;
}

function fitZoom() {
  const area = $("zoom-area"), canvas = $("zoom-canvas");
  zoom(100 * Math.min((area.clientWidth - 20) / canvas.width, (area.clientHeight - 20) / canvas.height));
}

function enlarge(kind) {
  if (state.loadedId !== state.current) return;
  const source = $(kind), canvas = $("zoom-canvas");
  canvas.width = source.width; canvas.height = source.height;
  canvas.getContext("2d").drawImage(source, 0, 0);
  $("zoom-title").textContent = `${VIEW_TITLES[kind]} · ${source.width} × ${source.height} px`;
  $("image-dialog").showModal();
  fitZoom();
  $("zoom-area").scrollTo(0, 0);
}

async function exportReviews(complete) {
  await exclusive(async () => {
    await flushDraft();
    const result = await api("/api/export", {complete});
    // Refresh all counts after a snapshot, including other browsers' saved work.
    const session = await api("/api/session");
    state.reviews = session.reviews;
    const changed = state.current && state.reviews[state.current].revision !== state.currentRevision;
    if (changed) {
      state.blocked = true;
      $("reload-review").hidden = false;
    }
    $("export-result").hidden = false;
    $("export-summary").textContent = `${result.final ? "Completed audit" : "Progress snapshot"}: ${result.completed} of ${result.total} reviewed.`;
    $("export-path").value = result.path;
    $("export-download").href = result.download_url;
    $("export-download").download = result.filename;
    const notice = result.final ? "Your completed audit CSV is ready for the summarize step." : "Progress exported. Unfinished reviews remain drafts.";
    message(changed ? `${notice} This photograph changed in another browser; reload its latest review to continue.` : notice, !!changed);
  });
}

async function start() {
  try {
    const session = await api("/api/session");
    state.items = session.items; state.reviews = session.reviews; state.schema = session.schema;
    $("reviewer").value = recall("reviewer") || "";
    buildFields();
    const cohorts = [...new Set(state.items.flatMap((i) => i.cohorts))].sort();
    for (const cohort of cohorts) {
      const option = document.createElement("option");
      option.value = cohort; option.textContent = COHORTS[cohort] || cohort;
      $("cohort-filter").append(option);
    }
    const last = recall("last-image");
    const first = state.items.some((i) => i.image_id === last) ? last
      : state.items.find((i) => !state.reviews[i.image_id].completed)?.image_id || state.items[0]?.image_id;
    await showRecord(first);
  } catch (error) { reportError(error); }
}

$("reviewer").addEventListener("input", () => { remember("reviewer", $("reviewer").value); edited(); });
$("notes").addEventListener("input", edited);
$("illumination-evidence").addEventListener("input", edited);
$("previous").addEventListener("click", () => move(-1));
$("next").addEventListener("click", () => move(1));
$("next-unreviewed").addEventListener("click", () => exclusive(async () => {
  await flushDraft();
  const items = filtered();
  const index = items.findIndex((i) => i.image_id === state.current);
  const ordered = [...items.slice(index + 1), ...items.slice(0, index + 1)];
  const target = ordered.find((i) => !state.reviews[i.image_id].completed);
  if (target) { message(""); await showRecord(target.image_id); }
}));
["status-filter", "cohort-filter"].forEach((id) => $(id).addEventListener("change", filterChanged));
let searchTimer;
$("search").addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(filterChanged, 300); });
$("review-form").addEventListener("submit", (event) => { event.preventDefault(); saveReview(true); });
$("save-only").addEventListener("click", () => saveReview(false));
$("export-draft").addEventListener("click", () => exportReviews(false));
$("export-final").addEventListener("click", () => exportReviews(true));
$("reload-review").addEventListener("click", () => {
  if (!confirm("Load the latest saved review? Unsaved changes in this browser will be discarded.")) return;
  exclusive(async () => {
    clearTimeout(state.timer);
    message("");
    await showRecord(state.current);
  });
});
document.querySelectorAll(".image-panel").forEach((button) => button.addEventListener("click", () => enlarge(button.dataset.view)));
$("zoom-range").addEventListener("input", () => zoom(Number($("zoom-range").value)));
$("zoom-fit").addEventListener("click", fitZoom);
$("zoom-actual").addEventListener("click", () => zoom(100));
$("zoom-close").addEventListener("click", () => $("image-dialog").close());
document.addEventListener("keydown", (event) => {
  if ($("image-dialog").open) return;
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    if (!$("save-next").disabled) saveReview(true);
  } else if (event.altKey && ["ArrowLeft", "ArrowRight"].includes(event.key)) {
    event.preventDefault();
    const direction = event.key === "ArrowLeft" ? -1 : 1;
    if (!$(direction < 0 ? "previous" : "next").disabled) move(direction);
  }
});
window.addEventListener("beforeunload", (event) => {
  if (state.dirty || state.pendingSave) { event.preventDefault(); event.returnValue = ""; }
});
start();
