# CzechLynx timed matching review (N3)

A small app for the timed reviewer trial (work item N3 in the main project's decision log). It measures how
long real matching decisions take, and how often reviewers get them wrong, for two workflows on CzechLynx
*calibration* photos:

- **W1**, full ranked matching: every catalogue animal, in similarity order, until a match or the end of the list;
- **N2**, the assisted workflow: a pruned shortlist, second checks, duplicate checks and audits of automatic
  decisions.

Everything runs locally with Python 3's standard library. No installation, GPU, model or external service is
needed. Use a current Firefox, Chrome, Edge or Safari browser.

This app is separate from the input-audit app in `../czechlynx-review-app` (`apps/czechlynx-review-app` in the repo). It never reads or writes that
app's database or exports, and refuses to store anything inside it.

## Before the first session (study lead)

1. Build the task set in the main project (CPU, a few minutes):

   ```bash
   cd /home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project
   venv/bin/python tools/build_czechlynx_timed_review_tasks.py --calibration-dir evaluations/czechlynx/facing_adaptation_calibration_classes_v1/clustering --raw-metadata data/czechlynx/raw/CzechLynxDataset-Metadata-Real.csv --out-dir evaluations/czechlynx/timed_review_tasks_v1_r2
   ```

   It fails unless it reproduces the saved N2 candidate row and the saved W1 decisions exactly.
2. Open `evaluations/czechlynx/timed_review_tasks_v1_r2/contact_sheet.html`. It shows practice images only
   (original and masked), so the main tasks stay unseen. Check that both variants render and show no
   identity overlays.
3. Commit the proposal and `pre_registration.json` before session 1 (the study lead's decision).
   After the first session, no task, cap or estimator may change.

## Start

```bash
cd /home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/apps/czechlynx-matching-review-app
bash run.sh
```

The default task set is the main project's `evaluations/czechlynx/timed_review_tasks_v1_r2`; use `--tasks-dir`
for another. The default port is **8767** (the input-audit app uses 8766). Open **http://localhost:8767** on
the workstation, or the LAN address the server prints from another computer on the same network. Ctrl+C
stops the server; saved answers remain. A database is bound to one task set: for a different task set, use
a separate copy of this app with an empty `data/` directory.

## Reviewers

Each reviewer uses one **slot**: R1 (the user) and R2 (a second reviewer) are the planned slots; R3 and R4
exist as spares. Odd slots start with a W1 block, even slots with an N2 block (ABBA order). Both reviewers
get the same 300 tasks in different random orders.

On the first visit, choose the slot and answer the short experience questionnaire. The same browser
remembers the slot.

## What a task looks like

- **Left:** the photos to identify (one camera, one day; a duplicate check shows a second catalogue animal).
- **Right:** one catalogue animal, four of its photos most similar to the left side. "Show more photos"
  adds up to 20 more.
- **Answer:** **Same (S)**, **Different (D)** or **Unsure (U)**. For an audit task the buttons read
  *Correct* / *Wrong*. Unsure is always allowed. "Same" ends the task; otherwise the next candidate appears,
  until the end of the list.
- **Tools:** click any photo to enlarge it (Fit, 100%, zoom slider). **V** switches between the original and
  the masked photo. **F** marks that the photos cannot identify the animal. **Backspace** undoes the last
  answer. **P** pauses the clock.
- After the final answer a confirmation bar appears: **Enter** goes to the next task; **Backspace** undoes.

Reviewers never see identity labels, file names, model scores or which route produced a task. A list length
("candidate 3 of 41") is shown, as a real tool would.

## Protocol and schedule

- **Practice:** 20 tasks with the reference answer shown after each answer; excluded from all results.
- **Main trial:** 12 blocks of about 25 tasks. Blocks alternate W1 and N2 in ABBA order.
- **Breaks:** a 2-minute untimed break after each block. After every 3 blocks the app ends the session:
  stop, and start the next session at least 2 hours later (at most 2 sessions a day).
- **Time:** about 15 minutes for practice and the questionnaire, then 4 sessions of about 40–50 minutes:
  about 3.5 hours per reviewer at placeholder-like speeds. One session every other day finishes in 1–2 weeks.
- **Pre-declared early stop:** after 8 blocks (two full ABBA cycles) the study lead may end the trial;
  the result is then labelled "reduced". No other mid-trial change is allowed.
- Work at your normal pace and accuracy, as on real photos. Pause (P) for interruptions.

## Timing rules

- Decision time runs from the moment a screen's photos are all displayed to the answer, minus pauses.
  Loading time is recorded separately; the next screen is prefetched.
- The clock pauses on **P**, when the page is hidden, when the window loses focus, and after 120 seconds
  without any input. A paused screen shows an overlay; click **Resume** or press Enter.
- Answers within 0.2 s of a screen appearing are ignored (accidental double presses).
- If the page is reloaded mid-task, that task restarts from its first screen; completed tasks stay complete.

## Export

Click **Export results** (bottom of the page). The app writes, under this app's `exports/`:

- `matching_decisions_<UTC stamp>_<id>.csv`: one row per answer, with no reference truth;
- an adjacent `.json` with every raw event, the reviewer profiles, the task-set fingerprint and the CSV hash.

Every export is a new file; earlier exports are never changed. Re-cost the trial in the main project:

```bash
venv/bin/python tools/recost_czechlynx_timed_review.py --tasks-dir evaluations/czechlynx/timed_review_tasks_v1_r2 --export /home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/apps/czechlynx-matching-review-app/exports/matching_decisions_STAMP_ID.csv --out-dir evaluations/czechlynx/timed_review_recost_v1
```

While the main project's `reid_demo/cluster.py` differs from the calibration report's binding (N1b commit `99e9a2da`), run this from a `git archive 7ade3225` export plus the N3 files, as the task build was; see the main project's `docs/czechlynx-benchmark.md`, section "Timed reviewer trial (N3)".

It refuses reviewers who completed neither all 12 blocks nor exactly the first 8, unless `--allow-incomplete`
is given for an interim look (the verdict is then not evaluated).

## Storage and protection

```text
czechlynx-matching-review-app/
  server.py  run.sh  README.md
  static/{index.html,app.js,styles.css}
  tests/test_matching_app.py
  data/matching.sqlite3        # created on first launch
  exports/matching_decisions_* # created only when exporting
```

The server checks the task set's manifest, `tasks.json` and `images.json` hashes at startup, and every image's
bytes before serving it. It never opens the task set's `truth.json`. Only allowlisted static files, task
images (by opaque token) and CSV exports have HTTP routes. Writes need the app's header and the same origin.
To back up, stop the server and copy this folder, including `data/` and `exports/`.

## Focused tests (user run)

Model-free checks with a synthetic task set in a temporary directory and a loopback server: the task set
loads without its truth file; changed files are refused; payloads hide identity, paths and truth; image bytes
are verified; storage refuses protected locations (including the input-audit app); the database is bound to
its task set; events are validated and idempotent; tasks complete once, in order, with valid answers; progress
survives a restart; exports are new files with exact columns; HTTP routes are allowlisted and writes guarded.

```bash
python3 -m unittest discover -s /home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/apps/czechlynx-matching-review-app/tests -v
```

The assistant ran static checks only (py_compile, Ruff F, `node --check`). The tests, a real browser
walkthrough and the reviewer sessions are the user's to run.
