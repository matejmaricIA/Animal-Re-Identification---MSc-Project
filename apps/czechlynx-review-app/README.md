# CzechLynx image review

A small app for the 321 photographs in the prepared CzechLynx input audit.
Open it from a computer on the workstation's network. Everything runs locally,
using Python 3's standard library. No installation, GPU, model or external service
is needed. Use a current Firefox, Chrome, Edge or Safari browser.

## Start

On the workstation:

```bash
cd /home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/apps/czechlynx-review-app
bash run.sh
```

Keep the terminal running. Open **http://192.168.8.110:8766** from another computer
on the same network, or **http://localhost:8766** on the workstation. The server
prints its available addresses; use the current LAN address if the workstation's
address changes. If another computer cannot connect, check that it is on the same
network and the workstation's firewall permits TCP port 8766.

Ctrl+C stops the server. Restart with the same command to resume saved reviews.
The default bind address is `0.0.0.0`; to listen on a specific LAN interface use:

```bash
bash run.sh --host 192.168.8.110 --port 8766
```

With a specific bind address, open that address on the workstation as well.

The default audit is the adjacent project's
`evaluations/czechlynx/input_audit_v1`. For a different audit, use a separate copy
of this app with its own empty `data/` directory and specify `--audit-dir`.
An existing database refuses to open against a different or changed audit.

## Review

1. Enter your reviewer name. This browser remembers it.
2. Inspect the original photograph, segmented crop and mask overlay. Click any
   panel to enlarge it; **Fit**, **100%** and the zoom slider help inspect markings.
   Green marks retained pixels; red outlines the crop. All three views use full
   resolution source pixels. Identity labels and identity-bearing filenames are
   hidden.
3. Choose each answer. **Unknown is valid** when the image is inconclusive.
   - **Animal visibility:** clear, partial, no clear animal, or unknown.
   - **Usable markings:** whether markings can help distinguish the individual.
   - **Crop/mask:** whether the retained area matches the visible animal.
   - **Blur/occlusion:** minor (including none), moderate, severe, or unknown.
   - **Flank:** the animal's anatomical left or right side, or unknown.
   - **Illumination:** colour/monochrome appearance, a verified label, or unknown.
     Verified day, night or IR requires supporting evidence. A monochrome image
     alone does not establish night or infrared.
4. Add notes if helpful. Changes automatically save as a **draft** after a short
   pause and before navigation. Drafts are not completed reviews.
5. Click **Save review** or **Save and next**. This records your name and the
   server's UTC timestamp. Completing a review requires all three image views to
   load successfully. Editing a completed review makes it a draft until saved
   complete again.

Use filters for remaining reviews, drafts, completed reviews, quality concerns
or sample cohorts. Search accepts any part of an opaque image ID. Next unreviewed
respects your current filters. Progress counts are for the entire audit.

Keyboard shortcuts: **Ctrl/⌘ + Enter** saves and advances; **Alt + Left/Right**
navigates. **Escape** closes an enlarged image.

Draft changes that have not reached the server trigger a browser warning when
closing the page. Wait for “Draft saved” or “Review complete” before closing.
If two browsers edit the same photograph, the stale browser must reload the latest
saved review before continuing; it cannot silently replace the other review.
Edits on different photographs can be saved independently. Counts from other
browsers refresh when exporting or reloading the page.

## Export and summarize

**Export progress** creates a dated CSV with all audit rows, preserving unknowns
and incomplete drafts. It is useful as a snapshot, but cannot pass the final audit
summarizer while reviews remain unfinished.

Once every photograph is reviewed, click **Export completed audit**. The app:

- Saves a new CSV under this app's `exports/` directory.
- Shows its absolute workstation path and offers a browser download.
- Writes an adjacent JSON with its hash, audit binding and saved row revisions.

Every export uses a unique name. The app never edits an earlier CSV. Copy the
displayed **workstation path** into the existing summarize command. Run this
yourself from the main project folder, replacing the example filename:

```bash
venv/bin/python tools/audit_czechlynx_inputs.py summarize \
  --audit-dir evaluations/czechlynx/input_audit_v1 \
  --reviews /home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/apps/czechlynx-review-app/exports/reviews_complete_TIMESTAMP_ID.csv \
  --out-dir evaluations/czechlynx/input_audit_review_v1
```

Use a new output directory if that summary directory already contains a report.
Keep the exported CSV at that path after summarizing: the existing receipt binds
its path and bytes. After further review edits, create a new export and summary.
The export JSON is a snapshot record; the existing project's **summarize** command
creates the review receipt required before training preparation.

## Storage and source protection

App files, reviews and exports stay outside the main repository:

```text
czechlynx-review-app/
  server.py
  run.sh
  static/{index.html,app.js,styles.css}
  tests/test_review_app.py
  data/reviews.sqlite3          # created on first launch
  exports/reviews_*.csv         # created only when exporting
  exports/reviews_*.json
```

The SQLite database stores current reviews and revision history. To back up the
app, stop the server and copy the app folder, including `data/` and `exports/`.
The app reads the existing audit/source files and never writes them. It checks
the saved audit, schema, metadata and implementation hashes at startup, binds the
database to that exact audit, and verifies image bytes before serving each image.
Only allowlisted static files, sample images and CSV exports have HTTP routes.

Quality annotations remain sidecar review data. They do not change identities,
protocol splits, image inclusion, training or count results. This review does not
establish field accuracy or resolve upstream training exposure.

## Focused tests — user run

The 13 model-free checks cover persistent state, explicit completion, valid
unknowns, illumination evidence, stale-edit conflicts, exact CSV columns and text
round trips, dated snapshot preservation, changed audit bindings, source byte
protection, HTTP route restrictions and origin validation. They use temporary
synthetic files and a temporary loopback server, with no model or dataset runs.

```bash
cd /home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project/apps/czechlynx-review-app
python3 -m unittest discover -s tests -v
```

The assistant has not run these tests. Real browser use and user-run test results
remain the validation steps.
