#!/usr/bin/env python3
"""Timed CzechLynx matching-review server (work item N3). Python standard library only.

Serves a precomputed timed-review task set: one query and one candidate animal per screen, answered
Same / Different / Unsure. Every client event is logged with the server's receive time. The server
never opens the task set's truth file, never exposes identity labels or file paths, and never writes
into the input-audit review app next to it.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import ipaddress
import json
import math
from pathlib import Path
import re
import socket
import sqlite3
import subprocess
import sys
import traceback
from urllib.parse import parse_qs, urlsplit
import uuid

APP_ROOT = Path(__file__).resolve().parent
DEFAULT_TASKS = APP_ROOT.parents[1] / "evaluations/czechlynx/timed_review_tasks_v1_r2"
INPUT_AUDIT_APP = APP_ROOT.parent / "czechlynx-review-app"
TASK_SET_KIND = "czechlynx_timed_review_tasks_v1"
EXPORT_KIND = "czechlynx_matching_review_export_v1"
ANSWERS = ("same", "different", "unsure")
VARIANTS = ("original", "masked")
REVIEWER = re.compile(r"R[1-9]\Z")
SESSION = re.compile(r"[0-9a-f]{16,64}\Z")
TOKEN = re.compile(r"[0-9a-f]{24}\Z")
EXPORT_NAME = re.compile(r"matching_decisions_[0-9]{8}T[0-9]{12}Z_[0-9a-f]{8}\.csv")
PROFILE_FIELDS = ("name", "years_identifying_individuals", "species", "lynx_or_felid_experience",
                  "czechlynx_exposure", "knows_system_internals", "device", "pointing_device", "notes")
EVENT_TYPES = ("session_start", "task_start", "screen_shown", "answer", "undo", "pause", "resume", "zoom_open",
               "zoom_close", "expand", "variant", "flag", "task_complete", "practice_feedback", "block_start",
               "break_start", "break_end", "session_override")
EVENT_FIELDS = {"task_id": str, "step": int, "answer": str, "to_step": int, "reason": str, "load_ms": float,
                "variant": str, "side": str, "value": bool, "width": int, "height": int, "dpr": float,
                "user_agent": str, "block": int}
ANSWER_COUNTS = ("decision_ms", "paused_ms", "undo_count", "zooms", "expands", "toggles")
CSV_COLUMNS = ("reviewer", "task_id", "practice", "block", "arm", "type", "step", "kind", "position", "list_length",
               "answer", "decision_ms", "paused_ms", "undo_count", "zooms", "expands", "toggles", "unidentifiable",
               "session_id", "completed_at")


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Conflict(ValueError):
    """The request contradicts saved progress (e.g. a task completed in another browser)."""


class TaskSet:
    """A complete, hash-verified task set. Only tasks.json and images.json are read; truth.json never."""

    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.manifest_path = self.root / "manifest.json"
        data = self.manifest_path.read_bytes()
        self.manifest_hash = digest_bytes(data)
        m = json.loads(data)
        if m.get("kind") != TASK_SET_KIND or m.get("status") != "complete" or not m.get("task_set_fingerprint"):
            raise ValueError("The task set must be a complete timed-review task set.")
        self.fingerprint = m["task_set_fingerprint"]
        for name in ("tasks.json", "images.json"):
            expected = m.get("artifacts", {}).get(name)
            if not expected or digest_file(self.root / name) != expected:
                raise ValueError(f"Changed or missing task-set artifact: {name}")
        doc = json.loads((self.root / "tasks.json").read_text())
        images = json.loads((self.root / "images.json").read_text())
        if doc.get("kind") != TASK_SET_KIND or images.get("kind") != TASK_SET_KIND:
            raise ValueError("Task-set files have the wrong kind.")
        self.tasks, self.orders, self.settings = doc["tasks"], doc["orders"], doc["settings"]
        self.images = {}
        protected = {self.root}
        for token, variants in images["images"].items():
            if not TOKEN.fullmatch(token) or set(variants) != set(VARIANTS):
                raise ValueError("Invalid image token or variants.")
            self.images[token] = {}
            for variant, record in variants.items():
                path = Path(record["path"]).resolve()
                if not path.is_file():
                    raise ValueError("A task image is missing.")
                self.images[token][variant] = (path, record["sha256"])
                protected.add(path.parent)
        self.protected = tuple(sorted(protected))
        self.sequences = {}
        for reviewer, order in self.orders.items():
            if not REVIEWER.fullmatch(reviewer):
                raise ValueError("Invalid reviewer slot.")
            sequence = [{"task_id": t, "practice": True, "block": 0} for t in order["practice"]]
            for block in order["blocks"]:
                sequence += [{"task_id": t, "practice": False, "block": block["block"]} for t in block["tasks"]]
            ids = [s["task_id"] for s in sequence]
            if len(ids) != len(set(ids)) or any(t not in self.tasks for t in ids):
                raise ValueError("A reviewer order repeats or names unknown tasks.")
            self.sequences[reviewer] = sequence
        for task_id, task in self.tasks.items():
            if task["task_id"] != task_id or not task["screens"] or not task["query"]:
                raise ValueError(f"Malformed task: {task_id}")
            photos = list(task["query"]) + [p for s in task["screens"] for p in s["exemplars"] + s["more"]]
            if any(p["token"] not in self.images for p in photos):
                raise ValueError(f"Task {task_id} references an unknown image.")

    def ensure_unchanged(self):
        if digest_file(self.manifest_path) != self.manifest_hash:
            raise ValueError("The task set changed. Restart with the intended task set before continuing.")

    def sequence(self, reviewer):
        if reviewer not in self.sequences:
            raise KeyError("Unknown reviewer slot.")
        return self.sequences[reviewer]

    def task_for(self, reviewer, task_id):
        if task_id not in {s["task_id"] for s in self.sequence(reviewer)}:
            raise KeyError("Unknown task for this reviewer.")
        return self.tasks[task_id]

    def image(self, token, variant):
        self.ensure_unchanged()
        if token not in self.images or variant not in VARIANTS:
            raise KeyError("Unknown image.")
        path, expected = self.images[token][variant]
        data = path.read_bytes()
        if digest_bytes(data) != expected:
            raise ValueError("This image changed since the task set was built.")
        mime = "image/png" if data.startswith(b"\x89PNG") else "image/jpeg"
        return data, mime


class Store:
    def __init__(self, taskset, app_root, protected=None):
        self.taskset = taskset
        self.root = Path(app_root).resolve()
        self.data_dir, self.export_dir = self.root / "data", self.root / "exports"
        guarded = list(taskset.protected) + [INPUT_AUDIT_APP.resolve()] + [Path(p).resolve() for p in (protected or ())]
        for output in (self.data_dir.resolve(), self.export_dir.resolve()):
            for path in guarded:
                if output == path or output.is_relative_to(path) or path.is_relative_to(output):
                    raise ValueError("Review data and exports must be outside the task set, image folders and "
                                     "the input-audit app.")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "matching.sqlite3"
        with self.connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS reviewers (code TEXT PRIMARY KEY, profile TEXT NOT NULL, "
                         "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS events (reviewer TEXT NOT NULL, session_id TEXT NOT NULL, "
                         "seq INTEGER NOT NULL, type TEXT NOT NULL, task_id TEXT, step INTEGER, t REAL NOT NULL, "
                         "wall TEXT NOT NULL, server_time TEXT NOT NULL, payload TEXT NOT NULL, "
                         "PRIMARY KEY(reviewer, session_id, seq))")
            conn.execute("CREATE TABLE IF NOT EXISTS completions (reviewer TEXT NOT NULL, task_id TEXT NOT NULL, "
                         "session_id TEXT NOT NULL, completed_at TEXT NOT NULL, unidentifiable INTEGER NOT NULL, "
                         "PRIMARY KEY(reviewer, task_id))")
            conn.execute("CREATE TABLE IF NOT EXISTS answers (reviewer TEXT NOT NULL, task_id TEXT NOT NULL, "
                         "step INTEGER NOT NULL, answer TEXT NOT NULL, decision_ms REAL NOT NULL, "
                         "paused_ms REAL NOT NULL, undo_count INTEGER NOT NULL, zooms INTEGER NOT NULL, "
                         "expands INTEGER NOT NULL, toggles INTEGER NOT NULL, PRIMARY KEY(reviewer, task_id, step))")
            binding = json.dumps({"version": 1, "task_manifest_sha256": taskset.manifest_hash,
                                  "task_set_fingerprint": taskset.fingerprint}, sort_keys=True)
            existing = conn.execute("SELECT value FROM settings WHERE key='binding'").fetchone()
            if existing and existing[0] != binding:
                raise ValueError("This app database belongs to a different task set. Use a separate app copy.")
            conn.execute("INSERT OR IGNORE INTO settings VALUES ('binding', ?)", (binding,))

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path, timeout=15)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # --- Reviewers and progress ------------------------------------------------------------

    def save_profile(self, reviewer, profile):
        self.taskset.sequence(reviewer)
        if not isinstance(profile, dict) or set(profile) - set(PROFILE_FIELDS):
            raise ValueError("Unsupported profile fields.")
        if any(not isinstance(v, str) or len(v) > 1000 for v in profile.values()):
            raise ValueError("Profile answers must be short text.")
        now = utc_now()
        with self.connect() as conn:
            conn.execute("INSERT INTO reviewers VALUES (?, ?, ?, ?) ON CONFLICT(code) DO UPDATE SET profile=excluded.profile,"
                         " updated_at=excluded.updated_at", (reviewer, json.dumps(profile, ensure_ascii=False), now, now))
        return {"reviewer": reviewer, "profile": profile}

    def completed(self, reviewer):
        with self.connect() as conn:
            return [r[0] for r in conn.execute("SELECT task_id FROM completions WHERE reviewer=?", (reviewer,))]

    def session(self, reviewer):
        self.taskset.ensure_unchanged()
        sequence = self.taskset.sequence(reviewer)
        with self.connect() as conn:
            row = conn.execute("SELECT profile FROM reviewers WHERE code=?", (reviewer,)).fetchone()
        return {"reviewer": reviewer, "profile": json.loads(row[0]) if row else None, "sequence": sequence,
                "completed": self.completed(reviewer), "settings": self.taskset.settings,
                "task_set": self.taskset.fingerprint[:12]}

    # --- Events --------------------------------------------------------------------------

    def add_events(self, reviewer, session_id, events):
        self.taskset.sequence(reviewer)
        if not isinstance(session_id, str) or not SESSION.fullmatch(session_id):
            raise ValueError("Invalid session ID.")
        if not isinstance(events, list) or not 0 < len(events) <= 200:
            raise ValueError("Send between 1 and 200 events.")
        allowed = {s["task_id"] for s in self.taskset.sequence(reviewer)}
        rows, now = [], utc_now()
        for event in events:
            if not isinstance(event, dict):
                raise ValueError("Events must be objects.")
            seq, kind, t, wall = event.get("seq"), event.get("type"), event.get("t"), event.get("wall")
            if type(seq) is not int or seq < 0 or kind not in EVENT_TYPES or isinstance(t, bool) \
                    or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0 \
                    or not isinstance(wall, str) or len(wall) > 40:
                raise ValueError("Each event needs seq, a known type, t and wall time.")
            payload = {k: v for k, v in event.items() if k not in ("seq", "type", "t", "wall")}
            for key, value in payload.items():
                expected = EVENT_FIELDS.get(key)
                if expected is None:
                    raise ValueError(f"Unsupported event field: {key}")
                ok = (isinstance(value, bool) if expected is bool else
                      type(value) is int if expected is int else
                      (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value))
                      if expected is float else isinstance(value, str) and len(value) <= 400)
                if not ok:
                    raise ValueError(f"Invalid event field: {key}")
            if "task_id" in payload and payload["task_id"] not in allowed:
                raise ValueError("Event names a task outside this reviewer's order.")
            if "answer" in payload and payload["answer"] not in ANSWERS:
                raise ValueError("Unsupported answer.")
            rows.append((reviewer, session_id, seq, kind, payload.get("task_id"), payload.get("step"), float(t), wall,
                         now, json.dumps(payload, ensure_ascii=False, sort_keys=True)))
        with self.connect() as conn:
            before = conn.total_changes
            conn.executemany("INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
            stored = conn.total_changes - before
        return {"stored": stored, "received": len(rows)}

    # --- Completion ----------------------------------------------------------------------

    def complete(self, reviewer, session_id, task_id, answers, unidentifiable):
        self.taskset.ensure_unchanged()
        sequence = self.taskset.sequence(reviewer)
        task = self.taskset.task_for(reviewer, task_id)
        if not isinstance(session_id, str) or not SESSION.fullmatch(session_id):
            raise ValueError("Invalid session ID.")
        if type(unidentifiable) is not bool or not isinstance(answers, list) or not answers:
            raise ValueError("Send the answers and the unidentifiable flag.")
        steps = []
        for i, a in enumerate(answers, 1):
            if not isinstance(a, dict) or set(a) != {"step", "answer", *ANSWER_COUNTS} or a["step"] != i \
                    or a["answer"] not in ANSWERS:
                raise ValueError("Answers must cover steps 1..k in order with supported values.")
            counts = [a[k] for k in ANSWER_COUNTS]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                   or v > 86_400_000 for v in counts):
                raise ValueError("Invalid timing or counts.")
            steps.append(a)
        last = steps[-1]["answer"]
        if len(steps) > len(task["screens"]) or any(a["answer"] == "same" for a in steps[:-1]) \
                or (last != "same" and len(steps) != len(task["screens"])):
            raise ValueError("A task ends at the first Same, or after the last candidate.")
        now = utc_now()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            done = {r[0] for r in conn.execute("SELECT task_id FROM completions WHERE reviewer=?", (reviewer,))}
            if task_id in done:
                raise Conflict("This task is already complete.")
            upcoming = next((s["task_id"] for s in sequence if s["task_id"] not in done), None)
            if task_id != upcoming:
                raise Conflict("Tasks must be completed in this reviewer's order. Reload to continue.")
            conn.execute("INSERT INTO completions VALUES (?, ?, ?, ?, ?)",
                         (reviewer, task_id, session_id, now, int(unidentifiable)))
            conn.executemany("INSERT INTO answers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                             [(reviewer, task_id, a["step"], a["answer"], float(a["decision_ms"]), float(a["paused_ms"]),
                               int(a["undo_count"]), int(a["zooms"]), int(a["expands"]), int(a["toggles"]))
                              for a in steps])
        return {"task_id": task_id, "completed_at": now}

    # --- Export --------------------------------------------------------------------------

    def export(self):
        self.taskset.ensure_unchanged()
        with self.connect() as conn:  # One read snapshot.
            conn.execute("BEGIN")
            reviewers = {code: json.loads(profile) for code, profile in conn.execute("SELECT code, profile FROM reviewers")}
            completions = [dict(zip(("reviewer", "task_id", "session_id", "completed_at", "unidentifiable"), r))
                           for r in conn.execute("SELECT * FROM completions ORDER BY reviewer, completed_at")]
            answers = [dict(zip(("reviewer", "task_id", "step", "answer", *ANSWER_COUNTS), r))
                       for r in conn.execute("SELECT * FROM answers ORDER BY reviewer, task_id, step")]
            events = []
            for r in conn.execute("SELECT * FROM events ORDER BY reviewer, session_id, seq"):
                reviewer, session_id, seq, kind, _, _, t, wall, server_time, payload = r
                events.append({"reviewer": reviewer, "session_id": session_id, "seq": seq, "type": kind, "t": t,
                               "wall": wall, "server_time": server_time, **json.loads(payload)})
        done = {(c["reviewer"], c["task_id"]): c for c in completions}
        position = {(r, s["task_id"]): s for r in self.taskset.sequences for s in self.taskset.sequences[r]}
        text = io.StringIO(newline="")
        writer = csv.DictWriter(text, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for a in answers:
            c, task = done[(a["reviewer"], a["task_id"])], self.taskset.tasks[a["task_id"]]
            seq, screen = position[(a["reviewer"], a["task_id"])], task["screens"][a["step"] - 1]
            writer.writerow({**{k: a[k] for k in ("reviewer", "task_id", "step", "answer", *ANSWER_COUNTS)},
                             "practice": seq["practice"], "block": seq["block"], "arm": task["arm"],
                             "type": task["type"], "kind": screen["kind"], "position": screen["position"],
                             "list_length": screen["list_length"], "unidentifiable": bool(c["unidentifiable"]),
                             "session_id": c["session_id"], "completed_at": c["completed_at"]})
        data = text.getvalue().encode("utf-8")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        name = f"matching_decisions_{stamp}_{uuid.uuid4().hex[:8]}.csv"
        path = self.export_dir / name
        with path.open("xb") as stream:
            stream.write(data)
        summary = {}
        main = {r: {s["task_id"] for s in seq if not s["practice"]} for r, seq in self.taskset.sequences.items()}
        for reviewer in sorted({c["reviewer"] for c in completions} | set(reviewers)):
            ids = {c["task_id"] for c in completions if c["reviewer"] == reviewer}
            blocks = {s["block"] for s in self.taskset.sequences[reviewer] if not s["practice"]}
            full = sum(all(s["task_id"] in ids for s in self.taskset.sequences[reviewer] if s["block"] == b)
                       for b in blocks)
            summary[reviewer] = {"profile": reviewers.get(reviewer), "main_completed": len(ids & main[reviewer]),
                                 "main_total": len(main[reviewer]), "practice_completed": len(ids - main[reviewer]),
                                 "blocks_completed": full}
        info = {"schema": 1, "kind": EXPORT_KIND, "created_at": utc_now(), "csv": name, "csv_sha256": digest_bytes(data),
                "task_set_dir": str(self.taskset.root), "task_manifest_sha256": self.taskset.manifest_hash,
                "task_set_fingerprint": self.taskset.fingerprint, "reviewers": summary, "completions": completions,
                "answers": answers, "events": events}
        with path.with_suffix(".json").open("x", encoding="utf-8") as stream:
            json.dump(info, stream, ensure_ascii=False, indent=1)
            stream.write("\n")
        return {"filename": name, "path": str(path), "json_path": str(path.with_suffix(".json")),
                "csv_sha256": info["csv_sha256"], "reviewers": summary, "download_url": f"/exports/{name}"}


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, taskset, store):
        self.taskset, self.store = taskset, store
        self.static = APP_ROOT / "static"
        self.allowed_hosts = {"localhost", "127.0.0.1", socket.gethostname().lower(), socket.getfqdn().lower()}
        try:
            self.allowed_hosts.update(socket.gethostbyname_ex(socket.gethostname())[2])
        except OSError:
            pass
        try:
            addresses = subprocess.run(["hostname", "-I"], check=True, capture_output=True,
                                       text=True, timeout=3).stdout.split()
            self.allowed_hosts.update(str(ipaddress.IPv4Address(ip)) for ip in addresses if ":" not in ip)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        if address[0] != "0.0.0.0":
            self.allowed_hosts.add(address[0].lower())
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server_version = "CzechLynxMatchingReview/1.0"

    def respond(self, status, data, content_type="application/json", extra=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; "
                         "img-src 'self' blob: data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; "
                         "form-action 'self'")
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def json(self, status, value):
        self.respond(status, json.dumps(value, ensure_ascii=False).encode("utf-8"))

    def valid_host(self):
        try:
            host = urlsplit("http://" + self.headers.get("Host", ""))
            return host.hostname in self.server.allowed_hosts and host.port == self.server.server_port
        except ValueError:
            return False

    def do_GET(self):
        try:
            if not self.valid_host():
                self.json(403, {"error": "Open the app using this workstation's LAN address or localhost."})
                return
            parts = urlsplit(self.path)
            path, query = parts.path, parse_qs(parts.query)
            reviewer = (query.get("reviewer") or [""])[0]
            if path == "/api/slots":
                self.server.taskset.ensure_unchanged()
                self.json(200, {"slots": sorted(self.server.taskset.orders),
                                "task_set": self.server.taskset.fingerprint[:12]})
            elif path == "/api/session":
                self.json(200, self.server.store.session(reviewer))
            elif path.startswith("/api/task/"):
                self.server.taskset.ensure_unchanged()
                self.json(200, self.server.taskset.task_for(reviewer, path.removeprefix("/api/task/")))
            elif path.startswith("/images/"):
                segments = path.split("/")
                if len(segments) != 4:
                    raise KeyError("Unknown image.")
                data, mime = self.server.taskset.image(segments[2], segments[3])
                self.respond(200, data, mime)
            elif path.startswith("/exports/"):
                name = path.removeprefix("/exports/")
                if not EXPORT_NAME.fullmatch(name):
                    raise KeyError("Unknown export.")
                target = (self.server.store.export_dir / name).resolve()
                if target.parent != self.server.store.export_dir.resolve():
                    raise KeyError("Unknown export.")
                self.respond(200, target.read_bytes(), "text/csv; charset=utf-8",
                             {"Content-Disposition": f'attachment; filename="{name}"'})
            elif path in ("/", "/index.html", "/app.js", "/styles.css"):
                name = "index.html" if path == "/" else path[1:]
                mime = {"index.html": "text/html", "app.js": "text/javascript", "styles.css": "text/css"}[name]
                self.respond(200, (self.server.static / name).read_bytes(), mime + "; charset=utf-8")
            elif path == "/favicon.ico":
                self.respond(204, b"", "image/x-icon")
            else:
                raise KeyError("Unknown page.")
        except (KeyError, FileNotFoundError):
            self.json(404, {"error": "Not found."})
        except ValueError as exc:
            self.json(409, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            traceback.print_exc()
            self.json(500, {"error": "Server error. Check the workstation terminal."})

    def do_POST(self):
        try:
            if (not self.valid_host() or self.headers.get("X-Review-App") != "1"
                    or self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json"):
                self.json(403, {"error": "Use the review app to save or export."})
                return
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                self.json(403, {"error": "Cross-origin writes are not allowed."})
                return
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 262144:
                raise ValueError("Invalid request size.")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object.")
            path, store = urlsplit(self.path).path, self.server.store
            if path == "/api/reviewers":
                if set(body) != {"reviewer", "profile"}:
                    raise ValueError("Invalid profile request.")
                self.json(200, store.save_profile(body["reviewer"], body["profile"]))
            elif path == "/api/events":
                if set(body) != {"reviewer", "session_id", "events"}:
                    raise ValueError("Invalid events request.")
                self.json(200, store.add_events(body["reviewer"], body["session_id"], body["events"]))
            elif path == "/api/complete":
                if set(body) != {"reviewer", "session_id", "task_id", "answers", "unidentifiable"}:
                    raise ValueError("Invalid completion request.")
                self.json(200, store.complete(body["reviewer"], body["session_id"], body["task_id"], body["answers"],
                                              body["unidentifiable"]))
            elif path == "/api/export":
                if body:
                    raise ValueError("Invalid export request.")
                self.json(200, store.export())
            else:
                raise KeyError("Unknown endpoint.")
        except Conflict as exc:
            self.json(409, {"error": str(exc), "conflict": True})
        except (KeyError, FileNotFoundError):
            self.json(404, {"error": "Not found."})
        except (ValueError, UnicodeDecodeError) as exc:
            self.json(400, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            traceback.print_exc()
            self.json(500, {"error": "Server error. Check the workstation terminal."})

    def log_message(self, format, *args):
        # No image requests or event bodies in the log.
        if not self.path.startswith("/images/") and not self.path.startswith("/api/events"):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--host", default="0.0.0.0", help="Bind address; default accepts LAN connections.")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    try:
        print("Checking the task set and preparing storage…", flush=True)
        taskset = TaskSet(args.tasks_dir)
        store = Store(taskset, APP_ROOT)
        server = ReviewServer((args.host, args.port), taskset, store)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Cannot start the matching-review app: {exc}", file=sys.stderr)
        return 1
    print(f"Ready: {len(taskset.tasks)} tasks, reviewer slots {', '.join(sorted(taskset.orders))}. "
          f"Data: {store.db_path}", flush=True)
    local_host = "localhost" if args.host == "0.0.0.0" else args.host
    print(f"On this computer: http://{local_host}:{server.server_port}", flush=True)
    for host in sorted(server.allowed_hosts):
        if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host) and not host.startswith("127."):
            print(f"On the same network: http://{host}:{server.server_port}", flush=True)
    print("Leave this terminal running. Ctrl+C stops the server; saved answers remain.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nMatching-review server stopped.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
