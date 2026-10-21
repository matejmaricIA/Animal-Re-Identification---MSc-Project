"""Model-free storage, protocol and HTTP checks for the timed matching-review app. Run with unittest."""
import csv
import hashlib
from http.client import HTTPConnection
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import unittest
import zlib

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import server

SECRET = "lynx_065"
SESSION = "ab" * 16


def png_pixel(red, green, blue):
    def chunk(kind, content):
        return struct.pack(">I", len(content)) + kind + content + struct.pack(">I", zlib.crc32(kind + content))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes([0, red, green, blue]))) + chunk(b"IEND", b""))


def token(i):
    return f"{i:024x}"


def photo(i):
    return {"token": token(i), "date": f"2020-01-0{i + 1}", "camera": "cam-1"}


def task(task_id, arm, kind, screens, practice=False, screen_kind=None):
    payload = {"task_id": task_id, "arm": arm, "type": kind, "practice": practice, "banner": "Banner",
               "query_label": "New photos", "candidate_label": "Catalogue animal", "query": [photo(0)],
               "screens": [{"step": s, "kind": screen_kind or ("rank1_comparison" if s == 1 else "further_comparison"),
                            "position": s, "list_length": screens, "exemplars": [photo(1), photo(2)],
                            "more": [photo(3)], "members": 3} for s in range(1, screens + 1)]}
    if practice:
        payload["feedback"] = ["different"] * screens
    return payload


def fixture(root):
    """A complete synthetic task set whose image folder and truth file carry an identity label."""
    images_dir = root / "project/data/raw" / SECRET
    tasks_dir = root / "project/evaluations/tasks"
    images_dir.mkdir(parents=True)
    tasks_dir.mkdir(parents=True)
    images = {}
    for i in range(4):
        original, masked = images_dir / f"{i}_{SECRET}.png", images_dir / f"{i}_masked.png"
        original.write_bytes(png_pixel(i, 10, 10))
        masked.write_bytes(png_pixel(i, 200, 10))
        images[token(i)] = {"original": {"path": str(original), "sha256": server.digest_file(original)},
                            "masked": {"path": str(masked), "sha256": server.digest_file(masked)}}
    tasks = {"p001": task("p001", "W1", "placement", 2, practice=True), "t001": task("t001", "W1", "placement", 3),
             "t002": task("t002", "N2", "audit", 1, screen_kind="audit_glance")}
    orders = {"R1": {"practice": ["p001"], "blocks": [{"block": 1, "arm": "W1", "arm_block": 1, "tasks": ["t001"]},
                                                      {"block": 2, "arm": "N2", "arm_block": 1, "tasks": ["t002"]}]},
              "R2": {"practice": ["p001"], "blocks": [{"block": 1, "arm": "N2", "arm_block": 1, "tasks": ["t002"]},
                                                      {"block": 2, "arm": "W1", "arm_block": 1, "tasks": ["t001"]}]}}
    files = {
        "tasks.json": {"schema": 1, "kind": server.TASK_SET_KIND, "tasks": tasks, "practice": ["p001"],
                       "main": ["t001", "t002"], "orders": orders,
                       "settings": {"idle_pause_seconds": 120, "block_size": 1, "blocks": 2, "early_stop_blocks": 1,
                                    "break_seconds": 120, "blocks_per_session": 3}},
        "images.json": {"schema": 1, "kind": server.TASK_SET_KIND, "images": images},
        "truth.json": {"tasks": {"t001": {"identity": SECRET, "screens_truth": [False, True, False]}}},
        "pre_registration.json": {"rule": "fixed"},
    }
    for name, value in files.items():
        (tasks_dir / name).write_text(json.dumps(value))
    artifacts = {name: server.digest_file(tasks_dir / name) for name in files}
    manifest = {"kind": server.TASK_SET_KIND, "status": "complete", "artifacts": artifacts,
                "task_set_fingerprint": hashlib.sha256(json.dumps(artifacts, sort_keys=True).encode()).hexdigest()}
    (tasks_dir / "manifest.json").write_text(json.dumps(manifest))
    return tasks_dir, images_dir


def answers(*values):
    return [{"step": i, "answer": v, "decision_ms": 1000.0 * i, "paused_ms": 0, "undo_count": 0, "zooms": 0,
             "expands": 0, "toggles": 0} for i, v in enumerate(values, 1)]


def event(seq, kind, **fields):
    return {"seq": seq, "type": kind, "t": float(seq * 100), "wall": "2026-10-20T10:00:00.000Z", **fields}


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tasks_dir, self.images_dir = fixture(self.root)
        self.taskset = server.TaskSet(self.tasks_dir)
        self.store = server.Store(self.taskset, self.root / "app")

    def finish_practice(self, reviewer="R1"):
        return self.store.complete(reviewer, SESSION, "p001", answers("different", "different"), False)

    def test_task_set_loads_without_the_truth_file(self):
        (self.tasks_dir / "truth.json").unlink()  # The app never needs (or opens) the truth file.
        taskset = server.TaskSet(self.tasks_dir)
        self.assertEqual(sorted(taskset.orders), ["R1", "R2"])
        self.assertEqual([s["task_id"] for s in taskset.sequence("R2")], ["p001", "t002", "t001"])

    def test_changed_task_files_are_refused(self):
        path = self.tasks_dir / "tasks.json"
        path.write_text(path.read_text().replace("Banner", "Changed"))
        with self.assertRaises(ValueError):
            server.TaskSet(self.tasks_dir)

    def test_browser_payloads_hide_identity_paths_and_truth(self):
        text = json.dumps([self.store.session("R1"), self.taskset.task_for("R1", "t001")])
        for leak in (SECRET, str(self.images_dir), "screens_truth", ".png"):
            self.assertNotIn(leak, text)

    def test_images_are_hash_checked_and_unknown_requests_rejected(self):
        data, mime = self.taskset.image(token(1), "masked")
        self.assertEqual(mime, "image/png")
        self.assertEqual(data, (self.images_dir / "1_masked.png").read_bytes())
        with self.assertRaises(KeyError):
            self.taskset.image(token(1), "mask")
        with self.assertRaises(KeyError):
            self.taskset.image("f" * 24, "original")
        (self.images_dir / "1_masked.png").write_bytes(png_pixel(9, 9, 9))
        with self.assertRaises(ValueError):
            self.taskset.image(token(1), "masked")

    def test_storage_refuses_protected_locations(self):
        for root in (self.tasks_dir / "app", self.images_dir / "app", self.root / "project"):
            with self.assertRaises(ValueError):
                server.Store(self.taskset, root)
        with self.assertRaises(ValueError):
            server.Store(self.taskset, self.root / "other", protected=[self.root / "other" / "data"])
        guarded = server.INPUT_AUDIT_APP / "copy"
        with self.assertRaises(ValueError):
            server.Store(self.taskset, guarded)

    def test_database_is_bound_to_its_task_set(self):
        manifest = self.tasks_dir / "manifest.json"
        value = json.loads(manifest.read_text())
        value["task_set_fingerprint"] = "0" * 64
        manifest.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            server.Store(server.TaskSet(self.tasks_dir), self.root / "app")

    def test_events_are_validated_and_idempotent(self):
        batch = [event(0, "session_start", width=1200, height=800, dpr=1, user_agent="test"),
                 event(1, "task_start", task_id="p001", block=0), event(2, "screen_shown", task_id="p001", step=1,
                                                                        load_ms=12.5)]
        self.assertEqual(self.store.add_events("R1", SESSION, batch)["stored"], 3)
        self.assertEqual(self.store.add_events("R1", SESSION, batch)["stored"], 0)
        bad = [[event(3, "teleport")], [event(3, "answer", answer="maybe")], [event(3, "answer", secret=1)],
               [event(3, "answer", task_id="nope")], [event(3, "screen_shown", step=True)],
               [{**event(3, "pause"), "t": float("nan")}]]
        for events in bad:
            with self.assertRaises(ValueError):
                self.store.add_events("R1", SESSION, events)
        with self.assertRaises(ValueError):
            self.store.add_events("R1", "not-hex", batch)
        with self.assertRaises(KeyError):
            self.store.add_events("R9", SESSION, batch)

    def test_tasks_complete_in_order_once_with_valid_answers(self):
        with self.assertRaises(server.Conflict):  # Practice comes first.
            self.store.complete("R1", SESSION, "t001", answers("same"), False)
        self.finish_practice()
        with self.assertRaises(server.Conflict):
            self.finish_practice()
        invalid = [answers("same", "different"), answers("different"), answers("different", "unsure"),
                   [{**answers("same")[0], "decision_ms": -1}], [{**answers("same")[0], "step": 2}],
                   answers("different", "different", "different", "different")]
        for values in invalid:
            with self.assertRaises(ValueError):
                self.store.complete("R1", SESSION, "t001", values, False)
        self.store.complete("R1", SESSION, "t001", answers("different", "same"), True)
        self.store.complete("R1", SESSION, "t002", answers("unsure"), False)
        self.assertEqual(sorted(self.store.completed("R1")), ["p001", "t001", "t002"])

    def test_progress_and_profile_survive_a_restart(self):
        self.store.save_profile("R1", {"name": "Reviewer", "pointing_device": "mouse"})
        self.finish_practice()
        restarted = server.Store(server.TaskSet(self.tasks_dir), self.root / "app")
        session = restarted.session("R1")
        self.assertEqual(session["completed"], ["p001"])
        self.assertEqual(session["profile"]["pointing_device"], "mouse")
        with self.assertRaises(ValueError):
            restarted.save_profile("R1", {"favourite_colour": "green"})
        with self.assertRaises(ValueError):
            restarted.save_profile("R1", {"notes": "x" * 1001})

    def test_export_snapshots_are_new_files_with_exact_columns(self):
        self.store.save_profile("R1", {"name": "Reviewer"})
        self.store.add_events("R1", SESSION, [event(0, "task_start", task_id="p001"),
                                              event(1, "answer", task_id="p001", step=1, answer="different")])
        self.finish_practice()
        self.store.complete("R1", SESSION, "t001", answers("different", "different", "unsure"), False)
        first = self.store.export()
        second = self.store.export()
        self.assertNotEqual(first["path"], second["path"])
        path = Path(first["path"])
        rows = list(csv.DictReader(io.StringIO(path.read_text())))
        self.assertEqual(list(rows[0]), list(server.CSV_COLUMNS))
        self.assertEqual([(r["task_id"], r["step"], r["kind"]) for r in rows if r["practice"] == "False"],
                         [("t001", "1", "rank1_comparison"), ("t001", "2", "further_comparison"),
                          ("t001", "3", "further_comparison")])
        info = json.loads(path.with_suffix(".json").read_text())
        self.assertEqual(info["kind"], server.EXPORT_KIND)
        self.assertEqual(info["csv_sha256"], server.digest_file(path))
        self.assertEqual(info["task_set_fingerprint"], self.taskset.fingerprint)
        self.assertEqual(info["reviewers"]["R1"]["main_completed"], 1)
        self.assertEqual(info["reviewers"]["R1"]["blocks_completed"], 1)
        self.assertEqual(info["events"][1]["answer"], "different")
        for text in (path.read_text(), path.with_suffix(".json").read_text()):
            self.assertNotIn(SECRET, text)


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        tasks_dir, _ = fixture(root)
        taskset = server.TaskSet(tasks_dir)
        self.store = server.Store(taskset, root / "app")
        self.httpd = server.ReviewServer(("127.0.0.1", 0), taskset, self.store)
        thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.port = self.httpd.server_port

    def request(self, method, path, body=None, headers=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = None if body is None else json.dumps(body).encode()
        base = {"Content-Type": "application/json", "X-Review-App": "1"} if method == "POST" else {}
        conn.request(method, path, body=payload, headers={**base, **(headers or {})})
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, response.getheader("Content-Type"), data

    def test_routes_are_allowlisted(self):
        for path in ("/", "/app.js", "/styles.css", "/api/slots", "/api/session?reviewer=R1",
                     "/api/task/p001?reviewer=R1", f"/images/{token(0)}/original"):
            status, _, _ = self.request("GET", path)
            self.assertEqual(status, 200, path)
        for path in ("/truth.json", "/tasks.json", "/server.py", f"/images/{token(0)}/x", "/api/task/zzz?reviewer=R1",
                     "/exports/../server.py", "/api/session?reviewer=R9"):
            status, _, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)
        status, mime, _ = self.request("GET", f"/images/{token(2)}/masked")
        self.assertEqual((status, mime), (200, "image/png"))
        status, _, _ = self.request("GET", "/", headers={"Host": f"evil.example:{self.port}"})
        self.assertEqual(status, 403)

    def test_writes_need_the_app_header_and_same_origin(self):
        body = {"reviewer": "R1", "session_id": SESSION, "events": [event(0, "task_start", task_id="p001")]}
        status, _, _ = self.request("POST", "/api/events", body, headers={"X-Review-App": "0"})
        self.assertEqual(status, 403)
        status, _, _ = self.request("POST", "/api/events", body, headers={"Origin": "http://evil.example"})
        self.assertEqual(status, 403)
        status, _, data = self.request("POST", "/api/events", body)
        self.assertEqual((status, json.loads(data)["stored"]), (200, 1))
        complete = {"reviewer": "R1", "session_id": SESSION, "task_id": "t001", "answers": answers("same"),
                    "unidentifiable": False}
        status, _, data = self.request("POST", "/api/complete", complete)
        self.assertEqual(status, 409)
        self.assertTrue(json.loads(data)["conflict"])
        status, _, data = self.request("POST", "/api/export", {})
        self.assertEqual(status, 200)
        name = json.loads(data)["filename"]
        status, mime, _ = self.request("GET", f"/exports/{name}")
        self.assertEqual((status, mime), (200, "text/csv; charset=utf-8"))


if __name__ == "__main__":
    unittest.main()
