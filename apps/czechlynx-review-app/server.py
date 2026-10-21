#!/usr/bin/env python3
"""Local CzechLynx audit review server. Python standard library only."""
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
import mimetypes
from pathlib import Path
import re
import socket
import sqlite3
import subprocess
import sys
import traceback
from urllib.parse import urlsplit
import uuid

APP_ROOT = Path(__file__).resolve().parent
DEFAULT_AUDIT = APP_ROOT.parents[1] / "evaluations/czechlynx/input_audit_v1"
FIELDS = ("animal_visibility", "usable_markings", "crop_mask", "blur_occlusion", "flank", "illumination")
COLUMNS = ("image_id", *FIELDS, "illumination_evidence", "reviewer", "reviewed_at", "notes")
EDITABLE = (*FIELDS, "illumination_evidence", "reviewer", "notes")
ID_PATTERN = re.compile(r"clx_[0-9a-f]{24}\Z")


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Conflict(ValueError):
    """Another browser saved this record first."""


class Audit:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.manifest_path = self.root / "manifest.json"
        data = self.manifest_path.read_bytes()
        self.manifest_hash = digest_bytes(data)
        m = json.loads(data)
        if (m.get("kind") != "czechlynx_input_audit_v1" or m.get("status") != "complete"
                or m.get("fingerprint") != fingerprint({k: v for k, v in m.items() if k != "fingerprint"})):
            raise ValueError("The audit must be complete and its fingerprint valid.")
        self.fingerprint = m["fingerprint"]
        self.binding = m["binding"]
        for name, expected in m["artifacts"].items():
            target = (self.root / name).resolve()
            if not target.is_relative_to(self.root) or digest_file(target) != expected:
                raise ValueError(f"Changed or invalid audit artifact: {name}")
        self.schema = json.loads((self.root / "review_schema.json").read_text())
        if tuple(self.schema["required"]) != COLUMNS or set(self.schema["choices"]) != set(FIELDS):
            raise ValueError("The audit review schema is not supported by this app.")
        self.choices = self.schema["choices"]
        if any("unknown" not in options for options in self.choices.values()):
            raise ValueError("Each review field must support unknown.")
        for key in ("metadata", "protocol_plan", "raw_metadata"):
            if digest_file(self.binding[key]) != self.binding[key + "_sha256"]:
                raise ValueError(f"Changed audit input: {key}")
        metadata = Path(self.binding["metadata"]).resolve()
        if digest_file(metadata.with_name("manifest.json")) != self.binding["prepared_manifest_sha256"]:
            raise ValueError("Changed prepared metadata manifest.")
        # Only source files listed by the saved audit are ever served. No pipeline imports.
        project = metadata.parents[4]
        for name, expected in m["implementation"].items():
            target = (project / name).resolve()
            if not target.is_relative_to(project) or digest_file(target) != expected:
                raise ValueError(f"Changed audit implementation: {name}")
        sample = m["sample"]
        ids = [row["image_id"] for row in sample]
        if not ids or len(ids) != len(set(ids)) or any(not ID_PATTERN.fullmatch(rid) for rid in ids):
            raise ValueError("Audit sample IDs must be unique CzechLynx image IDs.")
        checks = {row["image_id"]: row for row in m["checks"]}
        if len(checks) != len(m["checks"]) or set(checks) != set(ids):
            raise ValueError("Audit checks must cover the exact sample once.")
        with metadata.open(newline="", encoding="utf-8") as stream:
            records = {}
            for row in csv.DictReader(stream):
                if row["image_id"] in checks:
                    if row["image_id"] in records:
                        raise ValueError("Duplicate prepared sample record.")
                    records[row["image_id"]] = row
        if set(records) != set(ids):
            raise ValueError("Prepared metadata does not cover the sample.")
        self.items = []
        self.assets = {}
        source_root = Path(self.binding["images_root"]).resolve()
        prepared_root = metadata.parent.parent
        self.protected = (self.root, source_root, prepared_root, Path(self.binding["protocol_plan"]).resolve())
        for row in sample:
            rid = row["image_id"]
            check, prepared = checks[rid], records[rid]
            box = json.loads(prepared["bbox"])
            if (box != check["bbox"] or len(box) != 4 or check["annotation_alignment"] != "exact"
                    or prepared["image_sha256"] != check["image_sha256"]
                    or prepared["mask_sha256"] != check["mask_sha256"]
                    or prepared["source_sha256"] != check["source_sha256"]):
                raise ValueError(f"Inconsistent saved alignment check: {rid}")
            width, height = int(prepared["width"]), int(prepared["height"])
            if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                raise ValueError(f"Invalid source bounding box: {rid}")
            assets = {
                "original": (Path(check["source_path"]).resolve(), check["source_sha256"], source_root),
                "crop": (Path(prepared["path"]).resolve(), check["image_sha256"], prepared_root),
                "mask": (Path(prepared["mask_path"]).resolve(), check["mask_sha256"], prepared_root),
            }
            for path, _, root in assets.values():
                if not path.is_relative_to(root) or not path.is_file():
                    raise ValueError(f"Missing or out-of-root image: {rid}")
            self.assets[rid] = assets
            # Never expose identity labels or filenames containing identity labels.
            self.items.append({"image_id": rid, "cohorts": row["cohorts"], "bbox": box,
                               "width": width, "height": height,
                               "foreground_fraction": check["foreground_fraction"]})
        self.ids = tuple(ids)

    def ensure_unchanged(self):
        if digest_file(self.manifest_path) != self.manifest_hash:
            raise ValueError("Audit manifest changed. Restart with the intended audit before saving.")

    def asset(self, rid, kind):
        self.ensure_unchanged()
        if rid not in self.assets or kind not in self.assets[rid]:
            raise KeyError("Unknown image.")
        path, expected, allowed_root = self.assets[rid][kind]
        resolved = path.resolve()
        if not resolved.is_relative_to(allowed_root):
            raise ValueError("Image path changed outside its allowed root.")
        data = resolved.read_bytes()
        if digest_bytes(data) != expected:
            raise ValueError("This image changed since the audit. Restore it before reviewing.")
        return data, mimetypes.guess_type(path.name)[0] or "application/octet-stream"


class Store:
    def __init__(self, audit, app_root):
        self.audit = audit
        self.root = Path(app_root).resolve()
        self.data_dir, self.export_dir = self.root / "data", self.root / "exports"
        for output in (self.data_dir.resolve(), self.export_dir.resolve()):
            for protected in audit.protected:
                if output.is_relative_to(protected) or protected.is_relative_to(output):
                    raise ValueError("Review state and exports must be outside protected audit/source paths.")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "reviews.sqlite3"
        with self.connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS reviews (image_id TEXT PRIMARY KEY, payload TEXT NOT NULL, "
                         "completed INTEGER NOT NULL, revision INTEGER NOT NULL, updated_at TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS history (image_id TEXT NOT NULL, revision INTEGER NOT NULL, "
                         "payload TEXT NOT NULL, completed INTEGER NOT NULL, updated_at TEXT NOT NULL, "
                         "PRIMARY KEY(image_id, revision))")
            binding = json.dumps({"version": 1, "audit_manifest_sha256": audit.manifest_hash,
                                  "audit_fingerprint": audit.fingerprint, "ids": list(audit.ids),
                                  "schema": audit.schema}, sort_keys=True)
            existing = conn.execute("SELECT value FROM settings WHERE key='binding'").fetchone()
            if existing and existing[0] != binding:
                raise ValueError("This app database belongs to a different or changed audit. Use a separate app copy.")
            conn.execute("INSERT OR IGNORE INTO settings VALUES ('binding', ?)", (binding,))
            for rid in audit.ids:
                payload = self.blank(rid)
                conn.execute("INSERT OR IGNORE INTO reviews VALUES (?, ?, 0, 0, '')", (rid, json.dumps(payload)))
            actual = {r[0] for r in conn.execute("SELECT image_id FROM reviews")}
            if actual != set(audit.ids):
                raise ValueError("Review database IDs differ from the audit.")

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path, timeout=15)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @staticmethod
    def blank(rid):
        return {"image_id": rid, **{key: "unknown" for key in FIELDS},
                "illumination_evidence": "", "reviewer": "", "reviewed_at": "", "notes": ""}

    @staticmethod
    def unpack(row):
        return {"review": json.loads(row[0]), "completed": bool(row[1]),
                "revision": row[2], "updated_at": row[3]}

    def all(self):
        with self.connect() as conn:
            return {rid: self.unpack((payload, completed, revision, updated))
                    for rid, payload, completed, revision, updated in
                    conn.execute("SELECT image_id, payload, completed, revision, updated_at FROM reviews ORDER BY image_id")}

    def get(self, rid):
        with self.connect() as conn:
            row = conn.execute("SELECT payload, completed, revision, updated_at FROM reviews WHERE image_id=?", (rid,)).fetchone()
        if row is None:
            raise KeyError("Unknown review.")
        return self.unpack(row)

    def save(self, rid, values, revision, complete):
        self.audit.ensure_unchanged()
        if rid not in self.audit.ids:
            raise KeyError("Unknown review.")
        if type(revision) is not int or revision < 0 or type(complete) is not bool:
            raise ValueError("A nonnegative revision and boolean completion flag are required.")
        if not isinstance(values, dict) or set(values) != set(EDITABLE):
            raise ValueError("Provide exactly the supported review fields.")
        if any(not isinstance(v, str) or len(v) > (10000 if k == "notes" else 2000) for k, v in values.items()):
            raise ValueError("Invalid or excessively long review text.")
        if any(values[key] not in self.audit.choices[key] for key in FIELDS):
            raise ValueError("Unsupported review choice.")
        values = {**values, "reviewer": values["reviewer"].strip()}
        if complete and not values["reviewer"]:
            raise ValueError("Enter your reviewer name before completing a review.")
        if complete and values["illumination"].startswith("verified_") and not values["illumination_evidence"].strip():
            raise ValueError("Verified day/night/IR needs supporting evidence. Monochrome alone is insufficient.")
        now = utc_now()
        payload = {"image_id": rid, **values, "reviewed_at": now if complete else ""}
        encoded = json.dumps(payload, ensure_ascii=False)
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute("SELECT revision FROM reviews WHERE image_id=?", (rid,)).fetchone()[0]
            if current != revision:
                raise Conflict("This image was changed in another browser. Reload its latest review before saving.")
            conn.execute("UPDATE reviews SET payload=?, completed=?, revision=?, updated_at=? WHERE image_id=?",
                         (encoded, int(complete), revision + 1, now, rid))
            conn.execute("INSERT INTO history VALUES (?, ?, ?, ?, ?)", (rid, revision + 1, encoded, int(complete), now))
        return {"review": payload, "completed": complete, "revision": revision + 1, "updated_at": now}

    def export(self, complete):
        self.audit.ensure_unchanged()
        if type(complete) is not bool:
            raise ValueError("Export completion flag must be boolean.")
        rows = self.all()  # One SQLite read snapshot: concurrent saves cannot mix revisions.
        done = sum(row["completed"] for row in rows.values())
        if complete and done != len(self.audit.ids):
            raise ValueError(f"Complete all {len(self.audit.ids)} reviews before exporting the final audit ({done} complete).")
        text = io.StringIO(newline="")
        writer = csv.DictWriter(text, fieldnames=COLUMNS)
        writer.writeheader()
        for rid in self.audit.ids:
            writer.writerow(rows[rid]["review"])
        data = text.getvalue().encode("utf-8")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        name = f"reviews_{'complete' if complete else 'draft'}_{stamp}_{uuid.uuid4().hex[:8]}.csv"
        path = self.export_dir / name
        with path.open("xb") as stream:
            stream.write(data)
        info = {"filename": name, "path": str(path), "created_at": utc_now(), "completed": done,
                "total": len(rows), "final": complete, "csv_sha256": digest_bytes(data),
                "audit_dir": str(self.audit.root), "audit_manifest_sha256": self.audit.manifest_hash,
                "audit_fingerprint": self.audit.fingerprint,
                "revisions": {rid: row["revision"] for rid, row in rows.items()}}
        with path.with_suffix(".json").open("x", encoding="utf-8") as stream:
            json.dump(info, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        return {**{k: info[k] for k in ("filename", "path", "completed", "total", "final", "csv_sha256")},
                "download_url": f"/exports/{name}"}


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, audit, store):
        self.audit, self.store = audit, store
        self.static = APP_ROOT / "static"
        self.allowed_hosts = {"localhost", "127.0.0.1", socket.gethostname().lower(), socket.getfqdn().lower()}
        try:
            self.allowed_hosts.update(socket.gethostbyname_ex(socket.gethostname())[2])
        except OSError:
            pass
        # On Ubuntu the hostname often resolves only to 127.0.1.1. Enumerate LAN
        # interfaces as well, so normal browser Host headers pass validation.
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
    server_version = "CzechLynxReview/1.0"

    def respond(self, status, data, content_type="application/json", extra=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; "
                         "img-src 'self' blob: data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if extra:
            for name, value in extra.items():
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
            path = urlsplit(self.path).path
            if path == "/api/session":
                self.server.audit.ensure_unchanged()
                self.json(200, {"items": self.server.audit.items, "schema": self.server.audit.schema,
                                "reviews": self.server.store.all()})
            elif path.startswith("/api/reviews/"):
                self.json(200, self.server.store.get(path.removeprefix("/api/reviews/")))
            elif path.startswith("/images/"):
                parts = path.split("/")
                if len(parts) != 4:
                    raise KeyError("Unknown image.")
                data, mime = self.server.audit.asset(parts[2], parts[3])
                self.respond(200, data, mime)
            elif path.startswith("/exports/"):
                name = path.removeprefix("/exports/")
                if not re.fullmatch(r"reviews_(?:complete|draft)_[0-9]{8}T[0-9]{12}Z_[0-9a-f]{8}\.csv", name):
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
            if not 0 < length <= 65536:
                raise ValueError("Invalid request size.")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object.")
            path = urlsplit(self.path).path
            if path.startswith("/api/reviews/"):
                if set(body) != {"review", "revision", "complete"}:
                    raise ValueError("Invalid save request.")
                self.json(200, self.server.store.save(path.removeprefix("/api/reviews/"),
                          body["review"], body["revision"], body["complete"]))
            elif path == "/api/export":
                if set(body) != {"complete"}:
                    raise ValueError("Invalid export request.")
                self.json(200, self.server.store.export(body["complete"]))
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
        # No review text, identity-bearing source paths or image requests in logs.
        if not self.path.startswith("/images/"):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--host", default="0.0.0.0", help="Bind address; default accepts LAN connections.")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    try:
        print("Checking the saved audit and preparing review storage…", flush=True)
        audit = Audit(args.audit_dir)
        store = Store(audit, APP_ROOT)
        server = ReviewServer((args.host, args.port), audit, store)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Cannot start review app: {exc}", file=sys.stderr)
        return 1
    print(f"Ready: {len(audit.ids)} photographs. Reviews: {store.db_path}", flush=True)
    local_host = "localhost" if args.host == "0.0.0.0" else args.host
    print(f"On this computer: http://{local_host}:{server.server_port}", flush=True)
    for host in sorted(server.allowed_hosts):
        if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host) and not host.startswith("127."):
            print(f"On the same network: http://{host}:{server.server_port}", flush=True)
    print("Leave this terminal running. Ctrl+C stops the server; saved reviews remain.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nReview server stopped.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
