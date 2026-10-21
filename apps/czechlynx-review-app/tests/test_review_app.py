"""Model-free persistence, export and HTTP access checks. Run with unittest."""
import csv
import hashlib
from http.client import HTTPConnection
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

CHOICES = {
    "animal_visibility": ["clear", "partial", "no_clear_animal", "unknown"],
    "usable_markings": ["yes", "no", "unknown"], "crop_mask": ["correct", "issue", "unknown"],
    "blur_occlusion": ["minor", "moderate", "severe", "unknown"], "flank": ["left", "right", "unknown"],
    "illumination": ["colour", "monochrome", "verified_day", "verified_night", "verified_ir", "unknown"],
}


def png_pixel(red, green, blue):
    def chunk(kind, content):
        return struct.pack(">I", len(content)) + kind + content + struct.pack(">I", zlib.crc32(kind + content))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes([0, red, green, blue]))) + chunk(b"IEND", b""))


def fixture(root):
    project = root / "project"
    raw = project / "data/czechlynx/raw"
    prepared = project / "data/czechlynx/prepared_masked/masked"
    crops = prepared.parent / "masked_crops"
    audit = project / "evaluations/audit"
    for path in (raw, prepared, crops, audit):
        path.mkdir(parents=True)
    rows, sample, checks = [], [], []
    for i in range(3):
        rid = f"clx_{i:024x}"
        source, crop, mask = raw / f"identity_secret_{i}.png", crops / f"{rid}.png", crops / f"{rid}_mask.png"
        for path, data in ((source, png_pixel(i, 120, 20)), (crop, png_pixel(i, 120, 20)), (mask, png_pixel(255, 255, 255))):
            path.write_bytes(data)
        row = {"image_id": rid, "identity": f"secret_{i}", "path": str(crop), "mask_path": str(mask),
               "source_path": str(source.relative_to(raw)), "image_sha256": server.digest_file(crop),
               "mask_sha256": server.digest_file(mask), "source_sha256": server.digest_file(source),
               "bbox": "[0,0,1,1]", "width": "1", "height": "1"}
        rows.append(row)
        sample.append({"image_id": rid, "cohorts": ["calibration_uniform"]})
        checks.append({"image_id": rid, "annotation_alignment": "exact", "source_path": str(source),
                       "image_sha256": row["image_sha256"], "mask_sha256": row["mask_sha256"],
                       "source_sha256": row["source_sha256"], "bbox": [0, 0, 1, 1], "foreground_fraction": 1.0})
    metadata = prepared / "metadata.csv"
    with metadata.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    prepared_manifest = prepared / "manifest.json"
    prepared_manifest.write_text("{}")
    protocol = project / "protocol.json"; protocol.write_text("{}")
    raw_metadata = raw / "raw.csv"; raw_metadata.write_text("test\n")
    schema = audit / "review_schema.json"
    schema.write_text(json.dumps({"choices": CHOICES, "required": list(server.COLUMNS), "notes": "Unknown is valid."}))
    impl = project / "reid_demo/audit.py"; impl.parent.mkdir(); impl.write_text("# audit implementation\n")
    manifest = {"kind": "czechlynx_input_audit_v1", "status": "complete", "sample": sample, "checks": checks,
                "binding": {"images_root": str(raw), "metadata": str(metadata), "metadata_sha256": server.digest_file(metadata),
                            "prepared_manifest_sha256": server.digest_file(prepared_manifest),
                            "protocol_plan": str(protocol), "protocol_plan_sha256": server.digest_file(protocol),
                            "raw_metadata": str(raw_metadata), "raw_metadata_sha256": server.digest_file(raw_metadata)},
                "artifacts": {"review_schema.json": server.digest_file(schema)},
                "implementation": {"reid_demo/audit.py": server.digest_file(impl)}}
    manifest["fingerprint"] = server.fingerprint(manifest)
    (audit / "manifest.json").write_text(json.dumps(manifest))
    return audit, rows


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.audit_dir, self.rows = fixture(self.root)
        self.audit = server.Audit(self.audit_dir)
        self.store = server.Store(self.audit, self.root / "review-app")
        self.rid = self.audit.ids[0]
        self.values = {key: server.Store.blank(self.rid)[key] for key in server.EDITABLE}

    def complete(self, rid=None, **changes):
        rid = rid or self.rid
        record = self.store.get(rid)
        return self.store.save(rid, {**self.values, "reviewer": "Reviewer", **changes}, record["revision"], True)

    def test_defaults_unknown_and_drafts_never_marked_complete(self):
        self.assertEqual(0, self.store.get(self.rid)["revision"])
        draft = self.store.save(self.rid, {**self.values, "notes": "Still checking"}, 0, False)
        self.assertFalse(draft["completed"])
        self.assertEqual("", draft["review"]["reviewed_at"])
        self.assertEqual(1, draft["revision"])
        with self.assertRaisesRegex(ValueError, "reviewer"):
            self.store.save(self.rid, self.values, 1, True)

    def test_unknown_is_a_valid_completed_answer_and_state_survives_restart(self):
        saved = self.complete(notes='Comma, newline\nCroatian: čćžšđ "quote"')
        reopened = server.Store(server.Audit(self.audit_dir), self.root / "review-app")
        self.assertEqual(saved, reopened.get(self.rid))
        self.assertTrue(saved["completed"])
        self.assertEqual("unknown", saved["review"]["flank"])
        self.assertTrue(saved["review"]["reviewed_at"].endswith("+00:00"))

    def test_verified_illumination_requires_evidence_only_on_completion(self):
        draft = self.store.save(self.rid, {**self.values, "illumination": "verified_night"}, 0, False)
        with self.assertRaisesRegex(ValueError, "evidence"):
            self.store.save(self.rid, {**self.values, "reviewer": "R", "illumination": "verified_night"}, draft["revision"], True)
        self.complete(illumination="verified_ir", illumination_evidence="Camera IR mode recorded")

    def test_edit_completed_review_returns_to_draft_and_preserves_history(self):
        saved = self.complete()
        draft = self.store.save(self.rid, {**self.values, "reviewer": "R", "notes": "Changed"}, saved["revision"], False)
        self.assertFalse(draft["completed"])
        self.assertEqual("", draft["review"]["reviewed_at"])
        with self.store.connect() as conn:
            self.assertEqual([(1, 1), (2, 0)], conn.execute("SELECT revision, completed FROM history ORDER BY revision").fetchall())

    def test_stale_browser_cannot_overwrite_review(self):
        saved = self.complete()
        with self.assertRaises(server.Conflict):
            self.store.save(self.rid, {**self.values, "notes": "Stale browser"}, 0, False)
        self.assertEqual(saved, self.store.get(self.rid))

    def test_reject_invalid_fields_choices_revision_and_unknown_ids(self):
        for values in ({**self.values, "flank": "both"}, {**self.values, "identity": "leaked"}, {**self.values, "notes": 1}):
            with self.assertRaises(ValueError):
                self.store.save(self.rid, values, 0, False)
        with self.assertRaises(ValueError): self.store.save(self.rid, self.values, True, False)
        with self.assertRaises(KeyError): self.store.save("../manifest.json", self.values, 0, False)

    def test_exact_csv_schema_roundtrip_and_immutable_export_snapshots(self):
        with self.assertRaisesRegex(ValueError, "Complete all"):
            self.store.export(True)
        self.complete(notes='Newline\ncomma, "quotes" čćžšđ', illumination="monochrome")
        first = self.store.export(False)
        first_bytes = Path(first["path"]).read_bytes()
        self.complete(notes="Later edit")
        second = self.store.export(False)
        self.assertNotEqual(first["path"], second["path"])
        self.assertEqual(first_bytes, Path(first["path"]).read_bytes())
        with Path(first["path"]).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(list(server.COLUMNS), reader.fieldnames)
            rows = list(reader)
        self.assertEqual(3, len(rows))
        self.assertEqual(set(self.audit.ids), {r["image_id"] for r in rows})
        self.assertEqual('Newline\ncomma, "quotes" čćžšđ', rows[0]["notes"])
        self.assertEqual("", rows[1]["reviewed_at"])
        for rid in self.audit.ids[1:]: self.complete(rid)
        final = self.store.export(True)
        self.assertEqual(3, final["completed"])
        self.assertEqual(server.digest_file(final["path"]), final["csv_sha256"])

    def test_changed_audit_cannot_reuse_existing_database_or_save(self):
        path = self.audit_dir / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["extra"] = "changed audit"
        manifest["fingerprint"] = server.fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"})
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            self.store.save(self.rid, self.values, 0, False)
        with self.assertRaisesRegex(ValueError, "different or changed audit"):
            server.Store(server.Audit(self.audit_dir), self.root / "review-app")

    def test_sources_read_only_assets_verified_and_identity_hidden(self):
        before = {str(p): server.digest_file(p) for p in (self.root / "project").rglob("*") if p.is_file()}
        self.complete(); self.store.export(False)
        after = {str(p): server.digest_file(p) for p in (self.root / "project").rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertNotIn("secret", json.dumps(self.audit.items))
        for kind in ("original", "crop", "mask"):
            data, mime = self.audit.asset(self.rid, kind)
            self.assertTrue(data.startswith(b"\x89PNG")); self.assertEqual("image/png", mime)
        Path(self.rows[0]["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "image changed"):
            self.audit.asset(self.rid, "crop")
        with self.assertRaises(KeyError): self.audit.asset(self.rid, "../../manifest.json")

    def test_state_directory_cannot_be_under_protected_inputs(self):
        with self.assertRaisesRegex(ValueError, "protected"):
            server.Store(self.audit, self.audit.root / "review-app")


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        directory, _ = fixture(root)
        audit = server.Audit(directory)
        store = server.Store(audit, root / "review-app")
        self.http = server.ReviewServer(("127.0.0.1", 0), audit, store)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)
        self.port = self.http.server_port

    def stop(self):
        self.http.shutdown(); self.http.server_close(); self.thread.join(timeout=5)

    def request(self, method, path, body=None, headers=None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            connection.request(method, path, body, headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_http_assets_session_and_path_allowlist(self):
        status, _, body = self.request("GET", "/api/session")
        self.assertEqual(200, status)
        self.assertNotIn(b"identity_secret", body)
        self.assertEqual(3, len(json.loads(body)["items"]))
        for path in ("/", "/app.js", "/styles.css"):
            self.assertEqual(200, self.request("GET", path)[0])
        for path in ("/server.py", "/data/reviews.sqlite3", "/../server.py", "/images/../manifest.json", "/exports/../data/reviews.sqlite3"):
            self.assertEqual(404, self.request("GET", path)[0])

    def test_http_write_origin_guard_and_download(self):
        self.assertEqual(403, self.request("POST", "/api/export", '{"complete":false}')[0])
        headers = {"Content-Type": "application/json", "X-Review-App": "1", "Origin": "https://unrelated.example"}
        self.assertEqual(403, self.request("POST", "/api/export", '{"complete":false}', headers)[0])
        headers["Origin"] = f"http://127.0.0.1:{self.port}"
        status, _, body = self.request("POST", "/api/export", '{"complete":false}', headers)
        self.assertEqual(200, status)
        exported = json.loads(body)
        status, response_headers, data = self.request("GET", exported["download_url"])
        self.assertEqual(200, status)
        self.assertIn("attachment", response_headers["Content-Disposition"])
        self.assertEqual(hashlib.sha256(data).hexdigest(), exported["csv_sha256"])
        self.assertEqual(403, self.request("GET", "/api/session", headers={"Host": "unrelated.example"})[0])

    def test_http_save_and_conflicting_browser_response(self):
        rid = self.http.audit.ids[0]
        values = {key: server.Store.blank(rid)[key] for key in server.EDITABLE}
        values["reviewer"] = "Reviewer"
        payload = json.dumps({"review": values, "revision": 0, "complete": True})
        headers = {"Content-Type": "application/json", "X-Review-App": "1"}
        status, _, body = self.request("POST", f"/api/reviews/{rid}", payload, headers)
        self.assertEqual(200, status)
        self.assertTrue(json.loads(body)["completed"])
        status, _, body = self.request("POST", f"/api/reviews/{rid}", payload, headers)
        self.assertEqual(409, status)
        self.assertTrue(json.loads(body)["conflict"])


if __name__ == "__main__":
    unittest.main()
