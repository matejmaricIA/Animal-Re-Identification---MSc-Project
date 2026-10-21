"""Global-model contracts; no pretrained downloads or neural model inference."""
import argparse
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import reid_demo.benchmark_embeddings as adapters
import tools.evaluate_czechlynx as runner
from reid_demo.benchmark import file_digest, fingerprint, normalized, write_json


def options(**changes):
    return argparse.Namespace(**dict(
        {"global_model": "megadescriptor", "ckpt": None, "miewid_revision": None,
         "batch_size": 2, "workers": 0}, **changes))


@pytest.mark.parametrize("changes, message", [
    ({"global_model": "miewid", "ckpt": "custom.pt"}, "MegaDescriptor checkpoint"),
    ({"global_model": "miewid", "miewid_revision": "main"}, "commit hash"),
    ({"miewid_revision": adapters.MIEWID_REVISION}, "requires"),
])
def test_invalid_model_options_fail_before_loading(changes, message):
    with pytest.raises(ValueError, match=message):
        adapters.validate_global_options(options(**changes))


def test_miewid_defaults_to_immutable_revision():
    args = options(global_model="miewid")
    adapters.validate_global_options(args)
    assert args.miewid_revision == adapters.MIEWID_REVISION


def test_megadescriptor_adapter_preserves_checkpoint_loader(monkeypatch):
    import tools.evaluate_reid_embeddings as legacy
    sentinel = object()
    calls = []

    def load(args, device):
        calls.append((args.pretrained_megadescriptor, args.ckpt, device))
        return sentinel, {"kind": "scratch_checkpoint", "embedding_dim": 1536}

    monkeypatch.setattr(legacy, "_load_model", load)
    model, info, size = adapters.load_global_model(options(ckpt="custom.pt"), "cpu")
    assert model is sentinel and size == 384
    assert calls == [(False, "custom.pt", "cpu")]
    monkeypatch.setattr(legacy, "_load_model", lambda *a: (sentinel, {"missing_keys": ["backbone"]}))
    with pytest.raises(ValueError, match="architecture"):
        adapters.load_global_model(options(), "cpu")


def test_miewid_loader_pins_weights_and_code_and_freezes_parameters(monkeypatch):
    import torch
    from transformers import AutoModel
    model = torch.nn.Linear(3, 2)
    calls = []

    def load(name, **kwargs):
        calls.append((name, kwargs))
        return model, {}

    monkeypatch.setattr(AutoModel, "from_pretrained", load)
    loaded, info, size = adapters.load_global_model(options(global_model="miewid"), torch.device("cpu"))
    assert loaded is model and not loaded.training
    assert not any(p.requires_grad for p in loaded.parameters())
    assert size == 440 and info["embedding_dim"] == 2152
    assert info["remote_code_sha256"]
    name, kwargs = calls[0]
    assert name == adapters.MIEWID_MODEL_ID
    assert kwargs == dict(revision=adapters.MIEWID_REVISION, code_revision=adapters.MIEWID_REVISION,
                          trust_remote_code=True, use_safetensors=True, output_loading_info=True)


@pytest.mark.parametrize("key", ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
def test_miewid_rejects_partial_or_incompatible_weights(monkeypatch, key):
    import torch
    from transformers import AutoModel
    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *a, **kw: (torch.nn.Linear(3, 2), {key: ["bad"]}))
    with pytest.raises(ValueError, match="does not match"):
        adapters.load_miewid("cpu")


def test_global_preprocessing_cache_replay_and_model_separation(tmp_path, monkeypatch):
    import torch
    import tools.evaluate_reid_embeddings as legacy
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rows = []
    for i in range(2):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (8, 6), (128, 128, 128)).save(path)
        rows.append(dict(image_id=str(i), path=str(path), identity="secret",
                         image_sha256=file_digest(path), input_mode="masked"))
    frame = pd.DataFrame(rows)
    revision = [adapters.MIEWID_REVISION]
    widths = []
    state = torch.nn.Linear(1, 1).eval()

    def load(args, device):
        if args.global_model == "megadescriptor":
            return state, {"kind": "pretrained_megadescriptor_l_384", "embedding_dim": 3}, 384
        return state, {"kind": "pretrained_miewid_msv3", "embedding_dim": 3,
                       "model_id": adapters.MIEWID_MODEL_ID, "revision": revision[0],
                       "remote_code_sha256": {"modeling_miewid.py": "source-hash"}}, 440

    def extract(model, loader, *, device, amp):
        ids, labels, tensors = [], [], []
        for x, batch_labels, batch_ids, _, _ in loader:
            widths.append(x.shape[-1])
            assert x.shape[1] == 3 and x.shape[-1] == x.shape[-2]
            expected = (torch.tensor([128 / 255] * 3) - torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
            torch.testing.assert_close(x[0, :, 0, 0], expected)
            assert set(batch_labels) == {"unlabelled"}
            ids.extend(batch_ids)
            labels.extend(batch_labels)
            tensors.extend(torch.tensor([1., 2., float(int(rid) + 1)]) for rid in batch_ids)
        return torch.nn.functional.normalize(torch.stack(tensors), dim=1), labels, ids, [], []

    monkeypatch.setattr(runner, "load_global_model", load)
    monkeypatch.setattr(legacy, "_extract_embeddings", extract)
    first, mega_spec = runner.global_vectors(frame, options(), tmp_path / "cache")
    # The existing cache specification is preserved, including its implementation files.
    expected = {"kind": "pretrained_megadescriptor_l_384", "weights_sha256": runner.model_state_digest(state),
                "input_mode": "masked", "transform": "RGB; resize 384x384 bilinear; ImageNet normalization; L2",
                "amp": False, "torch": torch.__version__,
                **{name: runner.importlib.metadata.version(name) for name in ("timm", "torchvision", "pillow")},
                "implementation": runner.source_hashes(["tools/evaluate_reid_embeddings.py", "megadescriptor.py"]),
                "masked_preprocessing": runner.MASKED_PREPROCESSING}
    assert {k: v for k, v in mega_spec.items() if k != "dimension"} == expected
    assert (tmp_path / "cache" / ("global_" + fingerprint(expected)[:20] + ".sqlite")).exists()
    miew_args = options(global_model="miewid")
    second, spec = runner.global_vectors(frame, miew_args, tmp_path / "cache")
    np.testing.assert_array_equal(first, second)
    replay, replay_spec = runner.global_vectors(frame, miew_args, tmp_path / "cache")
    np.testing.assert_array_equal(second, replay)
    assert widths == [384, 440]  # Replay did not extract any images.
    assert spec == replay_spec and spec["dimension"] == 3
    assert spec["transform_size"] == 440 and "440x440" in spec["transform"]
    assert len(list((tmp_path / "cache").glob("global_*.sqlite"))) == 2
    revision[0] = "a" * 40
    runner.global_vectors(frame, miew_args, tmp_path / "cache")
    assert widths == [384, 440, 440]
    changed = frame.copy()
    changed.loc[0, "image_sha256"] = "changed"
    with pytest.raises(ValueError, match="Changed image"):
        runner.global_vectors(changed, miew_args, tmp_path / "cache")


def test_wrong_embedding_dimension_is_not_cached(tmp_path, monkeypatch):
    import torch
    import tools.evaluate_reid_embeddings as legacy
    path = tmp_path / "q.png"
    Image.new("RGB", (4, 4)).save(path)
    frame = pd.DataFrame([dict(image_id="q", identity="secret", path=str(path),
                               image_sha256=file_digest(path), input_mode="bbox")])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runner, "load_global_model", lambda *a: (
        torch.nn.Linear(1, 1), {"kind": "pretrained_megadescriptor_l_384", "embedding_dim": 1536}, 384))
    monkeypatch.setattr(legacy, "_extract_embeddings", lambda *a, **kw: (torch.ones(1, 10), [], ["q"], [], []))
    with pytest.raises(ValueError, match="feature dimension"):
        runner.global_vectors(frame, options(), tmp_path / "cache")


def test_masked_global_cli_comparison_and_threshold_isolation(tmp_path, monkeypatch):
    from reid_demo.benchmark_masks import MASKED_PREPROCESSING
    prepared = tmp_path / "prepared" / "masked"
    prepared.mkdir(parents=True)
    rows = []
    for identity in range(10):
        for encounter in range(4):
            rid = f"{identity}-{encounter}"
            path, mask_path = tmp_path / f"{rid}.png", tmp_path / f"{rid}_mask.png"
            image = Image.new("RGB", (8, 8), (identity * 20, encounter * 40, 50))
            image.putpixel((0, 0), (128, 128, 128))
            image.save(path)
            mask = Image.new("L", (8, 8), 255)
            mask.putpixel((0, 0), 0)
            mask.save(mask_path)
            rows.append(dict(image_id=rid, identity=str(identity), encounter_id=rid,
                             timestamp=f"2020-0{encounter + 1}-01", path=str(path), input_mode="masked",
                             source_sha256=rid, image_sha256=file_digest(path), bbox="[0,0,8,8]",
                             mask_path=str(mask_path), mask_sha256=file_digest(mask_path),
                             **{"split-time_open": "train" if encounter < 3 else "test"}))
    frame = pd.DataFrame(rows)
    metadata = prepared / "metadata.csv"
    frame.to_csv(metadata, index=False)
    write_json(prepared / "manifest.json", {"prepared_metadata_sha256": file_digest(metadata),
                                            "masked_preprocessing": MASKED_PREPROCESSING})
    write_json(prepared.parent / "data_audit.json", {"errors": []})
    calls = []

    def embeddings(images, args, cache):
        calls.append(args.global_model)
        assert set(images.input_mode) == {"masked"}
        matrix = np.stack([np.random.default_rng(int(fingerprint(rid)[:8], 16)).random(3) for rid in images.image_id])
        return normalized(matrix), {"kind": "pretrained_miewid_msv3" if args.global_model == "miewid" else "pretrained_megadescriptor_l_384",
                                    "weights_sha256": args.global_model, "dimension": 3}

    def no_local(*args, **kwargs):
        raise AssertionError("Global-only comparison must not run Fisher or local matching")

    monkeypatch.setattr(runner, "global_vectors", embeddings)
    monkeypatch.setattr(runner, "LocalFeatures", no_local)
    monkeypatch.setattr(runner, "fit_or_load_fisher_models", no_local)
    monkeypatch.setattr(runner, "PairVerifier", no_local)
    common = ["--metadata", str(metadata), "--cache-dir", str(tmp_path / "cache"),
              "--signals", "global", "--cluster-limit", "0", "--workers", "0"]
    baseline, current = tmp_path / "baseline", tmp_path / "miewid"
    assert runner.main(common + ["--out-dir", str(baseline)]) == 0
    assert runner.main(common + ["--out-dir", str(current), "--global-model", "miewid", "--compare-to", str(baseline)]) == 0
    old = json.loads((baseline / "manifest.json").read_text())
    new = json.loads((current / "manifest.json").read_text())
    assert old["gallery_ids"] == new["gallery_ids"] and old["query_ids"] == new["query_ids"]
    assert not {"global_model", "miewid_revision", "compare_to"}.intersection(old["config"])
    assert new["config"]["miewid_revision"] == adapters.MIEWID_REVISION
    assert "fisher_model" not in new and "local_features" not in new
    assert "Global model comparison" in (current / "index.html").read_text()
    assert "MiewID-msv3" in (current / "index.html").read_text()
    comparison = json.loads((current / "global_model_comparison.json").read_text())
    assert comparison["same_inputs"] and comparison["fixed_queries"] == comparison["worsened_queries"] == 0
    contract = json.loads((current / "thresholds.json").read_text())["contract"]
    with pytest.raises(ValueError, match="compatible"):
        runner.validate_thresholds(baseline / "thresholds.json", contract, frame.image_id)
    with pytest.raises(ValueError, match="differs"):
        runner.main(common + ["--out-dir", str(tmp_path / "different"), "--global-model", "miewid",
                              "--compare-to", str(baseline), "--max-identities", "5"])
    assert calls == ["megadescriptor", "miewid"]  # Input mismatch stopped before extraction.

    def failed_load(*args, **kwargs):
        raise RuntimeError("Simulated model download failure")

    monkeypatch.setattr(runner, "global_vectors", failed_load)
    failed = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="download failure"):
        runner.main(common + ["--out-dir", str(failed), "--global-model", "miewid"])
    failure = json.loads((failed / "manifest.json").read_text())
    assert failure["status"] == "failed" and "download failure" in failure["error"]


@pytest.mark.parametrize("extra", [
    ["--global-model", "miewid", "--ckpt", "wrong.pt"],
    ["--global-model", "miewid", "--miewid-revision", "main"],
    ["--compare-to", "reference", "--signals", "full-funnel"],
    ["--compare-to", "output"],
])
def test_cli_rejects_invalid_options_without_reading_data(extra):
    with pytest.raises(SystemExit) as exc:
        runner.main(["--metadata", "does-not-exist.csv", "--out-dir", "output", *extra])
    assert exc.value.code == 2
