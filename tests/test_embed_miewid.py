"""Tests for reid_demo.embed_miewid — MiewID in the T04 demo embedding service (N1e).

Logic-level tests use a stub model (no download, no GPU). One real-model test runs only
when REID_DEMO_REAL_MIEWID=1 and the pinned weights are already in the local Hugging
Face cache.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")
PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

import reid_demo.embed_miewid as em  # noqa: E402
from reid_demo.embed import (  # noqa: E402
    embed_records,
    embedding_cache_path,
    get_embedding_matrix,
)
from reid_demo.store import (  # noqa: E402
    DetectionRecord,
    connect,
    make_record_id,
    query_records,
    upsert_record,
)

DIM = 2152


class _StubMiewID(torch.nn.Module):
    """Records every input batch; returns deterministic RAW (non-unit) 2152-D vectors."""

    def __init__(self):
        super().__init__()
        self.batches = []

    def forward(self, x):
        self.batches.append(x.detach().clone())
        base = x.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        ramp = torch.arange(DIM, dtype=x.dtype, device=x.device) / DIM   # same device as x (CUDA too)
        return base.repeat(1, DIM) * 3.0 + ramp + 5.0


def _stub_loader(model):
    def _load(device, checkpoint_path=None):
        return model, {"kind": "stub_miewid", "embedding_dim": DIM}, 440
    return _load


@pytest.fixture
def stub(monkeypatch):
    model = _StubMiewID()
    monkeypatch.setattr(em, "load_demo_miewid", _stub_loader(model))
    return model


@pytest.fixture
def seeded(tmp_path):
    conn = connect(str(tmp_path / "reid.sqlite"))
    for i, colour in enumerate([(200, 30, 30), (30, 200, 30), (30, 30, 200)], start=1):
        crop = tmp_path / "crops" / f"p{i}.png"
        crop.parent.mkdir(exist_ok=True)
        Image.new("RGB", (64, 48), colour).save(crop)
        upsert_record(conn, DetectionRecord(
            record_id=make_record_id(f"p{i}", 1), source_image=f"src/p{i}.JPG",
            source_stem=f"p{i}", det_index=1, crop_path=str(crop),
            bbox_x=0.1, bbox_y=0.1, bbox_w=0.2, bbox_h=0.2, dataset="LynxDS"))
    return conn


def test_cache_label_binds_revision_and_preprocessing():
    from reid_demo.benchmark_embeddings import MIEWID_REVISION

    label = em.miewid_cache_label("miewid-msv3")
    assert MIEWID_REVISION[:12] in label
    assert label.endswith(f"-pp{em.MIEWID_PREPROCESSING_VERSION}")
    assert em.is_miewid("MiewID-msv3") and not em.is_miewid("megadescriptor-l-384")
    # MegaDescriptor cache paths are unchanged by N1e.
    assert embedding_cache_path("DS").endswith("DS_megadescriptor-l-384.pkl")


def test_checkpoint_label_binds_file_bytes(tmp_path):
    ckpt = tmp_path / "trial" / "best.pt"
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"weights-v1")
    first = em.miewid_cache_label("miewid-msv3", str(ckpt))
    ckpt.write_bytes(b"retrained-weights-v2")          # same path, new bytes
    second = em.miewid_cache_label("miewid-msv3", str(ckpt))
    assert "_ckpt-" in first and first != second
    with pytest.raises(FileNotFoundError):
        em.miewid_cache_label("miewid-msv3", str(tmp_path / "missing.pt"))


def test_checkpoint_without_manifest_refused(tmp_path):
    ckpt = tmp_path / "trial" / "best.pt"
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"weights")
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        em.load_demo_miewid(torch.device("cpu"), str(ckpt))


def test_embed_records_dispatches_to_miewid(stub, seeded, tmp_path):
    conn = seeded
    res = embed_records(conn, dataset="LynxDS", model_name="miewid-msv3",
                        cache_dir=str(tmp_path / "emb"))
    assert res.n_embedded == 3 and res.embedding_dim == DIM
    assert em.miewid_cache_label("miewid-msv3") in res.cache_path
    rows = query_records(conn, dataset="LynxDS")
    assert all(r.embedding_path == res.cache_path for r in rows)
    raw, _ids = get_embedding_matrix(conn, dataset="LynxDS", normalize=False)
    assert raw.shape == (3, DIM)
    assert not np.allclose(np.linalg.norm(raw, axis=1), 1.0)         # stored RAW (D2)
    unit, _ = get_embedding_matrix(conn, dataset="LynxDS", normalize=True)
    assert np.allclose(np.linalg.norm(unit, axis=1), 1.0, atol=1e-5)


def test_switch_model_recomputes_all_rows(stub, seeded, tmp_path, monkeypatch):
    import global_embedding

    monkeypatch.setattr(
        global_embedding, "extract_global_embeddings",
        lambda paths, model_name=None, device=None, checkpoint_path=None:
        {k: np.ones(1536, dtype=np.float32) for k in paths})
    conn = seeded
    cache_dir = str(tmp_path / "emb")
    first = embed_records(conn, dataset="LynxDS", cache_dir=cache_dir)
    assert first.embedding_dim == 1536
    second = embed_records(conn, dataset="LynxDS", model_name="miewid-msv3",
                           cache_dir=cache_dir)
    assert second.n_stale == 3 and second.n_embedded == 3 and second.n_skipped == 0
    matrix, _ = get_embedding_matrix(conn, dataset="LynxDS")
    assert matrix.shape == (3, DIM)


def test_preprocessing_matches_benchmark(stub, seeded):
    from reid_demo.benchmark_miewid_adaptation import build_transform

    rec = query_records(seeded, dataset="LynxDS")[0]
    out = em.extract_miewid_embeddings({rec.record_id: rec.crop_path},
                                       device=torch.device("cpu"))
    assert set(out) == {rec.record_id} and out[rec.record_id].shape == (DIM,)
    batch = stub.batches[0]
    assert tuple(batch.shape) == (1, 3, 440, 440)
    with Image.open(rec.crop_path) as im:
        expected = build_transform({"image_size": 440})(im.convert("RGB"))
    assert torch.allclose(batch[0], expected, atol=1e-6)
    # solid red crop -> ImageNet-normalised channel constants
    assert torch.allclose(batch[0, 0].mean(), torch.tensor((200 / 255 - 0.485) / 0.229), atol=1e-3)


def test_unreadable_images_are_omitted(stub, tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    good = tmp_path / "good.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(good)
    out = em.extract_miewid_embeddings({"bad": str(bad), "good": str(good)},
                                       device=torch.device("cpu"))
    assert set(out) == {"good"}


def test_run_demo_passes_embedding_model(tmp_path):
    from reid_demo.run_demo import _config_to_argv, _config_warnings, build_config, plan_stages

    class _Args:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    cfg = build_config(_Args(datasets=["LynxDS"], db=str(tmp_path / "s.sqlite"),
                             embedding_model="miewid-msv3",
                             embedding_checkpoint="models/x/best.pt"))
    embed = next(s for s in plan_stages(cfg) if s.name == "embed")
    assert embed.cli[embed.cli.index("--model") + 1] == "miewid-msv3"
    assert embed.cli[embed.cli.index("--checkpoint") + 1] == "models/x/best.pt"
    argv = _config_to_argv(cfg)
    assert "--embedding-model" in argv and "--embedding-checkpoint" in argv
    assert len(_config_warnings(cfg)) == 1                      # default DBSCAN eps
    cfg.backend = "leiden"
    assert _config_warnings(cfg) == []

    plain = build_config(_Args(datasets=["LynxDS"], db=str(tmp_path / "s.sqlite")))
    plain_embed = next(s for s in plan_stages(plain) if s.name == "embed")
    assert "--model" not in plain_embed.cli and "--checkpoint" not in plain_embed.cli


def _pinned_weights_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        from reid_demo.benchmark_embeddings import MIEWID_MODEL_ID, MIEWID_REVISION
        hit = try_to_load_from_cache(MIEWID_MODEL_ID, "config.json", revision=MIEWID_REVISION)
        return isinstance(hit, str)
    except Exception:
        return False


@pytest.mark.skipif(os.environ.get("REID_DEMO_REAL_MIEWID") != "1" or not _pinned_weights_cached(),
                    reason="set REID_DEMO_REAL_MIEWID=1 with the pinned MiewID weights cached")
def test_real_miewid_dim(tmp_path):
    img = tmp_path / "crop.png"
    Image.new("RGB", (120, 80), (90, 110, 70)).save(img)
    out = em.extract_miewid_embeddings({"crop": str(img)})
    assert out["crop"].shape == (DIM,) and np.isfinite(out["crop"]).all()
