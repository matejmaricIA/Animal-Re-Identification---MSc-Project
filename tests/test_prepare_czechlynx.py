import json
import zipfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from pycocotools import mask as mask_utils

from utils.prepare_czechlynx import SPLITS, canonical_metadata, extract_archive, mask_bbox, prepare
from reid_demo.benchmark import file_digest, make_protocol
from reid_demo.benchmark_masks import MASKED_PREPROCESSING
from utils.prepare_czechlynx import decode_mask


def metadata_rows():
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[2:9, 3:11] = 1
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode()
    return pd.DataFrame([
        dict(source="camera", path=f"camera{n}/same.jpg", unique_name=f"lynx{n}",
             encounter=n, date="04-03-2020", mask=json.dumps(rle),
             **{split: "train" if n == 1 else "test" for split in SPLITS})
        for n in (1, 2)
    ])


def test_stable_ids_disambiguate_basenames_and_preserve_splits():
    frame = metadata_rows()
    first = canonical_metadata(frame)
    second = canonical_metadata(frame.iloc[::-1])
    assert first.image_id.tolist() == second.image_id.tolist()
    assert first.image_id.nunique() == 2
    assert first.timestamp.tolist() == ["2020-03-04"] * 2
    for split in SPLITS:
        assert set(first[split]) == {"train", "test"}
    assert set(first.orientation) == {"unknown"}


@pytest.mark.parametrize("change,match", [
    ({"path": "../escape.jpg"}, "Unsafe"),
    ({"date": "2020-04-03"}, "Unparseable"),
    ({"split-time_open": None}, "split"),
    ({"encounter": None}, "present"),
])
def test_invalid_metadata_fails(change, match):
    frame = metadata_rows()
    for column, value in change.items():
        frame.loc[0, column] = value
    with pytest.raises(ValueError, match=match):
        canonical_metadata(frame)


def test_duplicate_source_path_fails():
    frame = metadata_rows()
    frame.loc[1, "path"] = frame.loc[0, "path"]
    with pytest.raises(ValueError, match="Duplicate"):
        canonical_metadata(frame)


def test_mask_dimensions_and_exact_bbox():
    value = metadata_rows().iloc[0]["mask"]
    assert mask_bbox(value, 16, 12) == (3, 2, 11, 9)
    with pytest.raises(ValueError, match="dimensions"):
        mask_bbox(value, 12, 16)
    empty = mask_utils.encode(np.asfortranarray(np.zeros((12, 16), np.uint8)))
    with pytest.raises(ValueError, match="empty"):
        mask_bbox(empty, 16, 12)


def test_prepare_frame_and_annotated_crop_with_common_exclusions(tmp_path):
    source = tmp_path / "source"
    frame = metadata_rows()
    for path in frame.path:
        target = source / path
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 12), (100, 60, 20)).save(target)
    csv = tmp_path / "source.csv"
    frame.to_csv(csv, index=False)
    output = tmp_path / "prepared"
    audit = prepare(csv, source, output)
    assert audit["retained_images"] == 2
    a = pd.read_csv(output / "frame/metadata.csv")
    b = pd.read_csv(output / "bbox/metadata.csv")
    assert a.image_id.tolist() == b.image_id.tolist()
    assert a.source_sha256.tolist() == b.source_sha256.tolist()
    assert a.image_sha256.tolist() != b.image_sha256.tolist()
    with Image.open(b.path.iloc[0]) as image:
        assert image.size == (8, 7)
    assert json.loads((output / "bbox/manifest.json").read_text())["annotation_assisted"]
    # A missing image is never silently dropped.
    (source / frame.path.iloc[0]).unlink()
    with pytest.raises(ValueError, match="invalid"):
        prepare(csv, source, output)
    audit = prepare(csv, source, output, on_error="exclude")
    assert audit["excluded_images"] == 1
    assert len(pd.read_csv(output / "frame/metadata.csv")) == 1
    assert len(pd.read_csv(output / "bbox/metadata.csv")) == 1


def test_archive_path_traversal_rejected(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_archive(archive, tmp_path)
    assert not (tmp_path.parent / "escape.txt").exists()


def test_masked_preparation_preserves_foreground_alignment_and_partitions(tmp_path):
    source = tmp_path / "source"
    base = metadata_rows().iloc[0].to_dict()
    frame = pd.DataFrame([
        {**base, "path": f"camera{identity}/{event}.jpg", "unique_name": f"lynx{identity}",
         "encounter": identity * 4 + event, "date": f"04-0{event + 1}-2020",
         **{split: "train" if event < 3 else "test" for split in SPLITS}}
        for identity in range(6) for event in range(4)])
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[2:9, 3:11] = 1
    mask[4:6, 6:8] = 0  # A hole inside the bbox must also become background.
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode()
    frame["mask"] = json.dumps(rle)
    for n, path in enumerate(frame.path):
        target = source / path
        target.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.random.default_rng(n).integers(0, 256, (12, 16, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(target)
    metadata = tmp_path / "source.csv"
    frame.to_csv(metadata, index=False)
    output = tmp_path / "prepared"
    prepare(metadata, source, output, modes=("bbox", "masked"))
    bbox = pd.read_csv(output / "bbox/metadata.csv")
    masked = pd.read_csv(output / "masked/metadata.csv")
    assert masked.image_id.tolist() == bbox.image_id.tolist()
    assert masked.bbox.tolist() == bbox.bbox.tolist()
    assert masked.source_sha256.tolist() == bbox.source_sha256.tolist()
    for stage in ("development", "test"):
        b_gallery, b_query, _ = make_protocol(bbox, split="time_open", stage=stage)
        m_gallery, m_query, _ = make_protocol(masked, split="time_open", stage=stage)
        assert b_gallery.image_id.tolist() == m_gallery.image_id.tolist()
        assert b_query.image_id.tolist() == m_query.image_id.tolist()
    for row in masked.to_dict("records"):
        with Image.open(source / row["source_path"]) as im:
            expected = np.asarray(im.convert("RGB").crop((3, 2, 11, 9)))
        with Image.open(row["path"]) as im:
            assert im.format == "PNG"
            actual = np.asarray(im)
        with Image.open(row["mask_path"]) as im:
            actual_mask = np.asarray(im)
        foreground = mask[2:9, 3:11] != 0
        np.testing.assert_array_equal(actual_mask, foreground.astype(np.uint8) * 255)
        np.testing.assert_array_equal(actual[foreground], expected[foreground])
        np.testing.assert_array_equal(actual[~foreground], np.full((4, 3), 128, dtype=np.uint8))
        assert file_digest(row["mask_path"]) == row["mask_sha256"]
    manifest = json.loads((output / "masked/manifest.json").read_text())
    assert manifest["annotation_assisted"]
    assert manifest["masked_preprocessing"] == MASKED_PREPROCESSING
    before = masked.image_sha256.tolist(), masked.mask_sha256.tolist()
    prepare(metadata, source, output, modes=("masked",))
    again = pd.read_csv(output / "masked/metadata.csv")
    assert before == (again.image_sha256.tolist(), again.mask_sha256.tolist())
    # Reusing a corrupt mask must fail, not bless the changed file with a new hash.
    from pathlib import Path
    Path(again.mask_path.iloc[0]).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="invalid images"):
        prepare(metadata, source, output, modes=("masked",))


def test_decode_mask_accepts_compressed_and_uncompressed_rle_and_rejects_empty():
    expected = np.array([[0, 1, 1], [0, 1, 0]], dtype=np.uint8)
    compressed = mask_utils.encode(np.asfortranarray(expected))
    np.testing.assert_array_equal(decode_mask(compressed, 3, 2), expected)
    uncompressed = {"size": [2, 3], "counts": [2, 3, 1]}
    np.testing.assert_array_equal(decode_mask(uncompressed, 3, 2), expected)
    with pytest.raises(ValueError, match="empty"):
        decode_mask({"size": [2, 3], "counts": [6]}, 3, 2)
    with pytest.raises(ValueError, match="run lengths"):
        decode_mask({"size": [2, 3], "counts": [7]}, 3, 2)
