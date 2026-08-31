import hashlib
import json
from pathlib import Path
import stat
import zipfile

import pytest

from mobilevla_r1_adapter.checkpoint_archive import (
    CheckpointArchiveError,
    extract_zip,
    inspect_zip,
    load_manifest,
    merge_parts,
    verify_parts,
)


def make_split_archive(tmp_path: Path, member_name="model/config.json", member_data=b"{}"):
    complete = tmp_path / "complete.zip"
    with zipfile.ZipFile(complete, "w") as archive:
        archive.writestr(member_name, member_data)
    payload = complete.read_bytes()
    boundary = max(1, len(payload) // 2)
    parts = []
    for suffix, data in (("aa", payload[:boundary]), ("ab", payload[boundary:])):
        name = f"weight.zip.part-{suffix}"
        (tmp_path / name).write_bytes(data)
        parts.append({"name": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"total_size": len(payload), "parts": parts}), encoding="utf-8")
    complete.unlink()
    return manifest_path


def test_verify_merge_inspect_and_extract(tmp_path):
    manifest = load_manifest(make_split_archive(tmp_path))
    assert len(verify_parts(tmp_path, manifest)) == 2
    merged = tmp_path / "weight.zip"
    digest = merge_parts(tmp_path, manifest, merged)
    assert digest == hashlib.sha256(merged.read_bytes()).hexdigest()
    assert inspect_zip(merged)[0]["path"] == "model/config.json"
    destination = tmp_path / "checkpoint"
    inventory = extract_zip(merged, destination)
    assert inventory[0]["path"] == "model/config.json"
    assert (destination / "CHECKPOINT_INVENTORY.json").is_file()


def test_missing_or_corrupt_part_is_rejected(tmp_path):
    manifest = load_manifest(make_split_archive(tmp_path))
    (tmp_path / "weight.zip.part-ab").unlink()
    with pytest.raises(CheckpointArchiveError, match="missing"):
        verify_parts(tmp_path, manifest)
    (tmp_path / "weight.zip.part-ab").write_bytes(b"wrong")
    with pytest.raises(CheckpointArchiveError, match="size mismatch"):
        verify_parts(tmp_path, manifest)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/windows", "..\\escape"])
def test_unsafe_archive_path_is_rejected(tmp_path, name):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(name, b"bad")
    with pytest.raises(CheckpointArchiveError, match="unsafe archive path"):
        inspect_zip(archive_path)


def test_symbolic_link_is_rejected(tmp_path):
    archive_path = tmp_path / "link.zip"
    info = zipfile.ZipInfo("model/link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(info, "target")
    with pytest.raises(CheckpointArchiveError, match="Symbolic|symbolic"):
        inspect_zip(archive_path)


def test_existing_outputs_are_not_overwritten(tmp_path):
    manifest = load_manifest(make_split_archive(tmp_path))
    merged = tmp_path / "weight.zip"
    merged.write_bytes(b"existing")
    with pytest.raises(CheckpointArchiveError, match="refusing to overwrite"):
        merge_parts(tmp_path, manifest, merged)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "user-file").write_text("keep", encoding="utf-8")
    with pytest.raises(CheckpointArchiveError, match="refusing to overwrite"):
        extract_zip(merged, checkpoint)
