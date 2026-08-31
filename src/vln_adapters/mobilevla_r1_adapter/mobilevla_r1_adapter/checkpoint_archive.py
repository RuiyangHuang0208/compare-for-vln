from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import zipfile


class CheckpointArchiveError(ValueError):
    pass


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    parts = manifest.get("parts")
    if not isinstance(parts, list) or not parts:
        raise CheckpointArchiveError("manifest has no archive parts")
    names = [part.get("name") for part in parts]
    if names != sorted(names) or len(names) != len(set(names)):
        raise CheckpointArchiveError("manifest part names must be unique and sorted")
    return manifest


def verify_parts(archive_dir: Path, manifest: dict) -> list[dict]:
    expected = {part["name"]: part for part in manifest["parts"]}
    actual = {path.name: path for path in archive_dir.glob("weight.zip.part-*") if path.is_file()}
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        raise CheckpointArchiveError(f"archive part mismatch: missing={missing}, unexpected={unexpected}")
    results = []
    for name, specification in expected.items():
        path = actual[name]
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != int(specification["size"]):
            raise CheckpointArchiveError(f"size mismatch for {name}: {size} != {specification['size']}")
        if digest != specification["sha256"]:
            raise CheckpointArchiveError(f"SHA256 mismatch for {name}: {digest}")
        results.append({"name": name, "size": size, "sha256": digest})
    if sum(item["size"] for item in results) != int(manifest["total_size"]):
        raise CheckpointArchiveError("verified part sizes do not match manifest total_size")
    return results


def merge_parts(archive_dir: Path, manifest: dict, destination: Path) -> str:
    if destination.exists():
        raise CheckpointArchiveError(f"refusing to overwrite merged archive: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        raise CheckpointArchiveError(f"remove stale partial archive first: {temporary}")
    digest = hashlib.sha256()
    try:
        with temporary.open("xb") as output:
            for part in manifest["parts"]:
                with (archive_dir / part["name"]).open("rb") as source:
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        output.write(chunk)
                        digest.update(chunk)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return digest.hexdigest()


def safe_member_path(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts or ".." in path.parts or ":" in path.parts[0]:
        raise CheckpointArchiveError(f"unsafe archive path: {name!r}")
    return path


def inspect_zip(path: Path) -> list[dict]:
    entries = []
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise CheckpointArchiveError(f"ZIP CRC failed for {bad!r}")
        for member in archive.infolist():
            safe_member_path(member.filename)
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise CheckpointArchiveError(f"symbolic link is not allowed: {member.filename!r}")
            entries.append(
                {
                    "path": member.filename,
                    "size": member.file_size,
                    "compressed_size": member.compress_size,
                    "directory": member.is_dir(),
                }
            )
    return entries


def extract_zip(path: Path, destination: Path) -> list[dict]:
    if destination.exists() and any(destination.iterdir()):
        raise CheckpointArchiveError(f"refusing to overwrite non-empty checkpoint directory: {destination}")
    destination_parent = destination.parent
    destination_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination_parent))
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                relative = safe_member_path(member.filename)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise CheckpointArchiveError(f"symbolic link is not allowed: {member.filename!r}")
                target = staging.joinpath(*relative.parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        files = sorted(path for path in staging.rglob("*") if path.is_file())
        inventory = [
            {
                "path": str(item.relative_to(staging)),
                "size": item.stat().st_size,
                "sha256": sha256_file(item),
            }
            for item in files
        ]
        (staging / "CHECKPOINT_INVENTORY.json").write_text(
            json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if destination.exists():
            destination.rmdir()
        staging.replace(destination)
        return inventory
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args():
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Verify and safely prepare MobileVLA-R1 checkpoint archives")
    parser.add_argument("--manifest", type=Path, default=package_root / "config" / "official_archive_manifest.json")
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--merged", type=Path)
    parser.add_argument("--extract-to", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        manifest = load_manifest(args.manifest.resolve())
        verified = verify_parts(args.archives.resolve(), manifest)
        report = {"verified_parts": verified, "total_size": sum(item["size"] for item in verified)}
        if args.merged is not None:
            merged = args.merged.resolve()
            report["merged_archive"] = str(merged)
            report["merged_sha256"] = merge_parts(args.archives.resolve(), manifest, merged)
            report["archive_entries"] = inspect_zip(merged)
            if args.extract_to is not None:
                report["checkpoint_files"] = extract_zip(merged, args.extract_to.resolve())
        elif args.extract_to is not None:
            raise CheckpointArchiveError("--extract-to requires --merged")
        print(json.dumps(report, indent=2, sort_keys=True))
    except (CheckpointArchiveError, FileNotFoundError, json.JSONDecodeError) as error:
        print(f"MobileVLA-R1 checkpoint preparation failed: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
