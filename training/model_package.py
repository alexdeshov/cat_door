import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

REQUIRED_METADATA = {"model_info.json", "thresholds.json", "checksums.sha256"}


def _unexpected_package_files(names: set[str]) -> list[str]:
    return sorted(
        name for name in names if name not in REQUIRED_METADATA and not name.endswith(".onnx")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_package(source_dir: Path, destination: Path) -> None:
    if source_dir.resolve() in destination.resolve().parents:
        raise ValueError("package destination must be outside source_dir")
    files = sorted(path for path in source_dir.iterdir() if path.is_file())
    names = {path.name for path in files}
    unexpected = _unexpected_package_files(names)
    if unexpected:
        raise ValueError(f"unexpected package files: {', '.join(unexpected)}")
    if not {"model_info.json", "thresholds.json"}.issubset(names):
        raise ValueError("model_info.json and thresholds.json are required")
    if not any(path.suffix == ".onnx" for path in files):
        raise ValueError("at least one ONNX model is required")

    checksums = [
        f"{sha256_file(path)}  {path.name}"
        for path in files
        if path.name != "checksums.sha256"
    ]
    checksum_path = source_dir / "checksums.sha256"
    checksum_path.write_text("\n".join(checksums) + "\n", encoding="ascii")
    files = sorted(path for path in source_dir.iterdir() if path.is_file())

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in files:
            archive.write(path, path.name)
    temporary.replace(destination)


def _parse_checksums(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or len(digest) != 64 or not name:
            raise ValueError("invalid checksums.sha256")
        result[name] = digest
    return result


def verify_package(package_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(package_path) as archive:
        names = archive.namelist()
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
                raise ValueError(f"unsafe package member: {name}")
        unexpected = _unexpected_package_files(set(names))
        if unexpected:
            raise ValueError(f"unexpected package files: {', '.join(unexpected)}")
        missing = REQUIRED_METADATA - set(names)
        if missing:
            raise ValueError(f"missing package files: {', '.join(sorted(missing))}")
        onnx_files = sorted(name for name in names if name.endswith(".onnx"))
        if not onnx_files:
            raise ValueError("package contains no ONNX models")
        checksums = _parse_checksums(archive.read("checksums.sha256").decode("ascii"))
        expected_checksum_names = set(names) - {"checksums.sha256"}
        if set(checksums) != expected_checksum_names:
            raise ValueError("checksums.sha256 must cover every package file exactly once")
        for name, expected in checksums.items():
            if name not in names:
                raise ValueError(f"checksummed file is missing: {name}")
            actual = hashlib.sha256(archive.read(name)).hexdigest()
            if actual != expected:
                raise ValueError(f"checksum mismatch: {name}")
        model_info = json.loads(archive.read("model_info.json"))
        thresholds = json.loads(archive.read("thresholds.json"))
    return {
        "valid": True,
        "onnx_files": onnx_files,
        "model_info": model_info,
        "thresholds": thresholds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or verify a CatDoor model package")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("source_dir", type=Path)
    build.add_argument("destination", type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("package", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        build_package(args.source_dir, args.destination)
        report = verify_package(args.destination)
    else:
        report = verify_package(args.package)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
