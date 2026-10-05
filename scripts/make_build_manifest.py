"""Generate a manifest of the image's build artifacts.

Run during the image build; ``scripts/verify.py`` re-hashes the same files at
verify time and compares them, so the one-shot verify job also proves that the
image contains the exact code artifacts that were tested.
"""

import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDED = ["app", "scripts", "tests"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    files = {}
    for directory in INCLUDED:
        base = os.path.join(ROOT, directory)
        for dirpath, _, names in os.walk(base):
            if "__pycache__" in dirpath:
                continue
            for name in sorted(names):
                if name.endswith((".pyc", ".pyo")):
                    continue
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, ROOT)
                files[rel] = sha256_file(full)
    out_dir = os.path.join(ROOT, "build-artifacts")
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "component": "delay-plan-compiler",
        "version": os.environ.get("BUILD_VERSION", "1.0.0"),
        "file_count": len(files),
        "files": files,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"manifest written: {len(files)} files")


if __name__ == "__main__":
    sys.exit(main())
