"""Check that code, configuration, data, model responses and fitted assessors are unchanged since the freeze.

    python tools/verify_freeze.py

Works on Windows, macOS and Linux. The freeze record stores relative paths in the notation of the
computer that created it (backslashes on Windows); this script compares paths in a normalised form.
It only reads files. It lives outside src/ so that adding it does not alter the sealed fingerprint.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uncertainty_project.common import digest, read_json  # noqa: E402
from uncertainty_project.experiment import _fingerprint, experiment_directory  # noqa: E402


def check(root=ROOT):
    record = read_json(experiment_directory(root) / "frozen_protocol.json")
    if digest(record["files"]) != record["protocol_id"]:
        raise RuntimeError("The freeze record does not match its protocol identifier.")
    frozen = {k.replace("\\", "/"): v for k, v in record["files"].items()}
    current = {k.replace("\\", "/"): v for k, v in _fingerprint(root).items()}
    changed = sorted(k for k in frozen.keys() & current.keys() if frozen[k] != current[k])
    missing = sorted(frozen.keys() - current.keys())
    added = sorted(current.keys() - frozen.keys())
    return record, changed, missing, added


def main():
    record, changed, missing, added = check()
    print(f"Protocol {record['protocol_id'][:16]}, frozen at {record['frozen_at']}, {len(record['files'])} files sealed.")
    if not (changed or missing or added):
        print("OK: every sealed file is byte-identical to the frozen state.")
        return 0
    for title, items in (("Changed", changed), ("Missing", missing), ("New", added)):
        if items:
            print(f"{title} ({len(items)}): " + ", ".join(items[:10]) + (" ..." if len(items) > 10 else ""))
    return 1


if __name__ == "__main__":
    sys.exit(main())
