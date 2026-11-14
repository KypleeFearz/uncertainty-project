from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

LABELS = ("entailment", "neutral", "contradiction")
LABEL_CODES = ("e", "n", "c")


def find_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for p in (start, *start.parents):
        if (p / "config/project.json").exists() and (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError("Open the notebook inside the extracted uncertainty_project folder.")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def digest(value):
    raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def config(root):
    return read_json(Path(root) / "config/project.json")
