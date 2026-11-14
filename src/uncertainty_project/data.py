from __future__ import annotations

import unicodedata
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .common import LABEL_CODES, config, digest, read_json, read_jsonl, write_json, write_jsonl


def normalized_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _fetch_source(root, name, spec):
    """Download one pinned file if missing and verify bytes and row count."""
    folder = Path(root) / "data/raw"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / spec["filename"]
    if not path.exists():
        req = Request(spec["url"], headers={"User-Agent": "HSG-Project14-research/0.1"})
        with urlopen(req, timeout=60) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024 or digest(raw) != spec["sha256"]:
            raise ValueError(f"Unexpected download for {name}; inspect the source before continuing.")
        path.write_bytes(raw)
    if digest(path.read_bytes()) != spec["sha256"]:
        raise ValueError(f"Checksum mismatch: {path}. Keep this file and inspect its provenance.")
    if len(read_jsonl(path)) != spec["expected_rows"]:
        raise ValueError(f"Unexpected number of rows in {name}.")
    return path


def download_sources(root):
    """Use pinned URLs and verify bytes. Existing mismatching files are never overwritten."""
    root = Path(root)
    sources = read_json(root / "config/data_sources.json")
    for name, spec in sources.items():
        _fetch_source(root, name, spec)
    return sources


def _audit_chaos_row(row, uid_prefix=""):
    """Checks shared by every ChaosNLI file: 100 votes, consistent counts, distribution and majority."""
    uid = uid_prefix + str(row["uid"])
    counts = np.asarray(row["label_count"], dtype=float)
    probs = np.asarray(row["label_dist"], dtype=float)
    if counts.shape != (3,) or np.any(counts < 0) or not np.all(counts == np.floor(counts)) or counts.sum() != 100:
        raise ValueError(f"Invalid vote counts: {uid}")
    if not np.allclose(probs, counts / 100, atol=1e-9):
        raise ValueError(f"Distribution and counts disagree: {uid}")
    if [row["label_counter"].get(k, 0) for k in LABEL_CODES] != counts.tolist():
        raise ValueError(f"Label order inconsistent: {uid}")
    chosen = LABEL_CODES.index(row["majority_label"])
    if counts[chosen] != counts.max():
        raise ValueError(f"Invalid majority label: {uid}")
    premise, hypothesis = row["example"]["premise"], row["example"]["hypothesis"]
    if not isinstance(premise, str) or not premise.strip() or not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ValueError(f"Missing text: {uid}")
    return {"uid": uid, "premise": premise, "hypothesis": hypothesis,
            "group_id": digest(normalized_text(premise)),
            "human_distribution": probs.tolist(), "human_counts": counts.astype(int).tolist(),
            "taxonomy_categories": None, "taxonomy_round": None}


def load_and_audit(root):
    root = Path(root)
    sources = download_sources(root)
    raw = read_jsonl(root / "data/raw" / sources["chaos_mnli"]["filename"])
    records = []
    seen = set()
    for row in raw:
        record = _audit_chaos_row(row)
        if record["uid"] in seen:
            raise ValueError(f"Duplicate example ID: {record['uid']}")
        seen.add(record["uid"])
        records.append(record)
    frame = pd.DataFrame(records).set_index("uid", drop=False)
    taxonomy_ids = set()
    unmatched = 0
    quote_format_differences = []
    for round_number in (1, 2):
        spec = sources[f"taxonomy_round{round_number}"]
        for row in read_jsonl(root / "data/raw" / spec["filename"]):
            uid = str(row["pairID"])
            if uid in taxonomy_ids:
                raise ValueError(f"Duplicate taxonomy example: {uid}")
            taxonomy_ids.add(uid)
            if uid not in frame.index:
                unmatched += 1
                continue
            for field in ("premise", "hypothesis"):
                left, right = normalized_text(row[field]), normalized_text(frame.at[uid, field])
                if left == right:
                    continue
                # The released taxonomy drops a leading ASCII quotation mark in 17 text fields.
                # Accept only that inspected formatting difference; preserve the original NLI text.
                if right.startswith("'") and left == right[1:]:
                    quote_format_differences.append({"uid": uid, "field": field})
                else:
                    raise ValueError(f"Taxonomy ID/text mismatch for {uid}, {field}")
            frame.at[uid, "taxonomy_categories"] = row[spec["label_field"]]
            frame.at[uid, "taxonomy_round"] = round_number
    frame = frame.reset_index(drop=True).sort_values("uid").reset_index(drop=True)
    # Pandas may coerce round numbers to float. Preserve JSON null, not NaN.
    frame["taxonomy_round"] = frame["taxonomy_round"].astype(object).where(frame["taxonomy_round"].notna(), None)
    report = {"examples": len(frame), "premise_groups": int(frame.group_id.nunique()),
              "votes_per_example": 100, "taxonomy_matches": int(frame.taxonomy_categories.notna().sum()),
              "taxonomy_outside_chaos_mnli": unmatched,
              "taxonomy_leading_quote_differences": quote_format_differences,
              "individual_chaos_annotator_ids_available": False,
              "label_order": list(LABEL_CODES), "data_digest": sources["chaos_mnli"]["sha256"]}
    if report["taxonomy_matches"] != 450:
        raise ValueError("The expected 450 matching taxonomy annotations were not recovered.")
    return frame, report


def make_splits(frame, seed=2026, train=0.6, dev=0.2, test=0.2):
    if not np.isclose(train + dev + test, 1) or min(train, dev, test) <= 0:
        raise ValueError("Split fractions must be positive and sum to one.")
    groups = np.array(sorted(frame.group_id.unique()))
    np.random.default_rng(seed).shuffle(groups)
    n_test, n_dev = round(len(groups) * test), round(len(groups) * dev)
    mapping = {g: "test" for g in groups[:n_test]}
    mapping.update({g: "dev" for g in groups[n_test:n_test + n_dev]})
    mapping.update({g: "train" for g in groups[n_test + n_dev:]})
    result = frame[["uid", "group_id"]].copy()
    result["split"] = result.group_id.map(mapping)
    if result.groupby("group_id").split.nunique().max() != 1:
        raise AssertionError("A premise appears in multiple splits.")
    return result.sort_values("uid").reset_index(drop=True)


def prepare_data(root):
    root = Path(root)
    cfg = config(root)
    frame, audit = load_and_audit(root)
    split = make_splits(frame, seed=cfg["seed"], **cfg["split"])
    spec = {"seed": cfg["seed"], "fractions": cfg["split"], "data_digest": audit["data_digest"],
            "assignment": split.to_dict("records")}
    path = root / "data/processed/splits.json"
    if path.exists() and read_json(path) != spec:
        raise ValueError("Existing split differs. Use a separate project copy for a different experimental split.")
    write_json(path, spec)
    full = frame.merge(split[["uid", "split"]], on="uid", validate="one_to_one")
    for name in ("train", "dev", "test"):
        rows = full.loc[full.split == name].drop(columns="split").to_dict("records")
        if name == "test":
            keys = ("uid", "premise", "hypothesis", "group_id")
            inputs = [{k: r[k] for k in keys} for r in rows]
            labels = [{k: v for k, v in r.items() if k not in ("premise", "hypothesis", "group_id")} for r in rows]
            write_jsonl(root / "data/processed/test_inputs.jsonl", inputs)
            write_jsonl(root / "data/processed/test_labels.jsonl", labels)
        else:
            write_jsonl(root / f"data/processed/{name}.jsonl", rows)
    audit["split_examples"] = {str(k): int(v) for k, v in full.split.value_counts().items()}
    write_json(root / "outputs/data_audit.json", audit)
    return audit


SPLIT_FILES = {"train": ("train.jsonl", None), "dev": ("dev.jsonl", None),
               "test": ("test_inputs.jsonl", "test_labels.jsonl"), "shift": ("shift_inputs.jsonl", "shift_labels.jsonl")}


def load_split(root, name, *, include_test_labels=False):
    """train/dev contain labels; test and shift (unfamiliar inputs) keep labels in a separate file."""
    root = Path(root)
    if name not in SPLIT_FILES:
        raise ValueError("Choose train, dev, test or shift.")
    inputs_file, labels_file = SPLIT_FILES[name]
    path = root / "data/processed" / inputs_file
    if not path.exists():
        raise FileNotFoundError("Run notebook 00 first." if name != "shift" else
                                "Prepare the unfamiliar-input sample in notebook 02 (prepare_shift) first.")
    frame = pd.DataFrame(read_jsonl(path))
    if labels_file and include_test_labels:
        frame = frame.merge(pd.DataFrame(read_jsonl(root / "data/processed" / labels_file)), on="uid", validate="one_to_one")
    return frame


def prepare_shift(root):
    """Unfamiliar inputs: a fixed random sample from a DIFFERENT ChaosNLI source (SNLI image captions).
    Prepared before freezing, so the sample is sealed with the method. Never used for fitting or tuning.
    IDs get a prefix so they can never collide with MNLI IDs in the response cache."""
    root = Path(root)
    cfg = config(root)
    shift = cfg.get("shift")
    if not shift:
        raise ValueError("No 'shift' section in config/project.json.")
    spec = read_json(root / "config/data_sources.json")[shift["source"]]
    path = _fetch_source(root, shift["source"], spec)
    prefix = shift["source"].replace("chaos_", "") + "_"
    rows = [_audit_chaos_row(r, uid_prefix=prefix) for r in read_jsonl(path)]
    if len({r["uid"] for r in rows}) != len(rows):
        raise ValueError("Duplicate IDs in the unfamiliar-input source.")
    # Exclude any pair whose normalised premise also occurs in the MNLI data (train/dev/test), before sampling.
    mnli_groups = {r["group_id"] for r in read_json(root / "data/processed/splits.json")["assignment"]}
    overlap = sum(r["group_id"] in mnli_groups for r in rows)
    rows = sorted((r for r in rows if r["group_id"] not in mnli_groups), key=lambda r: r["uid"])
    chosen = np.sort(np.random.default_rng(shift["seed"]).choice(len(rows), size=min(shift["size"], len(rows)), replace=False))
    sample = [rows[i] for i in chosen]
    keys = ("uid", "premise", "hypothesis", "group_id")
    inputs = [{k: r[k] for k in keys} for r in sample]
    labels = [{k: v for k, v in r.items() if k not in ("premise", "hypothesis", "group_id")} for r in sample]
    in_path, lab_path = root / "data/processed/shift_inputs.jsonl", root / "data/processed/shift_labels.jsonl"
    if in_path.exists() and read_jsonl(in_path) != inputs:
        raise ValueError("An existing unfamiliar-input sample differs. Do not replace it after freezing.")
    write_jsonl(in_path, inputs)
    write_jsonl(lab_path, labels)
    return {"source": shift["source"], "source_rows": len(rows) + overlap, "excluded_premise_overlap_with_mnli": overlap,
            "sample": len(sample),
            "premise_groups": len({r["group_id"] for r in sample}), "data_digest": spec["sha256"]}


def pilot_rows(root):
    cfg = config(root)
    frame = load_split(root, "train")
    return frame.sample(n=min(cfg["pilot_size"], len(frame)), random_state=cfg["seed"]).sort_values("uid").reset_index(drop=True)


def smoke_rows(root):
    """First few pilot pairs: a cheap technical check of formats, keys and storage."""
    return pilot_rows(root).head(config(root).get("smoke_size", 5)).reset_index(drop=True)
