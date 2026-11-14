"""
Apertus als drittes Modell (nachtraegliche Erweiterung nach dem Freeze).

Schickt exakt dieselben 5 Prompts wie bei GPT und Claude an
swiss-ai/apertus-v1.5-70b (Public AI) und speichert jede Antwort einzeln.
Der eingefrorene Code (src/, config/, data/processed/) wird nur GELESEN,
nie veraendert. Alles Neue landet in extension_apertus/outputs/.

Voraussetzung: In der Datei .env im Projektordner steht
    PUBLICAI_API_KEY=dein_key

Befehle (im Projektordner ausfuehren):
    python extension_apertus/run_apertus.py --check   # 1 Anfrage: funktioniert der Key?
    python extension_apertus/run_apertus.py --pilot   # 20 Anfragen auf Trainingsfaellen
    python extension_apertus/run_apertus.py           # richtiger Lauf: MNLI-Test + SNLI

Der richtige Lauf kann jederzeit abgebrochen (Ctrl+C) und neu gestartet werden,
fertige Antworten werden uebersprungen. Kostenbremse: --max-usd (Standard 6).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MODEL = "swiss-ai/apertus-v1.5-70b"
BASE_URL = os.environ.get("APERTUS_BASE_URL", "https://api.publicai.co/v1").rstrip("/")
PRICE_IN = 0.82   # USD pro 1 Mio. Input-Tokens (Public AI, Stand 7.11.2026)
PRICE_OUT = 2.92  # USD pro 1 Mio. Output-Tokens
TEMPERATURE = 0.0
MAX_TOKENS = 350
LABELS = ("entailment", "neutral", "contradiction")
CUES = ("lexical_ambiguity", "unclear_reference", "probabilistic_inference")

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs_v2"   # v2: Format per Text-Anweisung (Public AI blockiert json_schema seit 7.11.)

# Bei GPT/Claude kamen die Feldnamen ueber das Antwort-Schema der API (Structured Outputs).
# Public AI akzeptiert kein Schema mehr, deshalb dieselbe Information als Text am Ende
# der System-Nachricht. Der eigentliche Prompt bleibt unveraendert.
FORMAT_INSTRUCTION = (
    "\nRespond with a single JSON object and nothing else, using exactly these keys:\n"
    '{"entailment": <number>, "neutral": <number>, "contradiction": <number>, '
    '"lexical_ambiguity": <number>, "unclear_reference": <number>, '
    '"probabilistic_inference": <number>, "brief_reason": "<at most two sentences>"}'
)


# --------------------------------------------------------------------------- Projekt finden

def find_root():
    for p in (Path.cwd().resolve(), *Path.cwd().resolve().parents, HERE.parent, *HERE.parents):
        if (p / "config" / "project.json").exists() and (p / "src").exists():
            return p
    sys.exit("FEHLER: Projektordner nicht gefunden. Bitte im Ordner uncertainty_project ausfuehren.")


def load_key(root):
    key = os.environ.get("PUBLICAI_API_KEY", "").strip()
    if key:
        return key
    env = root / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("PUBLICAI_API_KEY") and "=" in line:
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    return value
    sys.exit("FEHLER: Kein PUBLICAI_API_KEY gefunden.\n"
             f"Oeffne {env} und fuege diese Zeile hinzu:\nPUBLICAI_API_KEY=dein_key")


# --------------------------------------------------------------------------- Prompts (aus deinem Code)

def load_prompts(root):
    """Prompts direkt aus dem eingefrorenen Code holen, damit sie identisch sind."""
    sys.path.insert(0, str(root / "src"))
    try:
        from uncertainty_project import api as project_api
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"FEHLER: Konnte uncertainty_project.api nicht laden ({type(exc).__name__}: {exc}).\n"
                 "Bitte die virtuelle Umgebung aktivieren (.venv) und nochmal starten.")
    def build(row, variant):
        messages = project_api.build_messages(row, variant)
        messages[0] = {"role": "system", "content": messages[0]["content"] + FORMAT_INSTRUCTION}
        return messages
    variants = len(project_api.PROMPT_VARIANTS)
    schema = None
    if hasattr(project_api, "NLIResult"):
        schema = project_api.NLIResult.model_json_schema()
    fingerprint = hashlib.sha256(json.dumps(
        {"system": project_api.SYSTEM_PROMPT, "variants": list(project_api.PROMPT_VARIANTS)},
        sort_keys=True).encode()).hexdigest()[:16]
    return build, variants, schema, fingerprint


# --------------------------------------------------------------------------- Daten

def read_rows(path):
    path = Path(path)
    rows = []
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    elif path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("rows", data.get("examples", []))
    elif path.suffix == ".csv":
        import csv
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if not isinstance(r, dict):
            return []
        ex = r.get("example") if isinstance(r.get("example"), dict) else {}
        uid = r.get("uid", r.get("pairID", r.get("id")))
        premise = r.get("premise", ex.get("premise"))
        hypothesis = r.get("hypothesis", ex.get("hypothesis"))
        if uid is None or not premise or not hypothesis:
            return []
        out.append({"uid": str(uid), "premise": str(premise), "hypothesis": str(hypothesis)})
    return out


def find_dataset(root, kind):
    processed = root / "data" / "processed"
    if kind == "mnli_test":
        direct = processed / "test_inputs.jsonl"
        if direct.exists():
            return direct
        words = ("test",)
    elif kind == "train":
        direct = processed / "train.jsonl"
        if direct.exists():
            return direct
        words = ("train",)
    else:
        words = ("snli", "shift")
    candidates = []
    for p in sorted((root / "data").rglob("*")):
        name = str(p.relative_to(root)).lower()
        if p.suffix in (".jsonl", ".json", ".csv") and any(w in name for w in words) and "label" not in p.name.lower():
            if kind == "mnli_test" and ("snli" in name or "shift" in name):
                continue
            if read_rows(p):
                candidates.append(p)
    return candidates[0] if candidates else None


# --------------------------------------------------------------------------- API

def post(key, payload):
    req = urllib.request.Request(
        BASE_URL + "/chat/completions", data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                 "User-Agent": "hsg-masterproject-apertus/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


HINTS = {401: "Key falsch. Zeile PUBLICAI_API_KEY in .env pruefen.",
         402: "Guthaben leer. Im Public-AI-Konto aufladen, dann einfach neu starten.",
         403: "Kein Zugriff (Guthaben oder Konto-Bestaetigung?).",
         404: "Modell nicht gefunden.",
         429: "Rate-Limit."}


class SkipItem(Exception):
    """Server lehnt genau diese Anfrage wiederholt ab: Fall ueberspringen, Lauf geht weiter."""


class Client:
    """Merkt sich, welche Optionen der Server akzeptiert (Thinking aus, JSON-Schema).
    Nach der ersten erfolgreichen Antwort sind die Einstellungen fixiert, damit sich
    mitten im Lauf nichts aendert."""

    def __init__(self, key, schema):
        self.key = key
        self.schema = schema
        self.locked = False
        self.thinking_off = True
        self.format_mode = "none"

    def _payload(self, messages, max_tokens):
        p = {"model": MODEL, "messages": messages, "temperature": TEMPERATURE, "max_tokens": max_tokens}
        if self.thinking_off:
            p["chat_template_kwargs"] = {"enable_thinking": False}
        if self.format_mode == "json_schema":
            p["response_format"] = {"type": "json_schema",
                                    "json_schema": {"name": "nli_result", "schema": self.schema}}
        elif self.format_mode == "json_object":
            p["response_format"] = {"type": "json_object"}
        return p

    def ask(self, messages, max_tokens=MAX_TOKENS):
        waits = [5, 15, 30, 60, 120]
        attempt = 0
        while True:
            try:
                start = time.time()
                data = post(self.key, self._payload(messages, max_tokens))
                self.locked = True
                return data, time.time() - start
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")[:400]
                if e.code == 400 and self.locked:
                    # Einstellungen haben schon funktioniert: Serverproblem, nicht Option.
                    if attempt < 3:
                        time.sleep(waits[attempt])
                        attempt += 1
                        continue
                    raise SkipItem(body)
                if e.code == 400:
                    # Nur ganz am Anfang: Option vom Server nicht unterstuetzt, schrittweise weglassen
                    if self.format_mode == "json_schema":
                        self.format_mode = "json_object"
                        continue
                    if self.thinking_off:
                        self.thinking_off = False
                        continue
                    if self.format_mode == "json_object":
                        self.format_mode = "none"
                        continue
                if e.code in (429, 500, 502, 503, 504) and attempt < len(waits):
                    time.sleep(waits[attempt])
                    attempt += 1
                    continue
                raise SystemExit(f"\nFEHLER HTTP {e.code}: {HINTS.get(e.code, 'siehe Serverantwort')}\n"
                                 f"Serverantwort: {body}\n"
                                 "Fertige Antworten sind gespeichert. Einfach neu starten, wenn behoben.")
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < len(waits):
                    time.sleep(waits[attempt])
                    attempt += 1
                    continue
                raise SystemExit(f"\nFEHLER: Keine Verbindung ({e}). Internet pruefen und neu starten.")


NUM = re.compile(r'"(entailment|neutral|contradiction|lexical_ambiguity|unclear_reference|probabilistic_inference)"'
                 r'\s*:\s*"?([0-9]*\.?[0-9]+(?:[eE]-?[0-9]+)?)')
REASON = re.compile(r'"brief_reason"\s*:\s*"((?:[^"\\]|\\.)*)')


def parse(content):
    """Werte aus der Antwort holen und pruefen.
    Gibt (parsed, info) zurueck, info = {"renormalized", "salvaged"}; bei Fehler (None, Grund).
    salvaged: Antwort wurde mitten in brief_reason abgeschnitten, die 6 Zahlen sind aber
    vollstaendig da. brief_reason wird in der Auswertung nicht verwendet."""
    text = (content or "").strip()
    obj, salvaged = None, False
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            obj = None
    if obj is None:
        found = dict(NUM.findall(text))
        if not all(k in found for k in LABELS + CUES):
            return None, "Zahlen unvollstaendig"
        m = REASON.search(text)
        obj = {**found, "brief_reason": (m.group(1) if m else "")}
        salvaged = True
    try:
        vals = {k: float(obj[k]) for k in LABELS + CUES}
    except (KeyError, TypeError, ValueError):
        return None, "Felder fehlen"
    if any(v < 0 or v > 1 for v in vals.values()):
        return None, "Werte ausserhalb 0..1"
    total = sum(vals[k] for k in LABELS)
    renorm = False
    if abs(total - 1) > 0.015:
        if total <= 0 or abs(total - 1) > 0.10:
            return None, f"Summe {total:.2f}"
        for k in LABELS:  # kleine Abweichung: normieren und markieren
            vals[k] = vals[k] / total
        renorm = True
    vals["brief_reason"] = str(obj.get("brief_reason", ""))[:1200]
    return vals, {"renormalized": renorm, "salvaged": salvaged}


# --------------------------------------------------------------------------- Lauf

def now():
    return datetime.now(timezone.utc).isoformat()


def run(items, folder, build, variants, client, prompt_fp, max_usd, label):
    folder.mkdir(parents=True, exist_ok=True)
    jobs = [(row, v) for row in items for v in range(variants)]
    todo = [(r, v) for r, v in jobs if not (folder / f"{r['uid'].replace('/', '_')}__v{v}.json").exists()]
    spent = 0.0
    tin_all = tout_all = 0
    fails = []
    skipped_in_row = 0
    total, already = len(jobs), len(jobs) - len(todo)
    print(f"--- apertus: {label} ({total} Anfragen, {already} schon fertig, {len(todo)} neu)", flush=True)
    t0 = time.time()
    for i, (row, v) in enumerate(todo, 1):
        messages = build(row, v)
        record = None
        for tries in range(2):
            try:
                data, secs = client.ask(messages, MAX_TOKENS)
            except SkipItem as exc:
                reason = {"reason": "HTTP 400 vom Server", "server": str(exc)[:300]}
                skipped_in_row += 1
                break
            usage = data.get("usage") or {}
            tin, tout = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            tin_all += tin
            tout_all += tout
            spent += tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
            choice = data["choices"][0]
            parsed, info = parse(choice["message"].get("content"))
            skipped_in_row = 0
            if parsed is not None:
                record = {"status": "ok", "uid": row["uid"], "variant": v, "model": MODEL,
                          "returned_model": data.get("model"), "parsed": parsed, **info,
                          "tries": tries + 1, "finish_reason": choice.get("finish_reason"),
                          "usage": {"input_tokens": tin, "output_tokens": tout}, "seconds": round(secs, 2),
                          "prompt_fingerprint": prompt_fp, "temperature": TEMPERATURE,
                          "thinking_off_sent": client.thinking_off, "format_mode": client.format_mode,
                          "format_instruction": True,
                          "received_at": now()}
                break
            raw = choice["message"].get("content") or ""
            reason = {"reason": info, "finish_reason": choice.get("finish_reason"),
                      "output_tokens": tout, "raw_start": raw[:300], "raw_end": raw[-300:]}
        if record is None:
            fails.append({"uid": row["uid"], "variant": v, **reason, "at": now()})
            (folder.parent / f"{folder.name}_failed.jsonl").open("a", encoding="utf-8").write(
                json.dumps(fails[-1]) + "\n")
        else:
            path = folder / f"{row['uid'].replace('/', '_')}__v{v}.json"
            path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        done = already + i
        if i == 1 or done % 50 == 0 or i == len(todo):
            rate = (time.time() - t0) / i
            left = (len(todo) - i) * rate / 60
            print(f"[apertus] {done}/{total} processed, {len(fails)} failed"
                  f"  (${spent:.3f}, ca. {left:.0f} Min uebrig)", flush=True)
        if skipped_in_row >= 15:
            print("\nSTOPP: Der Server lehnt seit 15 Faellen alles ab. Spaeter neu starten, "
                  "fertige Antworten bleiben gespeichert.")
            break
        if spent > max_usd:
            print(f"\nKOSTENBREMSE: ${spent:.2f} > ${max_usd}. Gestoppt. Neu starten mit --max-usd 10 o.ae.")
            break
    return {"label": label, "requested": len(jobs), "new_done": len(todo) - len(fails),
            "failed": len(fails), "input_tokens": tin_all, "output_tokens": tout_all,
            "usd_this_session": round(spent, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="1 Testanfrage")
    ap.add_argument("--pilot", action="store_true", help="20 Anfragen (4 Trainingsfaelle x 5 Prompts)")
    ap.add_argument("--snli", help="Pfad zur SNLI-Datei, falls nicht automatisch gefunden")
    ap.add_argument("--max-usd", type=float, default=6.0)
    args = ap.parse_args()

    root = find_root()
    key = load_key(root)
    build, variants, schema, prompt_fp = load_prompts(root)
    client = Client(key, schema)
    print(f"Projekt: {root}\nModell: {MODEL}\nPrompts aus deinem Code geladen ({variants} Varianten, "
          f"Fingerprint {prompt_fp}), Key endet auf ...{key[-4:]}")

    if args.check or args.pilot:
        train_path = find_dataset(root, "train")
        if not train_path:
            sys.exit("FEHLER: data/processed/train.jsonl nicht gefunden.")
        train = read_rows(train_path)

    if args.check:
        data, secs = client.ask(build(train[0], 0))
        content = data["choices"][0]["message"].get("content")
        parsed, info = parse(content)
        usage = data.get("usage") or {}
        cost = usage.get("prompt_tokens", 0) / 1e6 * PRICE_IN + usage.get("completion_tokens", 0) / 1e6 * PRICE_OUT
        print(f"\n=== VERBINDUNG OK ({secs:.1f} s) ===")
        print(f"Thinking aus: {client.thinking_off} | Format: {client.format_mode}")
        print(f"Antwort: {(content or '')[:400]}")
        print(f"Lesbar: {'ja' if parsed else 'nein (' + str(info) + ')'}")
        print(f"Tokens: {usage.get('prompt_tokens')} in / {usage.get('completion_tokens')} out, Kosten ${cost:.5f}")
        print(f"Hochrechnung richtiger Lauf (627 Faelle x 5): ca. ${cost * 3135:.2f}")
        print("\nNaechster Schritt: richtiger Lauf (ohne --check)")
        return

    if args.pilot:
        items = train[:4]
        res = run(items, OUT / "pilot", build, variants, client, prompt_fp, args.max_usd, "Pilot")
        files = list((OUT / "pilot").glob("*.json"))
        recs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
        res["pilot_ok_total"] = len(recs)
        res["pilot_salvaged_total"] = sum(1 for r in recs if r.get("salvaged"))
        per_call = res["usd_this_session"] / max(1, res["new_done"] + res["failed"])
        res["estimate_full_run_usd"] = round(per_call * 3135, 2)
        (OUT / "pilot_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"\nPilot fertig: {res['new_done']} neu ok, {res['failed']} Fehler, ${res['usd_this_session']:.3f}")
        print(f"Insgesamt im Pilot gespeichert: {res['pilot_ok_total']} von 20 (davon {res['pilot_salvaged_total']} gerettet)")
        if res["new_done"] + res["failed"]:
            print(f"Hochrechnung richtiger Lauf: ca. ${res['estimate_full_run_usd']}")
        print("\nBitte diese ganze Ausgabe an Claude schicken.")
        return

    mnli_path = find_dataset(root, "mnli_test")
    snli_path = Path(args.snli) if args.snli else find_dataset(root, "snli")
    if not mnli_path:
        sys.exit("FEHLER: data/processed/test_inputs.jsonl nicht gefunden.")
    mnli = read_rows(mnli_path)
    print(f"MNLI-Test: {mnli_path.relative_to(root)} ({len(mnli)} Faelle)")
    snli = read_rows(snli_path) if snli_path else []
    if snli:
        print(f"SNLI: {snli_path} ({len(snli)} Faelle)")
    else:
        print("WARNUNG: SNLI-Datei nicht gefunden, laufe nur MNLI. Spaeter mit --snli PFAD nachholen.")
    summaries = [run(mnli, OUT / "mnli_test", build, variants, client, prompt_fp, args.max_usd, "Test")]
    remaining = args.max_usd - summaries[0]["usd_this_session"]
    if snli and remaining > 0:
        summaries.append(run(snli, OUT / "snli", build, variants, client, prompt_fp, remaining, "unbekannte Eingaben (SNLI)"))
    summary = {"model": MODEL, "base_url": BASE_URL, "prompt_fingerprint": prompt_fp,
               "temperature": TEMPERATURE, "thinking_off_sent": client.thinking_off,
               "format_mode": client.format_mode, "finished_at": now(), "parts": summaries}
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\nFERTIG. Bitte den ganzen Ordner extension_apertus/outputs_v2 als ZIP an Claude schicken.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAbgebrochen. Fertige Antworten sind gespeichert, einfach neu starten.")
