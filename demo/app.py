"""Local demo app: python demo/app.py  ->  opens http://127.0.0.1:8765

Standard library only (no web framework). Runs on your own computer; nothing is published.
Stored examples need no API key. Live sentence pairs and LLM wording use the keys in .env.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import traceback
import webbrowser
from urllib.parse import urlsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from adapter import ACTION_TITLE, Adapter, model_settings  # noqa: E402

# Curated cases: each shows one behaviour of the frozen system, including its failures.
CURATED = [
    {"uid": "65353n", "model": "gpt", "tag": "Clear case → answer",
     "why": "97% of annotators chose entailment. The model agrees with this majority."},
    {"uid": "106411n", "model": "gpt", "tag": "Model unsure → hedge",
     "why": "Most people read 'diet was of wheaten bread' as 'only bread'. GPT says neutral, so the system flags its answer as uncertain."},
    {"uid": "66260c", "model": "gpt", "tag": "Genuinely debatable → alternatives",
     "why": "Annotators split across the labels. The two-signal rule chooses alternatives; both simpler rules hedge. This does not establish which readings are defensible."},
    {"uid": "13531e", "model": "claude", "tag": "Word meaning unclear → clarify",
     "why": "Annotators split 53/26/21 on whether feeling 'like a rat' means feeling 'weasily'. Annotated as lexical ambiguity. The simpler rules would hedge or answer."},
    {"uid": "95883e", "model": "gpt", "tag": "Failure: confident although debatable",
     "why": "Humans split 47/22/31, but GPT answers directly, and so would the simpler rules. Click \"Compare all models\": Claude shows alternatives."},
    {"uid": "snli_3552435734.jpg#0r1e", "model": "gpt", "tag": "Limit: other text source (SNLI)",
     "why": "Humans split 44/14/42, but the MNLI-trained system answers directly. This illustrates missed disagreement on the SNLI sample."},
]


def make_handler(adapter, curated):
    static = HERE / "static"

    class Handler(BaseHTTPRequestHandler):
        def _local_request(self):
            port = self.server.server_address[1]
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            origin = self.headers.get("Origin")
            return (self.headers.get("Host") in hosts and
                    (not origin or origin in {f"http://{h}" for h in hosts}) and
                    self.headers.get("Sec-Fetch-Site") != "cross-site")

        def log_message(self, *args):  # keep the console readable
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype + ("; charset=utf-8" if "json" in ctype or "html" in ctype else ""))
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self._local_request():
                return self._send(403, {"error": "Open this demo directly on localhost."})
            if self.path in ("/", "/index.html"):
                return self._send(200, (static / "index.html").read_bytes(), "text/html")
            if self.path == "/api/meta":
                ex = adapter.examples
                return self._send(200, {
                    "models": [{"key": k, "name": model_settings(adapter.root, k)["model"]} for k in adapter.keys],
                    "protocol_id": adapter.protocol_id, "curated": curated,
                    "counts": {"test": int((ex.split == "test").sum()), "shift": int((ex.split == "shift").sum())},
                    "actions": ACTION_TITLE, "apertus": adapter._apertus is not None})
            if self.path.startswith("/api/random"):
                split = "shift" if "shift" in self.path else "test"
                row = adapter.examples[adapter.examples.split == split].sample(1).iloc[0]
                return self._send(200, {"uid": row.uid})
            return self._send(404, {"error": "Not found"})

        def do_POST(self):
            if not self._local_request():
                return self._send(403, {"error": "Cross-site requests are not accepted."})
            if self.headers.get_content_type() != "application/json":
                return self._send(415, {"error": "Use application/json."})
            if self.path not in ("/api/assess", "/api/compare", "/api/apertus"):
                return self._send(404, {"error": "Not found"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 16384:
                    return self._send(413, {"error": "Invalid or oversized request."})
                req = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(req, dict):
                    raise ValueError("Expected a JSON object.")
                if self.path == "/api/apertus":  # post-hoc third model, stored cases only, no API call
                    return self._send(200, adapter.apertus(req.get("uid")))
                if self.path == "/api/compare":
                    return self._send(200, adapter.compare(uid=req.get("uid"), premise=req.get("premise"),
                                                           hypothesis=req.get("hypothesis")))
                result = adapter.assess(req.get("model", "gpt"), uid=req.get("uid"), premise=req.get("premise"),
                                        hypothesis=req.get("hypothesis"), wording=req.get("wording", "template"))
                return self._send(200, result)
            except (ValueError, LookupError) as exc:
                return self._send(400, {"error": str(exc)})
            except RuntimeError as exc:  # missing key, provider refused, invalid model output
                return self._send(502, {"error": str(exc)})
            except Exception as exc:  # never leak details that could contain credentials
                traceback.print_exc()
                return self._send(500, {"error": f"Unexpected {type(exc).__name__}. See the terminal for details."})

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Uncertain or Genuinely Debatable? Local demo.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    adapter = Adapter()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(adapter, CURATED))
    url = f"http://127.0.0.1:{args.port}"
    print(f"Demo running at {url}  (frozen protocol {adapter.protocol_id}). Stop with Ctrl+C.")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
