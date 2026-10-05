"""HTTP service for the delay-plan compiler.

Zero third-party dependencies (stdlib ``http.server``).  Endpoints:

* ``GET  /healthz``           -> liveness probe, 200 ``{"status": "ok"}``
* ``GET  /``                  -> service banner / version
* ``POST /api/delay-plans/compile`` -> compile a delay plan

The compile endpoint returns:

* 200 with the optimal plan on success;
* 400 for malformed requests or integer domains the exact solver rejects;
* 422 for well-formed but infeasible instances, with conflict intervals and
  **no** partial delay table;
* 404 / 405 for unknown routes / methods;
* 500 for unexpected internal errors.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.compiler import CompileError, compile_plan  # noqa: E402

SERVICE_NAME = "delay-plan-compiler"
SERVICE_VERSION = "1.0.0"
MAX_BODY_BYTES = 1 << 20  # 1 MiB is far beyond a 48-element request

logger = logging.getLogger(SERVICE_NAME)


class Handler(BaseHTTPRequestHandler):
    server_version = f"{SERVICE_NAME}/{SERVICE_VERSION}"
    protocol_version = "HTTP/1.1"
    timeout = 10

    # --- helpers ---------------------------------------------------------

    def _write_json(self, status: int, payload: dict):
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # route through stdlib logging
        logger.info("%s - %s", self.address_string(), fmt % args)

    # --- routing ---------------------------------------------------------

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/healthz":
            self._write_json(200, {"status": "ok", "service": SERVICE_NAME})
        elif path in ("/", "/api"):
            self._write_json(200, {
                "service": SERVICE_NAME,
                "version": SERVICE_VERSION,
                "endpoints": {
                    "health": "GET /healthz",
                    "compile": "POST /api/delay-plans/compile",
                },
            })
        else:
            self._write_json(404, {"error": "not_found", "path": path})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path != "/api/delay-plans/compile":
            self._write_json(404, {"error": "not_found", "path": path})
            return
        self._handle_compile()

    # Explicit responses for other common verbs.
    def do_PUT(self):
        self._method_not_allowed()

    def do_DELETE(self):
        self._method_not_allowed()

    def do_PATCH(self):
        self._method_not_allowed()

    def _method_not_allowed(self):
        self._write_json(405, {"error": "method_not_allowed",
                               "allowed": "GET, POST"})

    # --- compile endpoint ------------------------------------------------

    def _handle_compile(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._write_json(400, {"error": "bad_request",
                                   "message": "invalid Content-Length"})
            return
        if length <= 0:
            self._write_json(400, {"error": "bad_request",
                                   "message": "empty request body"})
            return
        if length > MAX_BODY_BYTES:
            self._write_json(413, {"error": "payload_too_large",
                                   "max_bytes": MAX_BODY_BYTES})
            return

        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._write_json(400, {"error": "invalid_json", "message": str(exc)})
            return

        try:
            plan = compile_plan(payload)
        except CompileError as exc:
            body = {"error": "infeasible" if exc.status == 422 else "bad_request",
                    "message": str(exc)}
            if exc.details:
                body.update(exc.details)
            # 422 bodies deliberately contain no "delays"/partial table.
            self._write_json(exc.status, body)
            return
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("compile failed")
            self._write_json(500, {"error": "internal_error",
                                   "message": str(exc)})
            return

        self._write_json(200, {"status": "ok", "plan": plan})


def build_server(host: str = "0.0.0.0", port: int = 8080) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None) -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    host = os.environ.get("HOST", "0.0.0.0")
    try:
        port = int(os.environ.get("API_PORT", os.environ.get("PORT", "8080")))
    except ValueError:
        logger.error("API_PORT must be an integer")
        return 2
    server = build_server(host, port)
    logger.info("%s %s listening on %s:%d", SERVICE_NAME, SERVICE_VERSION,
                host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
