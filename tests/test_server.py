"""HTTP-level tests for the compile service (in-process server)."""

import json
import os
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.server import build_server  # noqa: E402


class ServerHarness:
    def __init__(self):
        self.server = build_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def get(self, path):
        try:
            with urllib.request.urlopen(self.url(path), timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def post(self, path, payload, raw=None):
        data = raw if raw is not None else json.dumps(payload).encode()
        req = urllib.request.Request(self.url(path), data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())


GOOD_PAYLOAD = {
    "targets": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "delay_min": -50, "delay_max": 50, "max_step": 3, "max_ramps": 3,
    "anchors": [{"index": 0, "value": 0}, {"index": 11, "value": 11}],
}


class HttpTests(unittest.TestCase):
    def test_healthz(self):
        with ServerHarness() as h:
            status, body = h.get("/healthz")
            self.assertEqual(status, 200)
            self.assertEqual(body["status"], "ok")

    def test_compile_success(self):
        with ServerHarness() as h:
            status, body = h.post("/api/delay-plans/compile", GOOD_PAYLOAD)
            self.assertEqual(status, 200, body)
            self.assertEqual(body["status"], "ok")
            plan = body["plan"]
            self.assertEqual(len(plan["delays"]), 12)
            self.assertEqual(plan["delays"][0], 0)
            self.assertEqual(plan["delays"][11], 11)
            self.assertEqual(len(plan["errors"]), 12)
            self.assertLessEqual(plan["ramp_count"], 3)
            self.assertTrue(plan["ramps"])

    def test_infeasible_returns_422_with_conflicts_and_no_table(self):
        with ServerHarness() as h:
            p = dict(GOOD_PAYLOAD, max_step=0,
                     anchors=[{"index": 0, "value": 0},
                              {"index": 11, "value": 11}])
            status, body = h.post("/api/delay-plans/compile", p)
            self.assertEqual(status, 422)
            self.assertIn("conflicts", body)
            self.assertTrue(body["conflicts"])
            self.assertNotIn("delays", body)
            self.assertNotIn("plan", body)
            c = body["conflicts"][0]
            self.assertEqual(c["start"], 0)
            self.assertEqual(c["end"], 11)

    def test_bad_payload_returns_400(self):
        with ServerHarness() as h:
            status, body = h.post("/api/delay-plans/compile", {"foo": 1})
            self.assertEqual(status, 400)
            self.assertNotIn("delays", body)

    def test_invalid_json_returns_400(self):
        with ServerHarness() as h:
            status, body = h.post("/api/delay-plans/compile", None,
                                  raw=b"{not json")
            self.assertEqual(status, 400)
            self.assertEqual(body["error"], "invalid_json")

    def test_unknown_route(self):
        with ServerHarness() as h:
            status, body = h.get("/nope")
            self.assertEqual(status, 404)

    def test_method_not_allowed(self):
        with ServerHarness() as h:
            req = urllib.request.Request(h.url("/healthz"), method="PUT")
            try:
                urllib.request.urlopen(req, timeout=5)
                self.fail("expected 405")
            except urllib.error.HTTPError as e:
                self.assertEqual(e.code, 405)

    def test_typical_firmware_smoke_shape(self):
        # 16 elements, 3 anchors, step 4, 4 ramps: the canonical smoke case.
        targets = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
                   38, 40]
        payload = {
            "targets": targets, "delay_min": 0, "delay_max": 100,
            "max_step": 4, "max_ramps": 4,
            "anchors": [{"index": 0, "value": 10},
                        {"index": 8, "value": 26},
                        {"index": 15, "value": 40}],
        }
        with ServerHarness() as h:
            status, body = h.post("/api/delay-plans/compile", payload)
            self.assertEqual(status, 200, body)
            plan = body["plan"]
            self.assertEqual(len(plan["delays"]), 16)
            # adjacent deltas never exceed the hardware shift
            x = plan["delays"]
            self.assertTrue(all(abs(x[i + 1] - x[i]) <= 4
                                for i in range(15)))
            self.assertLessEqual(plan["ramp_count"], 4)
            # every ramp boundary is returned and consistent
            for r in plan["ramps"]:
                for i in range(r["start"], r["end"]):
                    self.assertEqual(x[i + 1] - x[i], r["delta"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
