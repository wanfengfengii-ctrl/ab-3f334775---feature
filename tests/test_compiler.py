"""Unit tests for the integer delay-plan compiler.

The key correctness test exhaustively enumerates every integer sequence in a
small domain and compares the compiler's plan against the brute-force
lexicographic minimum on the full objective tuple
(max abs error, total abs error, ramp count, sequence).
"""

import json
import os
import random
import sys
import threading
import time
import unittest
import urllib.request
import urllib.error
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import compiler  # noqa: E402
from app.compiler import CompileError, compile_plan, ramp_boundaries  # noqa: E402


def ramps_of(x):
    diffs = [x[i + 1] - x[i] for i in range(len(x) - 1)]
    runs = 1
    for a, b in zip(diffs, diffs[1:]):
        if b != a:
            runs += 1
    return runs


def brute_force_optimum(targets, lo, hi, step, max_ramps, anchors):
    """Return the optimal feasible sequence by full enumeration."""
    n = len(targets)
    best = None
    for x in product(range(lo, hi + 1), repeat=n):
        if any(x[i] != v for i, v in anchors.items()):
            continue
        if any(abs(x[i + 1] - x[i]) > step for i in range(n - 1)):
            continue
        r = ramps_of(x)
        if r > max_ramps:
            continue
        errs = [abs(x[i] - targets[i]) for i in range(n)]
        key = (max(errs), sum(errs), r, x)
        if best is None or key < best[0]:
            best = (key, list(x))
    return best


class BruteForceTests(unittest.TestCase):
    def setUp(self):
        self._saved = (compiler.MIN_ELEMENTS, compiler.MIN_ANCHORS)
        compiler.MIN_ELEMENTS = 4
        compiler.MIN_ANCHORS = 2

    def tearDown(self):
        compiler.MIN_ELEMENTS, compiler.MIN_ANCHORS = self._saved

    def test_random_instances_match_brute_force(self):
        rng = random.Random(20260930)
        trials = 400
        for trial in range(trials):
            n = rng.randint(4, 5)
            lo, hi = -2, 3
            step = rng.randint(1, 3)
            # Two anchors at random distinct positions with step-reachable
            # values (so the instance is structurally feasible).
            ai = sorted(rng.sample(range(n), 2))
            av = [rng.randint(lo, hi)]
            av.append(rng.randint(av[0] - step * (ai[1] - ai[0]),
                                  av[0] + step * (ai[1] - ai[0])))
            av[1] = max(lo, min(hi, av[1]))
            anchors = {ai[0]: av[0], ai[1]: av[1]}
            targets = [rng.randint(lo - 1, hi + 1) for _ in range(n)]
            max_ramps = rng.randint(1, n - 1)
            payload = {
                "targets": targets, "delay_min": lo, "delay_max": hi,
                "max_step": step, "max_ramps": max_ramps,
                "anchors": [{"index": i, "value": v}
                            for i, v in anchors.items()],
            }
            bf = brute_force_optimum(targets, lo, hi, step,
                                     max_ramps, anchors)
            if bf is None:
                with self.assertRaises(CompileError) as ctx:
                    compile_plan(payload)
                self.assertEqual(ctx.exception.status, 422)
                continue
            plan = compile_plan(payload)
            x = plan["delays"]
            key = (plan["max_abs_error"], plan["total_abs_error"],
                   plan["ramp_count"], tuple(x))
            self.assertEqual(key, bf[0],
                             msg=f"trial {trial}: {payload}\n{plan}")
            self.assertEqual(x, bf[1], msg=f"lex tie trial {trial}")

    def test_three_anchors_match_brute_force(self):
        rng = random.Random(4242)
        for _ in range(40):
            n = rng.randint(5, 6)
            lo, hi = -2, 3
            step = 2
            idxs = sorted(rng.sample(range(n), 3))
            vals = [rng.randint(lo, hi)]
            for k in (1, 2):
                gap = idxs[k] - idxs[k - 1]
                vals.append(max(lo, min(hi, rng.randint(
                    vals[-1] - step * gap, vals[-1] + step * gap))))
            anchors = dict(zip(idxs, vals))
            targets = [rng.randint(lo, hi) for _ in range(n)]
            max_ramps = rng.randint(2, n - 1)
            payload = {
                "targets": targets, "delay_min": lo, "delay_max": hi,
                "max_step": step, "max_ramps": max_ramps,
                "anchors": [{"index": i, "value": v}
                            for i, v in anchors.items()],
            }
            bf = brute_force_optimum(targets, lo, hi, step,
                                     max_ramps, anchors)
            if bf is None:
                with self.assertRaises(CompileError) as ctx:
                    compile_plan(payload)
                self.assertEqual(ctx.exception.status, 422)
                self.assertNotIn("delays", ctx.exception.details)
                return
            plan = compile_plan(payload)
            key = (plan["max_abs_error"], plan["total_abs_error"],
                   plan["ramp_count"], tuple(plan["delays"]))
            self.assertEqual(key, bf[0], msg=f"{payload}\n{plan}")


class PlanInvariantTests(unittest.TestCase):
    def _payload(self, **over):
        base = {
            "targets": [0, 2, 5, 9, 12, 14, 15, 14, 12, 9, 5, 2],
            "delay_min": -50, "delay_max": 50, "max_step": 4,
            "max_ramps": 4,
            "anchors": [{"index": 0, "value": 0},
                        {"index": 6, "value": 15},
                        {"index": 11, "value": 2}],
        }
        base.update(over)
        return base

    def test_plan_satisfies_all_constraints(self):
        payload = self._payload()
        plan = compile_plan(payload)
        x = plan["delays"]
        n = len(x)
        self.assertEqual(n, 12)
        self.assertEqual(len(plan["errors"]), n)
        for a in payload["anchors"]:
            self.assertEqual(x[a["index"]], a["value"])
        for i, v in enumerate(x):
            self.assertTrue(payload["delay_min"] <= v <= payload["delay_max"])
            self.assertEqual(plan["errors"][i], v - payload["targets"][i])
        for i in range(n - 1):
            self.assertLessEqual(abs(x[i + 1] - x[i]), payload["max_step"])
        self.assertEqual(plan["ramp_count"], ramps_of(x))
        self.assertLessEqual(plan["ramp_count"], payload["max_ramps"])
        # ramps partition the element range
        self.assertEqual(plan["ramps"][0]["start"], 0)
        self.assertEqual(plan["ramps"][-1]["end"], n - 1)
        for r, s in zip(plan["ramps"], plan["ramps"][1:]):
            self.assertEqual(r["end"], s["start"])
            self.assertNotEqual(r["delta"], s["delta"])
        for r in plan["ramps"]:
            for i in range(r["start"], r["end"]):
                self.assertEqual(x[i + 1] - x[i], r["delta"])
        self.assertEqual(plan["max_abs_error"],
                         max(abs(e) for e in plan["errors"]))
        self.assertEqual(plan["total_abs_error"],
                         sum(abs(e) for e in plan["errors"]))

    def test_deterministic(self):
        payload = self._payload()
        a = compile_plan(payload)
        b = compile_plan(payload)
        self.assertEqual(a, b)

    def test_48_elements_eight_anchors_performance(self):
        rng = random.Random(7)
        n = 48
        # smooth random walk targets inside a modest integer domain
        targets = []
        v = 100
        for _ in range(n):
            v += rng.randint(-6, 6)
            targets.append(v)
        idxs = [0, 7, 14, 21, 27, 33, 40, 47]
        anchors = [{"index": i, "value": targets[i]} for i in idxs]
        payload = {"targets": targets, "delay_min": min(targets) - 40,
                   "delay_max": max(targets) + 40, "max_step": 12,
                   "max_ramps": 8, "anchors": anchors}
        t0 = time.time()
        plan = compile_plan(payload)
        elapsed = time.time() - t0
        self.assertLess(elapsed, 5.0)
        self.assertEqual(len(plan["delays"]), n)
        for a in anchors:
            self.assertEqual(plan["delays"][a["index"]], a["value"])


class InfeasibilityTests(unittest.TestCase):
    BASE = dict(targets=[0] * 12, delay_min=-10, delay_max=10,
                max_step=1, max_ramps=3,
                anchors=[{"index": 0, "value": 0},
                         {"index": 11, "value": 0}])

    def test_anchor_out_of_bounds(self):
        p = dict(self.BASE, anchors=[{"index": 0, "value": -20},
                                     {"index": 11, "value": 0}])
        with self.assertRaises(CompileError) as ctx:
            compile_plan(p)
        self.assertEqual(ctx.exception.status, 422)
        conf = ctx.exception.details["conflicts"]
        self.assertTrue(any(c["kind"] == "anchor_out_of_bounds"
                            and c["start"] == c["end"] == 0 for c in conf))

    def test_step_unreachable_conflict_interval(self):
        p = dict(self.BASE, max_step=1,
                 anchors=[{"index": 0, "value": 0},
                          {"index": 5, "value": 10},
                          {"index": 11, "value": 0}])
        with self.assertRaises(CompileError) as ctx:
            compile_plan(p)
        self.assertEqual(ctx.exception.status, 422)
        kinds = {(c["kind"], c["start"], c["end"])
                 for c in ctx.exception.details["conflicts"]}
        self.assertIn(("step_unreachable", 0, 5), kinds)
        self.assertIn(("step_unreachable", 5, 11), kinds)

    def test_ramp_budget_conflict_without_partial_table(self):
        # Anchors force +2 on five edges then a decrease on six edges; a
        # single constant difference (one ramp) cannot satisfy both.
        p = dict(self.BASE, max_step=2, max_ramps=1,
                 anchors=[{"index": 0, "value": 0},
                          {"index": 5, "value": 10},
                          {"index": 11, "value": 0}])
        with self.assertRaises(CompileError) as ctx:
            compile_plan(p)
        self.assertEqual(ctx.exception.status, 422)
        conf = ctx.exception.details["conflicts"]
        self.assertTrue(any(c["kind"] == "ramp_budget" for c in conf))
        self.assertNotIn("delays", ctx.exception.details)

    def test_all_infeasible_errors_are_stable(self):
        p = dict(self.BASE, max_step=0,
                 anchors=[{"index": 0, "value": 0},
                          {"index": 11, "value": 3}])
        first = None
        for _ in range(3):
            try:
                compile_plan(p)
                self.fail("expected infeasibility")
            except CompileError as e:
                self.assertEqual(e.status, 422)
                self.assertNotIn("delays", e.details)
                blob = json.dumps(e.details, sort_keys=True)
                if first is None:
                    first = blob
                self.assertEqual(blob, first)


class ValidationTests(unittest.TestCase):
    def _ok(self, **over):
        base = {"targets": list(range(12)), "delay_min": 0, "delay_max": 100,
                "max_step": 5, "max_ramps": 3,
                "anchors": [{"index": 0, "value": 0},
                            {"index": 11, "value": 11}]}
        base.update(over)
        return base

    def test_wrong_target_count(self):
        with self.assertRaises(CompileError) as ctx:
            compile_plan(self._ok(targets=list(range(11))))
        self.assertEqual(ctx.exception.status, 400)

    def test_49_targets_rejected(self):
        with self.assertRaises(CompileError) as ctx:
            compile_plan(self._ok(targets=list(range(49))))
        self.assertEqual(ctx.exception.status, 400)

    def test_anchor_count_bounds(self):
        one = [{"index": 0, "value": 0}]
        with self.assertRaises(CompileError) as ctx:
            compile_plan(self._ok(anchors=one))
        self.assertEqual(ctx.exception.status, 400)
        nine = [{"index": i, "value": i} for i in range(9)]
        with self.assertRaises(CompileError) as ctx:
            compile_plan(self._ok(anchors=nine, targets=list(range(48)),
                                  max_ramps=10))
        self.assertEqual(ctx.exception.status, 400)

    def test_non_integer_rejected(self):
        with self.assertRaises(CompileError) as ctx:
            compile_plan(self._ok(max_step=2.5))
        self.assertEqual(ctx.exception.status, 400)

    def test_reversed_interval_rejected(self):
        with self.assertRaises(CompileError) as ctx:
            compile_plan(self._ok(delay_min=90, delay_max=10))
        self.assertEqual(ctx.exception.status, 400)

    def test_duplicate_anchor_conflict_rejected(self):
        with self.assertRaises(CompileError) as ctx:
            compile_plan(self._ok(anchors=[
                {"index": 3, "value": 3}, {"index": 3, "value": 9},
                {"index": 11, "value": 11}]))
        self.assertEqual(ctx.exception.status, 400)

    def test_exact_anchor_hits_even_when_target_differs(self):
        # Anchors deliberately disagree with the targets; they must win.
        plan = compile_plan(self._ok(
            anchors=[{"index": 0, "value": 2}, {"index": 11, "value": 9}]))
        self.assertEqual(plan["delays"][0], 2)
        self.assertEqual(plan["delays"][11], 9)


class RampBoundaryTests(unittest.TestCase):
    def test_runs(self):
        x = [0, 1, 2, 3, 3, 3, 2, 1, 0]
        ramps = ramp_boundaries(x)
        self.assertEqual(
            [(r["start"], r["end"], r["delta"]) for r in ramps],
            [(0, 3, 1), (3, 5, 0), (5, 8, -1)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
