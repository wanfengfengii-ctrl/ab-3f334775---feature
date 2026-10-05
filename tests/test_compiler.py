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


def brute_force_optimum(targets, lo, hi, step, max_ramps, anchors,
                        sync_windows=()):
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
        if any(sum(1 for i in range(s, e + 1) if x[i] == v) != 1
               for s, e, v in sync_windows):
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


class SyncWindowBruteForceTests(unittest.TestCase):
    """Exhaustive comparison with sync windows enabled."""

    def setUp(self):
        self._saved = (compiler.MIN_ELEMENTS, compiler.MIN_ANCHORS)
        compiler.MIN_ELEMENTS = 4
        compiler.MIN_ANCHORS = 2

    def tearDown(self):
        compiler.MIN_ELEMENTS, compiler.MIN_ANCHORS = self._saved

    def _run(self, seed, trials, n_choices, max_windows):
        rng = random.Random(seed)
        for trial in range(trials):
            n = rng.choice(n_choices)
            lo, hi = -2, 3
            step = rng.randint(1, 3)
            ai = sorted(rng.sample(range(n), 2))
            av = [rng.randint(lo, hi)]
            av.append(max(lo, min(hi, rng.randint(
                av[0] - step * (ai[1] - ai[0]),
                av[0] + step * (ai[1] - ai[0])))))
            anchors = {ai[0]: av[0], ai[1]: av[1]}
            targets = [rng.randint(lo - 1, hi + 1) for _ in range(n)]
            max_ramps = rng.randint(1, n - 1)

            # Build disjoint windows sorted by start from random boundary
            # pairs; skip a trial whose draw cannot form disjoint windows.
            nw = rng.randint(1, max_windows)
            bounds = sorted(rng.sample(range(n + 1), 2 * nw))
            windows = []
            ok = True
            for j in range(nw):
                s, e = bounds[2 * j], bounds[2 * j + 1] - 1
                if s > e or (windows and s <= windows[-1][1]):
                    ok = False
                    break
                windows.append((s, e, rng.randint(lo, hi)))
            if not ok:
                continue

            payload = {
                "targets": targets, "delay_min": lo, "delay_max": hi,
                "max_step": step, "max_ramps": max_ramps,
                "anchors": [{"index": i, "value": v}
                            for i, v in anchors.items()],
                "sync_windows": [{"start": s, "end": e, "value": v}
                                 for s, e, v in windows],
            }
            bf = brute_force_optimum(targets, lo, hi, step, max_ramps,
                                     anchors, windows)
            if bf is None:
                with self.assertRaises(CompileError) as ctx:
                    compile_plan(payload)
                self.assertEqual(ctx.exception.status, 422,
                                 msg=f"trial {trial}: {payload}")
                self.assertNotIn("delays", ctx.exception.details)
                continue
            plan = compile_plan(payload)
            x = plan["delays"]
            key = (plan["max_abs_error"], plan["total_abs_error"],
                   plan["ramp_count"], tuple(x))
            self.assertEqual(key, bf[0],
                             msg=f"trial {trial}: {payload}\n{plan}")
            self.assertEqual(x, bf[1], msg=f"lex tie trial {trial}")
            self.assertEqual(len(plan["sync_windows"]), len(windows))
            for (s, e, v), picked in zip(windows, plan["sync_windows"]):
                self.assertEqual(picked["value"], v)
                self.assertIn(picked["index"], range(s, e + 1))
                self.assertEqual(x[picked["index"]], v)
                self.assertEqual(sum(1 for i in range(s, e + 1)
                                     if x[i] == v), 1)

    def test_gapped_windows_match_brute_force(self):
        self._run(20261005, 500, (4, 5), 2)

    def test_three_adjacent_windows_match_brute_force(self):
        # n == 6 tiled into three adjacent two-element windows.
        self._run(31337, 250, (6,), 3)


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


class SyncWindowBehaviorTests(unittest.TestCase):
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

    def test_selected_indexes_returned_in_request_order(self):
        windows = [{"start": 2, "end": 4, "value": 8},
                   {"start": 7, "end": 9, "value": 10}]
        plan = compile_plan(self._payload(sync_windows=windows))
        picked = plan["sync_windows"]
        self.assertEqual([p["value"] for p in picked], [8, 10])
        x = plan["delays"]
        for w, p in zip(windows, picked):
            self.assertIn(p["index"], range(w["start"], w["end"] + 1))
            self.assertEqual(x[p["index"]], w["value"])
            self.assertEqual(sum(1 for i in range(w["start"], w["end"] + 1)
                                 if x[i] == w["value"]), 1)

    def test_exactly_one_hit_even_when_targets_prefer_more(self):
        # Targets equal the sync value at every window position; the
        # optimizer must move all but one off it.
        windows = [{"start": 2, "end": 5, "value": 12}]
        p = self._payload(sync_windows=windows)
        plan = compile_plan(p)
        x = plan["delays"]
        hits = [i for i in range(2, 6) if x[i] == 12]
        self.assertEqual(len(hits), 1)
        self.assertEqual(plan["sync_windows"][0]["index"], hits[0])

    def test_three_windows_including_singleton_and_adjacent(self):
        # Singleton window [1,1] sits directly adjacent to [2,3].
        windows = [{"start": 1, "end": 1, "value": 2},
                   {"start": 2, "end": 3, "value": 5},
                   {"start": 9, "end": 10, "value": 5}]
        plan = compile_plan(self._payload(max_ramps=6, sync_windows=windows))
        self.assertEqual(len(plan["sync_windows"]), 3)
        self.assertEqual(plan["sync_windows"][0],
                         {"index": 1, "value": 2})
        self.assertEqual(plan["sync_windows"][1]["value"], 5)
        self.assertIn(plan["sync_windows"][1]["index"], (2, 3))
        x = plan["delays"]
        for w in windows:
            self.assertEqual(sum(1 for i in range(w["start"], w["end"] + 1)
                                 if x[i] == w["value"]), 1)

    def test_omitted_field_keeps_response_shape(self):
        plan = compile_plan(self._payload())
        self.assertNotIn("sync_windows", plan)

    def test_window_honored_alongside_anchors_step_and_ramp_budget(self):
        windows = [{"start": 8, "end": 10, "value": 10}]
        plan = compile_plan(self._payload(sync_windows=windows))
        x = plan["delays"]
        for a in self._payload()["anchors"]:
            self.assertEqual(x[a["index"]], a["value"])
        self.assertTrue(all(abs(x[i + 1] - x[i]) <= 4
                            for i in range(len(x) - 1)))
        self.assertLessEqual(plan["ramp_count"], 4)
        self.assertEqual(sum(1 for i in range(8, 11) if x[i] == 10), 1)

    def test_deterministic_with_windows(self):
        p = self._payload(sync_windows=[{"start": 2, "end": 4, "value": 8}])
        self.assertEqual(compile_plan(p), compile_plan(p))


class SyncWindowInfeasibilityTests(unittest.TestCase):
    BASE = dict(targets=[0] * 12, delay_min=-10, delay_max=10,
                max_step=1, max_ramps=3,
                anchors=[{"index": 0, "value": 0},
                         {"index": 11, "value": 0}])

    def _assert_422(self, payload, reason=None, span=None):
        with self.assertRaises(CompileError) as ctx:
            compile_plan(payload)
        self.assertEqual(ctx.exception.status, 422)
        details = ctx.exception.details
        self.assertNotIn("delays", details)
        self.assertNotIn("ramps", details)
        conf = details["conflicts"]
        self.assertTrue(conf)
        self.assertTrue(all(c["kind"] == "sync_window" for c in conf))
        if reason is not None:
            self.assertTrue(any(c["reason"] == reason for c in conf))
        if span is not None:
            self.assertIn(span, {(c["start"], c["end"]) for c in conf})
        return conf

    def test_window_value_outside_anchor_cone(self):
        # With step 1 and both anchors pinned to 0, element 2 can only be
        # in [-2, 2]; asking for 9 is jointly impossible.
        p = dict(self.BASE,
                 sync_windows=[{"start": 2, "end": 4, "value": 9}])
        conf = self._assert_422(p, "no_admissible_value", (2, 4))
        self.assertEqual(conf[0]["value"], 9)

    def test_window_against_anchor_value_at_same_index(self):
        # The anchor at 5 fixes x[5]=0; a singleton window demanding x[5]=3
        # directly contradicts it.
        p = dict(self.BASE, max_step=5,
                 anchors=[{"index": 0, "value": 0},
                          {"index": 5, "value": 0},
                          {"index": 11, "value": 0}],
                 sync_windows=[{"start": 5, "end": 5, "value": 3}])
        self._assert_422(p, "no_admissible_value", (5, 5))

    def test_two_equal_anchors_inside_window_force_two_hits(self):
        p = dict(self.BASE, max_step=5,
                 anchors=[{"index": 3, "value": 4},
                          {"index": 7, "value": 4},
                          {"index": 11, "value": 4}],
                 sync_windows=[{"start": 2, "end": 8, "value": 4}])
        conf = self._assert_422(p, "multiple_forced_hits", (2, 8))
        self.assertEqual(conf[0]["forced_indices"][:2], [3, 7])

    def test_joint_ramp_budget_infeasibility(self):
        # One ramp with equal end anchors forces the constant zero plan;
        # hitting 4 once needs a departure and return (>= 3 ramps).
        p = dict(self.BASE, max_step=2, max_ramps=1,
                 sync_windows=[{"start": 3, "end": 8, "value": 4}])
        self._assert_422(p, "ramp_budget", (3, 8))

    def test_stable_422_across_calls(self):
        p = dict(self.BASE,
                 sync_windows=[{"start": 2, "end": 4, "value": 9}])
        blobs = set()
        for _ in range(3):
            try:
                compile_plan(p)
                self.fail("expected 422")
            except CompileError as e:
                self.assertEqual(e.status, 422)
                blobs.add(json.dumps(e.details, sort_keys=True))
        self.assertEqual(len(blobs), 1)


class SyncWindowValidationTests(unittest.TestCase):
    def _ok(self, **over):
        base = {"targets": list(range(12)), "delay_min": 0, "delay_max": 100,
                "max_step": 5, "max_ramps": 3,
                "anchors": [{"index": 0, "value": 0},
                            {"index": 11, "value": 11}]}
        base.update(over)
        return base

    def test_not_an_array(self):
        with self.assertRaises(CompileError) as ctx:
            compile_plan(self._ok(sync_windows={"start": 0}))
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.details["field"], "sync_windows")

    def test_empty_array(self):
        with self.assertRaises(CompileError) as ctx:
            compile_plan(self._ok(sync_windows=[]))
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.details["field"], "sync_windows")

    def test_four_windows_rejected(self):
        wins = [{"start": i, "end": i, "value": 0} for i in range(4)]
        with self.assertRaises(CompileError) as ctx:
            compile_plan(self._ok(sync_windows=wins))
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.details["field"], "sync_windows")

    def test_missing_field_localized(self):
        with self.assertRaises(CompileError) as ctx:
            compile_plan(self._ok(sync_windows=[{"start": 0, "end": 1}]))
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.details["field"],
                         "sync_windows[0].value")

    def test_non_integer_member(self):
        with self.assertRaises(CompileError) as ctx:
            compile_plan(self._ok(sync_windows="nope"))
        self.assertEqual(ctx.exception.status, 400)
        with self.assertRaises(CompileError) as ctx:
            compile_plan(self._ok(sync_windows=[
                {"start": 1.5, "end": 2, "value": 3}]))
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.details["field"],
                         "sync_windows[0].start")

    def test_index_out_of_range(self):
        with self.assertRaises(CompileError) as ctx:
            compile_plan(self._ok(sync_windows=[
                {"start": 0, "end": 12, "value": 3}]))
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.details["field"],
                         "sync_windows[0].end")
        with self.assertRaises(CompileError) as ctx:
            compile_plan(self._ok(sync_windows=[
                {"start": -1, "end": 2, "value": 3}]))
        self.assertEqual(ctx.exception.status, 400)

    def test_start_after_end(self):
        with self.assertRaises(CompileError) as ctx:
            compile_plan(self._ok(sync_windows=[
                {"start": 5, "end": 2, "value": 3}]))
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.details["field"],
                         "sync_windows[0].start")

    def test_value_outside_global_range(self):
        with self.assertRaises(CompileError) as ctx:
            compile_plan(self._ok(sync_windows=[
                {"start": 0, "end": 2, "value": 101}]))
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.details["field"],
                         "sync_windows[0].value")
        with self.assertRaises(CompileError) as ctx:
            compile_plan(self._ok(sync_windows=[
                {"start": 0, "end": 2, "value": -1}]))
        self.assertEqual(ctx.exception.status, 400)

    def test_unsorted_windows_rejected(self):
        with self.assertRaises(CompileError) as ctx:
            compile_plan(self._ok(sync_windows=[
                {"start": 6, "end": 7, "value": 6},
                {"start": 2, "end": 3, "value": 2}]))
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.details["field"],
                         "sync_windows[1].start")

    def test_overlapping_windows_rejected(self):
        with self.assertRaises(CompileError) as ctx:
            compile_plan(self._ok(sync_windows=[
                {"start": 2, "end": 5, "value": 3},
                {"start": 5, "end": 7, "value": 4}]))
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.details["field"],
                         "sync_windows[1].start")
        # touching with a gap of at least one element is fine (6 > 5)
        plan = compile_plan(self._ok(sync_windows=[
            {"start": 2, "end": 5, "value": 3},
            {"start": 6, "end": 7, "value": 4}]))
        self.assertEqual(len(plan["sync_windows"]), 2)

    def test_member_not_object(self):
        with self.assertRaises(CompileError) as ctx:
            compile_plan(self._ok(sync_windows=[42]))
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.details["field"], "sync_windows[0]")


class RampBoundaryTests(unittest.TestCase):
    def test_runs(self):
        x = [0, 1, 2, 3, 3, 3, 2, 1, 0]
        ramps = ramp_boundaries(x)
        self.assertEqual(
            [(r["start"], r["end"], r["delta"]) for r in ramps],
            [(0, 3, 1), (3, 5, 0), (5, 8, -1)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
