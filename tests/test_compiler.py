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


# ---------------------------------------------------------------------------
# Sync windows (clock-domain handoff): exactly one element per window
# ---------------------------------------------------------------------------


def _random_windows(rng, n, lo, hi):
    """1..3 non-overlapping zero-based closed intervals with in-range values."""
    k = rng.randint(1, min(3, n))
    wins = []
    cursor = 0
    for j in range(k):
        remaining = k - j - 1
        s = rng.randint(cursor, n - 1 - remaining)
        e = rng.randint(s, min(n - 1 - remaining, s + 2))
        wins.append((s, e, rng.randint(lo, hi)))
        cursor = e + 1
    return wins


def _planted_windows(rng, n, anchors, step):
    """Windows built around a feasible interpolation between the anchors.

    The interpolated sequence itself satisfies every sync window, so the
    resulting instance is structurally feasible (the ramp budget may still
    make it infeasible, which the brute-force comparison also covers).
    """
    items = sorted(anchors.items())
    x = [None] * n
    i0, v0 = items[0]
    for i in range(0, i0 + 1):
        x[i] = v0
    for (i0, v0), (i1, v1) in zip(items, items[1:]):
        gap = i1 - i0
        for j in range(i0, i1 + 1):
            x[j] = v0 + round((v1 - v0) * (j - i0) / gap)
    il, vl = items[-1]
    for i in range(il, n):
        x[i] = vl
    wins = []
    cursor = 0
    k = rng.randint(1, min(3, n))
    for j in range(k):
        remaining = k - j - 1
        p = rng.randint(cursor, n - 1 - remaining)
        v = x[p]
        s = p
        while s - 1 >= cursor and x[s - 1] != v and p - s < 2:
            s -= 1
        e = p
        while e + 1 <= n - 1 - remaining and x[e + 1] != v and e - p < 2:
            e += 1
        wins.append((s, e, v))
        cursor = e + 1
    return wins


def brute_force_optimum_sync(targets, lo, hi, step, max_ramps, anchors,
                             windows):
    """Brute-force optimum with the sync-window exactly-one constraint."""
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
               for s, e, v in windows):
            continue
        errs = [abs(x[i] - targets[i]) for i in range(n)]
        key = (max(errs), sum(errs), r, x)
        if best is None or key < best[0]:
            best = (key, list(x))
    return best


class SyncWindowValidationTests(unittest.TestCase):
    def _payload(self, **over):
        base = {
            "targets": list(range(12)), "delay_min": 0, "delay_max": 100,
            "max_step": 5, "max_ramps": 3,
            "anchors": [{"index": 0, "value": 0},
                        {"index": 11, "value": 11}],
        }
        base.update(over)
        return base

    def _bad(self, payload, field):
        with self.assertRaises(CompileError) as ctx:
            compile_plan(payload)
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(ctx.exception.details.get("field"), field)

    def test_not_an_array(self):
        self._bad(self._payload(sync_windows={"start": 0}), "sync_windows")

    def test_entry_count_bounds(self):
        self._bad(self._payload(sync_windows=[]), "sync_windows")
        four = [{"start": 3 * k, "end": 3 * k + 1, "value": 3 * k}
                for k in range(4)]
        self._bad(self._payload(sync_windows=four), "sync_windows")

    def test_entry_must_be_object(self):
        self._bad(self._payload(sync_windows=[[0, 1, 2]]), "sync_windows[0]")

    def test_entry_requires_keys(self):
        self._bad(self._payload(sync_windows=[{"start": 0, "end": 1}]),
                  "sync_windows[0]")

    def test_fields_must_be_integers(self):
        self._bad(self._payload(sync_windows=[
            {"start": 0.5, "end": 1, "value": 1}]), "sync_windows[0].start")
        self._bad(self._payload(sync_windows=[
            {"start": 0, "end": True, "value": 1}]), "sync_windows[0].end")
        self._bad(self._payload(sync_windows=[
            {"start": 0, "end": 1, "value": "1"}]), "sync_windows[0].value")

    def test_interval_must_fit_elements(self):
        self._bad(self._payload(sync_windows=[
            {"start": -1, "end": 2, "value": 1}]), "sync_windows[0].start")
        self._bad(self._payload(sync_windows=[
            {"start": 9, "end": 12, "value": 1}]), "sync_windows[0].end")

    def test_reversed_interval(self):
        self._bad(self._payload(sync_windows=[
            {"start": 4, "end": 2, "value": 1}]), "sync_windows[0].end")

    def test_value_must_be_in_global_range(self):
        self._bad(self._payload(sync_windows=[
            {"start": 0, "end": 2, "value": 101}]), "sync_windows[0].value")
        self._bad(self._payload(sync_windows=[
            {"start": 0, "end": 2, "value": -1}]), "sync_windows[0].value")

    def test_windows_must_be_sorted_and_disjoint(self):
        wins = [{"start": 0, "end": 3, "value": 1},
                {"start": 3, "end": 5, "value": 4}]
        self._bad(self._payload(sync_windows=wins), "sync_windows[1].start")
        wins = [{"start": 6, "end": 8, "value": 6},
                {"start": 1, "end": 2, "value": 1}]
        self._bad(self._payload(sync_windows=wins), "sync_windows[1].start")

    def test_adjacent_windows_allowed(self):
        plan = compile_plan(self._payload(sync_windows=[
            {"start": 0, "end": 1, "value": 0},
            {"start": 2, "end": 3, "value": 2}]))
        self.assertEqual(plan["sync_windows"],
                         [{"index": 0, "value": 0},
                          {"index": 2, "value": 2}])

    def test_none_is_treated_as_omitted(self):
        plan = compile_plan(self._payload(sync_windows=None))
        self.assertNotIn("sync_windows", plan)


class SyncWindowCompileTests(unittest.TestCase):
    def _payload(self, **over):
        base = {
            "targets": [5] * 12, "delay_min": 0, "delay_max": 10,
            "max_step": 2, "max_ramps": 4,
            "anchors": [{"index": 0, "value": 5},
                        {"index": 11, "value": 5}],
        }
        base.update(over)
        return base

    def test_window_forces_exactly_one_placement(self):
        plan = compile_plan(self._payload(
            sync_windows=[{"start": 4, "end": 6, "value": 7}]))
        x = plan["delays"]
        hits = [i for i in range(4, 7) if x[i] == 7]
        self.assertEqual(len(hits), 1)
        self.assertEqual(plan["sync_windows"],
                         [{"index": hits[0], "value": 7}])
        # the flat all-5 optimum is blocked, so the error budget must move
        self.assertEqual(plan["max_abs_error"], 2)
        self.assertTrue(all(abs(x[i + 1] - x[i]) <= 2 for i in range(11)))
        self.assertLessEqual(plan["ramp_count"], 4)

    def test_singleton_window_pins_the_element(self):
        plan = compile_plan(self._payload(
            sync_windows=[{"start": 5, "end": 5, "value": 3}]))
        self.assertEqual(plan["delays"][5], 3)
        self.assertEqual(plan["sync_windows"], [{"index": 5, "value": 3}])

    def test_multiple_windows_answer_in_request_order(self):
        windows = [(2, 3, 6), (5, 6, 4), (8, 9, 6)]
        plan = compile_plan(self._payload(
            max_ramps=11,
            sync_windows=[{"start": s, "end": e, "value": v}
                          for s, e, v in windows]))
        got = plan["sync_windows"]
        self.assertEqual([g["value"] for g in got], [6, 4, 6])
        x = plan["delays"]
        for (s, e, v), g in zip(windows, got):
            self.assertTrue(s <= g["index"] <= e)
            self.assertEqual(x[g["index"]], v)
            self.assertEqual(sum(1 for i in range(s, e + 1) if x[i] == v), 1)

    def test_anchor_inside_window_absorbs_the_placement(self):
        # The anchor at 0 forces x[0] == 0, so the window's single placement
        # is consumed there and no other element in [0,2] may be 0.
        plan = compile_plan(self._payload(
            targets=[0] * 12, max_step=1, max_ramps=11,
            anchors=[{"index": 0, "value": 0}, {"index": 11, "value": 0}],
            sync_windows=[{"start": 0, "end": 2, "value": 0}]))
        self.assertEqual(plan["sync_windows"], [{"index": 0, "value": 0}])
        self.assertNotEqual(plan["delays"][1], 0)
        self.assertNotEqual(plan["delays"][2], 0)

    def test_response_omits_key_without_windows(self):
        plan = compile_plan(self._payload())
        self.assertNotIn("sync_windows", plan)

    def test_deterministic_with_windows(self):
        p = self._payload(sync_windows=[{"start": 4, "end": 6, "value": 7}])
        self.assertEqual(compile_plan(p), compile_plan(p))


class SyncWindowInfeasibleTests(unittest.TestCase):
    BASE = dict(targets=[0] * 12, delay_min=0, delay_max=10,
                max_step=1, max_ramps=11,
                anchors=[{"index": 0, "value": 0},
                         {"index": 11, "value": 0}])

    def test_value_unreachable_in_window(self):
        p = dict(self.BASE, sync_windows=[{"start": 0, "end": 2, "value": 3}])
        with self.assertRaises(CompileError) as ctx:
            compile_plan(p)
        self.assertEqual(ctx.exception.status, 422)
        conf = ctx.exception.details["conflicts"]
        self.assertEqual(len(conf), 1)
        c = conf[0]
        self.assertEqual(c["kind"], "sync_window")
        self.assertEqual((c["start"], c["end"]), (0, 2))
        self.assertEqual(c["reason"], "value_unreachable_in_window")
        self.assertNotIn("delays", ctx.exception.details)
        self.assertNotIn("ramps", ctx.exception.details)

    def test_value_forced_twice_by_anchors(self):
        p = dict(self.BASE,
                 anchors=[{"index": 2, "value": 5},
                          {"index": 3, "value": 5},
                          {"index": 11, "value": 0}],
                 sync_windows=[{"start": 1, "end": 4, "value": 5}])
        with self.assertRaises(CompileError) as ctx:
            compile_plan(p)
        self.assertEqual(ctx.exception.status, 422)
        conf = ctx.exception.details["conflicts"]
        self.assertEqual(len(conf), 1)
        c = conf[0]
        self.assertEqual(c["kind"], "sync_window")
        self.assertEqual((c["start"], c["end"]), (1, 4))
        self.assertEqual(c["reason"], "value_forced_at_multiple_elements")

    def test_ramp_budget_interaction(self):
        # Feasible without the window (flat zeros, one ramp); the forced
        # placement needs at least three ramps.
        p = dict(self.BASE, max_ramps=2,
                 sync_windows=[{"start": 5, "end": 5, "value": 2}])
        with self.assertRaises(CompileError) as ctx:
            compile_plan(p)
        self.assertEqual(ctx.exception.status, 422)
        conf = ctx.exception.details["conflicts"]
        self.assertEqual(len(conf), 1)
        c = conf[0]
        self.assertEqual(c["kind"], "sync_window")
        self.assertEqual((c["start"], c["end"]), (5, 5))
        self.assertEqual(c["reason"], "infeasible_combination")
        self.assertNotIn("delays", ctx.exception.details)
        ok = dict(p)
        del ok["sync_windows"]
        self.assertEqual(compile_plan(ok)["delays"], [0] * 12)
        plan = compile_plan(dict(p, max_ramps=3))
        self.assertEqual(plan["delays"][5], 2)
        self.assertEqual(plan["sync_windows"], [{"index": 5, "value": 2}])

    def test_joint_window_conflict_reports_both_ranges(self):
        # Each window is satisfiable alone; together they need |3-1| in one
        # step.
        p = dict(self.BASE, sync_windows=[
            {"start": 2, "end": 2, "value": 1},
            {"start": 3, "end": 3, "value": 3}])
        with self.assertRaises(CompileError) as ctx:
            compile_plan(p)
        self.assertEqual(ctx.exception.status, 422)
        ranges = {(c["start"], c["end"])
                  for c in ctx.exception.details["conflicts"]}
        self.assertEqual(ranges, {(2, 2), (3, 3)})

    def test_infeasible_errors_are_stable_with_windows(self):
        p = dict(self.BASE, sync_windows=[{"start": 0, "end": 2, "value": 3}])
        first = None
        for _ in range(3):
            try:
                compile_plan(p)
                self.fail("expected infeasibility")
            except CompileError as e:
                self.assertEqual(e.status, 422)
                blob = json.dumps(e.details, sort_keys=True)
                if first is None:
                    first = blob
                self.assertEqual(blob, first)


class SyncBruteForceTests(unittest.TestCase):
    def setUp(self):
        self._saved = (compiler.MIN_ELEMENTS, compiler.MIN_ANCHORS)
        compiler.MIN_ELEMENTS = 4
        compiler.MIN_ANCHORS = 2

    def tearDown(self):
        compiler.MIN_ELEMENTS, compiler.MIN_ANCHORS = self._saved

    def test_random_sync_instances_match_brute_force(self):
        rng = random.Random(20261005)
        trials = 120
        for trial in range(trials):
            n = rng.randint(4, 6)
            lo, hi = -2, 3
            step = rng.randint(1, 3)
            ai = sorted(rng.sample(range(n), 2))
            av = [rng.randint(lo, hi)]
            av.append(rng.randint(av[0] - step * (ai[1] - ai[0]),
                                  av[0] + step * (ai[1] - ai[0])))
            av[1] = max(lo, min(hi, av[1]))
            anchors = {ai[0]: av[0], ai[1]: av[1]}
            targets = [rng.randint(lo - 1, hi + 1) for _ in range(n)]
            max_ramps = rng.randint(1, n - 1)
            if rng.random() < 0.5:
                windows = _random_windows(rng, n, lo, hi)
            else:
                windows = _planted_windows(rng, n, anchors, step)
            payload = {
                "targets": targets, "delay_min": lo, "delay_max": hi,
                "max_step": step, "max_ramps": max_ramps,
                "anchors": [{"index": i, "value": v}
                            for i, v in anchors.items()],
                "sync_windows": [{"start": s, "end": e, "value": v}
                                 for s, e, v in windows],
            }
            bf = brute_force_optimum_sync(targets, lo, hi, step,
                                          max_ramps, anchors, windows)
            if bf is None:
                with self.assertRaises(CompileError) as ctx:
                    compile_plan(payload)
                self.assertEqual(ctx.exception.status, 422,
                                 msg=f"trial {trial}: {payload}")
                continue
            plan = compile_plan(payload)
            x = plan["delays"]
            key = (plan["max_abs_error"], plan["total_abs_error"],
                   plan["ramp_count"], tuple(x))
            self.assertEqual(key, bf[0],
                             msg=f"trial {trial}: {payload}\n{plan}")
            self.assertEqual(x, bf[1], msg=f"lex tie trial {trial}")
            got = plan["sync_windows"]
            self.assertEqual(len(got), len(windows))
            for (s, e, v), entry in zip(windows, got):
                self.assertEqual(entry["value"], v)
                self.assertTrue(s <= entry["index"] <= e)
                self.assertEqual(x[entry["index"]], v)
                self.assertEqual(
                    sum(1 for i in range(s, e + 1) if x[i] == v), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
