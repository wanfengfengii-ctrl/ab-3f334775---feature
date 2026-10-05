"""Integer delay-plan compiler.

Compiles per-element target delays into a small number of integer *ramps*
that the probe firmware can apply.

Definitions (all arithmetic is exact integer arithmetic)
--------------------------------------------------------
A sequence ``x[0..n-1]`` is feasible when:

* every ``x[i]`` lies in the closed global delay interval ``[lo, hi]``;
* every anchor ``x[index] == value`` holds exactly;
* ``|x[i+1] - x[i]| <= max_step``;
* the number of maximal constant runs (ramps) of the adjacent-difference
  sequence ``d[i] = x[i+1] - x[i]`` does not exceed ``max_ramps``.

Optimization order, lexicographic (the first differing key decides):

1. minimize the maximum absolute error ``max_i |x[i] - target[i]|``;
2. then minimize the total absolute error ``sum_i |x[i] - target[i]|``;
3. then minimize the number of ramps actually used;
4. then minimize the delay sequence itself in lexicographic order.

Algorithms
----------
* Structural feasibility (global bounds, anchor cones under the step limit)
  is propagated as per-position integer bands; a violated anchor pair is
  reported as a localized conflict interval.
* The minimax error is found with exponential search followed by binary
  search; each feasibility test is a dynamic program over
  ``(position, value, last delta)`` states minimizing the ramp count.
  Transition cost is obtained in O(1) per predecessor value using the
  smallest / second-smallest predecessor ramp count, so each layer costs
  O(W * (2*max_step + 1)).
* With the optimal error budget fixed, a backward DP computes the best
  suffix cost ``(sum abs error, new ramps)`` for every state and the plan is
  recovered greedily, which yields the lexicographically smallest optimum.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_ELEMENTS = 12
MAX_ELEMENTS = 48
MIN_ANCHORS = 2
MAX_ANCHORS = 8

# Safety valve for pathological integer domains (firmware delay values are
# bounded in practice).  A DP working window wider than this many integer
# values, or a layer transition more expensive than this many predecessor
# checks, is rejected as unsupported rather than stalling the service.
MAX_WINDOW = 8_192
MAX_LAYER_OPS = 250_000


class CompileError(ValueError):
    """Invalid request or infeasible compilation.

    ``status`` is the suggested HTTP status: 400 for malformed input,
    422 for well-formed but infeasible instances, 500 for internal errors.
    """

    def __init__(self, message: str, status: int = 400, details=None):
        super().__init__(message)
        self.status = status
        self.details = details or {}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _as_int(name, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompileError(f"'{name}' must be an integer", 400, {"field": name})
    return value


@dataclass(frozen=True)
class Conflict:
    """An infeasibility interval.

    ``start``/``end`` are element indices.  For a conflict between two
    consecutive anchors they are the two anchor indices; for an anchor that
    violates the global delay interval, ``start == end``.
    """

    kind: str
    start: int
    end: int
    detail: dict

    def to_dict(self) -> dict:
        out = {"kind": self.kind, "start": self.start, "end": self.end}
        out.update(self.detail)
        return out


def validate_request(payload) -> dict:
    """Validate and normalize a compile request payload."""
    if not isinstance(payload, dict):
        raise CompileError("request body must be a JSON object")

    targets = payload.get("targets")
    if not isinstance(targets, list):
        raise CompileError("'targets' must be an array of integers", 400,
                           {"field": "targets"})
    n = len(targets)
    if not (MIN_ELEMENTS <= n <= MAX_ELEMENTS):
        raise CompileError(
            f"'targets' must contain between {MIN_ELEMENTS} and {MAX_ELEMENTS} "
            f"elements (got {n})", 400, {"field": "targets", "length": n})
    int_targets = [_as_int(f"targets[{i}]", v) for i, v in enumerate(targets)]

    if "delay_min" not in payload or "delay_max" not in payload:
        raise CompileError("'delay_min' and 'delay_max' are required", 400)
    lo = _as_int("delay_min", payload["delay_min"])
    hi = _as_int("delay_max", payload["delay_max"])
    if lo > hi:
        raise CompileError("'delay_min' must not exceed 'delay_max'", 400,
                           {"field": "delay_min"})

    max_step = _as_int("max_step", payload.get("max_step"))
    if max_step < 0:
        raise CompileError("'max_step' must be non-negative", 400,
                           {"field": "max_step"})

    max_ramps = _as_int("max_ramps", payload.get("max_ramps"))
    if not (1 <= max_ramps <= n - 1):
        raise CompileError(
            f"'max_ramps' must be between 1 and {n - 1} for {n} elements",
            400, {"field": "max_ramps"})

    anchors_raw = payload.get("anchors")
    if not isinstance(anchors_raw, list):
        raise CompileError("'anchors' must be an array", 400,
                           {"field": "anchors"})
    if not (MIN_ANCHORS <= len(anchors_raw) <= MAX_ANCHORS):
        raise CompileError(
            f"'anchors' must contain between {MIN_ANCHORS} and {MAX_ANCHORS} "
            f"entries (got {len(anchors_raw)})", 400, {"field": "anchors"})

    anchors = {}
    for k, a in enumerate(anchors_raw):
        if not isinstance(a, dict):
            raise CompileError(f"anchors[{k}] must be an object", 400,
                               {"field": f"anchors[{k}]"})
        if "index" not in a or "value" not in a:
            raise CompileError(
                f"anchors[{k}] requires 'index' and 'value'", 400,
                {"field": f"anchors[{k}]"})
        idx = _as_int(f"anchors[{k}].index", a["index"])
        val = _as_int(f"anchors[{k}].value", a["value"])
        if not (0 <= idx < n):
            raise CompileError(
                f"anchors[{k}].index={idx} out of range [0,{n - 1}]", 400,
                {"field": f"anchors[{k}].index"})
        if idx in anchors and anchors[idx] != val:
            raise CompileError(
                f"conflicting anchor values at index {idx}", 400,
                {"field": f"anchors[{k}].index", "index": idx})
        anchors[idx] = val

    return {
        "n": n,
        "targets": int_targets,
        "lo": lo,
        "hi": hi,
        "max_step": max_step,
        "max_ramps": max_ramps,
        "anchors": anchors,
    }


# ---------------------------------------------------------------------------
# Structural feasibility: bands from global bounds + anchor cones
# ---------------------------------------------------------------------------


@dataclass
class StructuralResult:
    feasible: bool
    conflicts: list
    bands: list  # [(lo_i, hi_i), ...] feasible integer value per position


def structural_check(req: dict) -> StructuralResult:
    n = req["n"]
    lo, hi = req["lo"], req["hi"]
    step = req["max_step"]
    anchors = req["anchors"]
    anchor_items = sorted(anchors.items())
    conflicts: list[Conflict] = []

    for idx, val in anchor_items:
        if val < lo or val > hi:
            conflicts.append(Conflict(
                "anchor_out_of_bounds", idx, idx,
                {"anchor_value": val, "delay_min": lo, "delay_max": hi}))
    if conflicts:
        return StructuralResult(False, conflicts, [(lo, hi)] * n)

    # NOTE: bands are deliberately *not* clipped to the extrema of targets and
    # anchor values: an optimal ramp fit can legitimately take values beyond
    # both (e.g. a shallow V between equal anchors whose targets sit at/above
    # the anchor value).  The error tube used by the DPs keeps every working
    # domain to width <= 2*E + 1 regardless.
    blo = [lo] * n
    bhi = [hi] * n

    first_idx, first_val = anchor_items[0]
    for i in range(first_idx):
        dist = first_idx - i
        blo[i] = max(blo[i], first_val - step * dist)
        bhi[i] = min(bhi[i], first_val + step * dist)
    blo[first_idx] = bhi[first_idx] = first_val

    unreachable_spans = []
    for (i0, v0), (i1, v1) in zip(anchor_items, anchor_items[1:]):
        gap = i1 - i0
        need = abs(v1 - v0)
        if need > step * gap:
            conflicts.append(Conflict(
                "step_unreachable", i0, i1,
                {"from_value": v0, "to_value": v1, "steps": gap,
                 "required_min_step": -(-need // gap),
                 "max_step": step, "min_total_change": need,
                 "max_total_change": step * gap}))
            unreachable_spans.append((i0, i1))
        for j in range(i0, i1 + 1):
            d1 = j - i0
            d2 = i1 - j
            blo[j] = max(blo[j], v0 - step * d1, v1 - step * d2)
            bhi[j] = min(bhi[j], v0 + step * d1, v1 + step * d2)

    last_idx, last_val = anchor_items[-1]
    for i in range(last_idx + 1, n):
        dist = i - last_idx
        blo[i] = max(blo[i], last_val - step * dist)
        bhi[i] = min(bhi[i], last_val + step * dist)

    bands = []
    for i in range(n):
        a, b = blo[i], bhi[i]
        if a > b and not any(s <= i <= e for s, e in unreachable_spans):
            conflicts.append(Conflict(
                "empty_band", i, i,
                {"delay_min": lo, "delay_max": hi, "max_step": step}))
        bands.append((a, b))

    return StructuralResult(not conflicts, conflicts, bands)


# ---------------------------------------------------------------------------
# Feasibility DP (minimize ramp count inside an absolute-error tube)
# ---------------------------------------------------------------------------


def _widths_under_budget(req, bands, budget):
    """Per-position integer ranges inside both the band and the error tube."""
    widths = []
    t = req["targets"]
    anchors = req["anchors"]
    for i, (a, b) in enumerate(bands):
        l = max(a, t[i] - budget)
        h = min(b, t[i] + budget)
        if i in anchors:
            v = anchors[i]
            if not (l <= v <= h):
                return None
            l = h = v
        if l > h:
            return None
        if h - l + 1 > MAX_WINDOW:
            raise CompileError(
                "integer delay domain too large to compile exactly "
                f"(more than {MAX_WINDOW} admissible values at element {i}); "
                "tighten 'delay_min'/'delay_max' or reduce the target spread",
                400, {"field": "delay_min", "element": i,
                      "window_width": h - l + 1})
        widths.append((l, h))
    return widths


def _best_two(p: dict):
    """Smallest value + its key, and the smallest value at another key."""
    m1 = None
    k1 = None
    m2 = None
    for k, v in p.items():
        if m1 is None or v < m1:
            m2 = m1
            m1, k1 = v, k
        elif m2 is None or v < m2:
            m2 = v
    return m1, k1, m2


def feasible_with_widths(req, widths) -> bool:
    """True iff some sequence fits ``widths`` while using <= max_ramps ramps."""
    step = req["max_step"]
    cap = req["max_ramps"]

    # prev[v] maps last-delta -> ramps used so far; None is the position-0
    # sentinel so the first concrete edge always starts ramp number one.
    prev = {v: {None: 0} for v in range(widths[0][0], widths[0][1] + 1)}
    stats = {u: _best_two(p) for u, p in prev.items()}

    for i in range(1, req["n"]):
        lo_w, hi_w = widths[i]
        plo, phi = widths[i - 1]
        if (hi_w - lo_w + 1) * min(2 * step + 1, phi - plo + 1) > MAX_LAYER_OPS:
            raise CompileError(
                "integer delay domain too large to compile exactly at "
                f"element {i}; tighten 'delay_min'/'delay_max' or reduce "
                "the target spread", 400, {"element": i})
        cur = {}
        for v in range(lo_w, hi_w + 1):
            entry = {}
            ua = max(plo, v - step)
            ub = min(phi, v + step)
            for u in range(ua, ub + 1):
                st = stats.get(u)
                if st is None:
                    continue
                m1, k1, m2 = st
                d = v - u
                cont = prev[u].get(d)              # keep the current ramp
                brk = m1 if k1 != d else m2         # start a new ramp here
                if brk is not None:
                    brk += 1
                best = None
                if cont is not None:
                    best = cont
                if brk is not None and (best is None or brk < best):
                    best = brk
                if best is not None and best <= cap:
                    old = entry.get(d)
                    if old is None or best < old:
                        entry[d] = best
            if entry:
                cur[v] = entry
        if not cur:
            return False
        prev = cur
        stats = {u: _best_two(p) for u, p in prev.items()}
    return True


def ramp_feasible_on_bands(req, bands):
    """Decide ramp-budget feasibility with no error tube.

    Returns True/False when the exact band DP fits the configured domain
    guard, or ``None`` when the integer domain is too large to decide here.
    """
    widths = []
    for i, (a, b) in enumerate(bands):
        if b - a + 1 > MAX_WINDOW:
            return None
        widths.append((a, b))
    try:
        return feasible_with_widths(req, widths)
    except CompileError:
        return None


# ---------------------------------------------------------------------------
# Backward optimization DP: (sum abs error, new ramps) suffix costs
# ---------------------------------------------------------------------------


def _incoming_keys(widths, i, step):
    """Map each value at position i to the possible deltas of edge (i-1)."""
    if i == 0:
        return {v: (None,) for v in range(widths[0][0], widths[0][1] + 1)}
    plo, phi = widths[i - 1]
    keys = {}
    for v in range(widths[i][0], widths[i][1] + 1):
        lo_u = max(plo, v - step)
        hi_u = min(phi, v + step)
        keys[v] = tuple(v - u for u in range(lo_u, hi_u + 1))
    return keys


def optimal_plan(req, widths) -> list:
    """Recover the lexicographically smallest optimal sequence.

    Assumes the feasibility DP has already proved that ``widths`` admits a
    sequence within the ramp budget.
    """
    n = req["n"]
    step = req["max_step"]
    cap = req["max_ramps"]
    targets = req["targets"]

    def vals(i):
        return range(widths[i][0], widths[i][1] + 1)

    # G[i][v] maps state (din, b) -> (S, T), where ``din`` is the delta of
    # the edge entering position i (None at i == 0) and ``b`` is the number
    # of *new* ramps still allowed on edges i..n-2.  S is the minimum sum of
    # absolute errors at positions i..n-1 over continuations using T <= b new
    # ramps; ties on S are broken by smaller T.
    G = [None] * n

    base = {}
    keys_last = _incoming_keys(widths, n - 1, step)
    for v in vals(n - 1):
        e = abs(v - targets[n - 1])
        base[v] = {(d, b): (e, 0) for d in keys_last[v]
                   for b in range(cap + 1)}
    G[n - 1] = base

    for i in range(n - 2, -1, -1):
        nlo, nhi = widths[i + 1]
        keys_here = _incoming_keys(widths, i, step)
        layer = {}
        for v in vals(i):
            ev = abs(v - targets[i])
            wa = max(nlo, v - step)
            wb = min(nhi, v + step)
            feasible_w = list(range(wa, wb + 1))

            # For every remaining budget b >= 1, precompute the best and the
            # second-best successor (distinct values) for starting a new ramp
            # at edge i, compared lexicographically on (S, T, w).
            best_new = {}
            for b in range(1, cap + 1):
                w1 = None
                p1 = None
                w2 = None
                p2 = None
                for w in feasible_w:
                    st = G[i + 1].get(w, {}).get((w - v, b - 1))
                    if st is None:
                        continue
                    cand = (st[0], st[1], w)
                    if p1 is None or cand < p1:
                        w2, p2 = w1, p1
                        w1, p1 = w, cand
                    elif w != w1 and (p2 is None or cand < p2):
                        w2, p2 = w, cand
                if p1 is not None:
                    best_new[b] = (p1, w1, p2, w2)

            cell = {}
            for din in keys_here[v]:
                cont_w = None if din is None else v + din
                for b in range(cap + 1):
                    best = None
                    # Continue the incoming ramp (edge delta == din).
                    if cont_w is not None:
                        st = G[i + 1].get(cont_w, {}).get((din, b))
                        if st is not None:
                            best = (ev + st[0], st[1])
                    # Start a new ramp at edge i.
                    if b >= 1:
                        pre = best_new.get(b)
                        if pre is not None:
                            p1, w1, p2, w2 = pre
                            s0, s1, bw = p1 if w1 != cont_w else (
                                p2 if p2 is not None else (None, None, None))
                            if s0 is not None:
                                cand = (ev + s0, 1 + s1)
                                if best is None or cand < best:
                                    best = cand
                    if best is not None:
                        cell[(din, b)] = best
            if cell:
                layer[v] = cell
        G[i] = layer

    # Greedy left-to-right: the prefix is fixed, so among feasible next
    # values minimizing the stored suffix pair and then the value itself
    # yields the globally lexicographically smallest optimum.
    x = [0] * n
    prev_din = None
    used_ramps = 0
    for i in range(n):
        if i == 0:
            choices = []
            for v in vals(0):
                st = G[0].get(v, {}).get((None, cap))
                if st is not None:
                    choices.append((st[0], st[1], v))
            if not choices:  # pragma: no cover - guarded by feasibility DP
                raise CompileError("internal reconstruction failure", 500)
            _, _, vi = min(choices, key=lambda z: (z[0], z[1], z[2]))
            x[0] = vi
            continue

        choices = []
        for v in vals(i):
            d = v - x[i - 1]
            if abs(d) > step:
                continue
            extra = 0 if d == prev_din else 1
            b_remaining = cap - used_ramps - extra
            if b_remaining < 0:
                continue
            st = G[i].get(v, {}).get((d, b_remaining))
            if st is None:
                continue
            total_ramps = used_ramps + extra + st[1]
            choices.append((st[0], total_ramps, v, d))
        if not choices:  # pragma: no cover - guarded by feasibility DP
            raise CompileError("internal reconstruction failure", 500)
        _, _, vi, di = min(choices, key=lambda z: (z[0], z[1], z[2]))
        if di != prev_din:
            used_ramps += 1
        prev_din = di
        x[i] = vi
    return x


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _ramp_budget_conflict(req):
    items = sorted(req["anchors"].items())
    segments = [
        {"start": i0, "end": i1, "total_change": v1 - v0, "steps": i1 - i0}
        for (i0, v0), (i1, v1) in zip(items, items[1:])
    ]
    return {"kind": "ramp_budget", "start": 0, "end": req["n"] - 1,
            "max_ramps": req["max_ramps"], "anchor_segments": segments}


def ramp_boundaries(x: list) -> list:
    """Maximal equal-difference runs as [{start, end, delta}, ...].

    ``start``/``end`` are the spanned element indices (both inclusive).
    """
    if len(x) < 2:
        return []
    ramps = []
    run_start = 0
    d = x[1] - x[0]
    for i in range(1, len(x) - 1):
        nd = x[i + 1] - x[i]
        if nd != d:
            ramps.append({"start": run_start, "end": i, "delta": d})
            run_start = i
            d = nd
    ramps.append({"start": run_start, "end": len(x) - 1, "delta": d})
    return ramps


def _saturation_budget(req, bands):
    """Smallest E for which the error tube contains every structural band."""
    t = req["targets"]
    e = 0
    for i, (a, b) in enumerate(bands):
        e = max(e, abs(a - t[i]), abs(b - t[i]))
    return e


def _probe(req, bands, budget):
    """Feasibility at ``budget``.

    Returns True (feasible), False (proven infeasible within the exact
    domain), or None (the exact DP domain exceeds the configured guard).
    """
    widths = _widths_under_budget(req, bands, budget)
    if widths is None:
        return False
    try:
        return feasible_with_widths(req, widths)
    except CompileError:
        return None


def compile_plan(payload) -> dict:
    """Compile a request payload into an optimal integer delay plan.

    Raises :class:`CompileError` for malformed input (400) or infeasible
    instances (422).  Infeasible responses never contain a delay table.
    """
    req = validate_request(payload)
    sr = structural_check(req)
    if not sr.feasible:
        raise CompileError(
            "no feasible delay plan: anchor/step conflict", 422,
            {"conflicts": [c.to_dict() for c in sr.conflicts]})
    bands = sr.bands
    t = req["targets"]
    esat = _saturation_budget(req, bands)

    # Ramp-budget feasibility without any error tube: once bands are fully
    # covered, ramp count is the only remaining restriction.
    if ramp_feasible_on_bands(req, bands) is False:
        raise CompileError(
            "no feasible delay plan within the ramp budget", 422,
            {"conflicts": [_ramp_budget_conflict(req)]})

    # Binary search the optimum budget within the exactly decidable domain.
    # Probe outcomes: True feasible, False infeasible, None means the exact
    # DP domain exceeds the configured guard.  The saturation budget is
    # feasible (ramp feasibility was established above); if probing it raises
    # the guard, locate the largest decidable budget and search below it.
    if _probe(req, bands, esat) is None:
        if _probe(req, bands, 0) is None:  # pragma: no cover - defensive
            raise _guard_refusal(req)
        safe, hi_guard = 0, esat
        while safe + 1 < hi_guard:
            mid = (safe + hi_guard) // 2
            if _probe(req, bands, mid) is None:
                hi_guard = mid
            else:
                safe = mid
        hi_e = safe
    else:
        hi_e = esat
    lo_e = 0

    while lo_e < hi_e:
        mid = (lo_e + hi_e) // 2
        verdict = _probe(req, bands, mid)
        if verdict is True:
            hi_e = mid
        else:
            # False: budget too small.  None cannot occur here because both
            # ends of the active interval are decidable (guard lies above).
            lo_e = mid + 1
    e_star = lo_e

    widths = _widths_under_budget(req, bands, e_star)
    if widths is None or not feasible_with_widths(req, widths):
        # Happens only when the optimum sits above every decidable budget.
        raise _guard_refusal(req)

    x = optimal_plan(req, widths)
    errors = [x[i] - t[i] for i in range(req["n"])]
    ramps = ramp_boundaries(x)
    return {
        "n": req["n"],
        "delays": x,
        "errors": errors,
        "ramps": ramps,
        "ramp_count": len(ramps),
        "max_abs_error": max(abs(e) for e in errors),
        "total_abs_error": sum(abs(e) for e in errors),
    }


def _guard_refusal(req):
    return CompileError(
        "integer delay domain too large to compile exactly; tighten "
        "'delay_min'/'delay_max' or reduce the target spread", 400,
        {"field": "delay_min"})
