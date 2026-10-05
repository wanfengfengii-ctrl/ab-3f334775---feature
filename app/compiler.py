"""Integer delay-plan compiler.

Compiles per-element target delays into a small number of integer *ramps*
that the probe firmware can apply.

Definitions (all arithmetic is exact integer arithmetic)
--------------------------------------------------------
A sequence ``x[0..n-1]`` is feasible when:

* every ``x[i]`` lies in the closed global delay interval ``[lo, hi]``;
* every anchor ``x[index] == value`` holds exactly;
* ``|x[i+1] - x[i]| <= max_step``;
* for every optional *sync window* ``[start, end]`` with value ``v``
  (windows are pairwise disjoint closed index intervals), exactly one
  index ``i`` in the window satisfies ``x[i] == v``;
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
MIN_SYNC_WINDOWS = 1
MAX_SYNC_WINDOWS = 3

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

    sync_windows = _validate_sync_windows(
        payload.get("sync_windows", None), n, lo, hi)

    return {
        "n": n,
        "targets": int_targets,
        "lo": lo,
        "hi": hi,
        "max_step": max_step,
        "max_ramps": max_ramps,
        "anchors": anchors,
        "sync_windows": sync_windows,
    }


def _validate_sync_windows(raw, n, lo, hi):
    """Validate the optional ``sync_windows`` field.

    Returns ``None`` when the field is omitted (legacy-compatible behavior)
    or a list of ``(start, end, value)`` tuples in request order.  Windows
    must number 1..3, be closed zero-based element intervals sorted by
    ``start`` with strictly increasing boundaries (hence non-overlapping),
    and carry an integer ``value`` within the global ``[lo, hi]`` range.
    """
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise CompileError("'sync_windows' must be an array", 400,
                           {"field": "sync_windows"})
    if not (MIN_SYNC_WINDOWS <= len(raw) <= MAX_SYNC_WINDOWS):
        raise CompileError(
            f"'sync_windows' must contain between {MIN_SYNC_WINDOWS} and "
            f"{MAX_SYNC_WINDOWS} entries (got {len(raw)})", 400,
            {"field": "sync_windows", "length": len(raw)})

    windows = []
    for k, w in enumerate(raw):
        if not isinstance(w, dict):
            raise CompileError(f"sync_windows[{k}] must be an object", 400,
                               {"field": f"sync_windows[{k}]"})
        for key in ("start", "end", "value"):
            if key not in w:
                raise CompileError(
                    f"sync_windows[{k}] requires 'start', 'end' and 'value'",
                    400, {"field": f"sync_windows[{k}].{key}"})
        start = _as_int(f"sync_windows[{k}].start", w["start"])
        end = _as_int(f"sync_windows[{k}].end", w["end"])
        value = _as_int(f"sync_windows[{k}].value", w["value"])
        if not (0 <= start < n):
            raise CompileError(
                f"sync_windows[{k}].start={start} out of range [0,{n - 1}]",
                400, {"field": f"sync_windows[{k}].start", "index": start})
        if not (0 <= end < n):
            raise CompileError(
                f"sync_windows[{k}].end={end} out of range [0,{n - 1}]",
                400, {"field": f"sync_windows[{k}].end", "index": end})
        if start > end:
            raise CompileError(
                f"sync_windows[{k}].start ({start}) must not exceed "
                f"sync_windows[{k}].end ({end})", 400,
                {"field": f"sync_windows[{k}].start", "start": start,
                 "end": end})
        if not (lo <= value <= hi):
            raise CompileError(
                f"sync_windows[{k}].value={value} outside the global delay "
                f"range [{lo},{hi}]", 400,
                {"field": f"sync_windows[{k}].value", "value": value,
                 "delay_min": lo, "delay_max": hi})
        if k > 0:
            p_start, p_end, _ = windows[k - 1]
            if start < p_start:
                raise CompileError(
                    "sync_windows must be sorted by 'start' in ascending "
                    f"order (sync_windows[{k - 1}].start={p_start}, "
                    f"sync_windows[{k}].start={start})", 400,
                    {"field": f"sync_windows[{k}].start",
                     "previous_start": p_start})
            if start <= p_end:
                raise CompileError(
                    f"sync_windows[{k}] overlaps sync_windows[{k - 1}]: "
                    f"[{start},{end}] vs [{p_start},{p_end}]", 400,
                    {"field": f"sync_windows[{k}].start",
                     "start": start, "previous_end": p_end})
        windows.append((start, end, value))
    return windows


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


def _sync_hit_map(req):
    """Map each position covered by a sync window to ``(window_index, value)``.

    Windows are disjoint (validated upstream), so every covered position maps
    to exactly one window.  Positions outside any window are absent.
    """
    hits = {}
    windows = req.get("sync_windows")
    if not windows:
        return hits
    for k, (start, end, value) in enumerate(windows):
        for i in range(start, end + 1):
            hits[i] = (k, value)
    return hits


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
    if req.get("sync_windows"):
        return _feasible_with_widths_sync(req, widths)
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


def _feasible_with_widths_sync(req, widths) -> bool:
    """Feasibility DP honoring the exactly-one-hit sync windows.

    The state is ``prev[v][delta][status] = ramps used`` (the incoming edge
    delta is ``None`` at position 0).  ``status`` is either ``None`` (before
    the first window, in a gap, or after the last window -- every window
    passed so far was hit exactly once) or ``(k, c)`` (currently inside
    window ``k`` with ``c`` hits so far, ``c in {0, 1}``).  At most one
    window is active at any position, so the extra dimension costs at most a
    factor of two; transitions between statuses are deterministic given the
    positions of the two endpoints.
    """
    step = req["max_step"]
    cap = req["max_ramps"]
    hit_map = _sync_hit_map(req)

    def win_at(i):
        return hit_map.get(i)

    # Best / second-best ramp count over incoming deltas, per summary status.
    def build_stats(layer):
        out = {}
        for u, entry in layer.items():
            by_status = {}
            for d0, sums in entry.items():
                for s0, r in sums.items():
                    bucket = by_status.setdefault(s0, [])
                    bucket.append((d0, r))
            stats_u = {}
            for s0, pairs in by_status.items():
                m1 = k1 = m2 = None
                for d0, r in pairs:
                    if m1 is None or r < m1:
                        m2 = m1
                        m1, k1 = r, d0
                    elif m2 is None or r < m2:
                        m2 = r
                stats_u[s0] = (m1, k1, m2)
            out[u] = stats_u
        return out

    lo0, hi0 = widths[0]
    w0 = win_at(0)
    prev = {}
    for v in range(lo0, hi0 + 1):
        if w0 is None:
            prev[v] = {None: {None: 0}}
        else:
            k, wval = w0
            c = 1 if v == wval else 0
            prev[v] = {None: {(k, c): 0}}

    for i in range(1, req["n"]):
        lo_w, hi_w = widths[i]
        plo, phi = widths[i - 1]
        if (hi_w - lo_w + 1) * min(2 * step + 1, phi - plo + 1) * 2 > MAX_LAYER_OPS:
            raise CompileError(
                "integer delay domain too large to compile exactly at "
                f"element {i}; tighten 'delay_min'/'delay_max' or reduce "
                "the target spread", 400, {"element": i})
        win = win_at(i)
        pwin = win_at(i - 1)
        stats = build_stats(prev)
        cur = {}
        for v in range(lo_w, hi_w + 1):
            ua = max(plo, v - step)
            ub = min(phi, v + step)
            entry = {}
            for u in range(ua, ub + 1):
                p_entry = prev.get(u)
                if p_entry is None:
                    continue
                stats_u = stats[u]
                d = v - u
                for s0, (m1, k1, m2) in stats_u.items():
                    # Determine the successor status (None == between windows).
                    if win is None:
                        if pwin is None:
                            s1 = None            # gap/suffix -> gap/suffix
                        else:
                            # Leaving a window: it must have exactly one hit.
                            if s0 != (pwin[0], 1):
                                continue
                            s1 = None
                    else:
                        k, wval = win
                        if pwin is not None and pwin[0] != k:
                            # Adjacent windows (no gap): the one that just
                            # ended must have exactly one hit.
                            if s0 != (pwin[0], 1):
                                continue
                            c0 = 0
                        elif pwin is None:
                            c0 = 0               # entering a window from a gap
                        else:
                            if s0 is None or s0[0] != k:
                                continue         # pragma: no cover - defensive
                            c0 = s0[1]
                        c1 = c0 + (1 if v == wval else 0)
                        if c1 > 1:
                            continue             # two hits can never recover
                        s1 = (k, c1)
                    cont = p_entry.get(d, {}).get(s0)
                    brk = m1 if k1 != d else m2
                    if brk is not None:
                        brk += 1
                    best = cont
                    if brk is not None and (best is None or brk < best):
                        best = brk
                    if best is None or best > cap:
                        continue
                    slot = entry.setdefault(d, {})
                    old = slot.get(s1)
                    if old is None or best < old:
                        slot[s1] = best
            if entry:
                cur[v] = entry
        if not cur:
            return False
        prev = cur

    # Acceptance: position n-1 must sit in gap/suffix status (all windows
    # closed with exactly one hit) -- or inside the final window with c == 1.
    last_win = win_at(req["n"] - 1)
    wanted = None if last_win is None else (last_win[0], 1)
    for p_entry in prev.values():
        for sums in p_entry.values():
            if wanted in sums:
                return True
    return False


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
    if req.get("sync_windows"):
        return _optimal_plan_sync(req, widths)
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


def _optimal_plan_sync(req, widths) -> list:
    """Backward-optimal recovery with the exactly-one-hit sync windows.

    Mirror of :func:`optimal_plan`; each state additionally carries the
    window-hit status ``s`` at its position: ``None`` in a gap (all earlier
    windows satisfied), or ``(k, c)`` inside window ``k`` with ``c`` hits.
    """
    n = req["n"]
    step = req["max_step"]
    cap = req["max_ramps"]
    targets = req["targets"]
    hit_map = _sync_hit_map(req)

    def win_at(i):
        return hit_map.get(i)

    def vals(i):
        return range(widths[i][0], widths[i][1] + 1)

    def statuses_at(i):
        w = win_at(i)
        return (None,) if w is None else ((w[0], 0), (w[0], 1))

    def incoming_deltas(i, v):
        if i == 0:
            return (None,)
        plo, phi = widths[i - 1]
        lo_u = max(plo, v - step)
        hi_u = min(phi, v + step)
        return tuple(v - u for u in range(lo_u, hi_u + 1))

    def next_status(s, i, w):
        """Window status at position ``i+1`` after choosing value ``w``."""
        wnow = win_at(i)
        wnext = win_at(i + 1)
        if wnext is None:
            if wnow is None:
                return None  # gap/prefix/suffix continues (s is always None)
            # Leaving a window: it must have accumulated exactly one hit.
            return None if s == (wnow[0], 1) else 0
        k, wval = wnext
        if wnow is not None and wnow[0] != k:
            if s != (wnow[0], 1):
                return 0                          # prior window unsatisfied
            c0 = 0
        elif wnow is None:
            if s is not None:
                return 0                          # pragma: no cover - defensive
            c0 = 0
        else:
            if s is None or s[0] != k:
                return 0
            c0 = s[1]
        c1 = c0 + (1 if w == wval else 0)
        return 0 if c1 > 1 else (k, c1)

    def initial_status(i, v):
        w = win_at(i)
        if w is None:
            return None
        return (w[0], 1 if v == w[1] else 0)

    # G[i][v] maps (din, status, b) -> (S, T) suffix cost (see optimal_plan).
    G = [None] * n

    base = {}
    last_win = win_at(n - 1)
    accept = None if last_win is None else (last_win[0], 1)
    keys_last = {v: incoming_deltas(n - 1, v) for v in vals(n - 1)}
    for v in vals(n - 1):
        e = abs(v - targets[n - 1])
        base[v] = {(d, accept, b): (e, 0)
                   for d in keys_last[v] for b in range(cap + 1)}
    G[n - 1] = base

    for i in range(n - 2, -1, -1):
        nlo, nhi = widths[i + 1]
        layer = {}
        for v in vals(i):
            ev = abs(v - targets[i])
            wa = max(nlo, v - step)
            wb = min(nhi, v + step)
            feasible_w = list(range(wa, wb + 1))
            deltas_here = incoming_deltas(i, v)

            # For every (b >= 1, status s at i) keep the best and the
            # second-best successor (distinct values) for starting a new
            # ramp at edge i, compared lexicographically on (S, T, w).
            best_new = {}
            for b in range(1, cap + 1):
                for s in statuses_at(i):
                    w1 = p1 = w2 = p2 = None
                    for w in feasible_w:
                        s1 = next_status(s, i, w)
                        if s1 == 0:
                            continue
                        st = G[i + 1].get(w, {}).get((w - v, s1, b - 1))
                        if st is None:
                            continue
                        cand = (st[0], st[1], w)
                        if p1 is None or cand < p1:
                            w2, p2 = w1, p1
                            w1, p1 = w, cand
                        elif w != w1 and (p2 is None or cand < p2):
                            w2, p2 = w, cand
                    if p1 is not None:
                        best_new[(b, s)] = (p1, w1, p2, w2)

            cell = {}
            for din in deltas_here:
                cont_w = None if din is None else v + din
                for s in statuses_at(i):
                    s_cont = (next_status(s, i, cont_w)
                              if cont_w is not None else None)
                    for b in range(cap + 1):
                        best = None
                        # Continue the incoming ramp (edge delta == din).
                        if cont_w is not None and s_cont != 0:
                            st = G[i + 1].get(cont_w, {}).get(
                                (din, s_cont, b))
                            if st is not None:
                                best = (ev + st[0], st[1])
                        # Start a new ramp at edge i.
                        if b >= 1:
                            pre = best_new.get((b, s))
                            if pre is not None:
                                p1, w1, p2, w2 = pre
                                s0, s1, _ = p1 if w1 != cont_w else (
                                    p2 if p2 is not None
                                    else (None, None, None))
                                if s0 is not None:
                                    cand = (ev + s0, 1 + s1)
                                    if best is None or cand < best:
                                        best = cand
                        if best is not None:
                            cell[(din, s, b)] = best
            if cell:
                layer[v] = cell
        G[i] = layer

    # Greedy left-to-right recovery (same tie-breaking as optimal_plan).
    x = [0] * n
    prev_din = None
    prev_status = None
    used_ramps = 0
    for i in range(n):
        if i == 0:
            choices = []
            for v in vals(i):
                s = initial_status(0, v)
                st = G[0].get(v, {}).get((None, s, cap))
                if st is not None:
                    choices.append((st[0], st[1], v))
            if not choices:  # pragma: no cover - guarded by feasibility DP
                raise CompileError("internal reconstruction failure", 500)
            _, _, vi = min(choices, key=lambda z: (z[0], z[1], z[2]))
            x[0] = vi
            prev_status = initial_status(0, vi)
            continue

        choices = []
        for v in vals(i):
            d = v - x[i - 1]
            if abs(d) > step:
                continue
            s = next_status(prev_status, i - 1, v)
            if s == 0:
                continue
            extra = 0 if d == prev_din else 1
            b_remaining = cap - used_ramps - extra
            if b_remaining < 0:
                continue
            st = G[i].get(v, {}).get((d, s, b_remaining))
            if st is None:
                continue
            total_ramps = used_ramps + extra + st[1]
            choices.append((st[0], total_ramps, v, d, s))
        if not choices:  # pragma: no cover - guided by feasibility DP
            raise CompileError("internal reconstruction failure", 500)
        _, _, vi, di, si = min(choices, key=lambda z: (z[0], z[1], z[2]))
        if di != prev_din:
            used_ramps += 1
        prev_din = di
        prev_status = si
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


def _sync_conflict(start, end, value, reason, **extra):
    out = {"kind": "sync_window", "start": start, "end": end,
           "value": value, "reason": reason}
    out.update(extra)
    return out


def _sync_structural_conflicts(req, bands):
    """Window conflicts decidable directly from the structural bands.

    * ``no_admissible_value`` -- no position in the window can take the
      required value under the anchor/step cones (typical window-vs-anchor
      joint infeasibility);
    * ``multiple_forced_hits`` -- at least two positions in the window are
      pinned to the required value (e.g. two equal anchors), so the
      exactly-one-hit rule must fail.
    """
    windows = req.get("sync_windows")
    if not windows:
        return []
    conflicts = []
    for s, e, value in windows:
        forced = [i for i in range(s, e + 1) if bands[i] == (value, value)]
        admissible = [i for i in range(s, e + 1)
                      if bands[i][0] <= value <= bands[i][1]]
        if len(forced) >= 2:
            conflicts.append(_sync_conflict(
                s, e, value, "multiple_forced_hits",
                forced_indices=forced))
        elif not admissible:
            conflicts.append(_sync_conflict(
                s, e, value, "no_admissible_value"))
    return conflicts


def _subset_req(req, windows, cap=None):
    sub = dict(req)
    sub["sync_windows"] = windows
    if cap is not None:
        sub["max_ramps"] = cap
    return sub


def _sync_diagnose_conflicts(req, bands):
    """Locate the windows responsible for full-domain infeasibility.

    Enumerating the <= 7 non-empty window subsets gives a stable, localized
    verdict: each inclusion-minimal infeasible subset is reported with its
    span and member windows, plus a ``ramp_budget`` reason when lifting the
    ramp cap to ``n - 1`` removes the conflict.
    """
    windows = req["sync_windows"]
    m = len(windows)

    def verdict(sub):
        return ramp_feasible_on_bands(sub, bands)

    results = {}
    for mask in range(1, 1 << m):
        subset = [windows[j] for j in range(m) if mask & (1 << j)]
        results[mask] = verdict(_subset_req(req, subset))

    if any(v is None for v in results.values()):
        # The integer-domain guard blocked exact localization; report the
        # full window set deterministically rather than guessing a subset.
        s, e = windows[0][0], windows[-1][1]
        return [_sync_conflict(
            s, e, windows[0][2], "infeasible",
            windows=[{"start": a, "end": b, "value": val}
                     for a, b, val in windows])]

    minimal = []
    for mask, v in results.items():
        if v is not False:
            continue
        submask = (mask - 1) & mask
        culprit = True
        while submask:
            if results.get(submask) is False:
                culprit = False
                break
            submask = (submask - 1) & mask
        if culprit:
            minimal.append(mask)

    conflicts = []
    for mask in minimal:
        members = [windows[j] for j in range(m) if mask & (1 << j)]
        loosened = verdict(_subset_req(req, members, cap=req["n"] - 1))
        reason = "ramp_budget" if loosened is True else "exactly_one_hit"
        conflicts.append(_sync_conflict(
            members[0][0], members[-1][1], members[0][2], reason,
            windows=[{"start": a, "end": b, "value": val}
                     for a, b, val in members]))
    if not conflicts:  # pragma: no cover - full set is known infeasible
        a, b, val = windows[0]
        conflicts.append(_sync_conflict(a, b, val, "infeasible"))
    return conflicts


def _selected_sync_windows(req, x):
    """The unique hit ``{index, value}`` per window, in request order."""
    selected = []
    for s, e, value in req["sync_windows"]:
        hits = [i for i in range(s, e + 1) if x[i] == value]
        selected.append({"index": hits[0], "value": value})
    return selected


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
    # covered, ramp count is the only remaining restriction.  Evaluate the
    # base instance (windows removed) first -- an infeasibility that already
    # exists without windows must not be attributed to them.
    base_req = dict(req, sync_windows=None) if req.get("sync_windows") else req
    if ramp_feasible_on_bands(base_req, bands) is False:
        raise CompileError(
            "no feasible delay plan within the ramp budget", 422,
            {"conflicts": [_ramp_budget_conflict(req)]})

    # Sync-window conflicts visible directly in the anchor/step cones.
    if req.get("sync_windows"):
        window_conflicts = _sync_structural_conflicts(req, bands)
        if window_conflicts:
            raise CompileError(
                "no feasible delay plan: sync window conflicts with anchor "
                "or step constraints", 422,
                {"conflicts": window_conflicts})

    # Window feasibility on the full bands: False localizes a genuine
    # exactly-one-hit / ramp-budget conflict (diagnosed per window subset);
    # None means the exact DP domain exceeds the guard and is handled by the
    # regular budget search below.
    if req.get("sync_windows"):
        window_verdict = ramp_feasible_on_bands(req, bands)
        if window_verdict is False:
            raise CompileError(
                "no feasible delay plan: sync window cannot be satisfied",
                422, {"conflicts": _sync_diagnose_conflicts(req, bands)})

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
    plan = {
        "n": req["n"],
        "delays": x,
        "errors": errors,
        "ramps": ramps,
        "ramp_count": len(ramps),
        "max_abs_error": max(abs(e) for e in errors),
        "total_abs_error": sum(abs(e) for e in errors),
    }
    if req.get("sync_windows"):
        # Present only when the request carried the field; one entry per
        # window in request order with the uniquely selected element.
        plan["sync_windows"] = _selected_sync_windows(req, x)
    return plan


def _guard_refusal(req):
    return CompileError(
        "integer delay domain too large to compile exactly; tighten "
        "'delay_min'/'delay_max' or reduce the target spread", 400,
        {"field": "delay_min"})
