"""gta6-trophy-rarity-forecast-01 entry (Python 3.13, standard library only).

Forecasts each GTA VI trophy's Exophase earn rate at the case's target.

Model (per trophy, in logit space unless noted):
  * Earned counts e(t) and players-tracked N(t) are each fitted as a power
    law in (1 + hours since launch) -- a straight line of log e (resp.
    log N) against log1p(hours) -- over the case history, recency
    weighted, slopes shrunk toward the tier mean and, when the history
    is flat (few distinct values), toward a prior. The counts forecast
    is rate_counts = e(target) / N(target). This is the part that sees
    dilution: Exophase refreshes the page on its own schedule, so when
    N jumps, rates fall even while more players earn the trophy.
  * logit(rate) is also fitted directly against log1p(hours), slope
    shrunk toward the tier mean; rate_trend is that line at the target.
  * The median is the logit-average of rate_counts and rate_trend.
  * The spread is a random-walk term growing with sqrt(horizon) plus a
    floor plus a term from each trophy's fit residual, so a refresh the
    history has not seen yet (the practice cases straddle one: every
    rate but the 100% one fell at the first refresh) still lands inside
    the 90% band more often than the house baseline's narrow band.

Deterministic: no randomness, cases processed in name order, every
aggregation iterates in sorted order. Pure stdlib, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Priors used when the history is flat (no distinct values to fit).
# N roughly doubled every few hours in the first refreshes after launch
# (638 -> 3,868 players tracked by the 04:30 UTC refresh on launch night,
# ~19,000 by midday), so with no observed movement we still assume the
# tracked population grows faster than earned counts do.
GAMMA_N_PRIOR = 1.2   # d log N / d log1p(hours)
GAMMA_E_PRIOR = 0.7   # d log earned / d log1p(hours)
SLOPE_PRIOR = 0.0     # d logit(rate) / d log1p(hours)

# Spread: sigma = sqrt((SIGMA0 * sqrt(days))**2 + SIGMA_FLOOR**2
#                      + (SIGMA_RESID * resid)**2), clamped.
SIGMA0 = 0.6
SIGMA_FLOOR = 0.35
SIGMA_RESID = 0.8
SIGMA_MIN = 0.25
SIGMA_MAX = 3.0

# Loose tier caps for the median (percent). The model drives the value;
# these only stop a wild extrapolation (platinum in week one stays tiny).
TIER_CAP = {"platinum": 3.0, "gold": 15.0, "silver": 60.0, "bronze": 99.9}

# Blend weight of the counts estimator vs the direct logit-trend estimator.
W_COUNTS = 0.5


def clamp(x: float, lo: float = 0.01, hi: float = 99.99) -> float:
    return min(hi, max(lo, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def wols(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_std).

    Degenerate input (fewer than 2 points, or all x equal) gives a flat
    line at the weighted mean of y and a residual of 0.
    """
    n = len(xs)
    sw = sum(ws)
    if n == 0 or sw <= 0:
        return 0.0, 0.0, 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        resid = math.sqrt(sum(w * (y - my) ** 2 for y, w in zip(ys, ws)) / sw) if n > 1 else 0.0
        return my, 0.0, resid
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / sw)
    else:
        resid = 0.0
    return a, b, resid


def richness(values) -> float:
    """0 when all values are equal, rising to 1 once 5+ distinct values."""
    distinct = len(set(values))
    return min(1.0, max(0.0, (distinct - 1) / 4.0))


def tier_mean(pairs):
    """Weighted mean of (value, weight) pairs; None when no weight."""
    sw = sum(w for _, w in pairs)
    if sw <= 0:
        return None
    return sum(v * w for v, w in pairs) / sw


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")

    release = datetime(2026, 11, 19, tzinfo=timezone.utc)
    try:
        rd = case.get("game", {}).get("release_date")
        if rd:
            release = datetime.fromisoformat(rd).replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        pass

    def hours_since_launch(iso: str) -> float:
        return max(0.0, (parse_time(iso) - release).total_seconds() / 3600.0)

    reads = []
    for h in history:
        n = h.get("players_tracked")
        if not n or n <= 0:
            continue
        t = hours_since_launch(h["at"])
        reads.append({"t": t, "x": math.log1p(t), "n": float(n), "rates": h["rates"]})
    if not reads:
        raise ValueError(f"{name} has no usable reads")
    last = reads[-1]
    t_last, x_last, n_last = last["t"], last["x"], last["n"]
    horizon_days = case["horizon_minutes"] / 1440.0
    t_tgt = max(t_last, hours_since_launch(case["target"]))
    x_tgt = math.log1p(t_tgt)

    # Recency weights: half-life 24h, so launch-night reads matter less
    # once two days of reads exist, but still anchor the fit.
    weights = [0.5 ** ((t_last - r["t"]) / 24.0) for r in reads]
    xs = [r["x"] for r in reads]

    # ---- players-tracked envelope: log N ~ a + gamma_N * x ----
    w_n = richness([r["n"] for r in reads])
    _, gamma_n_ols, _ = wols(xs, [math.log(r["n"]) for r in reads], weights)
    gamma_n = w_n * min(3.0, max(0.0, gamma_n_ols)) + (1.0 - w_n) * GAMMA_N_PRIOR
    n_pred = n_last * math.exp(gamma_n * (x_tgt - x_last))

    # ---- per-trophy fits ----
    per_trophy = []
    for i, trophy in enumerate(trophies):
        pts = []  # (x, weight, rate, earned)
        for r, w in zip(reads, weights):
            rate = r["rates"][i] if i < len(r["rates"]) else None
            if rate is None:
                continue
            earned = round(rate * r["n"] / 100.0)
            pts.append((r["x"], w, float(rate), earned))
        if not pts:
            per_trophy.append(None)
            continue
        xs_i = [p[0] for p in pts]
        ws_i = [p[1] for p in pts]
        # Direct logit-rate trend.
        a_r, b_r, resid_r = wols(xs_i, [logit(p[2]) for p in pts], ws_i)
        w_r = richness([p[2] for p in pts])
        # Earned-count trend: log(e + 0.5) ~ a + gamma_e * x.
        a_e, b_e, _ = wols(xs_i, [math.log(p[3] + 0.5) for p in pts], ws_i)
        w_e = richness([p[3] for p in pts])
        per_trophy.append({
            "tier": trophy.get("tier", "bronze"),
            "a_r": a_r, "b_r": b_r, "resid": resid_r, "w_r": w_r,
            "a_e": a_e, "b_e": b_e, "w_e": w_e,
        })

    # ---- tier-mean slopes (shrinkage targets) ----
    tiers = sorted({t["tier"] for t in per_trophy if t})
    tier_gamma_e = {}
    tier_slope = {}
    for tier in tiers:
        members = [t for t in per_trophy if t and t["tier"] == tier]
        ge = tier_mean([(t["b_e"], t["w_e"]) for t in members])
        sl = tier_mean([(t["b_r"], t["w_r"]) for t in members])
        tier_gamma_e[tier] = ge if ge is not None else GAMMA_E_PRIOR
        tier_slope[tier] = sl if sl is not None else SLOPE_PRIOR
    global_gamma_e = tier_mean(
        [(t["b_e"], t["w_e"]) for t in per_trophy if t]) or GAMMA_E_PRIOR
    global_slope = tier_mean(
        [(t["b_r"], t["w_r"]) for t in per_trophy if t]) or SLOPE_PRIOR

    rows = []
    for i, trophy in enumerate(trophies):
        info = per_trophy[i]
        if info is None:
            med = 1.0
            sigma = SIGMA_MAX
        else:
            tier = info["tier"]
            ge_tier = tier_gamma_e.get(tier, global_gamma_e)
            sl_tier = tier_slope.get(tier, global_slope)
            gamma_e = info["w_e"] * min(3.0, max(0.0, info["b_e"])) + (1.0 - info["w_e"]) * ge_tier
            slope = info["w_r"] * min(3.0, max(-3.0, info["b_r"])) + (1.0 - info["w_r"]) * sl_tier
            # Counts estimator: earned(target) / N(target).
            e_pred = math.exp(info["a_e"] + gamma_e * x_tgt) - 0.5
            e_pred = max(0.0, e_pred)
            r_counts = clamp(100.0 * e_pred / n_pred)
            # Direct trend estimator.
            r_trend = clamp(inv_logit(info["a_r"] + slope * x_tgt))
            med_logit = W_COUNTS * logit(r_counts) + (1.0 - W_COUNTS) * logit(r_trend)
            med = inv_logit(med_logit)
            cap = TIER_CAP.get(tier, 99.9)
            med = min(cap, max(0.01, med))
            sigma = math.sqrt(
                (SIGMA0 * math.sqrt(horizon_days)) ** 2
                + SIGMA_FLOOR ** 2
                + (SIGMA_RESID * info["resid"]) ** 2
            )
            sigma = min(SIGMA_MAX, max(SIGMA_MIN, sigma))
        lg = logit(med)
        qs = [round4(clamp(inv_logit(lg + z * sigma), 0.0, 100.0)) for z in Z]
        # Enforce non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def baseline_rows(case: dict) -> dict:
    """House-baseline fallback (persistence + random-walk spread).

    Used only if the model raises: a valid forecast beats a crash.
    """
    last = case["history"][-1]
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        lg = logit(rate if rate is not None else 0)
        qs = [round4(clamp(inv_logit(lg + z * spread), 0.0, 100.0)) for z in Z]
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception as exc:  # never crash a graded run
            print(f"model failed on {path.name} ({exc}); using baseline", file=sys.stderr)
            forecast = baseline_rows(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
