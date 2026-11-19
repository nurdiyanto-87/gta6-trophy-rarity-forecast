"""gta6-trophy-rarity-forecast-01 entry (Python 3.13, standard library only).

Forecasts each GTA VI trophy's Exophase earn rate at the case's target.

Model (per trophy, in logit space unless noted). Earn rates are the ratio
of two moving quantities -- players who earned the trophy, and players
Exophase tracks -- and both saturate: early growth is fast (launch-night
rates rose steeply while the tracked population jumped 638 -> 3,868 ->
~19,000 in half a day), then flattens. A straight log-linear extrapolation
of launch-night growth would overshoot the final by 2-3x, so the median
blends three estimators, all anchored at the last read:

  * counts: last_rate * exp((gamma_e - gamma_N) * dx), where gamma_e and
    gamma_N are the recent-window (last 36h) power-law slopes of earned
    counts and players-tracked against log1p(hours since launch). This is
    the part that sees dilution: when N jumps at an Exophase refresh,
    rates fall even while more players earn the trophy. With a flat
    history the slopes fall back to a prior drift (tracked grows faster
    than earned, harder for low-rate trophies), fitted on launch day.
  * saturating: a ceiling fit -- for a grid of ceilings U, fit
    log(r / (U - r)) linear in log1p(hours) and keep the U with the
    smallest residual -- read at the target. A rate cannot grow past its
    ceiling, so this flattens the launch-night rise instead of extending
    it forever.
  * recent trend: a logit-linear fit on the recent window only, its slope
    damped by span / (span + extrapolation distance), so a long
    extrapolation leans back toward persistence.

The blend weight follows how much the history actually moved: with a
flat history the counts estimator (dilution prior) carries 90%; with a
rich history the three share 30/52/18 (counts/saturating/recent). The
spread is a random-walk term growing with sqrt(horizon) plus a floor plus a term from each trophy's
fit residual, so a refresh the history has not seen yet (the practice
cases straddle one: every rate but the 100% one fell at the first
refresh) still lands inside the 90% band more often than the house
baseline's narrow band.

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

# Recent window for the counts and trend fits, in hours. The final case
# reads every 10 minutes late in its history, so this still holds 100+
# reads; with a shorter history it falls back to all reads.
RECENT_HOURS = 36.0

# Priors used when the history is flat (no distinct values to fit).
# N roughly doubled every few hours in the first refreshes after launch
# (638 -> 3,868 players tracked by the 04:30 UTC refresh on launch night,
# ~19,000 by midday), so with no observed movement we still assume the
# tracked population grows faster than earned counts do. Only the
# DIFFERENCE between the two growth rates moves a rate forecast, so the
# prior is stated as a drift: d logit(rate) / d log1p(hours). Launch-day
# data (03:00 -> 11:33 UTC) shows drift about -0.5 for the easiest
# trophies and about -1.5 for hard ones, so the prior drift is stronger
# for low-rate trophies; with a rich history the fitted slopes replace it.
GAMMA_N_PRIOR = 1.2   # d log N / d log1p(hours)
SLOPE_PRIOR = 0.0     # d logit(rate) / d log1p(hours)


def drift_prior(last_rate: float) -> float:
    """Prior d logit(rate) / d log1p(hours) for a flat history."""
    return -(0.55 + 0.75 * math.exp(-last_rate / 1.0))


# Spread: sigma = sqrt((SIGMA0 * sqrt(days))**2 + SIGMA_FLOOR**2
#                      + (SIGMA_RESID * resid)**2), clamped.
SIGMA0 = 0.6
SIGMA_FLOOR = 0.35
SIGMA_RESID = 0.8
SIGMA_MIN = 0.25
SIGMA_MAX = 3.0

# Loose tier caps for the median (percent). The model drives the value;
# these only stop a wild extrapolation (platinum in week one stays tiny).
# Bronze is uncapped (99.99) because a trophy can genuinely sit at 100%.
TIER_CAP = {"platinum": 3.0, "gold": 15.0, "silver": 60.0, "bronze": 99.99}

# Ceiling grid for the saturating fit, as multiples of the max observed rate.
CEILING_MULTS = [1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0, 100.0]

# Blend weights. With a flat history the counts estimator (dilution
# prior) carries W_COUNTS_FLAT; with a rich history it carries
# W_COUNTS_RICH, the rest split W_SAT_SHARE to the saturating fit and
# the remainder to the recent trend.
W_COUNTS_FLAT = 0.9
W_COUNTS_RICH = 0.3
W_SAT_SHARE = 0.75


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


def wsse(xs, ys, ws, a, b):
    """Weighted sum of squared errors of y ~ a + b*x."""
    return sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws))


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


def saturating_forecast(xs, rates, ws, x_tgt, last_rate):
    """Ceiling fit: r(t) = U / (1 + exp(-(a + b*x))) with U gridded.

    Returns the forecast rate at x_tgt, or last_rate when the fit does
    not apply (saturated, flat or falling history).
    """
    if last_rate >= 99.5 or len(set(rates)) < 3:
        return last_rate
    max_rate = max(rates)
    # Falling history: a rising ceiling curve is the wrong shape.
    _, slope_lin, _ = wols(xs, [logit(r) for r in rates], ws)
    if slope_lin <= 0:
        return last_rate
    best = None  # (sse, U, a, b)
    for mult in CEILING_MULTS:
        u = min(99.99, mult * max_rate)
        if u <= max_rate * 1.001:
            continue
        zs = []
        ok = True
        for r in rates:
            rc = min(max(r, 0.01), u - 0.01)
            zs.append(math.log(rc / (u - rc)))
        a, b, _ = wols(xs, zs, ws)
        sse = wsse(xs, zs, ws, a, b)
        if best is None or sse < best[0]:
            best = (sse, u, a, b)
    if best is None:
        return last_rate
    _, u, a, b = best
    z_tgt = a + b * x_tgt
    if z_tgt >= 0:
        return u / (1.0 + math.exp(-z_tgt))
    e = math.exp(z_tgt)
    return u * e / (1.0 + e)


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
    dx = x_tgt - x_last

    # Recent window (last 36h) for the counts and trend fits; all reads
    # when the history is shorter than that.
    recent = [r for r in reads if r["t"] >= t_last - RECENT_HOURS]
    if len(recent) < 4:
        recent = reads
    x_span_recent = max(0.05, recent[-1]["x"] - recent[0]["x"])
    # Extrapolation damping for the recent fits: a long extrapolation
    # leans back toward persistence.
    damp = x_span_recent / (x_span_recent + dx) if dx > 0 else 1.0

    weights_all = [0.5 ** ((t_last - r["t"]) / 24.0) for r in reads]
    weights_rec = [0.5 ** ((t_last - r["t"]) / 24.0) for r in recent]
    xs_all = [r["x"] for r in reads]
    xs_rec = [r["x"] for r in recent]

    # ---- players-tracked envelope on the recent window ----
    # Damping applies to fitted slopes only: a long extrapolation leans
    # back toward persistence. Priors (flat history) are not damped.
    w_n = richness([r["n"] for r in recent])
    _, gamma_n_ols, _ = wols(xs_rec, [math.log(r["n"]) for r in recent], weights_rec)
    gamma_n = w_n * min(3.0, max(0.0, gamma_n_ols)) * damp + (1.0 - w_n) * GAMMA_N_PRIOR

    # ---- per-trophy fits ----
    per_trophy = []
    for i, trophy in enumerate(trophies):
        pts_all = []  # (x, weight, rate) over all reads
        pts_rec = []  # (x, weight, rate, earned) over recent reads
        for r, w in zip(reads, weights_all):
            rate = r["rates"][i] if i < len(r["rates"]) else None
            if rate is None:
                continue
            pts_all.append((r["x"], w, float(rate)))
        for r, w in zip(recent, weights_rec):
            rate = r["rates"][i] if i < len(r["rates"]) else None
            if rate is None:
                continue
            earned = round(rate * r["n"] / 100.0)
            pts_rec.append((r["x"], w, float(rate), earned))
        if not pts_all:
            per_trophy.append(None)
            continue
        last_rate = pts_all[-1][2]
        w_r = richness([p[2] for p in pts_all])
        # Saturating ceiling fit on all reads.
        r_sat = saturating_forecast(
            [p[0] for p in pts_all], [p[2] for p in pts_all],
            [p[1] for p in pts_all], x_tgt, last_rate)
        # Recent-window logit trend, slope damped toward persistence.
        a_r, b_r, resid_r = wols(
            [p[0] for p in pts_rec], [logit(p[2]) for p in pts_rec],
            [p[1] for p in pts_rec])
        # Recent-window earned-count trend.
        a_e, b_e, _ = wols(
            [p[0] for p in pts_rec], [math.log(p[3] + 0.5) for p in pts_rec],
            [p[1] for p in pts_rec])
        w_e = richness([p[3] for p in pts_rec])
        per_trophy.append({
            "tier": trophy.get("tier", "bronze"),
            "last_rate": last_rate, "w_r": w_r,
            "r_sat": r_sat,
            "b_r": b_r, "resid": resid_r,
            "b_e": b_e, "w_e": w_e,
        })

    # ---- tier-mean earned slopes (shrinkage targets, recent window) ----
    # Fitted slopes are damped like the per-trophy ones; priors are not.
    tiers = sorted({t["tier"] for t in per_trophy if t})
    tier_gamma_e = {}
    tier_slope = {}
    for tier in tiers:
        members = [t for t in per_trophy if t and t["tier"] == tier]
        ge = tier_mean([(t["b_e"], t["w_e"]) for t in members])
        sl = tier_mean([(t["b_r"], t["w_r"]) for t in members])
        tier_gamma_e[tier] = ge * damp if ge is not None else None
        tier_slope[tier] = sl * damp if sl is not None else SLOPE_PRIOR
    global_slope = tier_mean(
        [(t["b_r"], t["w_r"]) for t in per_trophy if t])
    global_slope = global_slope * damp if global_slope is not None else SLOPE_PRIOR

    rows = []
    for i, trophy in enumerate(trophies):
        info = per_trophy[i]
        if info is None:
            med = 1.0
            sigma = SIGMA_MAX
        else:
            tier = info["tier"]
            last_rate = info["last_rate"]
            ge_tier = tier_gamma_e.get(tier)
            if ge_tier is None:
                # No variation anywhere: fall back to the drift prior,
                # i.e. earned grows slower than tracked by drift_prior.
                ge_tier = gamma_n + drift_prior(last_rate)
            sl_tier = tier_slope.get(tier, global_slope)
            gamma_e = info["w_e"] * min(3.0, max(0.0, info["b_e"])) * damp + (1.0 - info["w_e"]) * ge_tier
            slope = info["w_r"] * min(3.0, max(-3.0, info["b_r"])) * damp + (1.0 - info["w_r"]) * sl_tier
            # A trophy everyone has earned tracks N: its earned count
            # grows with the tracked population, it does not dilute.
            if last_rate >= 99.0:
                gamma_e = gamma_n
            # Counts estimator: last rate moved by the growth gap.
            r_counts = clamp(last_rate * math.exp((gamma_e - gamma_n) * dx))
            # Recent-trend estimator, damped toward persistence.
            r_rec = clamp(inv_logit(logit(last_rate) + slope * dx))
            # Blend: flat history -> counts dominate (dilution prior);
            # rich history -> saturating / recent / counts share.
            w_counts = W_COUNTS_FLAT - (W_COUNTS_FLAT - W_COUNTS_RICH) * info["w_r"]
            w_sat = (1.0 - w_counts) * W_SAT_SHARE
            w_rec = (1.0 - w_counts) * (1.0 - W_SAT_SHARE)
            med_logit = (w_counts * logit(r_counts)
                         + w_sat * logit(clamp(info["r_sat"]))
                         + w_rec * logit(r_rec))
            med = inv_logit(med_logit)
            cap = TIER_CAP.get(tier, 99.99)
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
