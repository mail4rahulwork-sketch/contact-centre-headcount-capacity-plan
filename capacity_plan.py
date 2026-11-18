"""
Long-Term Capacity Plan: When Do We Need to Hire?
-------------------------------------------------
A 26-week headcount (HC) plan for a SIMULATED contact centre.

Inputs (all simulated, no company data):
  - weekly contact forecast with growth and a festive peak in weeks 16-19
  - AHT, target occupancy, shrinkage
  - weekly attrition, hiring lead time (recruit + notice) and training time
  - new-hire ramp: productivity in the first weeks on the floor

The model computes required FTE per week, projects available FTE given
attrition and hiring, and compares three hiring strategies:
  A. Reactive  : raise a hiring request only when the gap appears
  B. Planned   : raise requests early enough (lead time + training + ramp)
                 that new hires are productive before the gap, including
                 expected attrition
  C. Planned + flex : plan permanent hiring to the baseline (no peak) and
                 cover the festive peak with a temporary batch that is
                 trained ahead and released after the peak

Author: Rahul
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

WEEKS = 26
AHT_SEC = 360
OCCUPANCY = 0.85
SHRINKAGE = 0.32
HOURS_PER_FTE = 40
ATTRITION_WEEKLY = 0.008          # ~3.4% per month
LEAD_WEEKS = 6                     # requisition -> joining
TRAIN_WEEKS = 3                    # classroom, 0% productive
RAMP = [0.5, 0.7, 0.9]             # productivity in first 3 weeks on the floor
START_HC = 120
BATCH = 10                         # hires come in training batches


def forecast():
    rng = np.random.default_rng(11)
    w = np.arange(WEEKS)
    base = 24000 * (1 + 0.006 * w)
    peak = np.where((w >= 15) & (w <= 18), 1.25, 1.0)
    return np.round(base * peak * (1 + rng.normal(0, 0.015, WEEKS)))


def required_fte(vol):
    workload_hrs = vol * AHT_SEC / 3600 / OCCUPANCY
    return workload_hrs / (HOURS_PER_FTE * (1 - SHRINKAGE))


def simulate(req, strategy):
    """Return weekly productive FTE, headcount and hires for a strategy."""
    hc = START_HC
    cohorts = []                    # (start_week_on_floor_minus_training, size)
    raised = np.zeros(WEEKS + LEAD_WEEKS + TRAIN_WEEKS + 10)
    prod, heads, hires = [], [], []
    pipeline = LEAD_WEEKS + TRAIN_WEEKS
    for t in range(WEEKS):
        # requests raised earlier join now (start training)
        joiners = int(raised[t])
        if joiners:
            cohorts.append((t, joiners))
        hc = hc * (1 - ATTRITION_WEEKLY) + joiners
        # productive FTE: tenured staff + ramping cohorts
        tenured = hc - sum(n for (s, n) in cohorts if t - s < TRAIN_WEEKS + len(RAMP))
        ramping = 0.0
        for s, n in cohorts:
            age = t - s
            if TRAIN_WEEKS <= age < TRAIN_WEEKS + len(RAMP):
                ramping += n * RAMP[age - TRAIN_WEEKS]
        p = tenured + ramping
        prod.append(p); heads.append(hc); hires.append(joiners)

        # decide new requisitions (they join after LEAD_WEEKS)
        if strategy == "A":
            gap = req[t] - p
            if gap > 0 and raised[t + 1: t + LEAD_WEEKS + 1].sum() == 0:
                raised[t + LEAD_WEEKS] += BATCH * math.ceil(gap / BATCH)
        else:
            look = t + pipeline + len(RAMP)  # week a request raised now is fully productive
            if look < WEEKS:
                keep = (1 - ATTRITION_WEEKLY) ** (look - t)
                in_pipe = raised[t + 1: t + LEAD_WEEKS + 1].sum() + sum(n for (s_, n) in cohorts if t - s_ < TRAIN_WEEKS + len(RAMP))
                future_prod = tenured * keep + in_pipe * keep
                need = max(req[look:min(WEEKS, look + 2)]) - future_prod
                if need > 0 and raised[t + LEAD_WEEKS] == 0:
                    raised[t + LEAD_WEEKS] += BATCH * math.ceil(need / BATCH)
    return np.array(prod), np.array(heads), np.array(hires)


def flex_plan(vol):
    """Strategy C: permanent plan on baseline volume + temporary peak batch."""
    base_vol = vol.copy()
    base_vol[15:19] = base_vol[15:19] / 1.25
    req_base = required_fte(base_vol)
    prod, heads, hires = simulate(req_base, "B")
    req = required_fte(vol)
    peak_gap = float((req[15:19] - prod[15:19]).max())
    temps = BATCH * math.ceil(max(0.0, peak_gap) / BATCH)
    join = 15 - TRAIN_WEEKS - len(RAMP)           # fully productive from week 16
    prod, heads, hires = prod.copy(), heads.copy(), hires.copy()
    hires[join] += temps
    for t in range(join, 19):
        age = t - join
        if age < TRAIN_WEEKS:
            f = 0.0
        elif age < TRAIN_WEEKS + len(RAMP):
            f = RAMP[age - TRAIN_WEEKS]
        else:
            f = 1.0
        prod[t] += temps * f
        heads[t] += temps
    return prod, heads, hires, temps


def main():
    vol = forecast()
    req = required_fte(vol)
    rows = []
    plans = {}
    for s, name in [("A", "Reactive hiring"), ("B", "Planned hiring (lead + training ahead)")]:
        prod, heads, hires = simulate(req, s)
        gap = prod - req
        short_weeks = int((gap < -0.5).sum())
        worst = float(gap.min())
        peak_cov = float((prod[15:19] / req[15:19]).mean() * 100)
        rows.append({
            "strategy": name,
            "weeks_understaffed": short_weeks,
            "worst_gap_FTE": round(worst, 1),
            "peak_coverage_%": round(peak_cov, 1),
            "total_hires": int(hires.sum()),
            "avg_overstaff_FTE": round(float(np.clip(gap, 0, None).mean()), 1),
            "post_peak_overstaff_FTE": round(float(np.clip(gap[19:], 0, None).mean()), 1),
        })
        plans[s] = (prod, heads, hires)
    prod, heads, hires, temps = flex_plan(vol)
    gap = prod - req
    rows.append({
        "strategy": f"Planned + flex ({temps} temps for the peak)",
        "weeks_understaffed": int((gap < -0.5).sum()),
        "worst_gap_FTE": round(float(gap.min()), 1),
        "peak_coverage_%": round(float((prod[15:19] / req[15:19]).mean() * 100), 1),
        "total_hires": int(hires.sum()),
        "avg_overstaff_FTE": round(float(np.clip(gap, 0, None).mean()), 1),
        "post_peak_overstaff_FTE": round(float(np.clip(gap[19:], 0, None).mean()), 1),
    })
    plans["C"] = (prod, heads, hires)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "strategy_comparison.csv", index=False)

    df = pd.DataFrame({
        "week": np.arange(1, WEEKS + 1), "forecast_contacts": vol.astype(int),
        "required_FTE": req.round(1),
        "reactive_productive_FTE": plans["A"][0].round(1), "reactive_hires": plans["A"][2],
        "planned_productive_FTE": plans["B"][0].round(1), "planned_hires": plans["B"][2],
        "flex_productive_FTE": plans["C"][0].round(1), "flex_hires": plans["C"][2],
    })
    df.to_csv(OUT / "weekly_plan.csv", index=False)

    lost_per_year = START_HC * (1 - (1 - ATTRITION_WEEKLY) ** 52)
    txt = f"""# Results (simulated)

Start HC {START_HC}, attrition {ATTRITION_WEEKLY*100:.1f}%/week (~{(1-(1-ATTRITION_WEEKLY)**4.33)*100:.1f}%/month),
hiring lead time {LEAD_WEEKS} weeks + {TRAIN_WEEKS} weeks training + {len(RAMP)}-week ramp.

{res.to_markdown(index=False)}

- Required FTE rises from {req[0]:.0f} (week 1) to a peak of {req.max():.0f} (week {int(req.argmax())+1}).
- Attrition alone removes ~{lost_per_year:.0f} of {START_HC} agents over a year if not backfilled.
- A new hire raised today is not fully productive for {LEAD_WEEKS + TRAIN_WEEKS + len(RAMP)} weeks.
"""
    (OUT / "RESULTS.md").write_text(txt)
    print(txt)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
