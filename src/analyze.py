#!/usr/bin/env python3
"""
analyze.py — align sampler CSV with windows.jsonl + benchmark JSONs -> J/token.

Pure stdlib. Engine-agnostic: it doesn't care what produced the energy or the
result JSON. It:
  1. reads the sampler CSV (cumulative energy),
  2. for each measured window, interpolates the energy counter at t_start/t_end
     and takes the delta (the correct way — deltas of a cumulative counter),
  3. pulls token counts from the benchmark result JSON,
  4. reports J/output-token, J/total-token, J/request, avg power,
     plus idle-ADJUSTED figures using the idle window.

Usage:
    python analyze.py --csv energy.csv --windows run/windows.jsonl --out run/summary
"""
import argparse, bisect, csv, json, os, sys

def load_samples(path):
    ts, energy_mj, power_mw, util = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                t = float(row["ts"]); e = float(row["energy_mj"])
            except (ValueError, KeyError):
                continue  # skip blanks / unsupported-field rows
            ts.append(t); energy_mj.append(e)
            try: power_mw.append(float(row["power_mw"]))
            except (ValueError, KeyError): power_mw.append(float("nan"))
            try: util.append(float(row["util_gpu"]))
            except (ValueError, KeyError): util.append(float("nan"))
    if len(ts) < 2:
        sys.exit("analyze: not enough samples in CSV")
    return ts, energy_mj, power_mw, util

def interp(ts, vals, t):
    """Linear-interpolate vals at time t. Clamp to ends with a warning."""
    if t <= ts[0]:
        print(f"  ! window edge {t:.1f} before sampling started ({ts[0]:.1f}); clamped", file=sys.stderr)
        return vals[0]
    if t >= ts[-1]:
        print(f"  ! window edge {t:.1f} after sampling ended ({ts[-1]:.1f}); clamped", file=sys.stderr)
        return vals[-1]
    i = bisect.bisect_left(ts, t)
    t0, t1 = ts[i-1], ts[i]
    v0, v1 = vals[i-1], vals[i]
    frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
    return v0 + frac * (v1 - v0)

def energy_J(ts, energy_mj, t0, t1):
    return (interp(ts, energy_mj, t1) - interp(ts, energy_mj, t0)) / 1000.0

def steady_power_W(ts, power_mw, util, t0, t1, util_thresh):
    """Average instantaneous power (W) over samples inside [t0,t1] whose GPU
    utilization is >= util_thresh.

    Why: avg_power_W = energy_J/dur averages the WHOLE window, including the
    client ramp-up before the GPU saturates and the drain as the last requests
    finish. Those low-power edges have a fixed duration but the window shrinks
    fast with concurrency (728s -> 46s here), so for short windows they dilute
    the mean and make high concurrency look *less* power-hungry than it is at
    steady state. Filtering on utilization keeps only the genuinely-loaded
    samples, so this number is comparable across concurrency levels.
    Returns (power_W, n_samples_used); power_W is None if nothing qualifies.
    """
    vals = [power_mw[i] / 1000.0
            for i in range(len(ts))
            if t0 <= ts[i] <= t1
            and power_mw[i] == power_mw[i]          # not NaN
            and util[i] == util[i] and util[i] >= util_thresh]
    if not vals:
        return None, 0
    return sum(vals) / len(vals), len(vals)

# Token fields differ across vLLM versions — try several, print what we used.
OUT_KEYS = ["total_output_tokens", "output_tokens", "total_generated_tokens"]
IN_KEYS  = ["total_input_tokens", "input_tokens", "total_prompt_tokens"]
REQ_KEYS = ["completed", "num_prompts", "successful_requests"]

def pick(d, keys):
    for k in keys:
        if k in d and d[k]:
            return d[k], k
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", default="summary")
    ap.add_argument("--util-thresh", type=float, default=90.0,
                    help="min GPU util%% for a sample to count toward steady_power_W")
    args = ap.parse_args()

    ts, energy_mj, power_mw, util = load_samples(args.csv)
    windows = [json.loads(l) for l in open(args.windows) if l.strip()]

    # idle power for the adjustment
    idle_w = None
    for w in windows:
        if w["phase"] == "idle":
            ej = energy_J(ts, energy_mj, w["t_start"], w["t_end"])
            idle_w = ej / (w["t_end"] - w["t_start"])
            print(f"idle power: {idle_w:.1f} W over {w['t_end']-w['t_start']:.0f}s")
    if idle_w is None:
        print("no idle window found — idle-adjusted columns will be blank")

    rows = []
    for w in windows:
        if w["phase"] != "measured":
            continue
        t0, t1 = w["t_start"], w["t_end"]
        dur = t1 - t0
        ej = energy_J(ts, energy_mj, t0, t1)
        avg_p = ej / dur
        steady_p, n_steady = steady_power_W(ts, power_mw, util, t0, t1, args.util_thresh)
        frac_steady = n_steady / max(1, sum(1 for tt in ts if t0 <= tt <= t1))

        out_tok = in_tok = req = None
        if w.get("result_json") and os.path.exists(w["result_json"]):
            d = json.load(open(w["result_json"]))
            out_tok, ok = pick(d, OUT_KEYS)
            in_tok, _  = pick(d, IN_KEYS)
            req, _     = pick(d, REQ_KEYS)
            if out_tok is None:
                print(f"  ! {w['result_json']}: no output-token field found; "
                      f"keys present: {list(d)[:12]}", file=sys.stderr)

        r = {"level": w["label"], "concurrency": w["concurrency"],
             "dur_s": round(dur, 1), "energy_J": round(ej, 1),
             "avg_power_W": round(avg_p, 1),
             "steady_power_W": round(steady_p, 1) if steady_p is not None else "",
             "steady_frac": round(frac_steady, 3),
             "out_tokens": out_tok, "in_tokens": in_tok, "requests": req}
        if out_tok:
            r["J_per_out_token"] = round(ej / out_tok, 4)
            if in_tok:
                r["J_per_total_token"] = round(ej / (out_tok + in_tok), 4)
            if idle_w is not None:
                dyn = ej - idle_w * dur
                r["J_per_out_token_dyn"] = round(dyn / out_tok, 4)
        if req:
            r["J_per_request"] = round(ej / req, 2)
        rows.append(r)

    # print a table
    cols = ["level", "concurrency", "dur_s", "energy_J", "avg_power_W",
            "steady_power_W", "steady_frac",
            "out_tokens", "J_per_out_token", "J_per_out_token_dyn",
            "J_per_total_token", "J_per_request"]
    print("\n" + "  ".join(f"{c:>18}" for c in cols))
    for r in rows:
        print("  ".join(f"{str(r.get(c,'')):>18}" for c in cols))

    with open(args.out + ".json", "w") as f:
        json.dump({"idle_power_W": idle_w, "rows": rows}, f, indent=2)
    with open(args.out + ".csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=cols); wtr.writeheader()
        for r in rows:
            wtr.writerow({c: r.get(c, "") for c in cols})
    print(f"\nsaved {args.out}.json / {args.out}.csv")

if __name__ == "__main__":
    main()
