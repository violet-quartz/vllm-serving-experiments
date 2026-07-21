#!/usr/bin/env python3
"""
plot.py — turn summary.csv + energy.csv into the two figures for your README.

  fig 1  J/token vs concurrency   (raw vs idle-adjusted "dynamic" — the story)
  fig 2  power vs time            (optionally shading each measured window)

Pure stdlib + matplotlib (Agg backend, no display needed on a headless server).

Usage:
    python plot.py --summary run/summary.csv --csv energy.csv \
                   --windows run/windows.jsonl --outdir run
"""
import argparse, csv, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return None

def plot_jtoken(summary_csv, out):
    rows = list(csv.DictReader(open(summary_csv)))
    rows = [r for r in rows if fnum(r.get("concurrency")) is not None]
    rows.sort(key=lambda r: fnum(r["concurrency"]))
    x     = [fnum(r["concurrency"]) for r in rows]
    raw   = [fnum(r.get("J_per_out_token")) for r in rows]
    dyn   = [fnum(r.get("J_per_out_token_dyn")) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, raw, "o-", label="J/token (raw, incl. idle)")
    if any(v is not None for v in dyn):
        ax.plot(x, dyn, "s--", label="J/token (dynamic, idle removed)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x); ax.set_xticklabels([str(int(v)) for v in x])
    ax.set_xlabel("concurrency"); ax.set_ylabel("J / output token")
    ax.set_title("Energy per output token vs concurrency")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"wrote {out}")

def plot_power(energy_csv, windows_path, out):
    ts, pw = [], []
    for r in csv.DictReader(open(energy_csv)):
        t, p = fnum(r.get("ts")), fnum(r.get("power_mw"))
        if t is not None and p is not None:
            ts.append(t); pw.append(p / 1000.0)  # mW -> W
    if not ts:
        print("no power samples; skipping power plot"); return
    t0 = ts[0]
    rel = [t - t0 for t in ts]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rel, pw, lw=0.8, color="tab:blue")
    ax.set_xlabel("time (s)"); ax.set_ylabel("GPU power (W)")
    ax.set_title("GPU power over the sweep")
    ax.grid(True, alpha=0.3)

    if windows_path and os.path.exists(windows_path):
        for w in (json.loads(l) for l in open(windows_path) if l.strip()):
            a, b = w["t_start"] - t0, w["t_end"] - t0
            if w["phase"] == "idle":
                ax.axvspan(a, b, color="gray", alpha=0.15)
                ax.text((a+b)/2, ax.get_ylim()[1]*0.95, "idle",
                        ha="center", va="top", fontsize=8, color="gray")
            else:
                ax.axvspan(a, b, color="tab:orange", alpha=0.12)
                ax.text((a+b)/2, ax.get_ylim()[1]*0.95, f"c{w['concurrency']}",
                        ha="center", va="top", fontsize=8, color="tab:orange")
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"wrote {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="summary.csv from analyze.py")
    ap.add_argument("--csv", required=True, help="energy.csv from the sampler")
    ap.add_argument("--windows", default=None, help="windows.jsonl (optional, for shading)")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    plot_jtoken(args.summary, os.path.join(args.outdir, "jtoken_vs_concurrency.png"))
    plot_power(args.csv, args.windows, os.path.join(args.outdir, "power_timeline.png"))

if __name__ == "__main__":
    main()
