#!/usr/bin/env python3
"""
plot.py — turn energy.csv + benchmark JSONs into figures for your README.

  fig 1  power vs time            (optionally shading each measured window)
  fig 2  throughput vs concurrency (output & total token throughput)
  fig 3  latency vs concurrency    (TTFT and TPOT, mean vs P99)

Each figure is drawn only when its inputs are present, so you can run with or
without the energy sampler:
  --csv      -> fig 1 (power over time)  omit to skip
  --windows  -> figs 2-3 (tput/lat)      omit to skip
figs 2-3 read the per-concurrency benchmark result JSONs pointed to by
windows.jsonl.

Pure stdlib + matplotlib (Agg backend, no display needed on a headless server).

Usage:
    # full run (with energy_sampler): power curve + throughput + latency
    python plot.py --csv energy.csv --windows run/windows.jsonl --outdir run

    # no energy_sampler: only throughput + latency
    python plot.py --windows run/windows.jsonl --outdir run
"""
import argparse, csv, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return None

def load_perf(windows_path):
    """From windows.jsonl, load each measured window's benchmark result JSON.

    Returns a list of dicts (one per concurrency, sorted) carrying the perf
    fields we plot. result_json paths in windows.jsonl are relative to the
    experiment dir, so resolve them against the windows file's directory too.
    """
    base = os.path.dirname(os.path.abspath(windows_path))
    rows = []
    for w in (json.loads(l) for l in open(windows_path) if l.strip()):
        if w.get("phase") != "measured" or not w.get("result_json"):
            continue
        p = w["result_json"]
        if not os.path.exists(p):
            p = os.path.join(base, os.path.basename(p))
        if not os.path.exists(p):
            print(f"  ! result json missing for {w['label']}: {w['result_json']}")
            continue
        d = json.load(open(p))
        rows.append({
            "concurrency":  fnum(w.get("concurrency")),
            "out_tput":     fnum(d.get("output_throughput")),
            "total_tput":   fnum(d.get("total_token_throughput")),
            "mean_ttft":    fnum(d.get("mean_ttft_ms")),
            "p99_ttft":     fnum(d.get("p99_ttft_ms")),
            "mean_tpot":    fnum(d.get("mean_tpot_ms")),
            "p99_tpot":     fnum(d.get("p99_tpot_ms")),
        })
    rows = [r for r in rows if r["concurrency"] is not None]
    rows.sort(key=lambda r: r["concurrency"])
    return rows

def _log2_xaxis(ax, x):
    ax.set_xscale("log", base=2)
    ax.set_xticks(x); ax.set_xticklabels([str(int(v)) for v in x])
    ax.set_xlabel("concurrency")

def plot_throughput(perf, out):
    if not perf:
        print("no perf rows; skipping throughput plot"); return
    x     = [r["concurrency"] for r in perf]
    out_t = [r["out_tput"]   for r in perf]
    tot_t = [r["total_tput"] for r in perf]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, out_t, "o-", color="tab:green",  label="output tokens/s")
    ax.plot(x, tot_t, "s--", color="tab:blue",  label="total tokens/s (in+out)")
    _log2_xaxis(ax, x)
    ax.set_ylabel("throughput (tok/s)")
    ax.set_title("Throughput vs concurrency")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"wrote {out}")

def plot_latency(perf, out):
    if not perf:
        print("no perf rows; skipping latency plot"); return
    x = [r["concurrency"] for r in perf]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))
    # TTFT (left)
    axL.plot(x, [r["mean_ttft"] for r in perf], "o-",  color="tab:orange", label="mean")
    axL.plot(x, [r["p99_ttft"]  for r in perf], "s--", color="tab:red",    label="P99")
    _log2_xaxis(axL, x)
    axL.set_ylabel("TTFT (ms)"); axL.set_title("Time to first token")
    axL.grid(True, which="both", alpha=0.3); axL.legend()
    # TPOT (right)
    axR.plot(x, [r["mean_tpot"] for r in perf], "o-",  color="tab:purple", label="mean")
    axR.plot(x, [r["p99_tpot"]  for r in perf], "s--", color="tab:brown",  label="P99")
    _log2_xaxis(axR, x)
    axR.set_ylabel("TPOT (ms)"); axR.set_title("Time per output token")
    axR.grid(True, which="both", alpha=0.3); axR.legend()

    fig.suptitle("Latency vs concurrency")
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

def _fmt(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"

def write_tables(perf, out):
    """Write the plotted numbers as Markdown tables so they can be pasted into docs."""
    if not perf:
        print("no data for tables; skipping results.md"); return
    lines = [
        "# Benchmark results",
        "",
        "## Throughput & latency vs concurrency",
        "",
        "| concurrency | output tok/s | total tok/s | mean TTFT (ms) | P99 TTFT (ms) | mean TPOT (ms) | P99 TPOT (ms) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in perf:
        lines.append(
            f"| {int(r['concurrency'])} | {_fmt(r['out_tput'])} | {_fmt(r['total_tput'])} "
            f"| {_fmt(r['mean_ttft'])} | {_fmt(r['p99_ttft'])} "
            f"| {_fmt(r['mean_tpot'])} | {_fmt(r['p99_tpot'])} |")
    lines.append("")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None,
                    help="energy.csv from the sampler (optional; needed for the power-over-time plot)")
    ap.add_argument("--windows", default=None,
                    help="windows.jsonl (needed for throughput/latency plots and power shading)")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Power-over-time plot needs the sampler CSV; skip it when running without energy_sampler.
    if args.csv:
        plot_power(args.csv, args.windows, os.path.join(args.outdir, "power_timeline.png"))
    else:
        print("no --csv; skipping power plot")

    # Throughput/latency come from the benchmark result JSONs, no energy data needed.
    perf = []
    if args.windows and os.path.exists(args.windows):
        perf = load_perf(args.windows)
        plot_throughput(perf, os.path.join(args.outdir, "throughput_vs_concurrency.png"))
        plot_latency(perf,    os.path.join(args.outdir, "latency_vs_concurrency.png"))
    else:
        print("no --windows; skipping throughput/latency plots")

    # Dump every plotted number as Markdown tables for pasting into docs.
    write_tables(perf, os.path.join(args.outdir, "results.md"))

if __name__ == "__main__":
    main()
