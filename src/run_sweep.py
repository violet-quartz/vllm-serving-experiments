#!/usr/bin/env python3
"""
run_sweep.py — drives sweep and records energy WINDOWS.

For each concurrency level it does:
  1. warmup run  (discarded — avoids cold-start / CUDA-graph capture in the numbers)
  2. measured run (records t_start / t_end around the real `vllm bench serve` call)
It also records ONE idle window up front (model loaded, no traffic) so analyze.py
can report both raw and idle-adjusted J/token.

It writes windows.jsonl (one JSON object per line). It does NOT read energy itself —
the standalone sampler does that. Same host + same wall clock => windows align cleanly.

The ONLY engine-specific code is build_bench_cmd(). Point it at SGLang/TGI/etc. by
editing that one function; everything downstream is unchanged.
"""
import argparse, json, os, signal, subprocess, sys, time

# ─────────────────────────────────────────────────────────────────────────────
# EDIT HERE to change engine / flags. Flag names vary across vLLM versions —
# verify with `vllm bench serve --help`. This is the one version-sensitive spot.
def build_bench_cmd(args, concurrency, num_prompts, result_path):
    return [
        "vllm", "bench", "serve",
        "--backend", "vllm",
        "--base-url", args.base_url,
        "--model", args.model,
        "--served-model-name", args.served_model_name,
        "--endpoint", args.endpoint,
        "--dataset-name", "sharegpt",
        "--dataset-path", args.dataset_path,
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(concurrency),
        "--ignore-eos",                       # optional: makes output length controllable
        "--save-result", "--result-filename", result_path,
    ]
# ─────────────────────────────────────────────────────────────────────────────

def run(cmd, quiet=False):
    print(f"[sweep] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL if quiet else None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model id the server was started with")
    ap.add_argument("--served-model-name", required=True, help="model name of the served model")
    ap.add_argument("--dataset-path", required=True, help="ShareGPT json path")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--endpoint", default="/v1/chat/completions")
    ap.add_argument("--concurrency", default="1,4,8,16,32")
    ap.add_argument("--num-prompts", type=int, default=500)
    ap.add_argument("--warmup-prompts", type=int, default=50)
    ap.add_argument("--idle-sec", type=float, default=30.0)
    ap.add_argument("--settle-sec", type=float, default=3.0, help="pause between phases")
    ap.add_argument("--outdir", default="run")
    ap.add_argument("--sampler-out", default=None, help="if set, start the sampler myself and write here")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    windows_path = os.path.join(args.outdir, "windows.jsonl")
    levels = [int(x) for x in args.concurrency.split(",")]

    # Optionally own the sampler process (otherwise run it yourself in another terminal).
    sampler = None
    if args.sampler_out:
        sampler = subprocess.Popen(
            [sys.executable, "energy_sampler.py", "--out", args.sampler_out,
             "--gpu", str(args.gpu), "--interval", "0.2"])
        time.sleep(2)  # let it self-check and start writing

    def record(label, concurrency, phase, t0, t1, result=None):
        rec = {"label": label, "concurrency": concurrency, "phase": phase,
               "t_start": t0, "t_end": t1, "result_json": result}
        with open(windows_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[sweep] window recorded: {label}/{phase} "
              f"({t1-t0:.1f}s){'' if not result else ' -> '+result}", flush=True)

    try:
        # --- idle window: model resident, zero traffic ---
        print(f"[sweep] idle window {args.idle_sec}s (no requests)...", flush=True)
        t0 = time.time(); time.sleep(args.idle_sec); t1 = time.time()
        record("idle", 0, "idle", t0, t1)
        time.sleep(args.settle_sec)

        for c in levels:
            # warmup (discarded)
            wu = os.path.join(args.outdir, f"warmup_c{c}.json")
            run(build_bench_cmd(args, c, args.warmup_prompts, wu), quiet=True)
            time.sleep(args.settle_sec)

            # measured
            res = os.path.join(args.outdir, f"result_c{c}.json")
            t0 = time.time()
            run(build_bench_cmd(args, c, args.num_prompts, res))
            t1 = time.time()
            record(f"c{c}", c, "measured", t0, t1, res)
            time.sleep(args.settle_sec)

    finally:
        if sampler:
            sampler.send_signal(signal.SIGINT)  # let it flush the CSV
            sampler.wait(timeout=10)

    print(f"[sweep] done. windows -> {windows_path}", flush=True)

if __name__ == "__main__":
    main()