#!/usr/bin/env python3
"""
energy_sampler.py — engine-agnostic GPU energy/power sampler (NVML).

Runs as a standalone background process. It knows NOTHING about vLLM/SGLang/TGI —
it only reads whole-GPU counters via NVML and appends timestamped rows to a CSV.
That's exactly why it's portable: swap the engine, this file doesn't change.

Usage:
    python energy_sampler.py --out energy.csv --gpu 0 --interval 0.2
Stop with Ctrl-C (SIGINT) or SIGTERM; the CSV is flushed on exit.

Correctness points:
- Energy comes from the CUMULATIVE counter nvmlDeviceGetTotalEnergyConsumption (mJ),
  NOT from integrating instantaneous power. Consumers take (end - start) deltas.
- power/util/clock/temp are sampled only for curves, plots, and sanity cross-checks.
- Uses time.time() (wall clock). The benchmark runner must use the SAME clock on the
  SAME host, so windows align with no drift.
"""
import argparse, csv, signal, sys, time
import pynvml

_STOP = False
def _handle(sig, frame):
    global _STOP
    _STOP = True

def safe(fn, default=""):
    """Read one NVML field; never let an unsupported field kill the loop."""
    try:
        return fn()
    except pynvml.NVMLError:
        return default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="energy.csv")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--interval", type=float, default=0.2, help="seconds between samples")
    args = ap.parse_args()

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)

    # --- startup self-check: prove the cumulative energy counter works here ---
    try:
        e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        time.sleep(0.3)
        e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        print(f"[sampler] energy counter OK: {e0} -> {e1} mJ (delta {e1-e0} mJ)", flush=True)
    except pynvml.NVMLError as ex:
        print(f"[sampler] FATAL: total-energy counter not supported on this instance: {ex}", flush=True)
        print("[sampler] this AutoDL image/virtualization can't read cumulative energy. "
              "Switch instance/image (you validated this back in stage 0).", flush=True)
        pynvml.nvmlShutdown()
        sys.exit(2)

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    f = open(args.out, "w", newline="")
    w = csv.writer(f)
    w.writerow(["ts", "iso", "energy_mj", "power_mw",
                "util_gpu", "mem_used_mib", "sm_clock_mhz", "temp_c"])

    print(f"[sampler] writing {args.out} every {args.interval}s. Ctrl-C to stop.", flush=True)
    next_t = time.time()
    while not _STOP:
        ts = time.time()
        energy = safe(lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(h))
        power  = safe(lambda: pynvml.nvmlDeviceGetPowerUsage(h))
        util   = safe(lambda: pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
        mem    = safe(lambda: pynvml.nvmlDeviceGetMemoryInfo(h).used // (1024*1024))
        clk    = safe(lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
        temp   = safe(lambda: pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
        w.writerow([f"{ts:.4f}", time.strftime("%H:%M:%S", time.localtime(ts)),
                    energy, power, util, mem, clk, temp])
        f.flush()  # flush every tick so a crash never loses the window
        next_t += args.interval
        sleep = next_t - time.time()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_t = time.time()  # we fell behind; resync instead of busy-looping

    f.close()
    pynvml.nvmlShutdown()
    print(f"[sampler] stopped, CSV closed: {args.out}", flush=True)

if __name__ == "__main__":
    main()
