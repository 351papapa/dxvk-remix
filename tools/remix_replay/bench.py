"""Replay benchmark: per-pass GPU times and converged-image check from an API recording.

  python bench.py <recording-dir> --runtime <dir> [--marker N] [--name NAME] [--set k=v ...]
                  [--save-baseline NAME | --baseline NAME] [--size WxH] [--hold N]

Each run replays the marker, holds the last frame (default 600 presents) and reads the last [PerfGpu]
interval, which then covers held frames only. Results go to <recording>/bench/<name>.json together
with the final screenshot. --save-baseline stores the run as a baseline; --baseline compares against
one: per-pass deltas, frame time, and screenshot PSNR. Exit code 1 when the image check fails or a
pass slows down by more than the thresholds.
"""
import argparse, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPLAY = HERE / "bin" / "remix_replay.exe"

PSNR_FAIL_DB = 40.0          # replay vs replay noise floor is ~56 dB
SLOWER_FAIL_PCT = 10.0       # per pass, only for passes above MIN_PASS_MS
MIN_PASS_MS = 0.5
# [PerfCpu] fields (p50 of each interval, averaged); CS-thread work for the API path.
CPU_KEYS = ["csThreadCpuMsPerFrame", "inject", "scenePrep", "retainedReplay", "apiDraws", "apiDraws.geometry",
            "apiDraws.instance", "apiDrawsPerFrame", "flush"]


def run_replay(args, rec: Path, runtime: Path):
    log = runtime / "rtx-remix" / "logs" / "remix-dxvk.log"
    if log.exists():
        log.unlink()
    shots = rec / "replay-screenshots"
    start = time.time()
    cmd = [str(REPLAY), str(rec), "--runtime", str(runtime), "--marker", str(args.marker),
           "--hold", str(args.hold), "--perf", "--screenshot", "--size", args.size]
    for kv in args.set or []:
        cmd += ["--set", kv]
    for at, kv in args.set_at or []:
        cmd += ["--set-at", at, kv]
    for at, kv in args.set_hold or []:
        cmd += ["--set-hold", at, kv]
    out = subprocess.run(cmd, capture_output=True, text=True, errors="replace").stdout
    summary = {}
    for key, pat in (("measuredMedianMs", r"measured frames .*?median ([\d.]+) ms"),
                     ("measuredP90Ms", r"measured frames .*?p90 ([\d.]+) ms"),
                     ("heldMedianMs", r"held frames .*?median ([\d.]+) ms"),
                     ("failedCalls", r"\((\d+) failed")):
        m = re.search(pat, out)
        if m:
            summary[key] = float(m.group(1))
    lines = log.read_text(errors="replace").splitlines() if log.exists() else []
    perf_lines = [l for l in lines if "[PerfGpu]" in l]
    passes = {}
    if perf_lines:
        for name, p50, p95, mx in re.findall(r" (\w+)=([\d.]+)/([\d.]+)/([\d.]+)", perf_lines[-1]):
            passes[name] = {"p50": float(p50), "p95": float(p95), "max": float(mx)}
    # CPU: intervals of the measured (moving) frames, i.e. before the held frames at the end.
    cpu_lines = [l for l in lines if "[PerfCpu]" in l]
    held_intervals = max(1, args.hold // 300)
    window = cpu_lines[-(held_intervals + 5):-held_intervals] if len(cpu_lines) > held_intervals else []
    cpu = {}
    for l in window:
        for key in CPU_KEYS:
            m = re.search(r"[ |]" + re.escape(key) + r"=([\d.]+)", l)
            if m:
                cpu.setdefault(key, []).append(float(m.group(1)))
    summary["cpu"] = {k: sum(v) / len(v) for k, v in cpu.items()}
    shot = None
    if shots.exists():
        new = [p for p in shots.glob("*.dds") if p.stat().st_mtime >= start]
        if new:
            shot = max(new, key=lambda p: p.stat().st_mtime)
    return summary, passes, shot, out


def psnr(a: Path, b: Path):
    out = subprocess.run([str(REPLAY), "--compare", str(a), str(b)], capture_output=True, text=True, errors="replace").stdout
    m = re.search(r"PSNR\s+(inf|[\d.]+)", out)
    if not m:
        return None, out.strip()
    return (float("inf") if m.group(1) == "inf" else float(m.group(1))), out.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--marker", type=int, default=1)
    ap.add_argument("--hold", type=int, default=600)
    ap.add_argument("--size", default="2560x1600")
    ap.add_argument("--set", action="append")
    ap.add_argument("--set-at", nargs=2, action="append", metavar=("FRAME", "KV"))
    # In-run A/B: SetConfigVariable before held present H (compare [PerfGpu] intervals between windows).
    ap.add_argument("--set-hold", nargs=2, action="append", metavar=("HOLD", "KV"))
    ap.add_argument("--name", default=time.strftime("run-%Y%m%d-%H%M%S"))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--save-baseline")
    g.add_argument("--baseline")
    args = ap.parse_args()

    rec, runtime = Path(args.recording), Path(args.runtime)
    bench = rec / "bench"
    bench.mkdir(exist_ok=True)
    summary, passes, shot, raw = run_replay(args, rec, runtime)
    if not passes:
        print(raw[-3000:])
        sys.exit("no [PerfGpu] data; see the replay output above")

    name = args.save_baseline or args.name
    result = {"name": name, "marker": args.marker, "set": args.set or [], "summary": summary, "passes": passes,
              "screenshot": None}
    if shot:
        dst = bench / f"{name}.dds"
        shutil.copyfile(shot, dst)
        result["screenshot"] = dst.name
    (bench / f"{name}.json").write_text(json.dumps(result, indent=2))

    total = sum(p["p50"] for p in passes.values())
    print(f"{name}: held frame median {summary.get('heldMedianMs', 0):.2f} ms, sum of pass p50 {total:.2f} ms, "
          f"failed calls {int(summary.get('failedCalls', 0))}")

    cpu = summary.get("cpu", {})
    if not args.baseline:
        for k, p in sorted(passes.items(), key=lambda kv: -kv[1]["p50"])[:15]:
            print(f"  {k:34s} {p['p50']:7.2f}")
        print("  CPU (moving frames): " + "  ".join(f"{k}={v:.2f}" for k, v in cpu.items()))
        return

    base = json.loads((bench / f"{args.baseline}.json").read_text())
    failed = False
    print(f"vs baseline {args.baseline}:")
    names = sorted(set(passes) | set(base["passes"]),
                   key=lambda k: -max(passes.get(k, {}).get("p50", 0), base["passes"].get(k, {}).get("p50", 0)))
    for k in names:
        a = base["passes"].get(k, {}).get("p50", 0.0)
        b = passes.get(k, {}).get("p50", 0.0)
        if max(a, b) < 0.05:
            continue
        pct = (b - a) / a * 100 if a > 0 else float("inf")
        flag = ""
        if max(a, b) >= MIN_PASS_MS and pct > SLOWER_FAIL_PCT:
            flag, failed = "  SLOWER", True
        print(f"  {k:34s} {a:7.2f} -> {b:7.2f}  ({b - a:+.2f} ms, {pct:+.0f}%){flag}")
    bt = sum(p["p50"] for p in base["passes"].values())
    print(f"  {'sum of pass p50':34s} {bt:7.2f} -> {total:7.2f}  ({total - bt:+.2f} ms)")
    base_cpu = base.get("summary", {}).get("cpu", {})
    if cpu or base_cpu:
        print("  CPU (moving frames, p50 averaged):")
        for k in CPU_KEYS:
            if k in cpu or k in base_cpu:
                a, b = base_cpu.get(k, 0.0), cpu.get(k, 0.0)
                print(f"    {k:32s} {a:7.2f} -> {b:7.2f}  ({b - a:+.2f})")
    if base.get("screenshot") and result["screenshot"]:
        value, text = psnr(bench / base["screenshot"], bench / result["screenshot"])
        ok = value is not None and value >= PSNR_FAIL_DB
        failed |= not ok
        print(f"  image PSNR vs baseline: {text if value is None else f'{value:.1f} dB'}  {'ok' if ok else 'IMAGE DIFFERS'}")
    else:
        print("  image check skipped: no screenshot")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
