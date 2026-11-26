#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend/Kunpeng encoder probe v2 - measures the per-process parallelism cap.

Usage (on the Ascend box, plain python3, no repo checkout needed):

    python3 ascend_enc_probe.py 2>&1 | tee enc_probe.log
    python3 ascend_enc_probe.py quick    # reduced grid

Self-contained: needs ``ffmpeg`` on PATH; ``numpy`` is required to synthesize
frames once (~285 MiB in /tmp); PyAV is used when importable.

What it measures and why:
  1. environment fingerprint (nproc, NUMA topology, ffmpeg build, encoders)
  2. CLI vs PyAV single-thread baseline (build health check)
  3. CLI single-process thread-scaling curve t=1..nproc + auto
     -> the per-process parallelism cap P (where the curve flattens)
  4. parallel batches of 4 ffmpeg processes (t=auto / t=8 / t=1)
     -> whether multi-process recovers the cores a single process cannot use
  5. PyAV scaling curve (FRAME threads + production default SLICE auto)

Paste the whole log back for analysis.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

W, H, N_FRAMES, FPS = 1344, 768, 96, 24
BIN = "/tmp/enc_probe_frames.bin"
OUT = "/tmp/enc_probe_out.mp4"
QUICK = "quick" in sys.argv[1:]


def sh(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout.strip()


def make_frames() -> None:
    want = N_FRAMES * H * W * 3
    if os.path.exists(BIN) and os.path.getsize(BIN) == want:
        return
    try:
        import numpy as np
    except ImportError:
        sys.exit("numpy is required to synthesize frames once: pip install numpy")
    base = np.linspace(0, 255, H * W * 3).reshape(H, W, 3).astype(np.uint16)
    with open(BIN, "wb") as f:
        for i in range(N_FRAMES):
            frame = ((base + (i * 13) % 255) & 255).astype(np.uint8)
            f.write(frame.tobytes())
    print(
        f"frames: wrote {N_FRAMES} synthetic {W}x{H} rgb24 frames ({os.path.getsize(BIN) / 2**20:.0f} MiB) to {BIN}",
        flush=True,
    )


def cli_cmd(threads: int | None, out: str) -> list[str]:
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{W}x{H}",
        "-r",
        str(FPS),
        "-i",
        BIN,
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
    ]
    if threads is not None:
        cmd += ["-threads", str(threads)]
    cmd.append(out)
    return cmd


def cli_encode(threads: int | None, out: str = OUT) -> float:
    t0 = time.perf_counter()
    r = subprocess.run(cli_cmd(threads, out), capture_output=True)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        print("FFMPEG FAILED: " + " ".join(cli_cmd(threads, out)), flush=True)
        print(r.stderr.decode(errors="replace")[-2000:], flush=True)
        sys.exit(1)
    return dt


def cli_batch(n: int, threads: int | None) -> float:
    """Run n concurrent ffmpeg encodes, return wall time of the whole batch."""
    procs = []
    t0 = time.perf_counter()
    for i in range(n):
        procs.append(
            subprocess.Popen(
                cli_cmd(threads, f"/tmp/enc_probe_p{i}.mp4"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        )
    for p in procs:
        if p.wait() != 0:
            print(f"FFMPEG FAILED in batch (t={threads})", flush=True)
            sys.exit(1)
    return time.perf_counter() - t0


def pyav_encode(thread_count: int | None, thread_type: str | None) -> float:
    import av
    import numpy as np

    arr = np.fromfile(BIN, np.uint8, N_FRAMES * H * W * 3)
    arr = arr.reshape(N_FRAMES, H, W, 3)
    t0 = time.perf_counter()
    with av.open("/tmp/enc_probe_pyav.mp4", "w", format="mp4") as c:
        s = c.add_stream("h264", rate=FPS)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.options = {"crf": "18"}
        if thread_count is not None:
            s.codec_context.thread_count = thread_count
        if thread_type is not None:
            s.codec_context.thread_type = thread_type
        for i in range(N_FRAMES):
            frame = av.VideoFrame.from_ndarray(arr[i], format="rgb24")
            for packet in s.encode(frame):
                c.mux(packet)
        for packet in s.encode():
            c.mux(packet)
    return time.perf_counter() - t0


def main() -> None:
    print("=== encoder probe v2 (per-process parallelism cap) ===", flush=True)
    print(f"uname: {sh('uname -a')}", flush=True)
    print(f"nproc: {sh('nproc')}", flush=True)
    print("--- lscpu (topology/NUMA) ---", flush=True)
    print(sh("lscpu | sed -n '1,25p'"), flush=True)
    print("--- ffmpeg version / configuration ---", flush=True)
    print(sh("ffmpeg -version | sed -n '1,4p'"), flush=True)
    print("--- h264-family encoders ---", flush=True)
    print(sh("ffmpeg -hide_banner -encoders 2>/dev/null | grep -i '264\\|265' | head -8"), flush=True)
    try:
        import av
        import numpy

        print(f"python extras: numpy {numpy.__version__}, av {av.__version__}", flush=True)
        have_pyav = True
    except ImportError as e:
        print(f"python extras: missing ({e}); PyAV section skipped", flush=True)
        have_pyav = False
    make_frames()

    nproc = int(sh("nproc") or 0) or (os.cpu_count() or 1)
    print(f"\n=== CLI single-process scaling curve (nproc={nproc}) ===", flush=True)
    grid = [1, 2, 4, 8, 16, 32, 64, 128]
    grid = [t for t in grid if t <= max(4, nproc)] + [None]  # None = auto
    if QUICK:
        grid = [1, 8, None]
    results: dict[int | None, float] = {}
    for t in grid:
        reps = 2 if (t is None or t <= 8) else 1
        best = min(cli_encode(t) for _ in range(reps))
        results[t] = best
        label = "auto" if t is None else str(t)
        print(
            f"CLI t={label:>4}: best {best:6.2f}s  {N_FRAMES / best:6.1f} fps  {best / N_FRAMES * 1000:5.0f} ms/frame",
            flush=True,
        )
    base = results[grid[0]]
    for t in grid:
        label = "auto" if t is None else str(t)
        print(f"speedup t={label:>4}: x{base / results[t]:5.2f}", flush=True)

    explicit = [(t, v) for t, v in results.items() if t is not None]
    best_time = min(v for _, v in explicit)
    cap = next((t for t, v in explicit if v <= best_time * 1.10), None)
    print(
        f"\nper-process cap P estimate (smallest t within 10% of best): "
        f"P = {cap}  (best explicit t: {min(explicit, key=lambda kv: kv[1])[0]})",
        flush=True,
    )

    print("\n=== multi-process batches (4 concurrent ffmpeg) ===", flush=True)
    for threads in [None] + ([8] if 8 in [t for t, _ in explicit] or True else []):
        label = "auto" if threads is None else str(threads)
        wall = cli_batch(4, threads)
        single = results.get(threads) or results.get(None)
        gain = 4 * single / wall
        print(
            f"batch 4 x t={label:>4}: wall {wall:6.2f}s  aggregate "
            f"{4 * N_FRAMES / wall:6.1f} fps  throughput gain vs single "
            f"x{gain:4.2f} (4.00 = perfect scaling, ~1.00 = cores shared/capped)",
            flush=True,
        )

    if not QUICK:
        wall = cli_batch(4, 1)
        single1 = results[1]
        print(
            f"batch 4 x t=   1: wall {wall:6.2f}s  aggregate "
            f"{4 * N_FRAMES / wall:6.1f} fps  throughput gain vs single t=1 "
            f"x{4 * single1 / wall:4.2f} (process mechanism overhead view)",
            flush=True,
        )

    if have_pyav:
        print("\n=== PyAV scaling curve (FRAME) + production default ===", flush=True)
        runs = [(1, "FRAME", 2), (4, "FRAME", 1), (16, "FRAME", 1)]
        if QUICK:
            runs = [(1, "FRAME", 2)]
        for tc, tt, reps in runs:
            if tc and tc > max(4, nproc):
                continue
            best = min(pyav_encode(tc, tt) for _ in range(reps))
            print(f"PyAV t={tc:>3} {tt}: best {best:6.2f}s  {N_FRAMES / best:6.1f} fps", flush=True)
        for tc, tt in [(0, None), (0, "FRAME")]:
            best = min(pyav_encode(tc, tt) for _ in range(2))
            note = "prod default (SLICE auto)" if tt is None else "FRAME auto"
            print(f"PyAV auto {note}: best {best:6.2f}s  {N_FRAMES / best:6.1f} fps", flush=True)

    print("\n=== summary ===", flush=True)
    print(f"nproc={nproc}, P estimate={cap}, CLI t=1={results[1]:.2f}s, CLI auto={results[None]:.2f}s", flush=True)
    print("Paste the full log back. Interpretation on our side.", flush=True)


if __name__ == "__main__":
    main()
