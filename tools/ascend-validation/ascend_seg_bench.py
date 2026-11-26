#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend/Kunpeng segmented-encoder benchmark (replicates the x86 matrix).

Measures, on identical synthetic content, the video-only matrix from the
x86 analysis: PyAV production default vs PyAV FRAME threads vs
SegmentedFFmpegEncoder (pc x t) grid. Wall time covers encoder
construction, streaming push (16-frame chunks) and finish()/container
close, i.e. the same end-to-end scope as the x86 numbers.

Usage (from the repo root of branch pr9017-ascend-validation):

    PYTHONPATH=. python3 tools/ascend-validation/ascend_seg_bench.py
    PYTHONPATH=. python3 tools/ascend-validation/ascend_seg_bench.py --quick
    PYTHONPATH=. python3 tools/ascend-validation/ascend_seg_bench.py --reps 3 --json out.json

Needs: numpy, ffmpeg on PATH; PyAV (``av``) for the PyAV rows; the repo
itself on PYTHONPATH for the segmented rows. Missing pieces are skipped
with a notice instead of failing the whole run.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

W, H, FPS, CRF = 1344, 768, 24, "18"
CHUNK = 16

try:
    import av
except ImportError:
    av = None

try:
    from vllm_omni.diffusion.utils.segmented_ffmpeg import SegmentedFFmpegEncoder
except Exception as e:  # noqa: BLE001
    print(
        f"NOTE: SegmentedFFmpegEncoder unavailable ({e!r}); "
        "run from the repo root with PYTHONPATH=. -- segmented rows will be skipped."
    )
    SegmentedFFmpegEncoder = None


def make_frames(n: int) -> np.ndarray:
    base = np.zeros((H, W, 3), np.uint8)
    base[..., 0] = np.linspace(8, 250, W, dtype=np.uint8)
    base[..., 1] = np.linspace(8, 250, H, dtype=np.uint8)[:, None]
    base[..., 2] = 128
    frames = np.empty((n, H, W, 3), np.uint8)
    for i in range(n):
        off = (i * 13) % 255
        frames[i] = ((base.astype(np.uint16) + off) & 0xFF).astype(np.uint8)
    return frames


def pyav_encode(frames: np.ndarray, out_path: str, *, thread_count=None, thread_type=None) -> None:
    container = av.open(out_path, "w", format="mp4")
    stream = container.add_stream("h264", rate=FPS)
    stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
    stream.options = {"crf": CRF}
    if thread_count is not None:
        stream.thread_count = thread_count
    if thread_type is not None:
        try:
            stream.thread_type = thread_type
        except TypeError:
            stream.thread_type = 1  # FF_THREAD_FRAME
    for i in range(frames.shape[0]):
        frame = av.VideoFrame.from_ndarray(frames[i], format="rgb24")
        frame.pts = i
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


def segmented_encode(frames: np.ndarray, *, pc: int, t: int) -> bytes:
    enc = SegmentedFFmpegEncoder(
        width=W,
        height=H,
        fps=FPS,
        crf=CRF,
        process_count=pc,
        per_process_threads=t,
        total_frames=frames.shape[0],
        max_pending=4,
    )
    try:
        for i in range(0, frames.shape[0], CHUNK):
            enc.push(frames[i : i + CHUNK])
        return enc.finish()
    except BaseException:
        enc.abort()
        raise


def probe(path: str) -> str:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,duration",
            "-of",
            "csv=p=0",
            path,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return r.stdout.strip()


def bench(name: str, fn, out_dir: str, frames: np.ndarray, reps: int) -> dict:
    best, kib, note = None, None, ""
    for rep in range(reps):
        out = os.path.join(out_dir, f"{name.replace(' ', '_').replace('=', '').replace(',', '_')}.mp4")
        t0 = time.perf_counter()
        result = fn(out, frames)
        dt = time.perf_counter() - t0
        if isinstance(result, (bytes, bytearray)):
            with open(out, "wb") as f:
                f.write(result)
        best = dt if best is None else min(best, dt)
        kib = os.path.getsize(out) // 1024
        if rep == 0:
            try:
                note = probe(out)
            except Exception as e:  # noqa: BLE001
                note = f"ffprobe failed: {e!r}"
    return {
        "config": name,
        "best_s": round(best, 3),
        "fps": round(frames.shape[0] / best, 1),
        "kib": kib,
        "ffprobe": note,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--quick", action="store_true", help="reduced grid")
    ap.add_argument("--json", default="ascend_seg_bench.json")
    args = ap.parse_args()

    print(
        f"arch={platform.machine()} nproc={os.cpu_count()} "
        f"av={getattr(av, '__version__', 'absent')} "
        f"numpy={np.__version__} frames={args.frames} reps={args.reps}"
    )
    v = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    print(v.stdout.splitlines()[0] if v.stdout else "ffmpeg not found")

    frames = make_frames(args.frames)
    out_dir = tempfile.mkdtemp(prefix="ascend_bench_")

    rows = []

    def run_rows(specs):
        for name, fn in specs:
            rows.append(bench(name, fn, out_dir, frames, args.reps))

    if av is not None:
        if args.quick:
            pyav_specs = [
                ("PyAV prod default", lambda out, f: pyav_encode(f, out)),
                ("PyAV FRAME t=16", lambda out, f: pyav_encode(f, out, thread_count=16, thread_type="FRAME")),
                ("PyAV FRAME auto", lambda out, f: pyav_encode(f, out, thread_count=0, thread_type="FRAME")),
            ]
        else:
            pyav_specs = [("PyAV prod default", lambda out, f: pyav_encode(f, out))]
            pyav_specs += [
                (f"PyAV FRAME t={t}", lambda out, f, t=t: pyav_encode(f, out, thread_count=t, thread_type="FRAME"))
                for t in (1, 4, 8, 16)
            ]
            pyav_specs += [("PyAV FRAME auto", lambda out, f: pyav_encode(f, out, thread_count=0, thread_type="FRAME"))]
        run_rows(pyav_specs)
    else:
        print("NOTE: PyAV (av) not importable -- PyAV rows skipped.")

    if SegmentedFFmpegEncoder is not None:
        if args.quick:
            grid = [(1, 4), (4, 1), (4, 4)]
        else:
            grid = [(1, t) for t in (1, 4, 8, 16)] + [(2, t) for t in (1, 4, 8)] + [(4, t) for t in (1, 2, 4)]
        run_rows(
            [
                (
                    f"segmented pc={pc} t={t}" + ("  (#9017 default)" if (pc, t) == (4, 1) else ""),
                    lambda out, f, pc=pc, t=t: segmented_encode(f, pc=pc, t=t),
                )
                for pc, t in grid
            ]
        )
    else:
        print("NOTE: segmented rows skipped.")

    shutil.rmtree(out_dir, ignore_errors=True)

    prod = next((r for r in rows if r["config"] == "PyAV prod default"), None)
    print(f"\n{'config':<28} {'best s':>7} {'fps':>6} {'KiB':>5}  ffprobe(frames,duration)  vs prod")
    for r in rows:
        vs = f"{prod['best_s'] / r['best_s']:.2f}x" if prod else "-"
        print(f"{r['config']:<28} {r['best_s']:7.2f} {r['fps']:6.0f} {r['kib']:5}  {r['ffprobe']:<24} {vs}")

    with open(args.json, "w") as f:
        json.dump(
            {
                "frames": args.frames,
                "reps": args.reps,
                "arch": platform.machine(),
                "nproc": os.cpu_count(),
                "rows": rows,
            },
            f,
            indent=2,
        )
    print(f"\nwrote {args.json}. Paste the full log and this JSON back.")
    if av is None or SegmentedFFmpegEncoder is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
