#!/usr/bin/env python3
"""Bar charts of GEMM runtime by optimization stage.

Data is the single-session `./run_gemm --all` output at 4096x4096x4096,
built natively for sm_120 (NVIDIA RTX 5060 Laptop). Mean of 20
CUDA-event-timed iterations after 3 warmups; stddev from the same run.

Keeping all stages from one session matters: absolute milliseconds are not
comparable across sessions on this laptop GPU because clocks are not locked.

Renders two views of the same numbers:
  gemm_progression.png         log y axis, every stage legible
  gemm_progression_linear.png  linear y axis, true proportions

Regenerate:  python3 plot_results.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless: no display under WSL/SSH
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- input data
# (label, mean_ms, stddev_ms) in pipeline order, slowest first.
# Labels are pre-wrapped so the x axis needs no rotation.
STAGES = [
    ("Naive",                 558.7595, 11.2086),
    ("Coalesced",             174.0754, 18.9683),
    ("Shared-mem\ntiled",     123.8070,  9.2490),
    ("1D reg\ntiling",         51.6262,  6.4807),
    ("2D reg\ntiling",         36.2836,  3.6833),
    ("Vectorized\n(float4)",   31.8531,  1.2128),
    ("Autotuned",              20.3874,  0.0738),
]
CUBLAS_MS = 16.8409

M = N = K = 4096
ITERS = 20
OPS = 2 * M * N * K          # 137,438,953,472 FLOP per GEMM
PEAK_TFLOPS = 15.4           # harness reference peak for this GPU
HERE = os.path.dirname(os.path.abspath(__file__))

NAIVE_MS = STAGES[0][1]
BEST_MS = STAGES[-1][1]
TITLE = "FP32 SGEMM 4096x4096x4096 on RTX 5060 Laptop (sm_120)"


def tflops(ms):
    """Achieved throughput in TFLOP/s for a given runtime in milliseconds."""
    return OPS / (ms * 1e-3) / 1e12


def render(log_scale, filename):
    """Draw the stage bar chart with either a log or a linear y axis."""
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    xpos = list(range(len(STAGES)))

    # Blue ramp: darker as the kernel gets faster.
    cmap = plt.get_cmap("Blues")
    colors = [cmap(0.32 + 0.60 * i / (len(STAGES) - 1))
              for i in range(len(STAGES))]

    ax.bar(xpos, [s[1] for s in STAGES], yerr=[s[2] for s in STAGES],
           width=0.68, color=colors, edgecolor="#1a1a1a", linewidth=0.6,
           error_kw=dict(ecolor="#555555", capsize=2.5, lw=0.9), zorder=3)

    # cuBLAS is the reference, not a stage, so it gets a line rather than a
    # bar. The label sits in the right margin, clear of every bar.
    ax.axhline(CUBLAS_MS, color="#c62828", ls="--", lw=1.6, zorder=4)
    ax.annotate("cuBLAS\n16.8 ms", xy=(1.0, CUBLAS_MS),
                xycoords=("axes fraction", "data"),
                xytext=(8, 0), textcoords="offset points",
                ha="left", va="center",
                fontsize=10, fontweight="bold", color="#c62828")

    # One compact label per bar: runtime and speedup over naive. Offsetting in
    # points rather than data units keeps this correct on both scales.
    for i, (_, ms, sd) in enumerate(STAGES):
        ax.annotate("%.1f ms\n%.1fx" % (ms, NAIVE_MS / ms),
                    xy=(i, ms + sd), xytext=(0, 5),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, linespacing=1.35)

    ax.set_xticks(xpos)
    ax.set_xticklabels([s[0] for s in STAGES], fontsize=10.5)
    ax.set_xlim(-0.65, len(STAGES) - 0.35)

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(9, 2000)
        ax.set_ylabel("Runtime (ms, log scale)", fontsize=11)
    else:
        ax.set_ylim(0, 670)        # headroom for the tallest bar's label
        ax.set_ylabel("Runtime (ms)", fontsize=11)

    ax.set_title(TITLE, fontsize=13.5, fontweight="bold", pad=14)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(axis="y", ls=":", color="#aaaaaa", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.text(0.012, 0.015,
             "Mean of %d timed iterations; error bars = stddev." % ITERS,
             fontsize=8.5, color="#666666", va="bottom")

    fig.subplots_adjust(left=0.085, right=0.885, top=0.91, bottom=0.145)
    out = os.path.join(HERE, filename)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------- main
written = [
    render(True, "gemm_progression.png"),
    render(False, "gemm_progression_linear.png"),
]

print("FP32 GEMM %dx%dx%d, %d iters, %d FLOP per GEMM" % (M, N, K, ITERS, OPS))
print("%-22s %10s %9s %8s %9s" % ("stage", "ms", "TFLOP/s", "vs naive", "% cuBLAS"))
for lab, ms, sd in STAGES:
    print("%-22s %10.4f %9.3f %7.2fx %8.1f%%"
          % (lab.replace("\n", " "), ms, tflops(ms), NAIVE_MS / ms,
             100.0 * CUBLAS_MS / ms))
print("%-22s %10.4f %9.3f %7.2fx %8.1f%%"
      % ("cuBLAS", CUBLAS_MS, tflops(CUBLAS_MS), NAIVE_MS / CUBLAS_MS, 100.0))
print("\nheadline: naive -> autotuned = %.1fx faster (%.1f ms -> %.1f ms), "
      "%.1f%% of cuBLAS, %.0f%% of %.1f TFLOP/s peak"
      % (NAIVE_MS / BEST_MS, NAIVE_MS, BEST_MS, 100.0 * CUBLAS_MS / BEST_MS,
         100.0 * tflops(BEST_MS) / PEAK_TFLOPS, PEAK_TFLOPS))
for path in written:
    print("wrote %s" % path)
