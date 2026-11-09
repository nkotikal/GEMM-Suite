# Autotune Summary

Not yet run. `bash sweep.sh` regenerates this file.

**10,659 tests** — 561 configs × 19 shapes, ~4.6 h.
`ratio` = kernel ms / cuBLAS ms, both timed in the same process. Below 1.00 beats cuBLAS.

## Best config per shape

| M | N | K | best config | ratio | TFLOP/s | max abs err |
|---:|---:|---:|---|---:|---:|---:|
| | | | *pending* | | | |

## Best single config overall

| rank | config | mean ratio | shapes |
|---:|---|---:|---:|
| | *pending* | | |

## Current default `BM128_BN64_BK32_TM4_TN4`

Baseline the sweep has to beat. Regenerated from `results.csv` on each run.

| M | N | K | ratio | TFLOP/s |
|---:|---:|---:|---:|---:|
| 32 | 32 | 32 | 0.7497 | 0.0042 |
| 64 | 64 | 64 | 0.6338 | 0.0308 |
| 128 | 128 | 128 | 0.7889 | 0.2597 |
| 256 | 256 | 256 | 1.0018 | 1.2871 |
| 512 | 512 | 512 | 1.6204 | 3.8732 |
| 1024 | 1024 | 1024 | 1.1994 | 5.9934 |
| 2048 | 2048 | 2048 | 1.2976 | 6.1191 |
| 4096 | 4096 | 4096 | 1.1163 | 5.6929 |
| 8192 | 8192 | 8192 | 1.6505 | 5.0842 |
| 8192 | 128 | 2048 | 1.3419 | 5.3626 |
| 128 | 8192 | 2048 | 1.2665 | 5.4548 |
| 4096 | 64 | 4096 | 1.0560 | 3.6413 |
| 64 | 4096 | 4096 | 2.6312 | 2.5400 |
| 256 | 256 | 8192 | 4.0608 | 1.7235 |
| 4096 | 4096 | 64 | 1.0532 | 3.8945 |
| 2048 | 512 | 1024 | 1.2579 | 5.9401 |
| 512 | 2048 | 1024 | 1.1916 | 5.9679 |
| 1024 | 4096 | 256 | 1.2398 | 5.5526 |
| 8192 | 8192 | 128 | 1.2516 | 4.7707 |

## Grid

| param | values |
|---|---|
| BM | 48 64 96 128 192 256 |
| BN | 48 64 96 128 192 |
| BK | 16 32 64 |
| TM | 2 4 6 8 |
| TN | 2 4 6 8 |
| pruned | `BK=8` (2 float4 chunks per row of A); `TM/TN` 2/2, 2/4, 4/2 (intensity 1.00–1.33 vs 4.00) |

Per-run data in `results.csv`.
