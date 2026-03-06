# TensorGR.jl Benchmark Suite

Ground-truth validation against published xAct (Mathematica) calculations.
Each benchmark reproduces specific equations from a peer-reviewed paper.

## Running

```bash
# Tier 1 only (default, ~25s)
julia --project=. benchmarks/run_all.jl

# Tiers 1-2
julia --project=. benchmarks/run_all.jl --tier 2

# All tiers
julia --project=. benchmarks/run_all.jl --tier 3

# Single benchmark
julia --project=. benchmarks/run_all.jl --bench 01

# With performance timing
TENSORGR_BENCH_PERF=1 julia --project=. benchmarks/run_all.jl

# Standalone
julia --project=. benchmarks/bench_01_xpert.jl
```

## Paper Reference

| # | Paper | arXiv | What it tests | Status |
|---|-------|-------|---------------|--------|
| **Tier 1: Core validation** |
| 01 | xPert (Brizuela+ 2009) | [0807.0824](https://arxiv.org/abs/0807.0824) | Perturbation engine, canonicalization | PASS |
| 02 | xTras (Nutma 2013) | [1308.3493](https://arxiv.org/abs/1308.3493) | Ansatz, VarD, contraction enumeration | PASS (1 broken) |
| 03 | xPand (Pitrou+ 2013) | [1302.6174](https://arxiv.org/abs/1302.6174) | Foliation, SVT decomposition | PASS |
| 04 | Conformal gravity (Grumiller+ 2013) | [1310.0819](https://arxiv.org/abs/1310.0819) | Weyl tensor, curvature algebra | PASS (1 broken) |
| **Tier 2: Stress tests** |
| 05 | 2nd-order Schwarzschild (Brizuela+ 2009) | [0903.1134](https://arxiv.org/abs/0903.1134) | Curved-bg perturbation, expression swell | stub |
| 06 | GW stress-energy (Stein & Yunes 2011) | [1012.3144](https://arxiv.org/abs/1012.3144) | Modified gravity, Levi-Civita | stub |
| 07 | Riemann correlator (Frob+ 2014) | [1403.3335](https://arxiv.org/abs/1403.3335) | de Sitter perturbation, Weyl decomp | stub |
| 08 | Galileon (Deffayet+ 2009) | [0901.1314](https://arxiv.org/abs/0901.1314) | Higher-derivative scalar-tensor | stub |
| **Tier 3: Stretch goals** |
| 09 | PSALTer (Barker+ 2024) | [2406.09500](https://arxiv.org/abs/2406.09500) | Spin-projection operators | stub |
| 10 | EFTofPNG (Levi & Steinhoff 2016) | [1705.06309](https://arxiv.org/abs/1705.06309) | Expression swell at high PN order | stub |
| 11 | Superfield (Green+ 2005) | [hep-th/0506161](https://arxiv.org/abs/hep-th/0506161) | High-rank canonicalization | stub |

## Findings

Issues discovered by this benchmark suite:

1. **Metric self-trace not resolved** (bench_04): `simplify(g[c,-c])` does not reduce
   to dimension=4 within products. Weyl trace-free identity requires manual
   `contract_curvature` + `contract_metrics` chain and still fails on metric
   self-contractions.

2. **`all_contractions` incomplete** (bench_02): The automatic contraction enumeration
   function exists but doesn't produce the complete set of independent contractions
   for the spin-2 Lagrangian problem.

## Adding a Benchmark

1. Create `bench_NN_shortname.jl` following the template in existing benchmarks
2. Add ground truth constants to `ground_truth.jl`
3. Add the filename to the appropriate tier list in `run_all.jl`
4. Run standalone first, then via `run_all.jl`

## Source Papers

TeX sources and PDFs are in `papers/` and `papers/src/` respectively.
