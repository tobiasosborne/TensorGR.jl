#!/usr/bin/env julia
# Master runner for TensorGR.jl benchmark suite.
#
# Usage:
#   julia --project=. benchmarks/run_all.jl              # Tier 1 only
#   julia --project=. benchmarks/run_all.jl --tier 2     # Tiers 1-2
#   julia --project=. benchmarks/run_all.jl --tier 3     # All tiers
#   julia --project=. benchmarks/run_all.jl --bench 01   # Single benchmark

using Test

cd(@__DIR__)

# Parse arguments
tier_max = 1
bench_filter = nothing
for (i, arg) in enumerate(ARGS)
    if arg == "--tier" && i < length(ARGS)
        global tier_max = parse(Int, ARGS[i+1])
    elseif arg == "--bench" && i < length(ARGS)
        global bench_filter = ARGS[i+1]
    end
end

const tier1 = ["bench_01_xpert.jl", "bench_02_xtras.jl", "bench_03_xpand.jl", "bench_04_conformal.jl"]
const tier2 = ["bench_05_schwarzschild2.jl", "bench_06_gw_stress.jl", "bench_07_riemann_corr.jl", "bench_08_galileon.jl"]
const tier3 = ["bench_09_psalter.jl", "bench_10_eftpng.jl", "bench_11_superfield.jl", "bench_12_6deriv_dS.jl", "bench_13_spectrum.jl"]

all_benches = String[]
tier_max >= 1 && append!(all_benches, tier1)
tier_max >= 2 && append!(all_benches, tier2)
tier_max >= 3 && append!(all_benches, tier3)

if bench_filter !== nothing
    all_benches = filter(b -> occursin(bench_filter, b), all_benches)
end

# Only include files that exist
all_benches = filter(isfile, all_benches)

if isempty(all_benches)
    println("No benchmark files found (tier=$tier_max, filter=$bench_filter)")
    exit(1)
end

println("Running $(length(all_benches)) benchmark(s) (tier ≤ $tier_max):\n")

@testset "TensorGR Benchmarks" begin
    for bench in all_benches
        println("─── $bench ───")
        include(bench)
        println()
    end
end
