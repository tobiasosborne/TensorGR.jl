# test_golden.jl — TGR-bhs5.9 / .10
#
# Orchestrates all committed golden-master cases under test/golden/data/*.json.
# Loads every case file, runs each case through golden_runner.run_case, and
# asserts. Gated behind ENV["TENSORGR_GOLDEN"] so the default test suite is
# unaffected until Phase 1 stabilizes.

using Test
using JSON
using TensorGR

const GOLDEN_DIR = joinpath(@__DIR__, "golden")

include(joinpath(GOLDEN_DIR, "consumers", "golden_loader.jl"))
include(joinpath(GOLDEN_DIR, "consumers", "golden_emitter.jl"))
include(joinpath(GOLDEN_DIR, "consumers", "golden_runner.jl"))

"""
    golden_case_files() -> Vector{String}

All committed case files under `data/`. Excludes probe/fixture artifacts
(filenames starting with `_`) and the conventions probe (legacy v0 schema).
"""
function golden_case_files()
    files = String[]
    for f in readdir(joinpath(GOLDEN_DIR, "data"); join=true)
        endswith(f, ".json") || continue
        base = basename(f)
        startswith(base, "_") && continue
        base == "conventions_probe.json" && continue
        push!(files, f)
    end
    sort(files)
end

function run_golden_file(path::AbstractString; registry)
    doc = JSON.parsefile(path)
    cases = get(doc, "cases", Any[])
    results = Tuple{String, Bool, Any}[]
    for c in cases
        name = get(c, "name", basename(path))
        ok, diff = run_case(c; registry=registry)
        push!(results, (name, ok, diff))
    end
    results
end

@testset "golden master cases" begin
    reg = TensorRegistry()
    with_registry(reg) do
        @manifold M4 dim=4 metric=g
        define_curvature_tensors!(reg, :M4, :g)

        files = golden_case_files()
        @test !isempty(files)

        for f in files
            @testset "$(basename(f))" begin
                results = run_golden_file(f; registry=reg)
                for (name, ok, diff) in results
                    if !ok
                        @info "Golden diff" name=name diff=diff
                    end
                    @test ok
                end
            end
        end
    end
end
