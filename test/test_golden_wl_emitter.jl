# test_golden_wl_emitter.jl — TGR-bhs5.7
#
# Red-green TDD for the xAct-side emitter.
# RED: test/golden/generators/common.wl does not yet exist, so the probe
# script cannot load it; the expected-fixture comparison fails.
# GREEN: common.wl + _emitter_probe.wl produce JSON that round-trips
# through our schema-v1 loader.
#
# Requires `wolframscript` on PATH. Skipped gracefully if missing.

using Test
using JSON

const COMMON_WL   = joinpath(@__DIR__, "golden", "generators", "common.wl")
const PROBE_WL    = joinpath(@__DIR__, "golden", "generators", "_emitter_probe.wl")
const PROBE_OUT   = joinpath(@__DIR__, "golden", "data", "_emitter_probe.actual.json")

function have_wolframscript()
    try
        read(`wolframscript -code "1+1"`, String)
        return true
    catch
        return false
    end
end

@testset "golden xAct emitter (common.wl)" begin
    @test isfile(COMMON_WL)
    @test isfile(PROBE_WL)

    if !have_wolframscript()
        @info "wolframscript not available; skipping live probe."
        return
    end

    # Remove any stale actual output so we're sure we see this run's result.
    isfile(PROBE_OUT) && rm(PROBE_OUT)

    # Run the probe. wolframscript exits non-zero on script error.
    result = try
        run(`wolframscript -f $PROBE_WL`)
    catch e
        @error "wolframscript probe failed" exception=e
        rethrow()
    end

    @test isfile(PROBE_OUT)

    # Parse probe output. Should be an array of emitted-JSON records.
    actual = JSON.parsefile(PROBE_OUT)
    @test actual isa AbstractVector
    @test !isempty(actual)

    # The probe emits at least three cases:
    #  1. "g_down2"   : g[-a, -b]          → Tensor g_{ab}, all-down
    #  2. "g_up2"     : g[a, b]            → Tensor g^{ab}, all-up
    #  3. "ric_trace" : g^{ab} R_{ab}      → contracted scalar, one dummy pair
    by_name = Dict(r["name"] => r["emitted"] for r in actual)
    @test haskey(by_name, "g_down2")
    @test haskey(by_name, "g_up2")
    @test haskey(by_name, "ric_trace")

    # --- case 1: g_down2 ---
    g_down = by_name["g_down2"]
    @test g_down["type"] == "tensor"
    @test g_down["name"] == "g"
    @test length(g_down["indices"]) == 2
    @test all(i -> i["pos"] == "down", g_down["indices"])
    @test all(i -> i["vbundle"] == "Tangent", g_down["indices"])
    @test [i["name"] for i in g_down["indices"]] == ["a", "b"]

    # --- case 2: g_up2 ---
    g_up = by_name["g_up2"]
    @test g_up["type"] == "tensor"
    @test g_up["name"] == "g"
    @test [i["pos"]  for i in g_up["indices"]] == ["up", "up"]
    @test [i["name"] for i in g_up["indices"]] == ["a", "b"]

    # --- case 3: ric_trace, dummy normalized to d1 ---
    #   g^{ab} R_{ab} → product with one Ricci trace; after norm, dummies d1,d2
    ric_trace = by_name["ric_trace"]
    @test ric_trace["type"] in ("product", "scalar", "tensor")
    # Serialized form should be stable & contain d1/d2 (not a/b) after norm.
    blob = JSON.json(ric_trace)
    @test occursin("d1", blob)
    @test occursin("d2", blob)
    @test !occursin("\"name\":\"a\"", blob)  # raw a should be gone
    @test !occursin("\"name\":\"b\"", blob)
end
