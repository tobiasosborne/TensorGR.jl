# test_golden_runner.jl — TGR-bhs5.8
#
# RED: runner missing.
# GREEN: test/golden/consumers/golden_runner.jl provides run_case(case) -> (ok, diff).
#
# Runner obligations:
#  - Dispatch on case["op"] (:simplify, :canonicalize, ...).
#  - Apply op to loaded input, re-canonicalize + normalize dummies on both
#    sides, then compare.
#  - On mismatch, return a human-readable JSON diff.
#  - Hand-built passing case → ok=true.
#  - Hand-built failing case (expected deliberately wrong) → ok=false, diff
#    references the mismatched branch.

using Test
using TensorGR

const LOADER_PATH  = joinpath(@__DIR__, "golden", "consumers", "golden_loader.jl")
const EMITTER_PATH = joinpath(@__DIR__, "golden", "consumers", "golden_emitter.jl")
const RUNNER_PATH  = joinpath(@__DIR__, "golden", "consumers", "golden_runner.jl")

@testset "golden runner v1" begin
    @test isfile(RUNNER_PATH)
    include(LOADER_PATH)
    include(EMITTER_PATH)
    include(RUNNER_PATH)

    reg = TensorRegistry()
    with_registry(reg) do
        @manifold M4 dim=4 metric=g
        define_curvature_tensors!(reg, :M4, :g)

        # Build a reference "first Bianchi 3-term" case by hand.
        riem(i1,i2,i3,i4) = Dict(
            "type" => "tensor",
            "name" => "Riem",
            "indices" => [
                Dict("name" => string(i1), "pos" => "down", "vbundle" => "Tangent"),
                Dict("name" => string(i2), "pos" => "down", "vbundle" => "Tangent"),
                Dict("name" => string(i3), "pos" => "down", "vbundle" => "Tangent"),
                Dict("name" => string(i4), "pos" => "down", "vbundle" => "Tangent"),
            ],
        )

        # --- passing case: simplify(0) == 0 ---
        pass_case = Dict(
            "name"     => "trivial_zero",
            "input"    => Dict("type" => "scalar",
                               "value" => Dict("rational" => Dict("num" => 0, "den" => 1))),
            "op"       => "simplify",
            "expected" => Dict("type" => "scalar",
                               "value" => Dict("rational" => Dict("num" => 0, "den" => 1))),
        )
        ok, diff = run_case(pass_case; registry=reg)
        @test ok
        @test diff === nothing

        # --- passing case: simplify(Riem_{abcd}) == Riem_{abcd} ---
        pass_tensor = Dict(
            "name"     => "riem_identity",
            "input"    => riem(:a, :b, :c, :d),
            "op"       => "simplify",
            "expected" => riem(:a, :b, :c, :d),
        )
        ok, diff = run_case(pass_tensor; registry=reg)
        @test ok

        # --- failing case: wrong expected ---
        fail_case = Dict(
            "name"     => "wrong_expected",
            "input"    => riem(:a, :b, :c, :d),
            "op"       => "simplify",
            "expected" => riem(:a, :b, :c, :e),  # wrong free index
        )
        ok, diff = run_case(fail_case; registry=reg)
        @test !ok
        @test diff !== nothing
        @test diff isa AbstractString
        @test occursin("expected", lowercase(diff)) || occursin("got", lowercase(diff))

        # --- unknown op should raise ---
        bad_op_case = Dict(
            "name"     => "bogus_op",
            "input"    => riem(:a, :b, :c, :d),
            "op"       => "bogus_nonexistent_op",
            "expected" => riem(:a, :b, :c, :d),
        )
        @test_throws Exception run_case(bad_op_case; registry=reg)
    end
end
