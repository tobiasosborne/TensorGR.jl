# test_golden_emitter.jl — TGR-bhs5.6
#
# Red-green TDD. RED: emitter does not exist.
# GREEN: test/golden/consumers/golden_emitter.jl implements
#   - emit_json(e; normalize=true) -> Dict
#   - normalize_dummies_golden(e) -> TensorExpr (d1..dN in AST order)

using Test
using TensorGR

const LOADER_PATH   = joinpath(@__DIR__, "golden", "consumers", "golden_loader.jl")
const EMITTER_PATH  = joinpath(@__DIR__, "golden", "consumers", "golden_emitter.jl")

@testset "golden emitter v1" begin
    @test isfile(EMITTER_PATH)
    include(LOADER_PATH)
    include(EMITTER_PATH)

    # --- round-trip WITHOUT normalization, structural equality preserved ---
    @testset "round-trip" begin
        exprs = [
            TScalar(:kappa),
            TScalar(3 // 7),
            Tensor(:Riem, [up(:a), down(:b), down(:c), down(:d)]),
            TProduct(1 // 2, TensorExpr[
                Tensor(:g, [down(:a), down(:b)]),
                Tensor(:RicScalar, TIndex[]),
            ]),
            TSum(TensorExpr[
                Tensor(:Ric, [down(:a), down(:b)]),
                Tensor(:Ein, [down(:a), down(:b)]),
            ]),
            TDeriv(down(:c), Tensor(:Ric, [down(:a), down(:b)]), :D),
        ]
        for e in exprs
            j = emit_json(e; normalize=false)
            @test load_expr(j) == e
        end
    end

    # --- normalization: R^a_a contraction should get dummy d1 ---
    @testset "dummy normalization" begin
        # Ric^a_a: contracted trace — one dummy pair.
        trace = Tensor(:Ric, [up(:a), down(:a)])
        norm  = normalize_dummies_golden(trace)
        @test norm == Tensor(:Ric, [up(:d1), down(:d1)])

        # Two independent dummy pairs in a product: R^{ab} R_{ab}
        # Indices appear at first use in order a(up), b(up), a(down), b(down)
        # → a=d1, b=d2.
        kretsch = TProduct(1 // 1, TensorExpr[
            Tensor(:Ric, [up(:a), up(:b)]),
            Tensor(:Ric, [down(:a), down(:b)]),
        ])
        normk = normalize_dummies_golden(kretsch)
        @test normk == TProduct(1 // 1, TensorExpr[
            Tensor(:Ric, [up(:d1), up(:d2)]),
            Tensor(:Ric, [down(:d1), down(:d2)]),
        ])

        # Free indices preserved (R_{ab} with free a,b — no dummies).
        free = Tensor(:Ric, [down(:a), down(:b)])
        @test normalize_dummies_golden(free) == free

        # Idempotence.
        @test normalize_dummies_golden(normalize_dummies_golden(kretsch)) ==
              normalize_dummies_golden(kretsch)
    end

    # --- emit_json(normalize=true) folds normalization into the serialization ---
    @testset "emit with normalization" begin
        trace = Tensor(:Ric, [up(:a), down(:a)])
        j = emit_json(trace)  # normalize=true default
        loaded = load_expr(j)
        @test loaded == Tensor(:Ric, [up(:d1), down(:d1)])
    end

    # --- sum: per-term normalization (each summand independent dummies) ---
    @testset "sum per-term normalization" begin
        # (R^a_a) + (R^b_b) — two terms, each with its own dummy pair.
        # After normalization each term should use d1 locally.
        e = TSum(TensorExpr[
            Tensor(:Ric, [up(:a), down(:a)]),
            Tensor(:Ric, [up(:b), down(:b)]),
        ])
        n = normalize_dummies_golden(e)
        expected = TSum(TensorExpr[
            Tensor(:Ric, [up(:d1), down(:d1)]),
            Tensor(:Ric, [up(:d1), down(:d1)]),
        ])
        @test n == expected
    end
end
