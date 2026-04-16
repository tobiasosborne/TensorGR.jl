# test_golden_loader.jl — TGR-bhs5.5
#
# Red-green TDD. RED first: loader does not yet exist.
# GREEN: `test/golden/consumers/golden_loader.jl` implements `load_expr`
# mapping schema v1 JSON dicts to TensorExpr nodes.

using Test
using TensorGR

const LOADER_PATH = joinpath(@__DIR__, "golden", "consumers", "golden_loader.jl")

@testset "golden loader v1" begin
    @test isfile(LOADER_PATH)
    include(LOADER_PATH)

    # load_expr should now be in scope.

    # --- fixture 1: scalar symbol ---
    f_scalar_sym = Dict(
        "type"  => "scalar",
        "value" => Dict("symbol" => "kappa"),
    )
    @test load_expr(f_scalar_sym) == TScalar(:kappa)

    # --- fixture 2: scalar rational ---
    f_scalar_rat = Dict(
        "type"  => "scalar",
        "value" => Dict("rational" => Dict("num" => 3, "den" => 7)),
    )
    @test load_expr(f_scalar_rat) == TScalar(3 // 7)

    # --- fixture 3: tensor with mixed indices, Riem^a_{bcd} ---
    f_tensor = Dict(
        "type" => "tensor",
        "name" => "Riem",
        "indices" => [
            Dict("name" => "a", "pos" => "up",   "vbundle" => "Tangent"),
            Dict("name" => "b", "pos" => "down", "vbundle" => "Tangent"),
            Dict("name" => "c", "pos" => "down", "vbundle" => "Tangent"),
            Dict("name" => "d", "pos" => "down", "vbundle" => "Tangent"),
        ],
    )
    expected_tensor = Tensor(:Riem, [up(:a), down(:b), down(:c), down(:d)])
    @test load_expr(f_tensor) == expected_tensor

    # --- fixture 4: product (1/2) * g_{ab} * R ---
    f_product = Dict(
        "type" => "product",
        "coef" => Dict("num" => 1, "den" => 2),
        "factors" => [
            Dict("type" => "tensor", "name" => "g",
                 "indices" => [Dict("name" => "a", "pos" => "down", "vbundle" => "Tangent"),
                               Dict("name" => "b", "pos" => "down", "vbundle" => "Tangent")]),
            Dict("type" => "tensor", "name" => "RicScalar", "indices" => []),
        ],
    )
    expected_product = TProduct(1 // 2, TensorExpr[
        Tensor(:g, [down(:a), down(:b)]),
        Tensor(:RicScalar, TIndex[]),
    ])
    @test load_expr(f_product) == expected_product

    # --- fixture 5: sum R_{ab} + G_{ab} ---
    f_sum = Dict(
        "type" => "sum",
        "terms" => [
            Dict("type" => "tensor", "name" => "Ric",
                 "indices" => [Dict("name" => "a", "pos" => "down", "vbundle" => "Tangent"),
                               Dict("name" => "b", "pos" => "down", "vbundle" => "Tangent")]),
            Dict("type" => "tensor", "name" => "Ein",
                 "indices" => [Dict("name" => "a", "pos" => "down", "vbundle" => "Tangent"),
                               Dict("name" => "b", "pos" => "down", "vbundle" => "Tangent")]),
        ],
    )
    expected_sum = TSum(TensorExpr[
        Tensor(:Ric, [down(:a), down(:b)]),
        Tensor(:Ein, [down(:a), down(:b)]),
    ])
    @test load_expr(f_sum) == expected_sum

    # --- fixture 6: deriv ∇_c R_{ab} under CovD D ---
    f_deriv = Dict(
        "type"  => "deriv",
        "covd"  => "D",
        "index" => Dict("name" => "c", "pos" => "down", "vbundle" => "Tangent"),
        "arg"   => Dict("type" => "tensor", "name" => "Ric",
                        "indices" => [Dict("name" => "a", "pos" => "down", "vbundle" => "Tangent"),
                                      Dict("name" => "b", "pos" => "down", "vbundle" => "Tangent")]),
    )
    expected_deriv = TDeriv(down(:c), Tensor(:Ric, [down(:a), down(:b)]), :D)
    @test load_expr(f_deriv) == expected_deriv

    # --- fixture 7: nested — sum of a product and a tensor ---
    f_nested = Dict(
        "type" => "sum",
        "terms" => [
            Dict("type" => "product",
                 "coef" => Dict("num" => -1, "den" => 2),
                 "factors" => [
                    Dict("type" => "tensor", "name" => "g",
                         "indices" => [Dict("name" => "a", "pos" => "down", "vbundle" => "Tangent"),
                                       Dict("name" => "b", "pos" => "down", "vbundle" => "Tangent")]),
                    Dict("type" => "tensor", "name" => "RicScalar", "indices" => []),
                 ]),
            Dict("type" => "tensor", "name" => "Ric",
                 "indices" => [Dict("name" => "a", "pos" => "down", "vbundle" => "Tangent"),
                               Dict("name" => "b", "pos" => "down", "vbundle" => "Tangent")]),
        ],
    )
    expected_nested = TSum(TensorExpr[
        TProduct(-1 // 2, TensorExpr[
            Tensor(:g, [down(:a), down(:b)]),
            Tensor(:RicScalar, TIndex[]),
        ]),
        Tensor(:Ric, [down(:a), down(:b)]),
    ])
    @test load_expr(f_nested) == expected_nested

    # --- malformed: missing "type" ---
    @test_throws Exception load_expr(Dict("name" => "Riem"))

    # --- malformed: unknown type tag ---
    @test_throws Exception load_expr(Dict("type" => "bogus"))
end
