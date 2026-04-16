# test_golden_schema.jl — TGR-bhs5.4
#
# Red-green TDD. RED first: schema/v1.json does not yet exist.
# GREEN: once schema is written, all assertions pass.
#
# Schema acceptance:
#   1. `test/golden/schema/v1.json` exists, parses as JSON.
#   2. Top-level declares $schema, title, and an Expr definition.
#   3. A canned VALID Expr fixture validates.
#   4. A canned MALFORMED Expr fixture is rejected.

using Test
using JSON
using JSONSchema

const SCHEMA_PATH = joinpath(@__DIR__, "golden", "schema", "v1.json")

"""
Minimal valid Tensor node under schema v1. `Riem_{abcd}`.
"""
const VALID_TENSOR = Dict(
    "type" => "tensor",
    "name" => "Riem",
    "indices" => [
        Dict("name" => "a", "pos" => "down", "vbundle" => "Tangent"),
        Dict("name" => "b", "pos" => "down", "vbundle" => "Tangent"),
        Dict("name" => "c", "pos" => "down", "vbundle" => "Tangent"),
        Dict("name" => "d", "pos" => "down", "vbundle" => "Tangent"),
    ],
)

"""
Minimal valid Product node under schema v1. `(1/2) * g_{ab} * R`.
"""
const VALID_PRODUCT = Dict(
    "type" => "product",
    "coef" => Dict("num" => 1, "den" => 2),
    "factors" => [
        Dict("type" => "tensor", "name" => "g",
             "indices" => [Dict("name" => "a", "pos" => "down", "vbundle" => "Tangent"),
                           Dict("name" => "b", "pos" => "down", "vbundle" => "Tangent")]),
        Dict("type" => "tensor", "name" => "RicScalar", "indices" => []),
    ],
)

"""
Invalid: Tensor node missing required "indices" key.
"""
const INVALID_TENSOR_NO_INDICES = Dict(
    "type" => "tensor",
    "name" => "Riem",
)

"""
Invalid: Product has bad coef (missing "den").
"""
const INVALID_PRODUCT_BAD_COEF = Dict(
    "type" => "product",
    "coef" => Dict("num" => 1),
    "factors" => [],
)

@testset "golden schema v1" begin
    @test isfile(SCHEMA_PATH)

    schema_doc = JSON.parsefile(SCHEMA_PATH)

    @test haskey(schema_doc, "\$schema")
    @test haskey(schema_doc, "title")
    @test occursin("Expr", get(schema_doc, "title", ""))
    @test haskey(schema_doc, "\$defs") || haskey(schema_doc, "definitions")

    sch = Schema(schema_doc)

    @testset "accepts valid fixtures" begin
        @test isvalid(sch, VALID_TENSOR)
        @test isvalid(sch, VALID_PRODUCT)
    end

    @testset "rejects malformed fixtures" begin
        @test !isvalid(sch, INVALID_TENSOR_NO_INDICES)
        @test !isvalid(sch, INVALID_PRODUCT_BAD_COEF)
    end
end
