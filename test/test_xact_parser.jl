#= Test the xAct Invar parser: MaxIndex/MaxDualIndex extraction,
#  Mathematica expression parser, and cross-check against our database.
#
#  Ground truth: xAct Invar.m source at reference/xAct/xAct/Invar/Invar.m.
=#

# Load the parser script (standalone, no module needed)
include(joinpath(@__DIR__, "..", "scripts", "parse_xact_invar.jl"))

const _INVAR_M_PATH = joinpath(@__DIR__, "..", "reference", "xAct", "xAct", "Invar", "Invar.m")

@testset "xAct Invar Parser" begin

    # ---- Part A: MaxIndex extraction ----
    @testset "MaxIndex extraction" begin
        d = extract_max_indices(_INVAR_M_PATH)

        # Verify the dict is non-empty
        @test length(d) > 10

        # Known algebraic values (non-product canonical forms at Level 1)
        @test d[[0]] == 1           # degree 1: just R
        @test d[[0,0]] == 3         # degree 2: 3 non-product
        @test d[[0,0,0]] == 9       # degree 3: 9 non-product
        @test d[[0,0,0,0]] == 38    # degree 4: 38 non-product
        @test d[[0,0,0,0,0]] == 204          # degree 5
        @test d[[0,0,0,0,0,0]] == 1613       # degree 6
        @test d[[0,0,0,0,0,0,0]] == 16532    # degree 7
        @test d[[0,0,0,0,0,0,0,0]] == 217395    # degree 8
        @test d[[0,0,0,0,0,0,0,0,0]] == 3406747 # degree 9

        # Known differential values
        @test d[[2]] == 2            # 2 derivs + 1 curvature
        @test d[[0,2]] == 12         # degree 2 with 2 derivs
        @test d[[1,1]] == 12
        @test d[[4]] == 12           # 4 derivs + 1 curvature
        @test d[[6]] == 105
        @test d[[8]] == 1155
        @test d[[10]] == 15120
    end

    # ---- Part A: MaxDualIndex extraction ----
    @testset "MaxDualIndex extraction" begin
        d = extract_max_dual_indices(_INVAR_M_PATH)

        # Verify the dict is non-empty
        @test length(d) > 5

        # Known algebraic dual values
        @test d[[0]] == 1            # degree 1
        @test d[[0,0]] == 4          # degree 2
        @test d[[0,0,0]] == 27       # degree 3
        @test d[[0,0,0,0]] == 232    # degree 4
        @test d[[0,0,0,0,0]] == 2582         # degree 5
        @test d[[0,0,0,0,0,0]] == 35090      # degree 6
        @test d[[0,0,0,0,0,0,0]] == 558323   # degree 7

        # Known differential dual values
        @test d[[2]] == 3
        @test d[[0,2]] == 58
        @test d[[1,1]] == 36
        @test d[[4]] == 32
        @test d[[6]] == 435
    end

    # ---- Part B: Mathematica expression parser ----
    @testset "Mathematica parser: integers" begin
        e = parse_mathematica("42")
        @test e isa MInt
        @test e.val == 42

        e = parse_mathematica("0")
        @test e isa MInt
        @test e.val == 0
    end

    @testset "Mathematica parser: lists" begin
        e = parse_mathematica("{1, 2, 3}")
        @test e isa MList
        @test length(e.items) == 3
        @test e.items[1] isa MInt && e.items[1].val == 1
        @test e.items[2] isa MInt && e.items[2].val == 2
        @test e.items[3] isa MInt && e.items[3].val == 3

        # Nested list
        e = parse_mathematica("{1, {5, 6, 7, 8, 1, 2, 3, 4}}")
        @test e isa MList
        @test length(e.items) == 2
        @test e.items[1] isa MInt && e.items[1].val == 1
        @test e.items[2] isa MList
        @test length(e.items[2].items) == 8
        @test e.items[2].items[1].val == 5
        @test e.items[2].items[8].val == 4
    end

    @testset "Mathematica parser: function calls" begin
        e = parse_mathematica("RInv[{0,0}, 4]")
        @test e isa MCall
        @test e.name == :RInv
        @test length(e.args) == 2
        @test e.args[1] isa MList
        @test length(e.args[1].items) == 2
        @test e.args[2] isa MInt && e.args[2].val == 4
    end

    @testset "Mathematica parser: rules" begin
        e = parse_mathematica("RInv[{0,0}, 4] -> 1/2 RInv[{0,0}, 3]")
        @test e isa MRule
        @test e.lhs isa MCall && e.lhs.name == :RInv
        # RHS: 1/2 * RInv[{0,0}, 3]  (implicit multiplication)
        rhs = e.rhs
        @test rhs isa MTimes
        @test rhs.factors[1] isa MRational
        @test rhs.factors[1].num == 1
        @test rhs.factors[1].den == 2
        @test rhs.factors[2] isa MCall && rhs.factors[2].name == :RInv
    end

    @testset "Mathematica parser: arithmetic" begin
        # Addition
        e = parse_mathematica("1 + 2")
        @test e isa MPlus
        @test length(e.terms) == 2

        # Subtraction
        e = parse_mathematica("3 - 1")
        @test e isa MPlus
        @test e.terms[2] isa MNeg
        @test e.terms[2].arg isa MInt && e.terms[2].arg.val == 1

        # Multiplication
        e = parse_mathematica("2 * 3")
        @test e isa MTimes
        @test length(e.factors) == 2

        # Negation
        e = parse_mathematica("-5")
        @test e isa MNeg
        @test e.arg isa MInt && e.arg.val == 5

        # Parentheses
        e = parse_mathematica("(1 + 2)")
        @test e isa MPlus
    end

    @testset "Mathematica parser: complex rule" begin
        # A rule with sum on RHS, like Invar database relations
        e = parse_mathematica("RInv[{0,0,0}, 10] -> 1/2 RInv[{0,0,0}, 3] + RInv[{0,0,0}, 5]")
        @test e isa MRule
        @test e.rhs isa MPlus
        @test length(e.rhs.terms) == 2
    end

    @testset "Mathematica parser: empty list" begin
        e = parse_mathematica("{}")
        @test e isa MList
        @test isempty(e.items)
    end

    # ---- Part C: Cross-check against our database ----
    @testset "Cross-check: algebraic MaxIndex vs our database" begin
        result = cross_check_counts(; invar_m=_INVAR_M_PATH)

        # We should have at least the degrees 2-7 matching
        @test length(result.matches) >= 5  # degrees 2-7 minus degree 1 (no db entry)

        # No mismatches for algebraic cases
        @test isempty(result.mismatches)

        # Degree 1 is missing from our database (trivial case)
        degree1_missing = any(m -> m.degree == 1, result.missing_in_ours)
        @test degree1_missing

        # Verify specific matches
        for m in result.matches
            if m.degree == 2
                @test m.xact_count == 3
                @test m.our_count == 3   # 4 total - 1 product (R^2) = 3
            elseif m.degree == 4
                @test m.xact_count == 38
                @test m.our_count == 38  # 57 total - 19 products = 38
            end
        end

        # xAct has many differential cases we don't track yet
        @test length(result.xact_only_cases) > 0
    end

    @testset "Cross-check: non-product count derivation" begin
        # Verify the product count subtraction is correct for key degrees.
        # These are the key identities:
        #   total_canonical = non_product + products
        #   MaxIndex = non_product (for algebraic cases)

        # Degree 2: 4 canonical - 1 product (R^2) = 3 non-product
        @test 4 - _PRODUCT_COUNTS[2] == 3

        # Degree 3: 13 canonical - 4 products = 9 non-product
        @test 13 - _PRODUCT_COUNTS[3] == 9

        # Degree 4: 57 canonical - 19 products = 38 non-product
        @test 57 - _PRODUCT_COUNTS[4] == 38

        # Degree 5: 288 canonical - 84 products = 204 non-product
        @test 288 - _PRODUCT_COUNTS[5] == 204

        # Degree 6: 2070 canonical - 457 products = 1613 non-product
        @test 2070 - _PRODUCT_COUNTS[6] == 1613

        # Degree 7: 19610 canonical - 3078 products = 16532 non-product
        @test 19610 - _PRODUCT_COUNTS[7] == 16532
    end
end
