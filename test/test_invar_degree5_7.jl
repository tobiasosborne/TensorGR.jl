#= Test the Invar degree 5-7 algebraic invariant database.
#
# Ground truth: Garcia-Parrado & Martin-Garcia, arXiv:0704.1756 (2007), Tables 1-2;
#               Martin-Garcia, Portugal & Manssur, arXiv:0802.1274 (2008), Table 1.
#
# Degrees 5-7 store counts only (no explicit RInv forms or relations),
# because full enumeration is computationally infeasible at these degrees.
=#

@testset "Invar Database: Degrees 5-7 Data" begin

    # ---- Degree 5 ----

    @testset "Degree 5: Level 1 (288 canonical forms)" begin
        cr1 = get_invar_relations(5, "0_0_0_0_0", 1)
        @test cr1.degree == 5
        @test cr1.case_key == "0_0_0_0_0"
        @test cr1.step == 1
        @test cr1.dim === nothing
        @test cr1.n_independent == 288
        @test cr1.n_dependent == 0
        @test isempty(cr1.relations)
    end

    @testset "Degree 5: Level 2 (75 independent)" begin
        cr2 = get_invar_relations(5, "0_0_0_0_0", 2)
        @test cr2.degree == 5
        @test cr2.step == 2
        @test cr2.n_independent == 75
        @test cr2.n_dependent == 213
        @test isempty(cr2.relations)  # counts only, no explicit relations
    end

    @testset "Degree 5: completeness (75 + 213 = 288)" begin
        cr2 = get_invar_relations(5, "0_0_0_0_0", 2)
        cr1 = get_invar_relations(5, "0_0_0_0_0", 1)
        @test cr2.n_independent + cr2.n_dependent == cr1.n_independent
    end

    # ---- Degree 6 ----

    @testset "Degree 6: Level 1 (2070 canonical forms)" begin
        cr1 = get_invar_relations(6, "0_0_0_0_0_0", 1)
        @test cr1.degree == 6
        @test cr1.case_key == "0_0_0_0_0_0"
        @test cr1.step == 1
        @test cr1.dim === nothing
        @test cr1.n_independent == 2070
        @test cr1.n_dependent == 0
        @test isempty(cr1.relations)
    end

    @testset "Degree 6: Level 2 (409 independent)" begin
        cr2 = get_invar_relations(6, "0_0_0_0_0_0", 2)
        @test cr2.degree == 6
        @test cr2.step == 2
        @test cr2.n_independent == 409
        @test cr2.n_dependent == 1661
        @test isempty(cr2.relations)
    end

    @testset "Degree 6: completeness (409 + 1661 = 2070)" begin
        cr2 = get_invar_relations(6, "0_0_0_0_0_0", 2)
        cr1 = get_invar_relations(6, "0_0_0_0_0_0", 1)
        @test cr2.n_independent + cr2.n_dependent == cr1.n_independent
    end

    # ---- Degree 7 ----

    @testset "Degree 7: Level 1 (19610 canonical forms)" begin
        cr1 = get_invar_relations(7, "0_0_0_0_0_0_0", 1)
        @test cr1.degree == 7
        @test cr1.case_key == "0_0_0_0_0_0_0"
        @test cr1.step == 1
        @test cr1.dim === nothing
        @test cr1.n_independent == 19610
        @test cr1.n_dependent == 0
        @test isempty(cr1.relations)
    end

    @testset "Degree 7: Level 2 (2247 independent)" begin
        cr2 = get_invar_relations(7, "0_0_0_0_0_0_0", 2)
        @test cr2.degree == 7
        @test cr2.step == 2
        @test cr2.n_independent == 2247
        @test cr2.n_dependent == 17363
        @test isempty(cr2.relations)
    end

    @testset "Degree 7: completeness (2247 + 17363 = 19610)" begin
        cr2 = get_invar_relations(7, "0_0_0_0_0_0_0", 2)
        cr1 = get_invar_relations(7, "0_0_0_0_0_0_0", 1)
        @test cr2.n_independent + cr2.n_dependent == cr1.n_independent
    end

    # ---- list_invar_cases includes degrees 5, 6, 7 ----

    @testset "list_invar_cases includes degree 5" begin
        cases = list_invar_cases(degree=5)
        @test length(cases) >= 2
        @test any(c -> c.step == 1 && c.n_independent == 288, cases)
        @test any(c -> c.step == 2 && c.n_independent == 75, cases)
    end

    @testset "list_invar_cases includes degree 6" begin
        cases = list_invar_cases(degree=6)
        @test length(cases) >= 2
        @test any(c -> c.step == 1 && c.n_independent == 2070, cases)
        @test any(c -> c.step == 2 && c.n_independent == 409, cases)
    end

    @testset "list_invar_cases includes degree 7" begin
        cases = list_invar_cases(degree=7)
        @test length(cases) >= 2
        @test any(c -> c.step == 1 && c.n_independent == 19610, cases)
        @test any(c -> c.step == 2 && c.n_independent == 2247, cases)
    end

    # ---- Case keys are correct ----

    @testset "Case keys match degree" begin
        @test TensorGR._algebraic_case_key(5) == "0_0_0_0_0"
        @test TensorGR._algebraic_case_key(6) == "0_0_0_0_0_0"
        @test TensorGR._algebraic_case_key(7) == "0_0_0_0_0_0_0"
    end

    # ---- Monotonicity: independent count increases with degree ----

    @testset "Monotonicity: independent counts increase with degree" begin
        indep = [
            get_invar_relations(d, TensorGR._algebraic_case_key(d), 2).n_independent
            for d in 2:7
        ]
        for i in 1:length(indep)-1
            @test indep[i] < indep[i+1]
        end
    end

    # ---- Canonical count >= independent count (always) ----

    @testset "Canonical >= independent for degrees 5-7" begin
        for d in 5:7
            key = TensorGR._algebraic_case_key(d)
            cr1 = get_invar_relations(d, key, 1)
            cr2 = get_invar_relations(d, key, 2)
            @test cr1.n_independent >= cr2.n_independent
        end
    end

    # ---- Count accessor functions ----

    @testset "degree5_7_canonical_count" begin
        @test degree5_7_canonical_count(5) == 288
        @test degree5_7_canonical_count(6) == 2070
        @test degree5_7_canonical_count(7) == 19610
        @test_throws ArgumentError degree5_7_canonical_count(4)
        @test_throws ArgumentError degree5_7_canonical_count(8)
    end

    @testset "degree5_7_independent_count" begin
        @test degree5_7_independent_count(5) == 75
        @test degree5_7_independent_count(6) == 409
        @test degree5_7_independent_count(7) == 2247
        @test_throws ArgumentError degree5_7_independent_count(4)
        @test_throws ArgumentError degree5_7_independent_count(8)
    end
end
