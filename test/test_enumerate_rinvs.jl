@testset "Enumerate Independent RInvs" begin
    using TensorGR: enumerate_independent_rinvs, enumerate_live_canonical_rinvs,
                    RInv, InvarRelation,
                    degree2_canonical_rinvs, degree2_independent_rinvs,
                    degree3_canonical_rinvs, degree3_independent_rinvs,
                    degree4_canonical_rinvs, degree4_independent_rinvs

    function _make_enum_registry()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
        end
        return reg
    end

    # ---- Input validation -------------------------------------------------------

    @testset "Input Validation" begin
        @test_throws ArgumentError enumerate_independent_rinvs(0)
        @test_throws ArgumentError enumerate_independent_rinvs(1)
        @test_throws ArgumentError enumerate_independent_rinvs(-1)
        @test_throws ArgumentError enumerate_independent_rinvs(2; level=0)
        @test_throws ArgumentError enumerate_independent_rinvs(2; level=3)
    end

    # ---- Degree 2: Ground truth (Fulling et al. 1992, Table 1) ------------------

    @testset "Degree-2, Level 1 (permutation symmetries)" begin
        result = enumerate_independent_rinvs(2; level=1)
        @test length(result.canonical) == 4
        @test length(result.independent) == 4
        @test isempty(result.relations)
        @test all(r -> r.canonical, result.canonical)
        @test all(r -> r.degree == 2, result.canonical)
    end

    @testset "Degree-2, Level 2 (Bianchi)" begin
        result = enumerate_independent_rinvs(2; level=2)
        @test length(result.canonical) == 4
        @test length(result.independent) == 3
        @test length(result.relations) == 1

        # Specific Bianchi relation: I4 = (1/2)*I3
        rel = result.relations[1]
        @test rel.lhs == [5, 7, 6, 8, 1, 3, 2, 4]
        @test length(rel.rhs) == 1
        @test rel.rhs[1][1] == 1 // 2
        @test rel.rhs[1][2] == [5, 6, 7, 8, 1, 2, 3, 4]
    end

    @testset "Degree-2 matches named accessors" begin
        result = enumerate_independent_rinvs(2; level=2)
        named_can = degree2_canonical_rinvs()
        named_indep = degree2_independent_rinvs()

        can_set = Set(r.contraction for r in result.canonical)
        named_can_set = Set(r.contraction for r in named_can)
        @test can_set == named_can_set

        indep_set = Set(r.contraction for r in result.independent)
        named_indep_set = Set(r.contraction for r in named_indep)
        @test indep_set == named_indep_set
    end

    # ---- Degree 3: Ground truth (Fulling et al. 1992, Table 2) ------------------

    @testset "Degree-3, Level 1" begin
        result = enumerate_independent_rinvs(3; level=1)
        @test length(result.canonical) == 13
        @test length(result.independent) == 13
        @test isempty(result.relations)
    end

    @testset "Degree-3, Level 2 (Bianchi)" begin
        result = enumerate_independent_rinvs(3; level=2)
        @test length(result.canonical) == 13
        @test length(result.independent) == 8
        @test length(result.relations) == 5
    end

    @testset "Degree-3 matches named accessors" begin
        result = enumerate_independent_rinvs(3; level=2)
        named_can = degree3_canonical_rinvs()
        named_indep = degree3_independent_rinvs()

        can_set = Set(r.contraction for r in result.canonical)
        named_can_set = Set(r.contraction for r in named_can)
        @test can_set == named_can_set

        indep_set = Set(r.contraction for r in result.independent)
        named_indep_set = Set(r.contraction for r in named_indep)
        @test indep_set == named_indep_set
    end

    # ---- Degree 4: Ground truth (Martin-Garcia et al. 2007, Tables 1-2) ---------

    @testset "Degree-4, Level 2 (Bianchi)" begin
        result = enumerate_independent_rinvs(4; level=2)
        @test length(result.canonical) == 57
        @test length(result.independent) == 26
        @test length(result.relations) == 31
    end

    @testset "Degree-4 matches named accessors" begin
        result = enumerate_independent_rinvs(4; level=2)
        named_can = degree4_canonical_rinvs()
        named_indep = degree4_independent_rinvs()

        can_set = Set(r.contraction for r in result.canonical)
        named_can_set = Set(r.contraction for r in named_can)
        @test can_set == named_can_set

        indep_set = Set(r.contraction for r in result.independent)
        named_indep_set = Set(r.contraction for r in named_indep)
        @test indep_set == named_indep_set
    end

    # ---- Structural properties ---------------------------------------------------

    @testset "Canonical form properties" begin
        for deg in [2, 3, 4]
            result = enumerate_independent_rinvs(deg; level=2)
            for r in result.canonical
                @test r.canonical
                @test r.degree == deg
                # Valid involution: sigma(sigma(i)) == i, no fixed points
                n = 4 * deg
                for i in 1:n
                    @test r.contraction[r.contraction[i]] == i
                    @test r.contraction[i] != i
                end
            end
            # All distinct
            contractions = [r.contraction for r in result.canonical]
            @test length(Set(contractions)) == length(contractions)
        end
    end

    @testset "Independent forms not in dependent LHS" begin
        for deg in [2, 3, 4]
            result = enumerate_independent_rinvs(deg; level=2)
            dep_lhs = Set(rel.lhs for rel in result.relations)
            for r in result.independent
                @test r.contraction ∉ dep_lhs
            end
        end
    end

    @testset "Relations: LHS is canonical, RHS coefficients rational" begin
        for deg in [2, 3, 4]
            result = enumerate_independent_rinvs(deg; level=2)
            can_set = Set(r.contraction for r in result.canonical)
            for rel in result.relations
                @test rel.lhs ∈ can_set
                for (coeff, c) in rel.rhs
                    @test coeff isa Rational{Int}
                    @test c ∈ can_set
                end
            end
        end
    end

    @testset "Counts: canonical = independent + dependent" begin
        for deg in [2, 3, 4]
            result = enumerate_independent_rinvs(deg; level=2)
            @test length(result.canonical) ==
                  length(result.independent) + length(result.relations)
        end
    end

    # ---- Live enumeration cross-validation (degrees 2-3) -------------------------

    @testset "Live enumeration degree 2" begin
        reg = _make_enum_registry()
        with_registry(reg) do
            live = enumerate_live_canonical_rinvs(2; registry=reg, metric=:g)
            db_result = enumerate_independent_rinvs(2; level=1)
            @test length(live) == length(db_result.canonical) == 4

            live_set = Set(r.contraction for r in live)
            db_set = Set(r.contraction for r in db_result.canonical)
            @test live_set == db_set
        end
    end

    @testset "Live enumeration degree 3" begin
        reg = _make_enum_registry()
        with_registry(reg) do
            live = enumerate_live_canonical_rinvs(3; registry=reg, metric=:g)
            db_result = enumerate_independent_rinvs(3; level=1)
            @test length(live) == length(db_result.canonical) == 13

            live_set = Set(r.contraction for r in live)
            db_set = Set(r.contraction for r in db_result.canonical)
            @test live_set == db_set
        end
    end

    # ---- Idempotency -------------------------------------------------------------

    @testset "Repeated calls return same result" begin
        r1 = enumerate_independent_rinvs(3; level=2)
        r2 = enumerate_independent_rinvs(3; level=2)
        @test Set(r.contraction for r in r1.canonical) ==
              Set(r.contraction for r in r2.canonical)
        @test Set(r.contraction for r in r1.independent) ==
              Set(r.contraction for r in r2.independent)
    end
end
