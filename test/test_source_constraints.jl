# Ground truth: Barker (2024) arXiv:2406.09500, PSALTer Section 7.
#
# Gauge invariance → source constraints:
# - Spin-1 gauge (vector gauge invariance) → ∂_μ J^μ = 0 (current conservation)
# - Spin-0w gauge (Weyl/conformal) → T^μ_μ = 0 (tracelessness)

@testset "Source constraints from gauge invariance" begin

    @testset "Fierz-Pauli (linearized GR): spin-1 + spin-0w gauge" begin
        # FP: a₂ = 5/2, a₁ = 0, a₀ₛ = -1, a₀w = 0
        K_FP = SpinSectorDecomposition(5//2, 0, -1//1, 0)
        sc = source_constraints(K_FP)

        @test length(sc) == 2  # spin-1 and spin-0w are gauge

        sectors = Set(c.sector for c in sc)
        @test :spin1 in sectors    # ∂_μ T^{μν} = 0
        @test :spin0w in sectors   # tracelessness not enforced in GR, but gauge mode

        # Check constraint types
        types = Set(c.constraint for c in sc)
        @test :transverse in types
        @test :traceless in types
    end

    @testset "Maxwell (U(1) gauge): spin-1 constraints" begin
        # For a vector field A_μ: only spin-1 is physical
        # All other sectors are zero
        K_maxwell = SpinSectorDecomposition(0, 1, 0, 0)
        sc = source_constraints(K_maxwell)

        # Spin-2, spin-0s, spin-0w are all gauge → 3 constraints
        @test length(sc) == 3
        sectors = Set(c.sector for c in sc)
        @test :spin2 in sectors
        @test :spin0s in sectors
        @test :spin0w in sectors
        @test !(:spin1 in sectors)
    end

    @testset "R² theory: only spin-0s physical" begin
        K_R2 = SpinSectorDecomposition(0, 0, 3, 0)
        sc = source_constraints(K_R2)

        # spin-2, spin-1, spin-0w are gauge → 3 constraints
        @test length(sc) == 3
        @test !any(c -> c.sector == :spin0s, sc)
    end

    @testset "All sectors physical: no constraints" begin
        K = SpinSectorDecomposition(1, 2, 3, 4)
        sc = source_constraints(K)
        @test isempty(sc)
    end

    @testset "All sectors gauge: 4 constraints" begin
        K = SpinSectorDecomposition(0, 0, 0, 0)
        sc = source_constraints(K)
        @test length(sc) == 4
    end

    @testset "Float tolerance" begin
        K = SpinSectorDecomposition(1.0, 1e-15, 2.0, 0.0)
        # Without tolerance: spin-1 is not gauge
        sc1 = source_constraints(K)
        spin1_constraint = any(c -> c.sector == :spin1, sc1)
        @test !spin1_constraint  # 1e-15 is not exactly zero

        # With tolerance: spin-1 treated as gauge
        sc2 = source_constraints(K; atol=1e-10)
        spin1_constraint2 = any(c -> c.sector == :spin1, sc2)
        @test spin1_constraint2
    end

    @testset "Constraint descriptions" begin
        K_FP = SpinSectorDecomposition(5//2, 0, -1//1, 0)
        sc = source_constraints(K_FP)
        for c in sc
            @test !isempty(c.description)
            @test c.description isa String
        end

        # Spin-1 should mention transverse/conservation
        spin1_c = first(c for c in sc if c.sector == :spin1)
        @test occursin("transverse", spin1_c.description)
    end

    @testset "Display" begin
        sc = SourceConstraint(:spin1, :transverse, "test")
        s = sprint(show, sc)
        @test occursin("SourceConstraint", s)
        @test occursin("spin1", s)
    end
end
