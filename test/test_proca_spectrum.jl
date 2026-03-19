# Ground truth: Barker (2024) arXiv:2406.09500, PSALTer Section 8.2.
#
# Proca theory: L = -¼ F_{μν}F^{μν} + ½ m² A_μ A^μ
#
# Kinetic kernel in momentum space:
#   K_{μν} = (k² + m²) θ_{μν} + m² ω_{μν}
#
# At generic k²:
#   a₁ (spin-1) = k² + m²   (transverse, 3 massive polarizations)
#   a₀ (spin-0) = m²         (longitudinal, physical for m≠0)
#
# Propagator:
#   G_{μν} = θ_{μν}/(k² + m²) + ω_{μν}/m²
#
# Spectrum: 3 massive spin-1 modes (d-1 = 3 in d=4)
# Unitarity: ghost-free if m² > 0, tachyon-free if m² > 0
# No source constraints (all sectors propagate)

@testset "PSALTer Validation: Proca theory spectrum" begin

    @testset "Proca kernel decomposition" begin
        # At k²=1, m²=1: a₁ = 2, a₀ = 1
        K_proca = SpinSectorDecomposition(0, 2, 0, 1)
        @test K_proca.spin1 == 2    # k² + m² = 1 + 1
        @test K_proca.spin0w == 1   # m²
        @test K_proca.spin2 == 0    # no spin-2 (vector field)
        @test K_proca.spin0s == 0   # no spin-0s
    end

    @testset "Proca propagator" begin
        K_proca = SpinSectorDecomposition(0, 2, 0, 1)
        prop = moore_penrose_propagator(K_proca)

        @test prop.spin1 == 1//2    # 1/(k²+m²) at k²=m²=1
        @test prop.spin0w == 1//1   # 1/m²
        @test prop.spin2 == 0       # absent
        @test prop.spin0s == 0      # absent
    end

    @testset "Proca has 2 propagating sectors" begin
        K_proca = SpinSectorDecomposition(0, 2, 0, 1)
        prop = moore_penrose_propagator(K_proca)

        propagating = count(x -> x != 0,
            [prop.spin2, prop.spin1, prop.spin0s, prop.spin0w])
        @test propagating == 2  # spin-1 + spin-0w (= 3 DOF in d=4)
    end

    @testset "Proca unitarity: healthy for m² > 0" begin
        K_proca = SpinSectorDecomposition(0, 2, 0, 1)
        ua = unitarity_conditions(K_proca)

        @test ua.unitary
        @test ua.ghost_free
        @test ua.tachyon_free

        @test ua.sectors[:spin1].ghost_free
        @test ua.sectors[:spin0w].ghost_free
    end

    @testset "Proca source constraints: only absent sectors" begin
        K_proca = SpinSectorDecomposition(0, 2, 0, 1)
        sc = source_constraints(K_proca)

        # Only spin-2 and spin-0s are absent → 2 constraints
        @test length(sc) == 2
        sectors = Set(c.sector for c in sc)
        @test :spin2 in sectors
        @test :spin0s in sectors
        # Physical sectors have no constraints
        @test !(:spin1 in sectors)
        @test !(:spin0w in sectors)
    end

    @testset "Proca vs Maxwell: mass lifts gauge degeneracy" begin
        K_maxwell = SpinSectorDecomposition(0, 1, 0, 0)
        K_proca = SpinSectorDecomposition(0, 2, 0, 1)

        # Maxwell: 1 propagating sector (spin-1 only)
        prop_m = moore_penrose_propagator(K_maxwell)
        n_maxwell = count(x -> x != 0,
            [prop_m.spin2, prop_m.spin1, prop_m.spin0s, prop_m.spin0w])
        @test n_maxwell == 1

        # Proca: 2 propagating sectors (spin-1 + spin-0w)
        prop_p = moore_penrose_propagator(K_proca)
        n_proca = count(x -> x != 0,
            [prop_p.spin2, prop_p.spin1, prop_p.spin0s, prop_p.spin0w])
        @test n_proca == 2

        # Mass lifts the gauge degeneracy: spin-0w goes from 0 to m²
        @test K_maxwell.spin0w == 0  # gauge
        @test K_proca.spin0w == 1    # massive
    end

    @testset "Tachyonic Proca: m² < 0" begin
        # Negative mass² should be flagged
        K_tachyon = SpinSectorDecomposition(0, 0, 0, -1)  # k²=0, m²=-1
        mass_sq = SpinSectorDecomposition(0, 0, 0, -1)
        @test !no_tachyon(mass_sq)
    end

    @testset "String match: Proca spectrum summary" begin
        K = SpinSectorDecomposition(0, 2, 0, 1)
        ua = unitarity_conditions(K)

        # Ground truth from Barker (2024):
        # "massive spin-1 particle with 3 polarizations"
        @test ua.unitary          # healthy
        @test !ua.sectors[:spin1].is_gauge  # spin-1 propagates
        @test !ua.sectors[:spin0w].is_gauge # spin-0w propagates (longitudinal)
    end
end
