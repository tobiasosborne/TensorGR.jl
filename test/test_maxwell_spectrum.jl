# Ground truth: Barker (2024) arXiv:2406.09500, PSALTer Section 8.1.
# Also: standard QFT textbook result.
#
# Maxwell theory: L = -¼ F_{μν} F^{μν}
#
# In momentum space, the kinetic kernel for A_μ is:
#   K_{μν} = k² θ_{μν} = k² (η_{μν} - k_μ k_ν/k²)
#
# Spin decomposition:
#   a₁ (spin-1) = k²   (transverse, physical photon)
#   a₀ (spin-0) = 0     (longitudinal, gauge mode)
#
# Propagator (Moore-Penrose):
#   G_{μν} = θ_{μν}/k²  (transverse propagator)
#
# Unitarity: ghost-free (positive residue), tachyon-free (massless pole)
# Source constraint: ∂_μ J^μ = 0 (current conservation)

@testset "PSALTer Validation: Maxwell theory spectrum" begin

    @testset "Maxwell kinetic kernel: spin-1 = k², spin-0 = 0" begin
        # At k²=1, the form factors are:
        K_maxwell = SpinSectorDecomposition(0, 1, 0, 0)
        # No spin-2 (vector field, not tensor)
        @test K_maxwell.spin2 == 0
        # Spin-1: physical photon, a₁ = k² → 1 at k²=1
        @test K_maxwell.spin1 == 1
        # Spin-0: gauge mode
        @test K_maxwell.spin0s == 0
        @test K_maxwell.spin0w == 0
    end

    @testset "Maxwell propagator: transverse 1/k²" begin
        K_maxwell = SpinSectorDecomposition(0, 1, 0, 0)
        prop = moore_penrose_propagator(K_maxwell)

        # Only spin-1 propagates
        @test prop.spin1 == 1//1   # 1/k² at k²=1
        # All other sectors: zero (gauge or absent)
        @test prop.spin2 == 0
        @test prop.spin0s == 0
        @test prop.spin0w == 0
    end

    @testset "Maxwell unitarity: healthy" begin
        K_maxwell = SpinSectorDecomposition(0, 1, 0, 0)
        ua = unitarity_conditions(K_maxwell)

        @test ua.unitary
        @test ua.ghost_free
        @test ua.tachyon_free

        # Spin-1: physical, ghost-free
        @test ua.sectors[:spin1].ghost_free
        @test !ua.sectors[:spin1].is_gauge

        # Spin-0: gauge mode
        @test ua.sectors[:spin0s].is_gauge
        @test ua.sectors[:spin0w].is_gauge
    end

    @testset "Maxwell source constraint: ∂_μ J^μ = 0" begin
        K_maxwell = SpinSectorDecomposition(0, 1, 0, 0)
        sc = source_constraints(K_maxwell)

        # 3 gauge sectors → 3 constraints
        @test length(sc) == 3

        # No constraint on spin-1 (physical sector)
        @test !any(c -> c.sector == :spin1, sc)

        # Spin-0 gauge sectors present constraints
        sectors = Set(c.sector for c in sc)
        @test :spin2 in sectors     # no spin-2 coupling (vector field)
        @test :spin0s in sectors    # no scalar coupling
        @test :spin0w in sectors    # traceless/longitudinal gauge
    end

    @testset "Maxwell: exactly 1 propagating DOF" begin
        K_maxwell = SpinSectorDecomposition(0, 1, 0, 0)
        prop = moore_penrose_propagator(K_maxwell)

        # Count propagating sectors (non-zero propagator)
        propagating = count(x -> x != 0,
            [prop.spin2, prop.spin1, prop.spin0s, prop.spin0w])
        @test propagating == 1  # only spin-1 (photon)
    end

    @testset "Proca comparison: massive vector has 2 sectors" begin
        # Proca (massive spin-1): L = -¼F² + ½m²A²
        # K_{μν} = k²θ_{μν} + m²ω_{μν}
        # → spin-1: k², spin-0: m²
        K_proca = SpinSectorDecomposition(0, 1, 0, 1)  # at k²=1, m²=1

        prop = moore_penrose_propagator(K_proca)
        @test prop.spin1 == 1//1   # 1/k²
        @test prop.spin0w == 1//1  # 1/m²

        # Proca has 2 propagating sectors (3 DOF = d-1 in d=4)
        propagating = count(x -> x != 0,
            [prop.spin2, prop.spin1, prop.spin0s, prop.spin0w])
        @test propagating == 2

        # No source constraints (no gauge modes... well, spin-2 and spin-0s are zero)
        ua = unitarity_conditions(K_proca)
        @test ua.ghost_free
    end

    @testset "String match: Maxwell spectrum summary" begin
        K = SpinSectorDecomposition(0, 1, 0, 0)
        prop = moore_penrose_propagator(K)
        ua = unitarity_conditions(K)
        sc = source_constraints(K)

        # Ground truth string matches from Barker (2024):
        @test prop.spin1 != 0    # "massless spin-1 particle"
        @test ua.unitary         # "no ghosts, no tachyons"
        @test length(sc) == 3    # 3 gauge constraints
    end
end
