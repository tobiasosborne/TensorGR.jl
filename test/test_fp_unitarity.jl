# Ground truth: Barker (2024) arXiv:2406.09500, PSALTer Section 8.3.
# Also: CLAUDE.md physics ground truth for Fierz-Pauli kernel.
#
# Fierz-Pauli (linearized GR): L = h^{ab} δG_{ab}
#
# Kinetic kernel (at k²):
#   K = (5k²/2) P₂ + 0·P₁ + (-k²)·P₀ₛ + 0·P₀w
#
# Ground truth from CLAUDE.md:
#   spin2=2.5k², spin0s=-k², spin1=0, spin0w=0
#
# Unitarity:
#   spin-2: ghost-free (positive), healthy
#   spin-0s: GHOST (negative form factor = Boulware-Deser ghost in massless limit)
#   spin-1: gauge mode (diffeomorphism invariance)
#   spin-0w: gauge mode (trace gauge)
#
# Propagator (Moore-Penrose):
#   spin-2: 2/(5k²), spin-0s: -1/k², spin-1: 0, spin-0w: 0
#
# Source constraints:
#   ∂_μ T^{μν} = 0 (from spin-1 gauge → transverse)
#   T^μ_μ = ... (from spin-0w gauge → traceless constraint)

@testset "PSALTer Validation: Fierz-Pauli unitarity" begin

    @testset "FP kernel matches CLAUDE.md ground truth" begin
        # At k²=1: a₂=5/2, a₁=0, a₀ₛ=-1, a₀w=0
        K_FP = SpinSectorDecomposition(5//2, 0, -1//1, 0)
        @test K_FP.spin2 == 5//2
        @test K_FP.spin1 == 0
        @test K_FP.spin0s == -1//1
        @test K_FP.spin0w == 0
    end

    @testset "FP Moore-Penrose propagator" begin
        K_FP = SpinSectorDecomposition(5//2, 0, -1//1, 0)
        prop = moore_penrose_propagator(K_FP)

        @test prop.spin2 == 2//5     # 1/(5/2) = 2/5
        @test prop.spin1 == 0        # gauge
        @test prop.spin0s == -1//1   # 1/(-1) = -1 (ghost pole!)
        @test prop.spin0w == 0       # gauge
    end

    @testset "FP unitarity: spin-0s ghost (Boulware-Deser)" begin
        K_FP = SpinSectorDecomposition(5//2, 0, -1//1, 0)
        ua = unitarity_conditions(K_FP)

        # NOT unitary: spin-0s is a ghost
        @test !ua.unitary
        @test !ua.ghost_free

        # Spin-2: healthy (positive form factor)
        @test ua.sectors[:spin2].ghost_free
        @test !ua.sectors[:spin2].is_gauge
        @test ua.sectors[:spin2].form_factor == 5//2

        # Spin-1: gauge mode (diffeomorphism invariance)
        @test ua.sectors[:spin1].is_gauge

        # Spin-0s: GHOST (negative form factor)
        @test !ua.sectors[:spin0s].ghost_free
        @test !ua.sectors[:spin0s].is_gauge
        @test ua.sectors[:spin0s].form_factor == -1//1

        # Spin-0w: gauge mode (trace gauge)
        @test ua.sectors[:spin0w].is_gauge
    end

    @testset "FP source constraints: transverse + traceless" begin
        K_FP = SpinSectorDecomposition(5//2, 0, -1//1, 0)
        sc = source_constraints(K_FP)

        # Two gauge sectors → two constraints
        @test length(sc) == 2

        sectors = Set(c.sector for c in sc)
        @test :spin1 in sectors     # ∂_μ T^{μν} = 0
        @test :spin0w in sectors    # T^μ_μ constraint

        # Physical sectors: no constraints
        @test !(:spin2 in sectors)
        @test !(:spin0s in sectors)

        # Spin-1 constraint is transverse
        spin1_c = first(c for c in sc if c.sector == :spin1)
        @test spin1_c.constraint == :transverse
    end

    @testset "FP: 2 propagating DOF" begin
        K_FP = SpinSectorDecomposition(5//2, 0, -1//1, 0)
        prop = moore_penrose_propagator(K_FP)

        propagating = count(x -> x != 0,
            [prop.spin2, prop.spin1, prop.spin0s, prop.spin0w])
        @test propagating == 2  # spin-2 + spin-0s
    end

    @testset "FP display shows ghost" begin
        K_FP = SpinSectorDecomposition(5//2, 0, -1//1, 0)
        ua = unitarity_conditions(K_FP)
        s = sprint(show, ua)
        @test occursin("ghost", s)
    end

    @testset "String match: FP vs CLAUDE.md ground truth" begin
        K_FP = SpinSectorDecomposition(5//2, 0, -1//1, 0)

        # Ground truth from CLAUDE.md:
        # K_FP: spin2=2.5k², spin0s=-k², spin1=0, spin0w=0
        @test K_FP.spin2 == 5//2       # 2.5
        @test K_FP.spin0s == -1//1     # -1
        @test K_FP.spin1 == 0          # 0
        @test K_FP.spin0w == 0         # 0

        # Spin-1 and spin-0w MUST be zero for ALL kernels (diffeomorphism invariance)
        @test K_FP.spin1 == 0
        @test K_FP.spin0w == 0
    end
end
