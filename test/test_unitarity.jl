# Ground truth: Barker (2024) arXiv:2406.09500, PSALTer Section 6.
#
# Unitarity conditions per spin sector:
# - Ghost-free: sign of kinetic form factor matches reference (positive)
# - Tachyon-free: no negative mass-squared poles
#
# Fierz-Pauli (linearized GR):
#   spin-2: a₂ = 5k²/2 > 0  → healthy
#   spin-0s: a₀ₛ = -k² < 0  → GHOST (scalar mode)
#   spin-1, spin-0w: gauge zero modes (excluded)

@testset "Unitarity conditions" begin

    @testset "Fierz-Pauli: spin-2 healthy, spin-0s ghost" begin
        K_FP = SpinSectorDecomposition(5//2, 0, -1//1, 0)
        ua = unitarity_conditions(K_FP)

        @test ua isa UnitarityAnalysis
        @test !ua.unitary          # FP has a ghost in spin-0s
        @test !ua.ghost_free       # ghost present

        # Spin-2: positive form factor → ghost-free
        @test ua.sectors[:spin2].ghost_free
        @test !ua.sectors[:spin2].is_gauge

        # Spin-1: gauge mode
        @test ua.sectors[:spin1].is_gauge

        # Spin-0s: negative form factor → GHOST
        @test !ua.sectors[:spin0s].ghost_free
        @test !ua.sectors[:spin0s].is_gauge

        # Spin-0w: gauge mode
        @test ua.sectors[:spin0w].is_gauge
    end

    @testset "Pure R² theory: spin-0s only, healthy" begin
        K_R2 = SpinSectorDecomposition(0, 0, 3, 0)
        ua = unitarity_conditions(K_R2)

        # Only spin-0s is physical, positive → unitary
        @test ua.unitary
        @test ua.ghost_free

        @test ua.sectors[:spin2].is_gauge
        @test ua.sectors[:spin1].is_gauge
        @test ua.sectors[:spin0s].ghost_free
        @test ua.sectors[:spin0w].is_gauge
    end

    @testset "All positive sectors: fully healthy" begin
        K = SpinSectorDecomposition(1, 2, 3, 4)
        ua = unitarity_conditions(K)
        @test ua.unitary
        @test ua.ghost_free
        @test ua.tachyon_free
        for (_, r) in ua.sectors
            @test r.ghost_free
            @test !r.is_gauge
        end
    end

    @testset "Mixed sign: ghost detected" begin
        K = SpinSectorDecomposition(1, -2, 3, 0)
        ua = unitarity_conditions(K)
        @test !ua.ghost_free
        @test !ua.unitary

        @test ua.sectors[:spin2].ghost_free      # positive
        @test !ua.sectors[:spin1].ghost_free      # negative → ghost
        @test ua.sectors[:spin0s].ghost_free      # positive
    end

    @testset "All gauge modes: trivially unitary" begin
        K = SpinSectorDecomposition(0, 0, 0, 0)
        ua = unitarity_conditions(K)
        @test ua.unitary
        for (_, r) in ua.sectors
            @test r.is_gauge
        end
    end

    @testset "Propagator stored in results" begin
        K = SpinSectorDecomposition(2, 0, 4, 0)
        ua = unitarity_conditions(K)

        @test ua.sectors[:spin2].propagator == 1//2
        @test ua.sectors[:spin1].propagator == 0   # gauge
        @test ua.sectors[:spin0s].propagator == 1//4
    end

    @testset "Display" begin
        K = SpinSectorDecomposition(5//2, 0, -1, 0)
        ua = unitarity_conditions(K)
        s = sprint(show, ua)
        @test occursin("UnitarityAnalysis", s)

        sr = ua.sectors[:spin2]
        s2 = sprint(show, sr)
        @test occursin("healthy", s2)

        sr_ghost = ua.sectors[:spin0s]
        s3 = sprint(show, sr_ghost)
        @test occursin("GHOST", s3)
    end

    @testset "Custom reference sign (negative convention)" begin
        # Some conventions use negative kinetic term
        K = SpinSectorDecomposition(-1, 0, -2, 0)
        ua_pos = unitarity_conditions(K; reference_sign=1)
        @test !ua_pos.ghost_free   # negative vs positive reference

        ua_neg = unitarity_conditions(K; reference_sign=-1)
        @test ua_neg.ghost_free    # negative matches negative reference
    end

    @testset "Float tolerance for gauge detection" begin
        K = SpinSectorDecomposition(1.0, 1e-15, 2.0, 0.0)
        ua = unitarity_conditions(K; atol=1e-10)
        @test ua.sectors[:spin1].is_gauge   # tiny value → gauge with tolerance
    end
end
