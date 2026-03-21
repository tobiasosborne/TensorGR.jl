using Test
using TensorGR

@testset "Proca Hamiltonian: 3 DOF" begin

    # ── Proca: massive vector field in 4D ─────────────────────────────
    # L = -(1/4)F_{ab}F^{ab} + (1/2)m²A_aA^a
    #
    # 4 canonical pairs (A_μ, π^μ) in 4D spacetime.
    # Primary constraint: π^0 ≈ 0 (A_0 has no time derivative in F_{ab})
    # Secondary: ∂_i π^i + m²A^0 ≈ 0 (modified Gauss law)
    # Both are SECOND-CLASS: {π^0, ∂_i π^i + m²A^0} = m²δ³(x-y) ≠ 0
    # 0 first-class (mass breaks gauge invariance)
    #
    # DOF = (2×4 - 2×0 - 2) / 2 = 3 (3 polarizations of massive vector)
    #
    # Ground truth: Henneaux & Teitelboim (1992) Sec 1.2.

    @testset "Proca DOF count" begin
        summary = dof_count(4, 0, 2)

        @test summary.config_dof == 3      # 3 massive vector polarizations
        @test summary.phase_dof == 6       # 2 × 3
        @test summary.n_canonical_pairs == 4
        @test summary.n_first_class == 0   # mass breaks gauge invariance
        @test summary.n_second_class == 2  # π^0 ≈ 0 and modified Gauss law
        @test summary.n_gauge == 0         # no gauge freedom
    end

    # ── Dirac formula explicit verification ───────────────────────────

    @testset "Dirac formula: (2×4 - 2×0 - 2)/2 = 3" begin
        summary = dof_count(4, 0, 2)

        phase_space_dim = 2 * summary.n_canonical_pairs
        expected = (phase_space_dim - 2 * summary.n_first_class - summary.n_second_class) ÷ 2
        @test expected == 3
        @test summary.config_dof == expected
    end

    # ── Phase DOF = 2 × config DOF ───────────────────────────────────

    @testset "phase_dof = 2 * config_dof" begin
        summary = dof_count(4, 0, 2)
        @test summary.phase_dof == 2 * summary.config_dof
    end

    # ── Description string ────────────────────────────────────────────

    @testset "description string is informative" begin
        summary = dof_count(4, 0, 2)
        @test occursin("4 canonical pairs", summary.description)
        @test occursin("0 first-class", summary.description)
        @test occursin("2 second-class", summary.description)
        @test occursin("3 config-space", summary.description)
        @test occursin("6 phase-space", summary.description)
    end

    # ── Massless limit: Maxwell (m → 0) ──────────────────────────────
    # When m → 0 the two second-class constraints become first-class
    # (gauge symmetry is restored), DOF → 2 (photon).

    @testset "massless limit: Maxwell (m → 0)" begin
        maxwell = dof_count(4, 2, 0)

        @test maxwell.config_dof == 2      # 2 photon polarizations
        @test maxwell.phase_dof == 4
        @test maxwell.n_first_class == 2   # Gauss law + primary
        @test maxwell.n_second_class == 0
        @test maxwell.n_gauge == 2         # U(1) gauge freedom
    end

    # ── Proca vs Maxwell comparison ──────────────────────────────────
    # Massive → massless: DOF decreases by 1 (Stückelberg mechanism).
    # The longitudinal polarization decouples as m → 0.

    @testset "Proca vs Maxwell: DOF differs by 1" begin
        proca   = dof_count(4, 0, 2)
        maxwell = dof_count(4, 2, 0)

        @test proca.config_dof == maxwell.config_dof + 1
        @test proca.n_gauge == 0
        @test maxwell.n_gauge == 2
    end

    # ── General massive spin-1: DOF = 2s+1 = 3 ──────────────────────

    @testset "massive spin-1: DOF = 2s+1 = 3" begin
        s = 1
        expected_dof = 2 * s + 1
        summary = dof_count(4, 0, 2)
        @test summary.config_dof == expected_dof
    end

end
