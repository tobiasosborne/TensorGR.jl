@testset "BH Second-Order Source Terms" begin
    using TensorGR: second_order_einstein_source, source_is_bilinear,
                    SourceModeCoupling, source_coupling_modes,
                    scalar_coupling_coefficient, vector_coupling_coefficient,
                    tensor_coupling_coefficient, count_coupling_modes,
                    MasterEquation, regge_wheeler_potential, regge_wheeler_equation,
                    zerilli_potential, zerilli_equation,
                    tortoise_coordinate, inverse_tortoise,
                    evaluate_potential, potential_at_horizon, potential_at_infinity,
                    MetricPerturbation, define_metric_perturbation!,
                    vacuum_background!,
                    δricci, δricci_scalar,
                    gaunt_integral, vector_gaunt, tensor_gaunt,
                    angular_selection_rule,
                    TensorRegistry, TensorProperties, SymmetrySpec,
                    Tensor, TProduct, TSum, TDeriv, TScalar,
                    up, down, with_registry, register_tensor!,
                    current_registry, has_tensor,
                    free_indices, simplify, set_vanishing!

    function _make_schwarzschild_registry()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            vacuum_background!(reg, :M4; metric=:g)
        end
        return reg
    end

    # ── Second-order Einstein source ─────────────────────────────────────

    @testset "second_order_einstein_source structure" begin
        reg = _make_schwarzschild_registry()
        with_registry(reg) do
            mp = define_metric_perturbation!(reg, :g, :h; curved=true)
            source = second_order_einstein_source(mp, down(:a), down(:b))

            @test source isa TSum
            # Three terms: δ²R_{ab}, -(1/2)g_{ab}δ²R, -(1/2)h_{ab}δ¹R
            @test length(source.terms) == 3
        end
    end

    @testset "second_order_einstein_source free indices" begin
        reg = _make_schwarzschild_registry()
        with_registry(reg) do
            mp = define_metric_perturbation!(reg, :g, :h; curved=true)
            source = second_order_einstein_source(mp, down(:a), down(:b))
            fi = free_indices(source)
            tangent_fi = filter(idx -> idx.vbundle === :Tangent, fi)
            @test length(tangent_fi) == 2
            @test Set(idx.name for idx in tangent_fi) == Set([:a, :b])
            @test all(idx -> idx.position === Down, tangent_fi)
        end
    end

    @testset "second_order_einstein_source non-zero on curved bg" begin
        reg = _make_schwarzschild_registry()
        with_registry(reg) do
            mp = define_metric_perturbation!(reg, :g, :h; curved=true)
            source = second_order_einstein_source(mp, down(:a), down(:b))
            @test source != TScalar(0 // 1)
        end
    end

    @testset "second_order_einstein_source index validation" begin
        reg = _make_schwarzschild_registry()
        with_registry(reg) do
            mp = define_metric_perturbation!(reg, :g, :h; curved=true)
            @test_throws ErrorException second_order_einstein_source(
                mp, up(:a), down(:b))
        end
    end

    @testset "source_is_bilinear in h" begin
        reg = _make_schwarzschild_registry()
        with_registry(reg) do
            mp = define_metric_perturbation!(reg, :g, :h; curved=true)
            # Every term in δ²G_{ab} must contain exactly 2 powers of h
            # This ensures S[0,0] = 0 (bilinear vanishes at zero)
            @test source_is_bilinear(mp)
        end
    end

    # ── Mode coupling infrastructure ─────────────────────────────────────

    @testset "source_coupling_modes: selection rules" begin
        # l=2, m=0, lmax=2
        couplings = source_coupling_modes(2, 0, 2)
        @test !isempty(couplings)

        # All couplings must satisfy selection rules
        for c in couplings
            @test c.l == 2
            @test c.m == 0
            @test c.m1 + c.m2 == 0  # m-conservation
            @test abs(c.l1 - c.l2) <= 2 <= c.l1 + c.l2  # triangle
            @test iseven(c.l1 + c.l2 + 2)  # parity
        end
    end

    @testset "source_coupling_modes: m-conservation" begin
        couplings = source_coupling_modes(2, 1, 3)
        for c in couplings
            @test c.m1 + c.m2 == 1
        end
    end

    @testset "source_coupling_modes: l=0,m=0 monopole" begin
        couplings = source_coupling_modes(0, 0, 2)
        # l=0 monopole can be sourced by l1=l2 (0+0, 1+1, 2+2)
        @test !isempty(couplings)
        for c in couplings
            @test c.l1 == c.l2  # triangle with l=0 requires l1=l2
        end
    end

    @testset "source_coupling_modes: odd×odd requires l>=1" begin
        couplings = source_coupling_modes(2, 0, 2)
        odd_couplings = filter(c -> c.parity1 === :odd, couplings)
        for c in odd_couplings
            @test c.l1 >= 1
            @test c.l2 >= 1
        end
    end

    @testset "source_coupling_modes: parity consistency" begin
        couplings = source_coupling_modes(2, 0, 3)
        for c in couplings
            # Only even×even and odd×odd (not even×odd)
            @test c.parity1 == c.parity2
        end
    end

    @testset "count_coupling_modes" begin
        n = count_coupling_modes(2, 0, 2)
        @test n > 0
        @test n == length(source_coupling_modes(2, 0, 2))
    end

    # ── Angular coupling coefficients ────────────────────────────────────

    @testset "scalar_coupling_coefficient: known values" begin
        # C^{00}_{00,00} = ∫ Y_{00} Y_{00} Y*_{00} dΩ
        # = (1/√(4π))^3 * 4π = 1/(4π)^{1/2}
        c = scalar_coupling_coefficient(0, 0, 0, 0, 0, 0)
        @test c ≈ 1 / sqrt(4π)

        # Selection rule: m1+m2 != m → vanishes
        c2 = scalar_coupling_coefficient(2, 0, 1, 1, 1, 1)
        @test c2 ≈ 0.0 atol=1e-14  # m1+m2=2 ≠ m=0
    end

    @testset "vector_coupling_coefficient: selection rules" begin
        # l1=0 or l2=0 → vanishes (vector harmonics need l>=1)
        c = vector_coupling_coefficient(2, 0, 0, 0, 2, 0)
        @test c ≈ 0.0

        # l1=1, l2=1 → non-zero for allowed l
        c2 = vector_coupling_coefficient(2, 0, 1, 0, 1, 0)
        # Should be non-zero (1+1→2 allowed, parity even)
        @test abs(c2) > 0
    end

    @testset "tensor_coupling_coefficient: cross-type vanishes" begin
        # Y-Z cross coupling vanishes
        c = tensor_coupling_coefficient(2, 0, 2, 0, 2, 0, :Y, :Z)
        @test c ≈ 0.0

        # Y-X cross coupling vanishes
        c2 = tensor_coupling_coefficient(2, 0, 2, 0, 2, 0, :Y, :X)
        @test c2 ≈ 0.0

        # Z-X cross coupling vanishes
        c3 = tensor_coupling_coefficient(2, 0, 2, 0, 2, 0, :Z, :X)
        @test c3 ≈ 0.0
    end

    @testset "tensor_coupling_coefficient: Y-Y non-zero" begin
        # Y-Y (metric × metric) = 2 * gaunt
        c = tensor_coupling_coefficient(0, 0, 2, 0, 2, 0, :Y, :Y)
        g = gaunt_integral(2, 0, 2, 0, 0, 0)
        @test c ≈ 2.0 * g
    end

    @testset "tensor_coupling_coefficient: Z-Z consistency" begin
        # Z-Z uses Q coefficient from Gleiser et al.
        c = tensor_coupling_coefficient(0, 0, 2, 0, 2, 0, :Z, :Z)
        g = gaunt_integral(2, 0, 2, 0, 0, 0)
        # Q(2,2,0) with L1=L2=6, L3=0: Q = (12)^2/8 + (36-12)/4 = 18+6 = 24
        @test c ≈ 24.0 * g
    end

    # ── Physics ground truth ─────────────────────────────────────────────

    @testset "angular_selection_rule: Brizuela triangle" begin
        # Quadrupole coupling: l1=2, l2=2 can source l=0,2,4
        @test angular_selection_rule(2, 2, 0)
        @test angular_selection_rule(2, 2, 2)
        @test angular_selection_rule(2, 2, 4)

        # l=1 violates parity (2+2+1=5 odd)
        @test !angular_selection_rule(2, 2, 1)

        # l=3 violates parity (2+2+3=7 odd)
        @test !angular_selection_rule(2, 2, 3)

        # l=5 violates triangle (2+2=4 < 5)
        @test !angular_selection_rule(2, 2, 5)
    end

    @testset "quadrupole self-coupling: l=2 modes" begin
        # l=2, m=0 sourced by l1=l2=2 quadrupole
        couplings = source_coupling_modes(2, 0, 2)
        has_22 = any(c -> c.l1 == 2 && c.l2 == 2, couplings)
        @test has_22

        # Check that the coupling coefficient is non-zero
        c = scalar_coupling_coefficient(2, 0, 2, 0, 2, 0)
        @test abs(c) > 0
    end

    # ══════════════════════════════════════════════════════════════════
    # Regge-Wheeler master equation (TGR-2yl)
    # ══════════════════════════════════════════════════════════════════

    @testset "regge_wheeler_equation construction" begin
        eq = regge_wheeler_equation(2)
        @test eq isa MasterEquation
        @test eq.parity === :odd
        @test eq.l == 2
        @test eq.potential_name === :RW
    end

    @testset "regge_wheeler_equation: l >= 2 required" begin
        @test_throws ErrorException regge_wheeler_equation(1)
        @test_throws ErrorException regge_wheeler_equation(0)
    end

    @testset "RW potential: known values at l=2, M=1" begin
        M = 1.0
        # V_RW(r) = (1-2/r)(6/r² - 6/r³) for l=2, M=1
        # At r=10: f=0.8, l(l+1)/r²=6/100=0.06, 6M/r³=6/1000=0.006
        #   V = 0.8*(0.06 - 0.006) = 0.0432
        V10 = regge_wheeler_potential(10.0, M, 2)
        @test V10 ≈ 0.8 * (6.0 / 100.0 - 6.0 / 1000.0)

        # At r=3M=3: f=1/3, l(l+1)/r²=6/9, 6M/r³=6/27
        #   V = (1/3)(2/3 - 2/9) = (1/3)(4/9) = 4/27
        V3 = regge_wheeler_potential(3.0, M, 2)
        @test V3 ≈ 4.0 / 27.0

        # At r=6M=6: f=2/3, l(l+1)/r²=6/36=1/6, 6M/r³=6/216=1/36
        #   V = (2/3)(1/6 - 1/36) = (2/3)(5/36) = 10/108 = 5/54
        V6 = regge_wheeler_potential(6.0, M, 2)
        @test V6 ≈ 5.0 / 54.0
    end

    @testset "RW potential: vanishes at horizon" begin
        # f(2M) = 0 → V(2M) = 0
        M = 1.0
        # Approach horizon from outside
        V = regge_wheeler_potential(2.0 + 1e-10, M, 2)
        @test abs(V) < 1e-5
        @test potential_at_horizon(regge_wheeler_equation(2), M) == 0.0
    end

    @testset "RW potential: decays at infinity" begin
        M = 1.0
        V100 = regge_wheeler_potential(100.0, M, 2)
        V1000 = regge_wheeler_potential(1000.0, M, 2)
        @test V1000 < V100  # monotonically decreasing at large r
        @test abs(V1000) < 1e-4
        @test potential_at_infinity(regge_wheeler_equation(2)) == 0.0
    end

    @testset "RW potential: positive outside horizon" begin
        # V_RW > 0 for r > 2M (for l >= 2)
        M = 1.0
        for r in [2.1, 3.0, 5.0, 10.0, 50.0, 100.0]
            @test regge_wheeler_potential(r, M, 2) > 0
        end
    end

    @testset "RW potential: l dependence" begin
        M = 1.0; r = 10.0
        V2 = regge_wheeler_potential(r, M, 2)
        V3 = regge_wheeler_potential(r, M, 3)
        V4 = regge_wheeler_potential(r, M, 4)
        # Higher l → larger barrier (more angular momentum)
        @test V3 > V2
        @test V4 > V3
    end

    @testset "evaluate_potential dispatch" begin
        eq = regge_wheeler_equation(2)
        V = evaluate_potential(eq, 10.0, 1.0)
        @test V ≈ regge_wheeler_potential(10.0, 1.0, 2)
    end

    # ══════════════════════════════════════════════════════════════════
    # Zerilli master equation (TGR-31k)
    # ══════════════════════════════════════════════════════════════════

    @testset "zerilli_equation construction" begin
        eq = zerilli_equation(2)
        @test eq isa MasterEquation
        @test eq.parity === :even
        @test eq.l == 2
        @test eq.potential_name === :Zerilli
    end

    @testset "zerilli_equation: l >= 2 required" begin
        @test_throws ErrorException zerilli_equation(1)
        @test_throws ErrorException zerilli_equation(0)
    end

    @testset "Zerilli potential: l=2 ground truth (Chandrasekhar)" begin
        # At l=2: lambda = (2-1)(2+2)/2 = 2
        # V_Z(r) = f · [24r³ + 24Mr² + 36M²r + 18M³] / [r³(2r+3M)²]
        # Ground truth: Chandrasekhar (1983) Eq 4.26; Brizuela et al. (2009) Eq 4.12
        M = 1.0

        # At r=10: f=0.8
        # num = 24*1000 + 24*100 + 36*10 + 18 = 26778
        # den = 1000 * (20+3)² = 529000
        # V = 0.8 * 26778/529000
        V10 = zerilli_potential(10.0, M, 2)
        num = 24.0 * 1000 + 24.0 * 100 + 36.0 * 10 + 18.0
        den = 1000.0 * (20.0 + 3.0)^2
        @test V10 ≈ 0.8 * num / den

        # V_Z should be positive outside horizon for l >= 2
        @test V10 > 0
    end

    @testset "Zerilli potential: vanishes at horizon" begin
        M = 1.0
        V = zerilli_potential(2.0 + 1e-10, M, 2)
        @test abs(V) < 1e-5
        @test potential_at_horizon(zerilli_equation(2), M) == 0.0
    end

    @testset "Zerilli potential: decays at infinity" begin
        M = 1.0
        V100 = zerilli_potential(100.0, M, 2)
        V1000 = zerilli_potential(1000.0, M, 2)
        @test V1000 < V100
        @test abs(V1000) < 1e-4
        @test potential_at_infinity(zerilli_equation(2)) == 0.0
    end

    @testset "Zerilli potential: positive outside horizon" begin
        M = 1.0
        for r in [2.1, 3.0, 5.0, 10.0, 50.0, 100.0]
            @test zerilli_potential(r, M, 2) > 0
        end
    end

    @testset "Zerilli potential: l dependence" begin
        M = 1.0; r = 10.0
        V2 = zerilli_potential(r, M, 2)
        V3 = zerilli_potential(r, M, 3)
        V4 = zerilli_potential(r, M, 4)
        @test V3 > V2
        @test V4 > V3
    end

    @testset "RW and Zerilli: isospectral (same l)" begin
        # RW and Zerilli potentials have the same QNM spectrum
        # but different potential shapes. However, at large r both
        # approach l(l+1)/r² (leading order).
        M = 1.0; r = 1000.0
        V_rw = regge_wheeler_potential(r, M, 2)
        V_z = zerilli_potential(r, M, 2)
        # Both → l(l+1)/r² = 6/10^6 = 6e-6 at large r
        @test abs(V_rw - 6.0 / r^2) / (6.0 / r^2) < 0.01
        @test abs(V_z - 6.0 / r^2) / (6.0 / r^2) < 0.01
    end

    # ── Tortoise coordinate ───────────────────────────────────────────

    @testset "tortoise_coordinate: basic values" begin
        M = 1.0
        # r* = r + 2M ln(r/2M - 1)
        # At r=4M=4: r* = 4 + 2*ln(1) = 4
        @test tortoise_coordinate(4.0, M) ≈ 4.0

        # At r=10: r* = 10 + 2*ln(4) ≈ 10 + 2.77 ≈ 12.77
        @test tortoise_coordinate(10.0, M) ≈ 10.0 + 2.0 * log(4.0)
    end

    @testset "tortoise_coordinate: diverges at horizon" begin
        M = 1.0
        # r* → -∞ as r → 2M+
        rstar = tortoise_coordinate(2.001, M)
        @test rstar < -10.0  # very negative near horizon
    end

    @testset "tortoise_coordinate: requires r > 2M" begin
        @test_throws ErrorException tortoise_coordinate(1.5, 1.0)
    end

    @testset "inverse_tortoise round-trip" begin
        M = 1.0
        for r in [3.0, 5.0, 10.0, 50.0]
            rstar = tortoise_coordinate(r, M)
            r_recovered = inverse_tortoise(rstar, M)
            @test r_recovered ≈ r atol=1e-10
        end
    end
end
