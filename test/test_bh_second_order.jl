@testset "BH Second-Order Source Terms" begin
    using TensorGR: second_order_einstein_source, source_is_bilinear,
                    SourceModeCoupling, source_coupling_modes,
                    scalar_coupling_coefficient, vector_coupling_coefficient,
                    tensor_coupling_coefficient, count_coupling_modes,
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
end
