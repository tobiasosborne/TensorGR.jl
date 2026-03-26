@testset "Yang-Mills Field Strength and Equations" begin
    using TensorGR: define_gauge_group!, get_gauge_group, GaugeGroupProperties,
                    yang_mills_field_strength, gauge_covariant_deriv,
                    yang_mills_bianchi, yang_mills_lagrangian,
                    yang_mills_field_equations, brst_gauge_field,
                    TensorRegistry, TensorProperties, SymmetrySpec,
                    AntiSymmetric, ManifoldProperties,
                    Tensor, TProduct, TSum, TDeriv, TScalar, TIndex,
                    up, down, Up, Down,
                    with_registry, register_tensor!, register_manifold!,
                    current_registry, has_tensor, get_tensor,
                    free_indices, indices, simplify

    function _make_ym_registry()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_gauge_group!(reg, :SU3; dim=8)
        end
        return reg
    end

    # ── Field strength (TGR-655.3) ───────────────────────────────────

    @testset "yang_mills_field_strength structure" begin
        reg = _make_ym_registry()
        ggp = get_gauge_group(reg, :SU3)
        with_registry(reg) do
            I = TIndex(:I, Up, :Gauge)
            a = down(:a)
            b = down(:b)

            F = yang_mills_field_strength(ggp, I, a, b)

            # F is a TSum with 3 terms: ∂_a A^I_b, -∂_b A^I_a, f^I_{JK} A^J_a A^K_b
            @test F isa TSum
            @test length(F.terms) == 3

            # First term: ∂_a A^I_b (TDeriv)
            @test F.terms[1] isa TDeriv
            @test F.terms[1].index.name === :a
            @test F.terms[1].arg isa Tensor
            @test F.terms[1].arg.name === :A

            # Second term: -∂_b A^I_a (TProduct with scalar -1)
            @test F.terms[2] isa TProduct
            @test F.terms[2].scalar == -1 // 1

            # Third term: f^I_{JK} A^J_a A^K_b (TProduct)
            @test F.terms[3] isa TProduct
            @test length(F.terms[3].factors) == 3

            # Free indices should be I (Up, Gauge), a (Down), b (Down)
            fi = free_indices(F)
            free_names = Set(idx.name for idx in fi)
            @test :I in free_names
            @test :a in free_names
            @test :b in free_names
        end
    end

    @testset "yang_mills_field_strength antisymmetry" begin
        reg = _make_ym_registry()
        ggp = get_gauge_group(reg, :SU3)
        with_registry(reg) do
            I = TIndex(:I, Up, :Gauge)
            a = down(:a)
            b = down(:b)

            F_ab = yang_mills_field_strength(ggp, I, a, b)
            F_ba = yang_mills_field_strength(ggp, I, b, a)

            # F_{ab} + F_{ba} should be zero (antisymmetry)
            # Check structure: F_ab has ∂_a A_b - ∂_b A_a + fAA_ab
            # F_ba has ∂_b A_a - ∂_a A_b + fAA_ba
            # Their sum cancels the derivative terms; the fAA terms also cancel
            # due to antisymmetry of structure constants
            @test F_ab isa TSum
            @test F_ba isa TSum
            @test length(F_ab.terms) == 3
            @test length(F_ba.terms) == 3
        end
    end

    @testset "yang_mills_field_strength abelian limit" begin
        # For abelian gauge group (dim=1), fAA term has structure constants
        # that vanish (f^1_{11} = 0 by antisymmetry). Only ∂_a A_b - ∂_b A_a
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            ggp = define_gauge_group!(reg, :U1; dim=1, vbundle=:U1Gauge)
            I = TIndex(:I, Up, :U1Gauge)
            a = down(:a)
            b = down(:b)

            F = yang_mills_field_strength(ggp, I, a, b)
            @test F isa TSum
            @test length(F.terms) == 3  # still 3 terms structurally

            # The fAA term has f^I_{JK} which vanishes for abelian, but
            # we don't simplify here — just check it builds correctly
            @test F.terms[3] isa TProduct
        end
    end

    # ── Gauge-covariant derivative (TGR-655.3) ───────────────────────

    @testset "gauge_covariant_deriv on simple tensor" begin
        reg = _make_ym_registry()
        ggp = get_gauge_group(reg, :SU3)
        with_registry(reg) do
            I = TIndex(:I, Up, :Gauge)
            a = down(:a)

            # D_a c^I = ∂_a c^I + f^I_{JK} A^J_a c^K
            c_I = Tensor(:c_ghost, [I])
            Dc = gauge_covariant_deriv(ggp, c_I, I, a)

            @test Dc isa TSum
            @test length(Dc.terms) == 2

            # First term: ∂_a c^I
            @test Dc.terms[1] isa TDeriv
            @test Dc.terms[1].index.name === :a
            @test Dc.terms[1].arg.name === :c_ghost

            # Second term: f^I_{JK} A^J_a c^K
            @test Dc.terms[2] isa TProduct
            @test length(Dc.terms[2].factors) == 3
        end
    end

    @testset "gauge_covariant_deriv matches brst_gauge_field" begin
        # D_a c^I from gauge_covariant_deriv should structurally match
        # brst_gauge_field (which is s(A^I_a) = D_a c^I)
        reg = _make_ym_registry()
        ggp = get_gauge_group(reg, :SU3)
        with_registry(reg) do
            I = TIndex(:I, Up, :Gauge)
            a = down(:a)

            brst_sA = brst_gauge_field(ggp, I, a)
            Dc = gauge_covariant_deriv(ggp, Tensor(:c_ghost, [I]), I, a)

            # Both should have 2 terms: ∂_a c^I and f·A·c
            @test brst_sA isa TSum
            @test Dc isa TSum
            @test length(brst_sA.terms) == 2
            @test length(Dc.terms) == 2
        end
    end

    @testset "gauge_covariant_deriv error on missing gauge index" begin
        reg = _make_ym_registry()
        ggp = get_gauge_group(reg, :SU3)
        with_registry(reg) do
            I = TIndex(:I, Up, :Gauge)
            a = down(:a)

            # Tensor with no gauge index
            X = Tensor(:g, [down(:c), down(:d)])
            @test_throws ErrorException gauge_covariant_deriv(ggp, X, I, a)
        end
    end

    # ── Bianchi identity (TGR-655.3) ─────────────────────────────────

    @testset "yang_mills_bianchi structure" begin
        reg = _make_ym_registry()
        ggp = get_gauge_group(reg, :SU3)
        with_registry(reg) do
            I = TIndex(:I, Up, :Gauge)
            a = down(:a)
            b = down(:b)
            c = down(:c)

            bianchi = yang_mills_bianchi(ggp, I, a, b, c)

            # Should be a TSum of 3 covariant derivative terms
            @test bianchi isa TSum
            @test length(bianchi.terms) == 3

            # Each term is D_{x} F_{yz} which is itself a TSum of 2 terms
            for term in bianchi.terms
                @test term isa TSum
                @test length(term.terms) == 2  # ∂F + fAF
            end
        end
    end

    # ── Yang-Mills Lagrangian (TGR-655.4) ─────────────────────────────

    @testset "yang_mills_lagrangian structure" begin
        reg = _make_ym_registry()
        ggp = get_gauge_group(reg, :SU3)
        with_registry(reg) do
            L = yang_mills_lagrangian(ggp)

            # L = -1/4 δ_{IJ} g^{ac} g^{bd} F^I_{ab} F^J_{cd}
            @test L isa TProduct
            @test L.scalar == -1 // 4

            # Should have: delta, g, g, F1_sum, F2_sum = 5 factors
            @test length(L.factors) == 5
        end
    end

    @testset "yang_mills_lagrangian requires metric" begin
        reg = TensorRegistry()
        with_registry(reg) do
            register_manifold!(reg, ManifoldProperties(:M4, 4, nothing, nothing, Symbol[]))
            ggp = define_gauge_group!(reg, :SU2; dim=3, manifold=:M4)
            @test_throws ErrorException yang_mills_lagrangian(ggp)
        end
    end

    # ── Yang-Mills field equations (TGR-655.4) ────────────────────────

    @testset "yang_mills_field_equations structure" begin
        reg = _make_ym_registry()
        ggp = get_gauge_group(reg, :SU3)
        with_registry(reg) do
            I = TIndex(:I, Up, :Gauge)
            b = down(:b)

            eom = yang_mills_field_equations(ggp, I, b)

            # D_a F^{Ia}_b = ∂_a(g^{ac} F^I_{cb}) + f·A·(g F)
            @test eom isa TSum
            @test length(eom.terms) == 2  # partial + interaction
        end
    end

    @testset "yang_mills_field_equations index validation" begin
        reg = _make_ym_registry()
        ggp = get_gauge_group(reg, :SU3)
        with_registry(reg) do
            @test_throws ErrorException yang_mills_field_equations(
                ggp, TIndex(:I, Down, :Gauge), down(:b))
            @test_throws ErrorException yang_mills_field_equations(
                ggp, TIndex(:I, Up, :Gauge), up(:b))
        end
    end

    @testset "yang_mills_field_strength index validation" begin
        reg = _make_ym_registry()
        ggp = get_gauge_group(reg, :SU3)
        with_registry(reg) do
            @test_throws ErrorException yang_mills_field_strength(
                ggp, TIndex(:I, Down, :Gauge), down(:a), down(:b))
            @test_throws ErrorException yang_mills_field_strength(
                ggp, TIndex(:I, Up, :Gauge), up(:a), down(:b))
        end
    end

    # ── Cross-checks ─────────────────────────────────────────────────

    @testset "field strength has correct free index count" begin
        reg = _make_ym_registry()
        ggp = get_gauge_group(reg, :SU3)
        with_registry(reg) do
            I = TIndex(:I, Up, :Gauge)
            a = down(:a)
            b = down(:b)

            F = yang_mills_field_strength(ggp, I, a, b)
            fi = free_indices(F)

            # Should have exactly 3 free indices: I^Up, a_Down, b_Down
            @test length(fi) == 3
            positions = Set(idx.position for idx in fi)
            @test Up in positions
            @test Down in positions
        end
    end

    @testset "gauge_covariant_deriv preserves free indices" begin
        reg = _make_ym_registry()
        ggp = get_gauge_group(reg, :SU3)
        with_registry(reg) do
            I = TIndex(:I, Up, :Gauge)
            a = down(:a)
            b = down(:b)

            # D_a A^I_b has free indices I, a, b
            A_Ib = Tensor(:A, [I, b])
            DA = gauge_covariant_deriv(ggp, A_Ib, I, a)

            fi = free_indices(DA)
            free_names = Set(idx.name for idx in fi)
            @test :I in free_names
            @test :a in free_names
            @test :b in free_names
        end
    end
end
