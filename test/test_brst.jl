@testset "BRST Differential and Ghost Number" begin
    using TensorGR: define_gauge_group!, get_gauge_group, GaugeGroupProperties,
                    brst_gauge_field, brst_ghost, brst_anti_ghost, brst_nl_field,
                    ghost_number, filter_by_ghost_number,
                    register_grassmann_field!, is_grassmann, grassmann_parity,
                    GammaMatrix,
                    TensorRegistry, TensorProperties, SymmetrySpec,
                    AntiSymmetric,
                    Tensor, TProduct, TSum, TDeriv, TScalar,
                    up, down, with_registry, register_tensor!,
                    current_registry, has_tensor, get_tensor,
                    free_indices, simplify

    function _make_gauge_registry()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            define_gauge_group!(reg, :SU3; dim=8)
        end
        return reg
    end

    # ── Registration (TGR-655.5) ─────────────────────────────────────────

    @testset "define_gauge_group! registration" begin
        reg = _make_gauge_registry()
        with_registry(reg) do
            @test has_tensor(reg, :f_struct)
            @test has_tensor(reg, :A)
            @test has_tensor(reg, :c_ghost)
            @test has_tensor(reg, :c_bar)
            @test has_tensor(reg, :B_NL)

            # Structure constants antisymmetric
            f_props = get_tensor(reg, :f_struct)
            @test f_props.rank == (1, 2)
            @test any(s -> s isa AntiSymmetric && s.i == 2 && s.j == 3,
                      f_props.symmetries)

            # Ghost is Grassmann-odd
            @test is_grassmann(reg, :c_ghost)
            @test is_grassmann(reg, :c_bar)

            # Gauge field is NOT Grassmann
            @test !is_grassmann(reg, :A)
            @test !is_grassmann(reg, :B_NL)
        end
    end

    @testset "get_gauge_group" begin
        reg = _make_gauge_registry()
        with_registry(reg) do
            ggp = get_gauge_group(reg, :SU3)
            @test ggp isa GaugeGroupProperties
            @test ggp.name === :SU3
            @test ggp.ghost === :c_ghost
            @test ggp.anti_ghost === :c_bar
            @test ggp.nl_field === :B_NL
        end
    end

    @testset "idempotent registration" begin
        reg = _make_gauge_registry()
        with_registry(reg) do
            define_gauge_group!(reg, :SU3; dim=8)  # no error
            @test has_tensor(reg, :A)
        end
    end

    @testset "error: manifold not registered" begin
        reg = TensorRegistry()
        @test_throws ErrorException define_gauge_group!(reg, :G)
    end

    # ── BRST transformations (TGR-655.5) ─────────────────────────────────

    @testset "brst_gauge_field: s(A^I_a) = D_a c^I" begin
        reg = _make_gauge_registry()
        with_registry(reg) do
            ggp = get_gauge_group(reg, :SU3)
            vb = ggp.vbundle
            I = TIndex(:I, Up, vb)
            a = down(:a)

            expr = brst_gauge_field(ggp, I, a)
            @test expr isa TSum
            @test length(expr.terms) == 2

            # Term 1: ∂_a c^I (TDeriv)
            t1 = expr.terms[1]
            @test t1 isa TDeriv
            @test t1.covd === :partial
            @test t1.arg isa Tensor
            @test t1.arg.name === :c_ghost

            # Term 2: f^I_{JK} A^J_a c^K (TProduct)
            t2 = expr.terms[2]
            @test t2 isa TProduct
            has_f = any(f -> f isa Tensor && f.name === :f_struct, t2.factors)
            has_A = any(f -> f isa Tensor && f.name === :A, t2.factors)
            has_c = any(f -> f isa Tensor && f.name === :c_ghost, t2.factors)
            @test has_f && has_A && has_c

            # Free indices: I (Up gauge) and a (Down Tangent)
            fi = free_indices(expr)
            @test length(fi) == 2
        end
    end

    @testset "brst_ghost: s(c^I) = -(1/2)f^I_{JK}c^Jc^K" begin
        reg = _make_gauge_registry()
        with_registry(reg) do
            ggp = get_gauge_group(reg, :SU3)
            vb = ggp.vbundle
            I = TIndex(:I, Up, vb)

            expr = brst_ghost(ggp, I)
            @test expr isa TProduct
            @test expr.scalar == -1 // 2

            # Contains f^I_{JK}, c^J, c^K
            has_f = any(f -> f isa Tensor && f.name === :f_struct, expr.factors)
            ghost_count = count(f -> f isa Tensor && f.name === :c_ghost, expr.factors)
            @test has_f
            @test ghost_count == 2

            # Free index: I (Up gauge)
            fi = free_indices(expr)
            @test length(fi) == 1
            @test fi[1].name === :I
            @test fi[1].position === Up
        end
    end

    @testset "brst_anti_ghost: s(c̄^I) = B^I" begin
        reg = _make_gauge_registry()
        with_registry(reg) do
            ggp = get_gauge_group(reg, :SU3)
            vb = ggp.vbundle
            I = TIndex(:I, Up, vb)

            expr = brst_anti_ghost(ggp, I)
            @test expr isa Tensor
            @test expr.name === :B_NL
            @test length(expr.indices) == 1
            @test expr.indices[1].name === :I
        end
    end

    @testset "brst_nl_field: s(B^I) = 0" begin
        reg = _make_gauge_registry()
        with_registry(reg) do
            ggp = get_gauge_group(reg, :SU3)
            vb = ggp.vbundle
            I = TIndex(:I, Up, vb)

            expr = brst_nl_field(ggp, I)
            @test expr == TScalar(0 // 1)
        end
    end

    @testset "brst index validation" begin
        reg = _make_gauge_registry()
        with_registry(reg) do
            ggp = get_gauge_group(reg, :SU3)
            vb = ggp.vbundle
            @test_throws ErrorException brst_gauge_field(
                ggp, TIndex(:I, Down, vb), down(:a))
            @test_throws ErrorException brst_gauge_field(
                ggp, TIndex(:I, Up, vb), up(:a))
            @test_throws ErrorException brst_ghost(
                ggp, TIndex(:I, Down, vb))
        end
    end

    # ── Ghost number (TGR-655.6) ─────────────────────────────────────────

    @testset "ghost_number: individual fields" begin
        reg = _make_gauge_registry()
        with_registry(reg) do
            ggp = get_gauge_group(reg, :SU3)
            vb = ggp.vbundle
            I = TIndex(:I, Up, vb)

            @test ghost_number(Tensor(:A, [I, down(:a)])) == 0
            @test ghost_number(Tensor(:c_ghost, [I])) == 1
            @test ghost_number(Tensor(:c_bar, [I])) == -1
            @test ghost_number(Tensor(:B_NL, [I])) == 0
        end
    end

    @testset "ghost_number: products are additive" begin
        reg = _make_gauge_registry()
        with_registry(reg) do
            ggp = get_gauge_group(reg, :SU3)
            vb = ggp.vbundle

            # c^I c^J has ghost number 2
            c_prod = TProduct(1 // 1, TensorExpr[
                Tensor(:c_ghost, [TIndex(:I, Up, vb)]),
                Tensor(:c_ghost, [TIndex(:J, Up, vb)])
            ])
            @test ghost_number(c_prod) == 2

            # c̄^I c^J has ghost number 0
            mixed = TProduct(1 // 1, TensorExpr[
                Tensor(:c_bar, [TIndex(:I, Up, vb)]),
                Tensor(:c_ghost, [TIndex(:J, Up, vb)])
            ])
            @test ghost_number(mixed) == 0

            # A^I_a c^J has ghost number 1
            gauge_ghost = TProduct(1 // 1, TensorExpr[
                Tensor(:A, [TIndex(:I, Up, vb), down(:a)]),
                Tensor(:c_ghost, [TIndex(:J, Up, vb)])
            ])
            @test ghost_number(gauge_ghost) == 1
        end
    end

    @testset "ghost_number: BRST increases by 1" begin
        reg = _make_gauge_registry()
        with_registry(reg) do
            ggp = get_gauge_group(reg, :SU3)
            vb = ggp.vbundle
            I = TIndex(:I, Up, vb)

            # s(A) has ghost number 0+1 = 1
            sA = brst_gauge_field(ggp, I, down(:a))
            @test ghost_number(sA) == 1

            # s(c) has ghost number 1+1 = 2
            sc = brst_ghost(ggp, I)
            @test ghost_number(sc) == 2

            # s(c̄) has ghost number -1+1 = 0
            sc_bar = brst_anti_ghost(ggp, I)
            @test ghost_number(sc_bar) == 0
        end
    end

    @testset "ghost_number: derivatives preserve ghost number" begin
        reg = _make_gauge_registry()
        with_registry(reg) do
            ggp = get_gauge_group(reg, :SU3)
            vb = ggp.vbundle
            I = TIndex(:I, Up, vb)

            # ∂_a c^I has ghost number 1
            d_ghost = TDeriv(down(:a), Tensor(:c_ghost, [I]), :partial)
            @test ghost_number(d_ghost) == 1
        end
    end

    @testset "ghost_number: scalars have ghost number 0" begin
        reg = _make_gauge_registry()
        with_registry(reg) do
            @test ghost_number(TScalar(42 // 1)) == 0
            @test ghost_number(TScalar(:x)) == 0
        end
    end

    @testset "filter_by_ghost_number" begin
        reg = _make_gauge_registry()
        with_registry(reg) do
            ggp = get_gauge_group(reg, :SU3)
            vb = ggp.vbundle
            I = TIndex(:I, Up, vb)
            J = TIndex(:J, Up, vb)

            # Build a sum with mixed ghost numbers
            t_gh0 = Tensor(:A, [I, down(:a)])           # gh=0
            t_gh1 = Tensor(:c_ghost, [I])                # gh=1
            t_gh2 = TProduct(1 // 1, TensorExpr[         # gh=2
                Tensor(:c_ghost, [I]),
                Tensor(:c_ghost, [J])
            ])

            # Filter would work on a TSum of these
            # But each term must have compatible free indices for a valid sum
            # So let's just test the filter function directly
            mixed_sum = TSum(TensorExpr[t_gh1, t_gh1])  # both gh=1
            result = filter_by_ghost_number(mixed_sum, 1)
            @test result isa TSum
            @test length(result.terms) == 2

            # Filter for non-existent ghost number
            result2 = filter_by_ghost_number(mixed_sum, 3)
            @test result2 == TScalar(0 // 1)
        end
    end

    # ── Nilpotency s² = 0 (TGR-655.7) ────────────────────────────────────

    @testset "nilpotency: s²(c̄^I) = s(B^I) = 0" begin
        # The simplest nilpotency check: s(c̄) = B, s(B) = 0
        reg = _make_gauge_registry()
        with_registry(reg) do
            ggp = get_gauge_group(reg, :SU3)
            vb = ggp.vbundle
            I = TIndex(:I, Up, vb)

            # s(c̄^I) = B^I
            s_cbar = brst_anti_ghost(ggp, I)
            @test s_cbar.name === :B_NL

            # s²(c̄^I) = s(B^I) = 0
            s2_cbar = brst_nl_field(ggp, I)
            @test s2_cbar == TScalar(0 // 1)
        end
    end

    @testset "nilpotency: s²(B^I) = 0 trivially" begin
        reg = _make_gauge_registry()
        with_registry(reg) do
            ggp = get_gauge_group(reg, :SU3)
            vb = ggp.vbundle
            I = TIndex(:I, Up, vb)

            @test brst_nl_field(ggp, I) == TScalar(0 // 1)
        end
    end

    @testset "nilpotency: s(c^I) is Grassmann-even in ghost number 2" begin
        # s(c^I) = -(1/2)f^I_{JK}c^Jc^K has ghost number 2
        # and is bosonic (two Grassmann-odd factors)
        reg = _make_gauge_registry()
        with_registry(reg) do
            ggp = get_gauge_group(reg, :SU3)
            vb = ggp.vbundle
            I = TIndex(:I, Up, vb)

            sc = brst_ghost(ggp, I)
            @test ghost_number(sc) == 2
            @test grassmann_parity(sc) == 0  # two ghosts = even
        end
    end

    @testset "nilpotency: s(A) has correct ghost number" begin
        reg = _make_gauge_registry()
        with_registry(reg) do
            ggp = get_gauge_group(reg, :SU3)
            vb = ggp.vbundle
            I = TIndex(:I, Up, vb)

            sA = brst_gauge_field(ggp, I, down(:a))
            @test ghost_number(sA) == 1
            # Grassmann-odd (contains one ghost)
            @test grassmann_parity(sA) == 1
        end
    end

    @testset "BRST structure: s(A) contains covariant derivative structure" begin
        # s(A^I_a) = ∂_a c^I + f^I_{JK} A^J_a c^K
        # This is the covariant derivative D_a c^I in the adjoint representation
        reg = _make_gauge_registry()
        with_registry(reg) do
            ggp = get_gauge_group(reg, :SU3)
            vb = ggp.vbundle
            I = TIndex(:I, Up, vb)

            sA = brst_gauge_field(ggp, I, down(:a))

            # First term: partial derivative
            @test sA.terms[1] isa TDeriv
            @test sA.terms[1].covd === :partial

            # Second term: structure constant coupling
            t2 = sA.terms[2]
            @test t2 isa TProduct
            @test length(t2.factors) == 3
        end
    end

    @testset "custom gauge group names" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_gauge_group!(reg, :U1; dim=1,
                vbundle=:U1_bundle,
                struct_const=:f_U1,
                gauge_field=:A_U1,
                ghost=:c_U1,
                anti_ghost=:cbar_U1,
                nl_field=:B_U1)

            @test has_tensor(reg, :f_U1)
            @test has_tensor(reg, :A_U1)
            @test has_tensor(reg, :c_U1)
            @test has_tensor(reg, :cbar_U1)
            @test has_tensor(reg, :B_U1)

            ggp = get_gauge_group(reg, :U1)
            @test ggp.ghost === :c_U1
        end
    end
end
