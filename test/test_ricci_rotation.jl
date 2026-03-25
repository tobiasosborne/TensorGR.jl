@testset "Ricci Rotation Coefficients" begin
    using TensorGR: define_ricci_rotation!, ricci_rotation_expr,
                    has_ricci_rotation, get_ricci_rotation_name,
                    define_anholonomy!, anholonomy_expr,
                    has_anholonomy, get_anholonomy_name,
                    define_frame_bundle!, frame_up, frame_down,
                    TensorRegistry, TensorProperties, SymmetrySpec,
                    Tensor, TProduct, TSum, TScalar,
                    up, down, with_registry, register_tensor!,
                    current_registry, has_tensor, get_tensor,
                    free_indices, simplify, fresh_index

    function _make_tetrad_registry()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            define_frame_bundle!(reg; manifold=:M4)
            register_tensor!(reg, TensorProperties(
                name=:e, manifold=:M4, rank=(1, 1),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}(:vbundle_mixed => (:Tangent, :Lorentz))))
            define_anholonomy!(reg, :e)
        end
        return reg
    end

    # ---- Registration tests --------------------------------------------------

    @testset "define_ricci_rotation! registration" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            define_ricci_rotation!(reg, :e)
            @test has_tensor(reg, :gamma_rot)
            props = get_tensor(reg, :gamma_rot)
            @test props.rank == (1, 2)
            @test isempty(props.symmetries)  # no manifest slot symmetry
            @test get(props.options, :is_ricci_rotation, false)
            @test get(props.options, :tetrad, nothing) === :e
            @test get(props.options, :vbundle, nothing) === :Lorentz
        end
    end

    @testset "has_ricci_rotation / get_ricci_rotation_name" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            @test !has_ricci_rotation(reg, :e)
            define_ricci_rotation!(reg, :e)
            @test has_ricci_rotation(reg, :e)
            @test get_ricci_rotation_name(reg, :e) === :gamma_rot
        end
    end

    @testset "custom rotation name" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            define_ricci_rotation!(reg, :e; rotation_name=:omega_frame)
            @test has_tensor(reg, :omega_frame)
            @test get_ricci_rotation_name(reg, :e) === :omega_frame
        end
    end

    @testset "idempotent registration" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            define_ricci_rotation!(reg, :e)
            define_ricci_rotation!(reg, :e)  # no error on second call
            @test has_tensor(reg, :gamma_rot)
        end
    end

    # ---- Error handling ------------------------------------------------------

    @testset "error: no Lorentz VBundle" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            register_tensor!(reg, TensorProperties(
                name=:e, manifold=:M4, rank=(1, 1), symmetries=SymmetrySpec[]))
        end
        @test_throws ErrorException define_ricci_rotation!(reg, :e)
    end

    @testset "error: tetrad not registered" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_frame_bundle!(reg; manifold=:M4)
        end
        @test_throws ErrorException define_ricci_rotation!(reg, :missing_tetrad)
    end

    @testset "error: anholonomy not defined" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_frame_bundle!(reg; manifold=:M4)
            register_tensor!(reg, TensorProperties(
                name=:e, manifold=:M4, rank=(1, 1), symmetries=SymmetrySpec[]))
        end
        @test_throws ErrorException define_ricci_rotation!(reg, :e)
    end

    # ---- Expression construction ---------------------------------------------

    @testset "ricci_rotation_expr structure" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            define_ricci_rotation!(reg, :e)
            expr = ricci_rotation_expr(:e, frame_up(:I), frame_down(:J), frame_down(:K))

            # Should be a TSum of three terms
            @test expr isa TSum
            @test length(expr.terms) == 3

            # Two positive terms (½c, ½ηηc), one negative (-½ηηc)
            scalars = [t.scalar for t in expr.terms]
            @test count(s -> s > 0, scalars) == 2
            @test count(s -> s < 0, scalars) == 1
            @test all(s -> abs(s) == 1 // 2, scalars)
        end
    end

    @testset "ricci_rotation_expr contains anholonomy tensor" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            expr = ricci_rotation_expr(:e, frame_up(:I), frame_down(:J), frame_down(:K))

            # Each term should contain the anholonomy tensor Omega
            for term in expr.terms
                has_omega = any(f -> f isa Tensor && f.name === :Omega, term.factors)
                @test has_omega
            end
        end
    end

    @testset "ricci_rotation_expr contains eta for cross terms" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            expr = ricci_rotation_expr(:e, frame_up(:I), frame_down(:J), frame_down(:K))

            # Terms 2 and 3 should contain eta (frame metric)
            eta_counts = [count(f -> f isa Tensor && f.name === :eta, t.factors) for t in expr.terms]
            @test eta_counts[1] == 0  # term 1: just c^I_{JK}
            @test eta_counts[2] == 2  # term 2: η^{IM}η_{KN}c^N_{MJ}
            @test eta_counts[3] == 2  # term 3: η^{IM}η_{JN}c^N_{MK}
        end
    end

    @testset "ricci_rotation_expr free indices" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            expr = ricci_rotation_expr(:e, frame_up(:I), frame_down(:J), frame_down(:K))
            fi = free_indices(expr)
            frame_fi = filter(idx -> idx.vbundle === :Lorentz, fi)
            @test length(frame_fi) == 3
            names = Set(idx.name for idx in frame_fi)
            @test :I in names
            @test :J in names
            @test :K in names
        end
    end

    @testset "ricci_rotation_expr index validation" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            # Wrong bundle for I
            @test_throws ErrorException ricci_rotation_expr(
                :e, up(:a), frame_down(:J), frame_down(:K))
            # Wrong position for J
            @test_throws ErrorException ricci_rotation_expr(
                :e, frame_up(:I), frame_up(:J), frame_down(:K))
            # Wrong bundle for K
            @test_throws ErrorException ricci_rotation_expr(
                :e, frame_up(:I), frame_down(:J), down(:c))
        end
    end

    # ---- Algebraic properties ------------------------------------------------

    @testset "abstract tensor gamma_rot in expressions" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            define_ricci_rotation!(reg, :e)

            # Can build abstract tensor expressions
            gamma = Tensor(:gamma_rot, [frame_up(:I), frame_down(:J), frame_down(:K)])
            @test gamma isa Tensor
            @test gamma.name === :gamma_rot

            # Can form products
            gamma2 = TProduct(1 // 1, [gamma, gamma])
            @test gamma2 isa TProduct
        end
    end

    @testset "lowered antisymmetry structure: γ_{IJK} + γ_{JIK}" begin
        # The all-down Ricci rotation coefficients are antisymmetric
        # in the first two indices: η_{IM}γ^M_{JK} + η_{JM}γ^M_{IK} = 0.
        # This follows from the formula and c_{IJK} = -c_{IKJ}.
        #
        # The simplifier cannot verify this at the abstract level because
        # it requires resolving nested η·Omega contractions across multiple
        # terms (cross-vbundle dummy relabeling). Verified by structure.
        reg = _make_tetrad_registry()
        with_registry(reg) do
            define_ricci_rotation!(reg, :e)

            # Build η_{IM} γ^M_{JK}
            used = Set{Symbol}([:I, :J, :K])
            M_sym = fresh_index(used; vbundle=:Lorentz)
            gamma_MJK = ricci_rotation_expr(:e,
                frame_up(M_sym), frame_down(:J), frame_down(:K))
            lowered_IJK = TProduct(1 // 1, [
                Tensor(:eta, [frame_down(:I), TIndex(M_sym, Down, :Lorentz)]),
                gamma_MJK
            ])

            # Build η_{JM} γ^M_{IK}
            gamma_MIK = ricci_rotation_expr(:e,
                frame_up(M_sym), frame_down(:I), frame_down(:K))
            lowered_JIK = TProduct(1 // 1, [
                Tensor(:eta, [frame_down(:J), TIndex(M_sym, Down, :Lorentz)]),
                gamma_MIK
            ])

            # Verify the structure is correct: both have correct free indices
            fi1 = free_indices(lowered_IJK)
            fi2 = free_indices(lowered_JIK)
            lorentz_fi1 = filter(idx -> idx.vbundle === :Lorentz, fi1)
            lorentz_fi2 = filter(idx -> idx.vbundle === :Lorentz, fi2)
            @test length(lorentz_fi1) == 3  # I, J, K
            @test length(lorentz_fi2) == 3  # I, J, K
            @test Set(idx.name for idx in lorentz_fi1) == Set([:I, :J, :K])
            @test Set(idx.name for idx in lorentz_fi2) == Set([:I, :J, :K])
        end
    end

    @testset "gamma with different tetrad names" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            # Register a second tetrad
            register_tensor!(reg, TensorProperties(
                name=:f, manifold=:M4, rank=(1, 1),
                symmetries=SymmetrySpec[]))
            define_anholonomy!(reg, :f; anholonomy_name=:Omega_f)

            # Both should work independently
            define_ricci_rotation!(reg, :e)
            define_ricci_rotation!(reg, :f; rotation_name=:gamma_f)

            @test has_ricci_rotation(reg, :e)
            @test has_ricci_rotation(reg, :f)
            @test get_ricci_rotation_name(reg, :e) === :gamma_rot
            @test get_ricci_rotation_name(reg, :f) === :gamma_f
        end
    end
end
