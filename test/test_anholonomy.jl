@testset "Anholonomy Coefficients" begin
    using TensorGR: define_anholonomy!, anholonomy_expr, has_anholonomy,
                    get_anholonomy_name,
                    define_frame_bundle!, frame_up, frame_down,
                    TensorRegistry, TensorProperties, SymmetrySpec,
                    AntiSymmetric, Tensor, TProduct, TSum, TDeriv, TScalar,
                    up, down, with_registry, register_tensor!,
                    current_registry, has_tensor, get_tensor,
                    free_indices, simplify

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
        end
        return reg
    end

    # ---- Registration tests --------------------------------------------------

    @testset "define_anholonomy! registration" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            define_anholonomy!(reg, :e)
            @test has_tensor(reg, :Omega)
            props = get_tensor(reg, :Omega)
            @test props.rank == (1, 2)
            @test any(s -> s isa AntiSymmetric && s.i == 2 && s.j == 3,
                      props.symmetries)
            @test get(props.options, :is_anholonomy, false)
            @test get(props.options, :tetrad, nothing) === :e
        end
    end

    @testset "has_anholonomy / get_anholonomy_name" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            @test !has_anholonomy(reg, :e)
            define_anholonomy!(reg, :e)
            @test has_anholonomy(reg, :e)
            @test get_anholonomy_name(reg, :e) === :Omega
        end
    end

    @testset "custom anholonomy name" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            define_anholonomy!(reg, :e; anholonomy_name=:c_anh)
            @test has_tensor(reg, :c_anh)
            @test get_anholonomy_name(reg, :e) === :c_anh
        end
    end

    @testset "idempotent registration" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            define_anholonomy!(reg, :e)
            define_anholonomy!(reg, :e)  # no error on second call
            @test has_tensor(reg, :Omega)
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
        @test_throws ErrorException define_anholonomy!(reg, :e)
    end

    @testset "error: tetrad not registered" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_frame_bundle!(reg; manifold=:M4)
        end
        @test_throws ErrorException define_anholonomy!(reg, :missing_tetrad)
    end

    # ---- Expression construction ----------------------------------------------

    @testset "anholonomy_expr structure" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            define_anholonomy!(reg, :e)
            expr = anholonomy_expr(:e, frame_up(:I), frame_down(:J), frame_down(:K))

            # Should be a TSum of two terms
            @test expr isa TSum
            @test length(expr.terms) == 2

            # First term positive, second negative
            t1, t2 = expr.terms
            @test t1 isa TProduct
            @test t2 isa TProduct
            @test t1.scalar > 0
            @test t2.scalar < 0
        end
    end

    @testset "anholonomy_expr contains TDeriv" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            expr = anholonomy_expr(:e, frame_up(:I), frame_down(:J), frame_down(:K))

            # Each term should contain a TDeriv (partial derivative)
            for term in expr.terms
                has_deriv = any(f -> f isa TDeriv, term.factors)
                @test has_deriv
            end
        end
    end

    @testset "anholonomy_expr partial derivatives" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            expr = anholonomy_expr(:e, frame_up(:I), frame_down(:J), frame_down(:K))

            # All derivatives should be :partial
            for term in expr.terms
                for f in term.factors
                    if f isa TDeriv
                        @test f.covd === :partial
                    end
                end
            end
        end
    end

    @testset "anholonomy_expr index validation" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            # Wrong bundle for I
            @test_throws ErrorException anholonomy_expr(
                :e, up(:a), frame_down(:J), frame_down(:K))
            # Wrong position for J
            @test_throws ErrorException anholonomy_expr(
                :e, frame_up(:I), frame_up(:J), frame_down(:K))
            # Wrong bundle for K
            @test_throws ErrorException anholonomy_expr(
                :e, frame_up(:I), frame_down(:J), down(:c))
        end
    end

    @testset "anholonomy_expr uses tetrad name" begin
        reg = _make_tetrad_registry()
        # Register a second tetrad with different name
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:h, manifold=:M4, rank=(1, 1),
                symmetries=SymmetrySpec[]))
            expr_e = anholonomy_expr(:e, frame_up(:I), frame_down(:J), frame_down(:K))
            expr_h = anholonomy_expr(:h, frame_up(:I), frame_down(:J), frame_down(:K))

            # Different tetrad names should produce different expressions
            tensors_e = Symbol[]
            for t in expr_e.terms
                for f in t.factors
                    if f isa Tensor
                        push!(tensors_e, f.name)
                    elseif f isa TDeriv && f.arg isa Tensor
                        push!(tensors_e, f.arg.name)
                    end
                end
            end
            @test all(n -> n == :e, tensors_e)

            tensors_h = Symbol[]
            for t in expr_h.terms
                for f in t.factors
                    if f isa Tensor
                        push!(tensors_h, f.name)
                    elseif f isa TDeriv && f.arg isa Tensor
                        push!(tensors_h, f.arg.name)
                    end
                end
            end
            @test all(n -> n == :h, tensors_h)
        end
    end

    # ---- Antisymmetry property -----------------------------------------------

    @testset "antisymmetry: c^I_{JK} = -c^I_{KJ}" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            define_anholonomy!(reg, :e)

            # Build c^I_{JK} + c^I_{KJ} — should be structurally zero
            c_JK = anholonomy_expr(:e, frame_up(:I), frame_down(:J), frame_down(:K))
            c_KJ = anholonomy_expr(:e, frame_up(:I), frame_down(:K), frame_down(:J))

            # Sum: each has 2 terms, with swapped J↔K in the second call
            # c_JK = e^I_a (e^b_J ∂_b e^a_K - e^b_K ∂_b e^a_J)
            # c_KJ = e^I_a (e^b_K ∂_b e^a_J - e^b_J ∂_b e^a_K)
            # c_JK + c_KJ = 0 algebraically

            total = TSum([c_JK.terms..., c_KJ.terms...])
            result = simplify(total; registry=reg)
            @test result == TScalar(0 // 1) || result == TScalar(0)
        end
    end

    # ---- Frame index count ---------------------------------------------------

    @testset "expression has correct free frame indices" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            expr = anholonomy_expr(:e, frame_up(:I), frame_down(:J), frame_down(:K))
            fi = free_indices(expr)
            frame_fi = filter(idx -> idx.vbundle === :Lorentz, fi)
            @test length(frame_fi) == 3
            names = Set(idx.name for idx in frame_fi)
            @test :I ∈ names
            @test :J ∈ names
            @test :K ∈ names
        end
    end

    # ---- Abstract tensor usage -----------------------------------------------

    @testset "Omega tensor can be used in expressions" begin
        reg = _make_tetrad_registry()
        with_registry(reg) do
            define_anholonomy!(reg, :e)

            # Build c^I_{JK} as abstract tensor
            c = Tensor(:Omega, [frame_up(:I), frame_down(:J), frame_down(:K)])
            @test c isa Tensor
            @test c.name === :Omega

            # Can form products and sums
            c2 = TProduct(1 // 1, [c, c])
            @test c2 isa TProduct

            # Canonicalization respects antisymmetry
            c_swap = Tensor(:Omega, [frame_up(:I), frame_down(:K), frame_down(:J)])
            summed = TSum([c, c_swap])
            result = simplify(summed; registry=reg)
            @test result == TScalar(0 // 1) || result == TScalar(0)
        end
    end
end
