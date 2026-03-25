@testset "Spin Connection for Dirac Fields" begin
    using TensorGR: define_spin_connection!, spin_connection_expr,
                    has_spin_connection, get_spin_connection_name,
                    dirac_covd_expr, dirac_bar_covd_expr,
                    define_fermion!, get_conjugate_name,
                    define_ricci_rotation!, has_ricci_rotation,
                    define_anholonomy!,
                    define_frame_bundle!, frame_up, frame_down,
                    GammaMatrix,
                    TensorRegistry, TensorProperties, SymmetrySpec,
                    AntiSymmetric,
                    Tensor, TProduct, TSum, TDeriv, TScalar,
                    up, down, with_registry, register_tensor!,
                    current_registry, has_tensor, get_tensor,
                    free_indices, indices

    function _make_spin_connection_registry()
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
            define_ricci_rotation!(reg, :e)
            define_spin_connection!(reg, :e)
            define_fermion!(reg, :psi)
        end
        return reg
    end

    # ---- Registration --------------------------------------------------------

    @testset "define_spin_connection! registration" begin
        reg = _make_spin_connection_registry()
        with_registry(reg) do
            @test has_tensor(reg, :omega_spin)
            props = get_tensor(reg, :omega_spin)
            @test props.rank == (2, 1)
            @test any(s -> s isa AntiSymmetric && s.i == 1 && s.j == 2,
                      props.symmetries)
            @test get(props.options, :is_spin_connection, false)
            @test get(props.options, :tetrad, nothing) === :e
        end
    end

    @testset "has_spin_connection / get_spin_connection_name" begin
        reg = _make_spin_connection_registry()
        with_registry(reg) do
            @test has_spin_connection(reg, :e)
            @test get_spin_connection_name(reg, :e) === :omega_spin
        end
    end

    @testset "idempotent registration" begin
        reg = _make_spin_connection_registry()
        with_registry(reg) do
            define_spin_connection!(reg, :e)  # no error
            @test has_tensor(reg, :omega_spin)
        end
    end

    @testset "error: missing Lorentz VBundle" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            register_tensor!(reg, TensorProperties(
                name=:e, manifold=:M4, rank=(1, 1), symmetries=SymmetrySpec[]))
        end
        @test_throws ErrorException define_spin_connection!(reg, :e)
    end

    @testset "error: missing Ricci rotation" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_frame_bundle!(reg; manifold=:M4)
            register_tensor!(reg, TensorProperties(
                name=:e, manifold=:M4, rank=(1, 1), symmetries=SymmetrySpec[]))
            define_anholonomy!(reg, :e)
            # Don't define Ricci rotation
        end
        @test_throws ErrorException define_spin_connection!(reg, :e)
    end

    # ---- spin_connection_expr ------------------------------------------------

    @testset "spin_connection_expr structure" begin
        reg = _make_spin_connection_registry()
        with_registry(reg) do
            expr = spin_connection_expr(:e, down(:a), frame_up(:I), frame_up(:J))
            @test expr isa TProduct
            # Should have 3 factors: tetrad, eta, gamma_rot
            @test length(expr.factors) == 3

            # Should contain the tetrad
            has_tetrad = any(f -> f isa Tensor && f.name === :e, expr.factors)
            @test has_tetrad

            # Should contain eta (frame metric)
            has_eta = any(f -> f isa Tensor && f.name === :eta, expr.factors)
            @test has_eta

            # Should contain gamma_rot
            has_gamma = any(f -> f isa Tensor && f.name === :gamma_rot, expr.factors)
            @test has_gamma
        end
    end

    @testset "spin_connection_expr free indices" begin
        reg = _make_spin_connection_registry()
        with_registry(reg) do
            expr = spin_connection_expr(:e, down(:a), frame_up(:I), frame_up(:J))
            fi = free_indices(expr)

            tangent_fi = filter(idx -> idx.vbundle === :Tangent, fi)
            lorentz_fi = filter(idx -> idx.vbundle === :Lorentz, fi)

            @test length(tangent_fi) == 1
            @test tangent_fi[1].name === :a
            @test tangent_fi[1].position === Down

            @test length(lorentz_fi) == 2
            names = Set(idx.name for idx in lorentz_fi)
            @test :I in names && :J in names
            @test all(idx -> idx.position === Up, lorentz_fi)
        end
    end

    @testset "spin_connection_expr index validation" begin
        reg = _make_spin_connection_registry()
        with_registry(reg) do
            @test_throws ErrorException spin_connection_expr(
                :e, up(:a), frame_up(:I), frame_up(:J))
            @test_throws ErrorException spin_connection_expr(
                :e, down(:a), frame_down(:I), frame_up(:J))
        end
    end

    # ---- dirac_covd_expr -----------------------------------------------------

    @testset "dirac_covd_expr structure" begin
        reg = _make_spin_connection_registry()
        with_registry(reg) do
            expr = dirac_covd_expr(:psi, down(:a), :e)
            @test expr isa TSum
            @test length(expr.terms) == 2

            # Term 1: partial derivative ∂_a ψ
            t1 = expr.terms[1]
            @test t1 isa TDeriv
            @test t1.covd === :partial

            # Term 2: (1/4) ω_a^{IJ} γ_I γ_J ψ
            t2 = expr.terms[2]
            @test t2 isa TProduct
            @test t2.scalar == 1 // 4

            # Should contain omega_spin tensor
            has_omega = any(f -> f isa Tensor && f.name === :omega_spin, t2.factors)
            @test has_omega

            # Should contain two gamma matrices
            gamma_count = count(f -> f isa GammaMatrix, t2.factors)
            @test gamma_count == 2

            # Should contain psi
            has_psi = any(f -> f isa Tensor && f.name === :psi, t2.factors)
            @test has_psi
        end
    end

    @testset "dirac_covd_expr free indices" begin
        reg = _make_spin_connection_registry()
        with_registry(reg) do
            expr = dirac_covd_expr(:psi, down(:a), :e)
            fi = free_indices(expr)
            tangent_fi = filter(idx -> idx.vbundle === :Tangent, fi)
            lorentz_fi = filter(idx -> idx.vbundle === :Lorentz, fi)
            @test length(tangent_fi) == 1
            @test tangent_fi[1].position === Down
            # Lorentz dummy indices I,J must be fully contracted
            @test isempty(lorentz_fi)
        end
    end

    @testset "dirac_covd_expr index validation" begin
        reg = _make_spin_connection_registry()
        with_registry(reg) do
            @test_throws ErrorException dirac_covd_expr(:psi, up(:a), :e)
        end
    end

    # ---- dirac_bar_covd_expr -------------------------------------------------

    @testset "dirac_bar_covd_expr structure" begin
        reg = _make_spin_connection_registry()
        with_registry(reg) do
            expr = dirac_bar_covd_expr(:psi, down(:a), :e)
            @test expr isa TSum
            @test length(expr.terms) == 2

            # Term 1: ∂_a ψ̄
            t1 = expr.terms[1]
            @test t1 isa TDeriv
            @test t1.arg isa Tensor
            @test t1.arg.name === :psi_bar

            # Term 2: -(1/4) ω γ γ ψ̄  (negative sign!)
            t2 = expr.terms[2]
            @test t2 isa TProduct
            @test t2.scalar == -1 // 4

            # Should contain psi_bar
            has_psi_bar = any(f -> f isa Tensor && f.name === :psi_bar, t2.factors)
            @test has_psi_bar
        end
    end

    @testset "dirac_bar_covd_expr free index" begin
        reg = _make_spin_connection_registry()
        with_registry(reg) do
            expr = dirac_bar_covd_expr(:psi, down(:a), :e)
            fi = free_indices(expr)
            tangent_fi = filter(idx -> idx.vbundle === :Tangent, fi)
            @test length(tangent_fi) == 1
            @test tangent_fi[1].position === Down
        end
    end
end
