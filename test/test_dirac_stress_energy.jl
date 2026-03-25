@testset "Dirac Stress-Energy Tensor" begin
    using TensorGR: define_dirac_stress_energy!, dirac_stress_energy_expr,
                    dirac_stress_trace_expr, get_dirac_stress_energy,
                    DiracStressEnergy,
                    define_fermion!, is_grassmann, grassmann_parity,
                    GammaMatrix,
                    TensorRegistry, TensorProperties, SymmetrySpec,
                    Symmetric,
                    Tensor, TProduct, TSum, TDeriv, TScalar,
                    up, down, with_registry, register_tensor!,
                    current_registry, has_tensor, get_tensor,
                    free_indices

    function _make_stress_energy_registry()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            define_fermion!(reg, :psi)
        end
        return reg
    end

    # ---- Registration --------------------------------------------------------

    @testset "define_dirac_stress_energy! registration" begin
        reg = _make_stress_energy_registry()
        with_registry(reg) do
            dse = define_dirac_stress_energy!(reg, :T_D;
                manifold=:M4, metric=:g, field=:psi)
            @test dse isa DiracStressEnergy
            @test dse.name === :T_D
            @test dse.field === :psi
            @test dse.conjugate === :psi_bar

            @test has_tensor(reg, :T_D)
            props = get_tensor(reg, :T_D)
            @test props.rank == (2, 0)
            @test any(s -> s isa Symmetric && s.i == 1 && s.j == 2,
                      props.symmetries)
            @test get(props.options, :is_stress_energy, false)
            @test get(props.options, :matter_type, nothing) === :dirac
        end
    end

    @testset "get_dirac_stress_energy" begin
        reg = _make_stress_energy_registry()
        with_registry(reg) do
            define_dirac_stress_energy!(reg, :T_D;
                manifold=:M4, metric=:g, field=:psi)
            dse = get_dirac_stress_energy(reg, :T_D)
            @test dse.name === :T_D
            @test dse.field === :psi
        end
    end

    @testset "error: field not Grassmann" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            register_tensor!(reg, TensorProperties(
                name=:phi, manifold=:M4, rank=(0, 0), symmetries=SymmetrySpec[]))
        end
        @test_throws ErrorException define_dirac_stress_energy!(
            reg, :T; manifold=:M4, metric=:g, field=:phi)
    end

    @testset "error: manifold not registered" begin
        reg = TensorRegistry()
        @test_throws ErrorException define_dirac_stress_energy!(
            reg, :T; manifold=:M4, metric=:g, field=:psi)
    end

    # ---- Expression construction ---------------------------------------------

    @testset "dirac_stress_energy_expr structure" begin
        reg = _make_stress_energy_registry()
        with_registry(reg) do
            dse = define_dirac_stress_energy!(reg, :T_D;
                manifold=:M4, metric=:g, field=:psi)
            expr = dirac_stress_energy_expr(up(:a), up(:b), dse)

            @test expr isa TSum
            # Four symmetrized terms
            @test length(expr.terms) == 4

            # All terms should contain i (imaginary unit)
            for term in expr.terms
                has_im = any(f -> f isa TScalar && f.val === :im, term.factors)
                @test has_im
            end

            # Two terms positive (i/4), two negative (-i/4)
            positive = count(t -> t.scalar > 0, expr.terms)
            negative = count(t -> t.scalar < 0, expr.terms)
            @test positive == 2
            @test negative == 2
        end
    end

    @testset "dirac_stress_energy_expr free indices" begin
        reg = _make_stress_energy_registry()
        with_registry(reg) do
            dse = define_dirac_stress_energy!(reg, :T_D;
                manifold=:M4, metric=:g, field=:psi)
            expr = dirac_stress_energy_expr(up(:a), up(:b), dse)

            fi = free_indices(expr)
            tangent_fi = filter(idx -> idx.vbundle === :Tangent, fi)
            @test length(tangent_fi) == 2
            @test Set(idx.name for idx in tangent_fi) == Set([:a, :b])
            @test all(idx -> idx.position === Up, tangent_fi)
        end
    end

    @testset "dirac_stress_energy_expr contains derivatives" begin
        reg = _make_stress_energy_registry()
        with_registry(reg) do
            dse = define_dirac_stress_energy!(reg, :T_D;
                manifold=:M4, metric=:g, field=:psi)
            expr = dirac_stress_energy_expr(up(:a), up(:b), dse)

            # Each term should contain exactly one TDeriv
            for term in expr.terms
                deriv_count = count(f -> f isa TDeriv, term.factors)
                @test deriv_count == 1
            end
        end
    end

    @testset "dirac_stress_energy_expr contains gamma matrices" begin
        reg = _make_stress_energy_registry()
        with_registry(reg) do
            dse = define_dirac_stress_energy!(reg, :T_D;
                manifold=:M4, metric=:g, field=:psi)
            expr = dirac_stress_energy_expr(up(:a), up(:b), dse)

            # Each term should contain exactly one GammaMatrix
            for term in expr.terms
                gamma_count = count(f -> f isa GammaMatrix, term.factors)
                @test gamma_count == 1
            end
        end
    end

    @testset "dirac_stress_energy_expr index validation" begin
        reg = _make_stress_energy_registry()
        with_registry(reg) do
            dse = define_dirac_stress_energy!(reg, :T_D;
                manifold=:M4, metric=:g, field=:psi)
            @test_throws ErrorException dirac_stress_energy_expr(
                down(:a), up(:b), dse)
        end
    end

    # ---- Trace ---------------------------------------------------------------

    @testset "dirac_stress_trace_expr: T^a_a = mψ̄ψ" begin
        reg = _make_stress_energy_registry()
        with_registry(reg) do
            dse = define_dirac_stress_energy!(reg, :T_D;
                manifold=:M4, metric=:g, field=:psi)
            trace = dirac_stress_trace_expr(dse)

            @test trace isa TProduct
            # Should contain mass, psi_bar, psi
            has_mass = any(f -> f isa TScalar && f.val === :m, trace.factors)
            has_psi_bar = any(f -> f isa Tensor && f.name === :psi_bar, trace.factors)
            has_psi = any(f -> f isa Tensor && f.name === :psi, trace.factors)
            @test has_mass
            @test has_psi_bar
            @test has_psi

            # No free indices (scalar)
            @test isempty(free_indices(trace))

            # Grassmann-even (mass * ψ̄ψ)
            @test grassmann_parity(trace) == 0
        end
    end
end
