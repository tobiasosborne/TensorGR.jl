@testset "Dirac Field" begin
    using TensorGR: define_fermion!, dirac_bar, is_fermion, get_conjugate_name,
                    scalar_bilinear, vector_bilinear, axial_bilinear,
                    pseudo_bilinear, dirac_kinetic_expr, dirac_equation_expr,
                    register_grassmann_field!, is_grassmann, grassmann_parity,
                    GammaMatrix, Gamma5,
                    TensorRegistry, TensorProperties, SymmetrySpec,
                    Tensor, TProduct, TSum, TDeriv, TScalar,
                    up, down, with_registry, register_tensor!,
                    current_registry, has_tensor, get_tensor,
                    free_indices

    function _make_dirac_registry()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_fermion!(reg, :psi)
        end
        return reg
    end

    # ---- Registration --------------------------------------------------------

    @testset "define_fermion! registration" begin
        reg = _make_dirac_registry()
        with_registry(reg) do
            @test has_tensor(reg, :psi)
            @test has_tensor(reg, :psi_bar)
            @test has_tensor(reg, :m)

            # psi is Grassmann-odd
            @test is_grassmann(reg, :psi)
            @test is_grassmann(reg, :psi_bar)

            # psi is a fermion
            @test is_fermion(reg, :psi)
            @test is_fermion(reg, :psi_bar)

            # Rank (0,0) — suppressed spinor indices
            props = get_tensor(reg, :psi)
            @test props.rank == (0, 0)

            # Conjugate linkage
            @test get(props.options, :conjugate_field, nothing) === :psi_bar
            bar_props = get_tensor(reg, :psi_bar)
            @test get(bar_props.options, :conjugate_field, nothing) === :psi
            @test get(bar_props.options, :is_conjugate, false)
        end
    end

    @testset "idempotent registration" begin
        reg = _make_dirac_registry()
        with_registry(reg) do
            define_fermion!(reg, :psi)  # no error
            @test has_tensor(reg, :psi)
        end
    end

    @testset "custom mass name" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_fermion!(reg, :chi; mass=:M_chi)
            @test has_tensor(reg, :M_chi)
            props = get_tensor(reg, :chi)
            @test get(props.options, :mass, nothing) === :M_chi
        end
    end

    @testset "error: manifold not registered" begin
        reg = TensorRegistry()
        @test_throws ErrorException define_fermion!(reg, :psi)
    end

    # ---- dirac_bar -----------------------------------------------------------

    @testset "dirac_bar returns conjugate" begin
        reg = _make_dirac_registry()
        with_registry(reg) do
            psi = Tensor(:psi, TIndex[])
            bar = dirac_bar(psi)
            @test bar isa Tensor
            @test bar.name === :psi_bar
            @test isempty(bar.indices)
        end
    end

    @testset "dirac_bar error on unregistered field" begin
        reg = _make_dirac_registry()
        with_registry(reg) do
            @test_throws ErrorException dirac_bar(Tensor(:unknown, TIndex[]))
        end
    end

    # ---- Bilinears -----------------------------------------------------------

    @testset "scalar_bilinear: ψ̄ψ" begin
        reg = _make_dirac_registry()
        with_registry(reg) do
            sb = scalar_bilinear(:psi)
            @test sb isa TProduct
            @test length(sb.factors) == 2
            @test sb.factors[1].name === :psi_bar
            @test sb.factors[2].name === :psi

            # No free indices (scalar)
            @test isempty(free_indices(sb))

            # Grassmann parity: even (two odd factors)
            @test grassmann_parity(sb) == 0
        end
    end

    @testset "vector_bilinear: ψ̄γ^aψ" begin
        reg = _make_dirac_registry()
        with_registry(reg) do
            vb = vector_bilinear(:psi, up(:a))
            @test vb isa TProduct
            @test length(vb.factors) == 3
            @test vb.factors[1].name === :psi_bar
            @test vb.factors[2] isa GammaMatrix
            @test vb.factors[3].name === :psi

            # One free Tangent index
            fi = free_indices(vb)
            @test length(fi) == 1
            @test fi[1].name === :a
            @test fi[1].position === Up

            # Grassmann-even
            @test grassmann_parity(vb) == 0
        end
    end

    @testset "axial_bilinear: ψ̄γ^aγ⁵ψ" begin
        reg = _make_dirac_registry()
        with_registry(reg) do
            ab = axial_bilinear(:psi, up(:a))
            @test ab isa TProduct
            @test length(ab.factors) == 4
            @test any(f -> f isa Gamma5, ab.factors)

            fi = free_indices(ab)
            @test length(fi) == 1
            @test fi[1].name === :a
        end
    end

    @testset "pseudo_bilinear: ψ̄γ⁵ψ" begin
        reg = _make_dirac_registry()
        with_registry(reg) do
            pb = pseudo_bilinear(:psi)
            @test pb isa TProduct
            @test length(pb.factors) == 3
            @test any(f -> f isa Gamma5, pb.factors)
            @test isempty(free_indices(pb))
        end
    end

    # ---- Kinetic term --------------------------------------------------------

    @testset "dirac_kinetic_expr structure" begin
        reg = _make_dirac_registry()
        with_registry(reg) do
            L = dirac_kinetic_expr(:psi)
            @test L isa TSum
            @test length(L.terms) == 2

            # No free indices (Lagrangian is a scalar)
            fi = free_indices(L)
            @test isempty(fi)

            # First term contains a TDeriv (kinetic)
            t1 = L.terms[1]
            has_deriv = any(f -> f isa TDeriv, t1.factors)
            @test has_deriv

            # First term contains i (imaginary unit)
            has_im = any(f -> f isa TScalar && f.val === :im, t1.factors)
            @test has_im

            # First term contains GammaMatrix
            has_gamma = any(f -> f isa GammaMatrix, t1.factors)
            @test has_gamma

            # Second term is the mass term (no derivative)
            t2 = L.terms[2]
            @test t2.scalar < 0  # negative mass term
            has_mass = any(f -> f isa TScalar && f.val === :m, t2.factors)
            @test has_mass
        end
    end

    # ---- Dirac equation ------------------------------------------------------

    @testset "dirac_equation_expr structure" begin
        reg = _make_dirac_registry()
        with_registry(reg) do
            eq = dirac_equation_expr(:psi)
            @test eq isa TSum
            @test length(eq.terms) == 2

            # Should have no free indices (spinor equation)
            fi = free_indices(eq)
            @test isempty(fi)

            # First term: iγ^a∂_aψ
            t1 = eq.terms[1]
            has_gamma = any(f -> f isa GammaMatrix, t1.factors)
            has_deriv = any(f -> f isa TDeriv, t1.factors)
            @test has_gamma && has_deriv

            # Second term: -mψ
            t2 = eq.terms[2]
            @test t2.scalar < 0
        end
    end

    # ---- Majorana field ------------------------------------------------------

    @testset "Majorana field registration" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_fermion!(reg, :lambda; type=:majorana)
            @test has_tensor(reg, :lambda)
            @test has_tensor(reg, :lambda_bar)
            @test is_grassmann(reg, :lambda)

            props = get_tensor(reg, :lambda)
            @test get(props.options, :fermion_type, nothing) === :majorana
        end
    end

    # ---- Weyl field ----------------------------------------------------------

    @testset "Weyl field registration (no auto-conjugate)" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_fermion!(reg, :psi_L; type=:weyl_left)
            @test has_tensor(reg, :psi_L)
            @test !has_tensor(reg, :psi_L_bar)  # no auto-conjugate for Weyl
            @test is_grassmann(reg, :psi_L)
        end
    end
end
