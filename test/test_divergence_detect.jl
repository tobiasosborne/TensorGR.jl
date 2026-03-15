@testset "Divergence Detection (CPS)" begin

    # ── Helper: standard 4D GR registry ──
    function make_div_registry()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :dM4,
            [:a, :b, :c, :d, :e, :f, :m, :n, :p, :q, :r, :s]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:dM4, manifold=:M4, rank=(1, 1),
            options=Dict{Symbol,Any}(:is_delta => true)))
        define_curvature_tensors!(reg, :M4, :g)
        define_covd!(reg, :D; manifold=:M4, metric=:g)
        # Test tensors
        register_tensor!(reg, TensorProperties(name=:V, manifold=:M4, rank=(1, 0)))
        register_tensor!(reg, TensorProperties(name=:T, manifold=:M4, rank=(0, 2),
            symmetries=SymmetrySpec[AntiSymmetric(1, 2)]))
        register_tensor!(reg, TensorProperties(name=:xi, manifold=:M4, rank=(1, 0)))
        register_tensor!(reg, TensorProperties(name=:phi, manifold=:M4, rank=(0, 0)))
        reg
    end

    reg = make_div_registry()

    with_registry(reg) do

        V_up = Tensor(:V, [up(:a)])
        phi = Tensor(:phi, TIndex[])
        xi_up = Tensor(:xi, [up(:a)])

        # ── Test 1: D_a(V^a) is a divergence ──
        @testset "D_a(V^a) is divergence" begin
            expr = TDeriv(down(:a), V_up, :D)
            @test is_divergence(expr, :D; registry=reg) == true
        end

        # ── Test 2: bare tensor is not a divergence ──
        @testset "bare tensor is not divergence" begin
            @test is_divergence(V_up, :D; registry=reg) == false
        end

        # ── Test 3: G^{ab} xi_b is NOT a divergence ──
        @testset "Einstein * vector is not divergence" begin
            G_up = Tensor(:Ein, [up(:a), up(:b)])
            xi_down = Tensor(:xi, [down(:b)])
            expr = G_up * xi_down
            @test is_divergence(expr, :D; registry=reg) == false
        end

        # ── Test 4: D_a(phi) is not a divergence (no Up index in arg) ──
        @testset "D_a(phi) is not divergence" begin
            expr = TDeriv(down(:a), phi, :D)
            @test is_divergence(expr, :D; registry=reg) == false
        end

        # ── Test 5: nabla_nu(nabla^mu xi^nu - nabla^nu xi^mu) is a divergence ──
        # This is the Noether charge term: nabla_nu(Q^{mu nu}) where
        # Q^{mu nu} = nabla^mu xi^nu - nabla^nu xi^mu
        @testset "Noether charge divergence" begin
            # Q^{mu nu} = nabla^mu xi^nu - nabla^nu xi^mu
            xi_nu = Tensor(:xi, [up(:b)])
            xi_mu = Tensor(:xi, [up(:a)])
            term1 = TDeriv(up(:a), xi_nu, :D)   # nabla^mu xi^nu
            term2 = TDeriv(up(:b), xi_mu, :D)   # nabla^nu xi^mu
            Q = term1 - term2  # Q^{mu nu} = nabla^mu xi^nu - nabla^nu xi^mu

            # nabla_nu(Q^{mu nu}) = D_b(Q^{a b})
            expr = TDeriv(down(:b), Q, :D)

            # D_b acts on Q which has up(:b) via the xi terms
            @test is_divergence(expr, :D; registry=reg) == true
        end

        # ── Test 6: extract_divergence returns the vector argument ──
        @testset "extract_divergence returns V" begin
            expr = TDeriv(down(:a), V_up, :D)
            ok, V = extract_divergence(expr, :D; registry=reg)
            @test ok == true
            @test V == V_up
        end

        # ── Test 7: extract_divergence on non-divergence returns nothing ──
        @testset "extract_divergence on non-divergence" begin
            ok, V = extract_divergence(V_up, :D; registry=reg)
            @test ok == false
            @test V === nothing
        end

        # ── Test 8: split_divergence separates a sum ──
        @testset "split_divergence on mixed sum" begin
            # div_term = D_a(V^a)
            div_term = TDeriv(down(:a), V_up, :D)
            # non_div_term = G^{ab} xi_b
            G_up = Tensor(:Ein, [up(:a), up(:b)])
            xi_down = Tensor(:xi, [down(:b)])
            non_div_term = G_up * xi_down

            expr = div_term + non_div_term
            div_part, non_div_part = split_divergence(expr, :D; registry=reg)

            # div_part should be D_a(V^a)
            @test is_divergence(div_part, :D; registry=reg) == true
            # non_div_part should be the Einstein * xi term
            @test is_divergence(non_div_part, :D; registry=reg) == false
            @test !(non_div_part isa TScalar && non_div_part.val == 0 // 1)
        end

        # ── Test 9: scalar * D_a(V^a) is a divergence ──
        @testset "scalar * divergence is divergence" begin
            div_expr = TDeriv(down(:a), V_up, :D)
            expr = TProduct(3 // 2, TensorExpr[phi, div_expr])
            @test is_divergence(expr, :D; registry=reg) == true
        end

        # ── Test 10: partial derivative divergence (covd = :partial) ──
        @testset "partial derivative divergence" begin
            expr = TDeriv(down(:a), V_up, :partial)
            @test is_divergence(expr, :D; registry=reg) == true
        end

        # ── Test 11: split_divergence on pure divergence gives zero remainder ──
        @testset "split pure divergence" begin
            expr = TDeriv(down(:a), V_up, :D)
            div_part, non_div_part = split_divergence(expr, :D; registry=reg)
            @test div_part == expr
            @test non_div_part == TScalar(0 // 1)
        end

        # ── Test 12: split_divergence on pure non-divergence gives zero div part ──
        @testset "split pure non-divergence" begin
            div_part, non_div_part = split_divergence(phi, :D; registry=reg)
            @test div_part == TScalar(0 // 1)
            @test non_div_part == phi
        end

        # ── Test 13: D_b(T^{ab}) with antisymmetric T is a divergence ──
        @testset "divergence of antisymmetric tensor" begin
            T_up = Tensor(:T, [up(:a), up(:b)])
            expr = TDeriv(down(:b), T_up, :D)
            @test is_divergence(expr, :D; registry=reg) == true
            ok, V = extract_divergence(expr, :D; registry=reg)
            @test ok == true
            @test V == T_up
        end

        # ── Test 14: extract_divergence on product gives correct vector ──
        @testset "extract_divergence on product" begin
            div_expr = TDeriv(down(:a), V_up, :D)
            expr = TProduct(1 // 1, TensorExpr[phi, div_expr])
            ok, V = extract_divergence(expr, :D; registry=reg)
            @test ok == true
            # V should incorporate phi and V_up
            @test V isa TProduct || V isa Tensor
        end
    end
end
