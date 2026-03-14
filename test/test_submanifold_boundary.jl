@testset "Submanifold & Boundary" begin

    # ── GHY boundary term ──
    @testset "GHY boundary term" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            define_hypersurface!(reg, :Sigma; ambient=:M4, metric=:g,
                                 normal_name=:n, extrinsic_name=:K, signature=-1)

            S_ghy = ghy_boundary_term(reg, :Sigma)

            # GHY = 2 K = 2 g^{ab} K_{ab}
            # Outermost should be a TProduct with scalar coefficient 2
            @test S_ghy isa TProduct
            @test S_ghy.scalar == 2 // 1

            # The inner product g^{ab} K_{ab} must contain :g and :K tensors
            tensor_names = Symbol[]
            for f in S_ghy.factors
                if f isa Tensor
                    push!(tensor_names, f.name)
                elseif f isa TProduct
                    for ff in f.factors
                        if ff isa Tensor
                            push!(tensor_names, ff.name)
                        end
                    end
                end
            end
            @test :g in tensor_names
            @test :K in tensor_names
        end
    end

    @testset "GHY boundary term uses registered names" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=G
            define_curvature_tensors!(reg, :M4, :G)
            define_hypersurface!(reg, :bdy; ambient=:M4, metric=:G,
                                 normal_name=:nu, extrinsic_name=:Kext, signature=1)

            S = ghy_boundary_term(reg, :bdy)
            @test S isa TProduct
            @test S.scalar == 2 // 1

            # Should use the custom extrinsic curvature name :Kext
            tensor_names = Symbol[]
            for f in S.factors
                if f isa Tensor
                    push!(tensor_names, f.name)
                elseif f isa TProduct
                    for ff in f.factors
                        if ff isa Tensor
                            push!(tensor_names, ff.name)
                        end
                    end
                end
            end
            @test :Kext in tensor_names
            @test :G in tensor_names
        end
    end

    @testset "GHY rejects codimension > 1" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_submanifold!(reg, :S2; ambient=:M4, metric=:g, codimension=2)

            @test_throws ErrorException ghy_boundary_term(reg, :S2)
        end
    end

    @testset "GHY rejects unknown hypersurface" begin
        reg = TensorRegistry()
        @test_throws ErrorException ghy_boundary_term(reg, :nonexistent)
    end

    # ── Gauss equation ──
    @testset "Gauss equation structure" begin
        ge = gauss_equation(down(:a), down(:b), down(:c), down(:d);
                            Riem=:Riem, K=:K, signature=-1)
        @test ge isa TSum
        # 3 terms: Riem_{abcd} + sigma K_{ac}K_{bd} - sigma K_{ad}K_{bc}
        @test length(ge.terms) == 3

        # First term is the ambient Riemann tensor
        riem_term = ge.terms[1]
        @test riem_term isa Tensor
        @test riem_term.name == :Riem

        # Other two are K*K products
        for t in ge.terms[2:3]
            @test t isa TProduct
            K_factors = [f for f in t.factors if f isa Tensor && f.name == :K]
            @test length(K_factors) == 2
        end
    end

    @testset "Gauss equation spacelike normal" begin
        ge = gauss_equation(down(:a), down(:b), down(:c), down(:d);
                            Riem=:R, K=:Kext, signature=1)
        @test ge isa TSum
        @test length(ge.terms) == 3

        # Check that the Riemann tensor has the specified name
        @test ge.terms[1] isa Tensor
        @test ge.terms[1].name == :R

        # K*K terms should use the specified extrinsic curvature name
        for t in ge.terms[2:3]
            K_factors = [f for f in t.factors if f isa Tensor && f.name == :Kext]
            @test length(K_factors) == 2
        end
    end

    # ── Codazzi equation ──
    @testset "Codazzi equation structure" begin
        ce = codazzi_equation(down(:a), down(:b), down(:c);
                              Riem=:Riem, K=:K, normal=:n, signature=-1)
        @test ce isa TProduct

        # Should be sigma * Riem_{dabc} * n^d  (2 tensor factors)
        tensor_factors = [f for f in ce.factors if f isa Tensor]
        @test length(tensor_factors) == 2

        riem_factor = [f for f in tensor_factors if f.name == :Riem]
        @test length(riem_factor) == 1
        @test length(riem_factor[1].indices) == 4

        normal_factor = [f for f in tensor_factors if f.name == :n]
        @test length(normal_factor) == 1
        @test normal_factor[1].indices[1].position == Up
    end

    @testset "Codazzi equation avoids index clashes" begin
        # The dummy index for normal contraction must not clash with a, b, c
        ce = codazzi_equation(down(:a), down(:b), down(:c);
                              Riem=:Riem, K=:K, normal=:n, signature=-1)
        riem_factor = nothing
        normal_factor = nothing
        for f in ce.factors
            if f isa Tensor && f.name == :Riem
                riem_factor = f
            elseif f isa Tensor && f.name == :n
                normal_factor = f
            end
        end
        # The normal index and first Riemann index should share a name (contracted)
        @test normal_factor.indices[1].name == riem_factor.indices[1].name
        # That name must not be any of the free indices
        dummy_name = normal_factor.indices[1].name
        @test dummy_name != :a
        @test dummy_name != :b
        @test dummy_name != :c
    end

    # ── ibp_with_boundary ──
    @testset "ibp_with_boundary returns tuple" begin
        phi = Tensor(:phi, TIndex[])
        psi = Tensor(:psi, TIndex[])
        d_phi = TDeriv(down(:a), phi)
        expr = TProduct(1 // 1, TensorExpr[d_phi, psi])

        result = ibp_with_boundary(expr, :phi)
        @test result isa Tuple
        @test length(result) == 2
        bulk, boundary = result
        @test bulk isa TensorExpr
        @test boundary isa TensorExpr
    end

    @testset "ibp_with_boundary bulk matches ibp_product" begin
        phi = Tensor(:phi, TIndex[])
        psi = Tensor(:psi, TIndex[])
        d_phi = TDeriv(down(:a), phi)
        expr = TProduct(1 // 1, TensorExpr[d_phi, psi])

        bulk, _ = ibp_with_boundary(expr, :phi)
        ibp_result = ibp_product(expr, :phi)

        # The bulk should have the same scalar sign as ibp_product
        @test bulk isa TProduct
        @test bulk.scalar == -1 // 1
        @test ibp_result isa TProduct
        @test ibp_result.scalar == -1 // 1
    end

    @testset "ibp_with_boundary boundary is nonzero" begin
        phi = Tensor(:phi, TIndex[])
        psi = Tensor(:psi, TIndex[])
        d_phi = TDeriv(down(:a), phi)
        expr = TProduct(1 // 1, TensorExpr[d_phi, psi])

        _, boundary = ibp_with_boundary(expr, :phi)
        # boundary = expr - bulk, should not be identically zero for a derivative expression
        @test boundary != TScalar(0 // 1)
    end

    @testset "ibp_with_boundary no derivative gives zero boundary" begin
        phi = Tensor(:phi, TIndex[])
        psi = Tensor(:psi, TIndex[])
        expr = TProduct(1 // 1, TensorExpr[phi, psi])

        bulk, boundary = ibp_with_boundary(expr, :phi)
        @test bulk == expr
        @test boundary == TScalar(0 // 1)
    end

    @testset "ibp_with_boundary on plain tensor" begin
        T = Tensor(:T, [down(:a), down(:b)])
        bulk, boundary = ibp_with_boundary(T, :T)
        @test bulk == T
        @test boundary == TScalar(0 // 1)
    end

    @testset "ibp_with_boundary on TScalar" begin
        s = TScalar(5 // 1)
        bulk, boundary = ibp_with_boundary(s, :phi)
        @test bulk == s
        @test boundary == TScalar(0 // 1)
    end

    @testset "ibp_with_boundary on TDeriv (standalone)" begin
        phi = Tensor(:phi, TIndex[])
        d_phi = TDeriv(down(:a), phi)
        bulk, boundary = ibp_with_boundary(d_phi, :phi)
        @test bulk == d_phi
        @test boundary == TScalar(0 // 1)
    end

    @testset "ibp_with_boundary on TSum" begin
        phi = Tensor(:phi, TIndex[])
        psi = Tensor(:psi, TIndex[])
        chi = Tensor(:chi, TIndex[])
        d_phi = TDeriv(down(:a), phi)
        term1 = TProduct(1 // 1, TensorExpr[d_phi, psi])
        term2 = TProduct(1 // 1, TensorExpr[d_phi, chi])
        expr = TSum(TensorExpr[term1, term2])

        bulk, boundary = ibp_with_boundary(expr, :phi)
        @test bulk isa TSum
        @test boundary isa TSum
    end

    # ── define_submanifold! (codimension > 1) ──
    @testset "define_submanifold! codimension 2" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            sp = define_submanifold!(reg, :S2; ambient=:M4, metric=:g,
                                     codimension=2)

            @test sp isa SubmanifoldProperties
            @test sp.codimension == 2
            @test sp.dim_ambient == 4
            @test sp.dim_surface == 2
            @test length(sp.normal_names) == 2
            @test length(sp.extrinsic_names) == 2
            @test length(sp.signatures) == 2

            # Normals and extrinsic curvatures should be registered
            for nn in sp.normal_names
                @test has_tensor(reg, nn)
            end
            for kn in sp.extrinsic_names
                @test has_tensor(reg, kn)
            end

            # Induced metric and projector should be registered
            @test has_tensor(reg, :gamma)  || has_tensor(reg, :γ)
            @test has_tensor(reg, :P_hs)
        end
    end

    @testset "define_submanifold! custom names" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M5 dim=5 metric=G
            sp = define_submanifold!(reg, :S3; ambient=:M5, metric=:G,
                                     codimension=2,
                                     normal_names=[:m1, :m2],
                                     extrinsic_names=[:L1, :L2],
                                     induced_name=:h,
                                     signatures=[1, -1])

            @test sp.normal_names == [:m1, :m2]
            @test sp.extrinsic_names == [:L1, :L2]
            @test sp.signatures == [1, -1]
            @test has_tensor(reg, :m1)
            @test has_tensor(reg, :m2)
            @test has_tensor(reg, :L1)
            @test has_tensor(reg, :L2)
            @test has_tensor(reg, :h)
        end
    end

    @testset "define_submanifold! codimension validation" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            # codimension must be >= 1
            @test_throws AssertionError define_submanifold!(reg, :bad;
                ambient=:M4, metric=:g, codimension=0)
            # codimension must be < dim
            @test_throws AssertionError define_submanifold!(reg, :bad2;
                ambient=:M4, metric=:g, codimension=4)
        end
    end

    # ── Induced metric (multi-normal) ──
    @testset "Induced metric codimension 2" begin
        gamma = induced_metric_expr(down(:a), down(:b), :g,
                                     [:n1, :n2], [1, -1])
        @test gamma isa TSum
        # g_{ab} - sigma1 n1_a n1_b - sigma2 n2_a n2_b = 3 terms
        @test length(gamma.terms) == 3
    end

    # ── Projector (multi-normal) ──
    @testset "Projector codimension 2" begin
        P = projector_expr(up(:a), down(:b), [:n1, :n2], [1, -1])
        @test P isa TSum
        # delta^a_b - sigma1 n1^a n1_b - sigma2 n2^a n2_b = 3 terms
        @test length(P.terms) == 3
    end

    # ── Backward compatibility ──
    @testset "HypersurfaceProperties alias" begin
        @test HypersurfaceProperties === SubmanifoldProperties
    end

    @testset "define_hypersurface! backward compat" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            hs = define_hypersurface!(reg, :Sigma; ambient=:M4, metric=:g,
                                       normal_name=:n, extrinsic_name=:K,
                                       induced_name=:gamma, signature=-1)

            @test hs isa SubmanifoldProperties
            @test hs isa HypersurfaceProperties
            @test hs.codimension == 1
            @test hs.signature == -1
            @test hs.dim_surface == 3
            @test has_tensor(reg, :n)
            @test has_tensor(reg, :K)
            @test has_tensor(reg, :gamma)
        end
    end

    @testset "Extrinsic curvature expr" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_hypersurface!(reg, :Sigma; ambient=:M4, metric=:g,
                                  normal_name=:n, signature=-1)

            K_ab = extrinsic_curvature_expr(down(:a), down(:b), :n, :g;
                                             registry=reg)
            # K_{ab} = -∂_a n_b, so it should be a negated TDeriv
            @test K_ab isa TProduct
            @test K_ab.scalar == -1 // 1
        end
    end

end
