@testset "Chain continuations" begin

    # ── Poisson brackets (TGR-vdm.3) ──
    @testset "Poisson bracket" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d,:e,:f,:g,:h,:i,:j]))
        with_registry(reg) do
            adm = define_adm!(reg)
            cp = adm_canonical_pair(adm)
            @test cp isa CanonicalPair
            @test cp.config == :gamma_adm
            @test cp.momentum == :pi_gamma_adm

            fb = fundamental_bracket(cp; registry=reg)
            @test fb isa TensorExpr

            s = sprint(show, cp)
            @test occursin("CanonicalPair", s)
        end
    end

    # ── Hassan-Rosen potential (TGR-wq0.3) ──
    @testset "Hassan-Rosen potential" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d,:e,:f,:g,:h,:i,:j]))
        with_registry(reg) do
            bs = define_bimetric!(reg, :g_phys, :f_phys)
            params = HassanRosenParams(; m_sq=:m2, beta0=1, beta1=:b1, beta2=:b2)

            V = hassan_rosen_potential(bs, params; registry=reg)
            @test V isa TensorExpr

            # Elementary symmetric polynomials
            S_name = Symbol(:S_g_phys_f_phys)
            @test elementary_symmetric(0, S_name) == TScalar(1 // 1)
            e1 = elementary_symmetric(1, S_name)
            @test e1 isa Tensor  # Tr(S) = S^a_a
            e2 = elementary_symmetric(2, S_name)
            @test e2 isa TensorExpr
            e3 = elementary_symmetric(3, S_name)
            @test e3 isa TensorExpr
            e4 = elementary_symmetric(4, S_name)
            @test e4 isa TScalar  # det(S) symbolic

            s = sprint(show, params)
            @test occursin("HR", s)
        end
    end

    # ── Torsion decomposition (TGR-swh.3) ──
    @testset "Torsion decomposition" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d,:e,:f,:g,:h,:i,:j]))
        define_curvature_tensors!(reg, :M4, :g)
        with_registry(reg) do
            ac = define_affine_connection!(reg, :Gamma_gen)
            td = decompose_torsion!(reg, ac)
            @test td isa TorsionDecomposition

            # All components registered
            @test has_tensor(reg, td.vector)
            @test has_tensor(reg, td.axial)
            @test has_tensor(reg, td.tensor)

            # Correct ranks
            @test get_tensor(reg, td.vector).rank == (0, 1)
            @test get_tensor(reg, td.axial).rank == (1, 0)
            @test get_tensor(reg, td.tensor).rank == (1, 2)

            # Tensor part is antisymmetric in lower indices
            @test !isempty(get_tensor(reg, td.tensor).symmetries)

            # Torsion vector expression
            tv = torsion_vector_expr(ac; registry=reg)
            @test tv isa Tensor

            s = sprint(show, td)
            @test occursin("TorsionDecomp", s)
        end
    end

    # ── Gamma matrix traces (TGR-dai.3) ──
    @testset "Gamma matrix traces" begin
        # Tr(empty) = 4
        @test gamma_chain_trace(GammaMatrix[]) == TScalar(4 // 1)

        # Tr(γ^a) = 0
        @test gamma_chain_trace([GammaMatrix(up(:a))]) == TScalar(0 // 1)

        # Tr(γ^a γ^b) = 4 g^{ab}
        tr2 = gamma_chain_trace([GammaMatrix(up(:a)), GammaMatrix(up(:b))])
        @test tr2 isa TProduct
        @test tr2.scalar == 4 // 1

        # Tr(γ^a γ^b γ^c) = 0 (odd)
        @test gamma_chain_trace([GammaMatrix(up(:a)), GammaMatrix(up(:b)),
                                 GammaMatrix(up(:c))]) == TScalar(0 // 1)

        # Tr(γ^a γ^b γ^c γ^d) = 4(g^{ab}g^{cd} - g^{ac}g^{bd} + g^{ad}g^{bc})
        tr4 = gamma_chain_trace([GammaMatrix(up(:a)), GammaMatrix(up(:b)),
                                 GammaMatrix(up(:c)), GammaMatrix(up(:d))])
        @test tr4 isa TProduct
        @test tr4.scalar == 4 // 1
    end

    @testset "trace_identity_2" begin
        t = trace_identity_2(up(:a), up(:b))
        @test t isa TProduct
        @test t.scalar == 4 // 1
    end

    @testset "trace_identity_4" begin
        t = trace_identity_4(up(:a), up(:b), up(:c), up(:d))
        @test t isa TProduct
        @test t.scalar == 4 // 1
    end

    @testset "slash notation" begin
        v = Tensor(:k, [down(:a)])
        s = slash(v)
        @test s isa TProduct
        # Should contain a GammaMatrix
        has_gamma = any(f -> f isa GammaMatrix, s.factors)
        @test has_gamma
    end

    @testset "slash: rank check" begin
        T = Tensor(:T, [down(:a), down(:b)])
        @test_throws ErrorException slash(T)
    end

    @testset "Tr(6 gammas) recursive" begin
        gammas6 = [GammaMatrix(up(Symbol(:a, i))) for i in 1:6]
        tr6 = gamma_chain_trace(gammas6)
        @test tr6 isa TensorExpr
        # Should be a sum of metric products
    end

end
