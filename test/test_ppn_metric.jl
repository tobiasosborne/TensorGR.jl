@testset "PPN Metric Ansatz" begin

    # ── PPNParameters construction ──
    @testset "PPNParameters struct" begin
        p = PPNParameters(1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
        @test p.gamma == 1
        @test p.beta == 1
        @test p.xi == 0
        @test p.alpha1 == 0
        @test p.alpha2 == 0
        @test p.alpha3 == 0
        @test p.zeta1 == 0
        @test p.zeta2 == 0
        @test p.zeta3 == 0
        @test p.zeta4 == 0
    end

    @testset "ppn_gr() convenience" begin
        p = ppn_gr()
        @test p.gamma == 1
        @test p.beta == 1
        @test p.xi == 0
        @test p.alpha1 == 0
        @test is_gr(p)
    end

    @testset "is_gr detection" begin
        @test is_gr(ppn_gr())
        @test !is_gr(PPNParameters(1, 2, 0, 0, 0, 0, 0, 0, 0, 0))
        @test !is_gr(PPNParameters(2, 1, 0, 0, 0, 0, 0, 0, 0, 0))
        @test !is_gr(PPNParameters(1, 1, 1, 0, 0, 0, 0, 0, 0, 0))
    end

    # ── PPN potentials registration ──
    @testset "define_ppn_potentials!" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_ppn_potentials!(reg; manifold=:M4)

            # Scalars
            @test has_tensor(reg, :U)
            @test has_tensor(reg, :Phi_W)
            @test has_tensor(reg, :Phi_1)
            @test has_tensor(reg, :Phi_2)
            @test has_tensor(reg, :Phi_3)
            @test has_tensor(reg, :Phi_4)
            @test has_tensor(reg, :A_ppn)

            # Vectors
            @test has_tensor(reg, :V_ppn)
            @test has_tensor(reg, :W_ppn)

            # Superpotential (rank-2)
            @test has_tensor(reg, :U_ppn)

            # Check ranks
            @test get_tensor(reg, :U).rank == (0, 0)
            @test get_tensor(reg, :V_ppn).rank == (0, 1)
            @test get_tensor(reg, :W_ppn).rank == (0, 1)
            @test get_tensor(reg, :U_ppn).rank == (0, 2)

            # Check options
            @test get_tensor(reg, :U).options[:is_ppn_potential] == true
        end
    end

    # ── Full PPN metric with GR parameters ──
    @testset "GR metric (order=2)" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_ppn_potentials!(reg; manifold=:M4)

            params = ppn_gr()
            metric = ppn_metric_ansatz(params, reg; order=2)

            # Check all three components exist
            @test haskey(metric, (:time, :time))
            @test haskey(metric, (:time, :space))
            @test haskey(metric, (:space, :space))

            # ── g_{00} for GR ──
            # g_{00} = -1 + 2U - 2*1*U^2 + 0 + (2+2+0+0-0)*Phi_1 + 2*(3-2+1+0+0)*Phi_2
            #          + 2*(1+0)*Phi_3 + 2*(3+0-0)*Phi_4 - (0-0)*A_ppn
            # = -1 + 2U - 2U^2 + 4*Phi_1 + 4*Phi_2 + 2*Phi_3 + 6*Phi_4
            g00 = metric[(:time, :time)]
            @test g00 isa TSum

            # Count terms: -1, 2U, -2U^2, 4Phi_1, 4Phi_2, 2Phi_3, 6Phi_4
            # (A_ppn coefficient is 0 so it's dropped)
            @test length(g00.terms) == 7

            # ── g_{0i} for GR ──
            # V_i coeff: -1/2*(4+3+0-0+0-0) = -1/2*7 = -7/2
            # W_i coeff: -1/2*(1+0-0+0) = -1/2
            g0i = metric[(:time, :space)]
            @test g0i isa TSum
            @test length(g0i.terms) == 2

            # ── g_{ij} for GR ──
            # g_{ij} = (1 + 2*1*U)*delta_{ij} = delta_{ij} + 2*U*delta_{ij}
            gij = metric[(:space, :space)]
            @test gij isa TSum
            @test length(gij.terms) == 2
        end
    end

    # ── g_{00} correctness ──
    @testset "g_{00} U and U^2 terms" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_ppn_potentials!(reg; manifold=:M4)

            params = ppn_gr()
            metric = ppn_metric_ansatz(params, reg; order=2)
            g00 = metric[(:time, :time)]

            # First term: -1
            @test g00.terms[1] isa TScalar
            @test g00.terms[1].val == -1 // 1

            # Second term: 2U
            t2 = g00.terms[2]
            @test t2 isa TProduct
            @test t2.scalar == 2 // 1
            @test length(t2.factors) == 1
            @test t2.factors[1] isa Tensor
            @test t2.factors[1].name == :U

            # Third term: -2*beta*U^2 = -2*U*U (for beta=1)
            t3 = g00.terms[3]
            @test t3 isa TProduct
            @test t3.scalar == -2 // 1
            @test length(t3.factors) == 2
            @test all(f -> f isa Tensor && f.name == :U, t3.factors)
        end
    end

    # ── g_{0i} V_i and W_i terms ──
    @testset "g_{0i} V_i and W_i terms" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_ppn_potentials!(reg; manifold=:M4)

            params = ppn_gr()
            metric = ppn_metric_ansatz(params, reg; order=2)
            g0i = metric[(:time, :space)]

            @test g0i isa TSum
            @test length(g0i.terms) == 2

            # Find V_ppn and W_ppn terms
            v_term = nothing
            w_term = nothing
            for t in g0i.terms
                if t isa TProduct
                    for f in t.factors
                        if f isa Tensor && f.name == :V_ppn
                            v_term = t
                        elseif f isa Tensor && f.name == :W_ppn
                            w_term = t
                        end
                    end
                end
            end

            # V_i coefficient: -1/2*(4*1 + 3 + 0 - 0 + 0 - 0) = -7/2
            @test v_term !== nothing
            @test v_term.scalar == -7 // 2

            # W_i coefficient: -1/2*(1 + 0 - 0 + 0) = -1/2
            @test w_term !== nothing
            @test w_term.scalar == -1 // 2

            # Both carry spatial index :i (Down)
            v_idx = v_term.factors[end].indices[1]
            @test v_idx.position == Down
            @test v_idx.name == :i

            w_idx = w_term.factors[end].indices[1]
            @test w_idx.position == Down
            @test w_idx.name == :i
        end
    end

    # ── g_{ij} = (1 + 2*gamma*U)*delta_{ij} ──
    @testset "g_{ij} spatial metric" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_ppn_potentials!(reg; manifold=:M4)

            params = ppn_gr()
            metric = ppn_metric_ansatz(params, reg; order=2)
            gij = metric[(:space, :space)]

            @test gij isa TSum
            @test length(gij.terms) == 2

            # First term: delta_{ij}
            @test gij.terms[1] isa Tensor
            @test gij.terms[1].name == :delta
            @test length(gij.terms[1].indices) == 2
            @test gij.terms[1].indices[1].name == :i
            @test gij.terms[1].indices[2].name == :j
            @test gij.terms[1].indices[1].position == Down
            @test gij.terms[1].indices[2].position == Down

            # Second term: 2*gamma*U*delta_{ij} = 2*U*delta_{ij} for gamma=1
            t2 = gij.terms[2]
            @test t2 isa TProduct
            @test t2.scalar == 2 // 1
            # Should have U and delta as factors
            factor_names = Set(f isa Tensor ? f.name : :_scalar for f in t2.factors)
            @test :U in factor_names
            @test :delta in factor_names
        end
    end

    # ── Order=1 (1PN) ──
    @testset "order=1 metric" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_ppn_potentials!(reg; manifold=:M4)

            params = ppn_gr()
            metric = ppn_metric_ansatz(params, reg; order=1)

            # g_{00} at 1PN: -1 + 2U (just 2 terms in the sum)
            g00 = metric[(:time, :time)]
            @test g00 isa TSum
            @test length(g00.terms) == 2

            # g_{0i} at 1PN: 0
            g0i = metric[(:time, :space)]
            @test g0i isa TScalar
            @test g0i.val == 0 // 1

            # g_{ij} at 1PN: same as 2PN (gamma*U term is already 1PN)
            gij = metric[(:space, :space)]
            @test gij isa TSum
            @test length(gij.terms) == 2
        end
    end

    # ── Non-GR parameters (Brans-Dicke) ──
    @testset "non-GR parameters" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_ppn_potentials!(reg; manifold=:M4)

            # Brans-Dicke: gamma = (1+omega)/(2+omega), beta = 1
            # For omega=3: gamma = 4/5
            bd_gamma = 4 // 5
            params = PPNParameters(bd_gamma, 1, 0, 0, 0, 0, 0, 0, 0, 0)
            @test !is_gr(params)

            metric = ppn_metric_ansatz(params, reg; order=2)
            g00 = metric[(:time, :time)]
            @test g00 isa TSum

            # g_{ij} should have 2*gamma = 8/5 coefficient on U*delta
            gij = metric[(:space, :space)]
            @test gij isa TSum
            t_U_delta = gij.terms[2]
            @test t_U_delta isa TProduct
            @test t_U_delta.scalar == 8 // 5   # 2 * 4/5

            # g_{0i}: V_i coeff = -1/2*(4*4/5 + 3 + 0 - 0 + 0 - 0) = -1/2*(31/5) = -31/10
            g0i = metric[(:time, :space)]
            v_term = nothing
            for t in g0i.terms
                if t isa TProduct
                    for f in t.factors
                        if f isa Tensor && f.name == :V_ppn
                            v_term = t
                        end
                    end
                end
            end
            @test v_term !== nothing
            @test v_term.scalar == -31 // 10
        end
    end

    # ── Symbolic parameters ──
    @testset "symbolic PPN parameters" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_ppn_potentials!(reg; manifold=:M4)

            # Use symbolic gamma (a Symbol, not a number)
            params = PPNParameters(:gamma_ppn, 1, 0, 0, 0, 0, 0, 0, 0, 0)
            @test !is_gr(params)

            metric = ppn_metric_ansatz(params, reg; order=2)

            # g_{ij} should contain symbolic gamma
            gij = metric[(:space, :space)]
            @test gij isa TSum
            @test length(gij.terms) == 2

            # The U*delta term should contain TScalar(:gamma_ppn)
            t2 = gij.terms[2]
            @test t2 isa TProduct
            has_gamma = any(f -> f isa TScalar && f.val == :gamma_ppn, t2.factors)
            @test has_gamma
        end
    end

    # ── Error handling ──
    @testset "error handling" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g

            # Calling ppn_metric_ansatz without defining potentials
            @test_throws ErrorException ppn_metric_ansatz(ppn_gr(), reg)

            # Invalid order
            define_ppn_potentials!(reg; manifold=:M4)
            @test_throws ErrorException ppn_metric_ansatz(ppn_gr(), reg; order=3)

            # Missing manifold
            reg2 = TensorRegistry()
            @test_throws ErrorException define_ppn_potentials!(reg2; manifold=:M4)
        end
    end

    # ── PPN parameters with all zeros except gamma, beta ──
    @testset "vanishing PPN parameters drop terms" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_ppn_potentials!(reg; manifold=:M4)

            # All PPN params zero except gamma=1, beta=1 (GR)
            # should give well-formed metric
            params = ppn_gr()
            metric = ppn_metric_ansatz(params, reg; order=2)

            # In g_{00}, A_ppn should NOT appear (coeff = -(0 - 0) = 0)
            g00 = metric[(:time, :time)]
            has_a = false
            for t in g00.terms
                if t isa TProduct
                    for f in t.factors
                        if f isa Tensor && f.name == :A_ppn
                            has_a = true
                        end
                    end
                end
            end
            @test !has_a
        end
    end

    # ── Show method ──
    @testset "PPNParameters show" begin
        p = ppn_gr()
        s = sprint(show, p)
        @test occursin("gamma=1", s)
        @test occursin("beta=1", s)
        @test occursin("PPNParameters", s)
    end

    # ── Conservation and frame tests (TGR-bgl.3) ──
    @testset "is_fully_conservative" begin
        @test is_fully_conservative(ppn_gr())
        @test is_fully_conservative(PPNParameters(2, 1, 0, 0, 0, 0, 0, 0, 0, 0))
        @test !is_fully_conservative(PPNParameters(1, 1, 0, 0, 0, 0, 1, 0, 0, 0))
        @test !is_fully_conservative(PPNParameters(1, 1, 0, 0, 0, 0, 0, 1, 0, 0))
        @test !is_fully_conservative(PPNParameters(1, 1, 0, 0, 0, 0, 0, 0, 1, 0))
        @test !is_fully_conservative(PPNParameters(1, 1, 0, 0, 0, 0, 0, 0, 0, 1))
    end

    @testset "is_preferred_frame_free" begin
        @test is_preferred_frame_free(ppn_gr())
        @test is_preferred_frame_free(PPNParameters(1, 2, 0, 0, 0, 0, 0, 0, 0, 0))
        @test !is_preferred_frame_free(PPNParameters(1, 1, 0, 1, 0, 0, 0, 0, 0, 0))
        @test !is_preferred_frame_free(PPNParameters(1, 1, 0, 0, 1, 0, 0, 0, 0, 0))
        @test !is_preferred_frame_free(PPNParameters(1, 1, 0, 0, 0, 1, 0, 0, 0, 0))
    end

    @testset "is_preferred_location_free" begin
        @test is_preferred_location_free(ppn_gr())
        @test !is_preferred_location_free(PPNParameters(1, 1, 1, 0, 0, 0, 0, 0, 0, 0))
    end

    @testset "is_semi_conservative" begin
        @test is_semi_conservative(ppn_gr())
        @test !is_semi_conservative(PPNParameters(1, 1, 0, 1, 0, 0, 0, 0, 0, 0))
        @test !is_semi_conservative(PPNParameters(1, 1, 0, 0, 0, 0, 1, 0, 0, 0))
    end

    # ── Named constructors (TGR-bgl.3) ──
    @testset "named constructors" begin
        p_gr = PPNParameters(:GR)
        @test is_gr(p_gr)

        p_bd = PPNParameters(:BransDicke; omega=1000000)
        @test p_bd.beta == 1
        @test p_bd.gamma ≈ 1.0 atol=1e-5
        @test is_fully_conservative(p_bd)

        p_bd3 = PPNParameters(:BransDicke; omega=3//1)
        @test p_bd3.gamma == 4 // 5
        @test p_bd3.beta == 1
        @test !is_gr(p_bd3)

        @test_throws ErrorException PPNParameters(:BransDicke)

        p_nord = PPNParameters(:Nordtvedt; omega=3//1, beta=2//1)
        @test p_nord.gamma == 4 // 5
        @test p_nord.beta == 2 // 1

        p_rosen = PPNParameters(:Rosen)
        @test p_rosen.gamma == 1
        @test p_rosen.alpha1 == -2
        @test !is_preferred_frame_free(p_rosen)
        @test is_fully_conservative(p_rosen)

        @test_throws ErrorException PPNParameters(:Unknown)
    end

end
