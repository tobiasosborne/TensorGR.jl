@testset "Spinor Metric epsilon_{AB}" begin

    # Helper: set up a fresh registry with manifold + spinor bundles + spin metric
    function _spin_metric_registry()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_spinor_bundles!(reg; manifold=:M4)
            define_spin_metric!(reg; manifold=:M4)
        end
        reg
    end

    @testset "define_spin_metric! registers tensors" begin
        reg = _spin_metric_registry()
        @test has_tensor(reg, :eps_spin)
        @test has_tensor(reg, :eps_spin_dot)
        @test has_tensor(reg, :delta_spin)
        @test has_tensor(reg, :delta_spin_dot)
    end

    @testset "eps_spin is antisymmetric metric" begin
        reg = _spin_metric_registry()
        props = get_tensor(reg, :eps_spin)
        @test props.is_metric == true
        @test props.rank == (0, 2)
        @test length(props.symmetries) == 1
        @test props.symmetries[1] isa AntiSymmetric
    end

    @testset "eps_spin_dot is antisymmetric metric" begin
        reg = _spin_metric_registry()
        props = get_tensor(reg, :eps_spin_dot)
        @test props.is_metric == true
        @test props.rank == (0, 2)
        @test length(props.symmetries) == 1
        @test props.symmetries[1] isa AntiSymmetric
    end

    @testset "delta_spin properties" begin
        reg = _spin_metric_registry()
        props = get_tensor(reg, :delta_spin)
        @test props.is_delta == true
        @test props.rank == (1, 1)
        @test get(props.options, :vbundle_dim, nothing) == 2
    end

    @testset "delta_spin_dot properties" begin
        reg = _spin_metric_registry()
        props = get_tensor(reg, :delta_spin_dot)
        @test props.is_delta == true
        @test props.rank == (1, 1)
        @test get(props.options, :vbundle_dim, nothing) == 2
    end

    @testset "metric_cache populated by vbundle" begin
        reg = _spin_metric_registry()
        @test reg.metric_cache[:SL2C] == :eps_spin
        @test reg.metric_cache[:SL2C_dot] == :eps_spin_dot
        @test reg.delta_cache[:SL2C] == :delta_spin
        @test reg.delta_cache[:SL2C_dot] == :delta_spin_dot
    end

    @testset "spin_metric() constructor" begin
        reg = _spin_metric_registry()
        with_registry(reg) do
            eps = spin_metric(:SL2C; registry=reg)
            @test eps isa Tensor
            @test eps.name == :eps_spin
            @test length(eps.indices) == 2
            @test eps.indices[1].position == Down
            @test eps.indices[2].position == Down
            @test eps.indices[1].vbundle == :SL2C
            @test eps.indices[2].vbundle == :SL2C

            eps_dot = spin_metric(:SL2C_dot; registry=reg)
            @test eps_dot.name == :eps_spin_dot
            @test eps_dot.indices[1].vbundle == :SL2C_dot
            @test eps_dot.indices[2].vbundle == :SL2C_dot
        end
    end

    @testset "epsilon_{AB} = -epsilon_{BA} after canonicalize" begin
        reg = _spin_metric_registry()
        with_registry(reg) do
            # eps_{AB} and eps_{BA} should differ by a sign
            eps_AB = Tensor(:eps_spin, [spin_down(:A), spin_down(:B)])
            eps_BA = Tensor(:eps_spin, [spin_down(:B), spin_down(:A)])

            # Both should canonicalize to the same form up to sign
            # eps_{AB} + eps_{BA} = 0
            s = simplify(eps_AB + eps_BA; registry=reg)
            @test s == TScalar(0 // 1)
        end
    end

    @testset "epsilon^{AC} epsilon_{CB} -> delta^A_B" begin
        reg = _spin_metric_registry()
        with_registry(reg) do
            # eps^{AC} eps_{CB}
            eps_up = Tensor(:eps_spin, [spin_up(:A), spin_up(:C)])
            eps_dn = Tensor(:eps_spin, [spin_down(:C), spin_down(:B)])
            prod = eps_up * eps_dn

            result = contract_metrics(prod)
            # Should contract to delta^A_B
            @test result isa Tensor
            @test result.name == :delta_spin
            @test length(result.indices) == 2
            # One Up, one Down
            positions = Set(idx.position for idx in result.indices)
            @test Up in positions
            @test Down in positions
        end
    end

    @testset "Self-trace: delta^A_A = 2 (dim of SL2C)" begin
        reg = _spin_metric_registry()
        with_registry(reg) do
            delta = Tensor(:delta_spin, [spin_up(:A), spin_down(:A)])
            result = contract_metrics(delta)
            @test result == TScalar(2 // 1)
        end
    end

    @testset "Self-trace: eps^{AB} delta_{BA} (trace through contraction) = 2" begin
        reg = _spin_metric_registry()
        with_registry(reg) do
            # epsilon^{AB} epsilon_{AB} = epsilon^{AB} epsilon_{AB}
            # For a 2-dim antisymmetric metric, eps^{AB}eps_{AB} = dim = 2
            eps_up = Tensor(:eps_spin, [spin_up(:A), spin_up(:B)])
            eps_dn = Tensor(:eps_spin, [spin_down(:A), spin_down(:B)])
            prod = eps_up * eps_dn

            result = simplify(prod; registry=reg)
            @test result == TScalar(2 // 1)
        end
    end

    @testset "Dotted metric: epsilon_{A'B'} = -epsilon_{B'A'}" begin
        reg = _spin_metric_registry()
        with_registry(reg) do
            eps_AB = Tensor(:eps_spin_dot, [spin_dot_down(:Ap), spin_dot_down(:Bp)])
            eps_BA = Tensor(:eps_spin_dot, [spin_dot_down(:Bp), spin_dot_down(:Ap)])

            s = simplify(eps_AB + eps_BA; registry=reg)
            @test s == TScalar(0 // 1)
        end
    end

    @testset "Dotted metric: eps^{A'C'} eps_{C'B'} -> delta^{A'}_{B'}" begin
        reg = _spin_metric_registry()
        with_registry(reg) do
            eps_up = Tensor(:eps_spin_dot, [spin_dot_up(:Ap), spin_dot_up(:Cp)])
            eps_dn = Tensor(:eps_spin_dot, [spin_dot_down(:Cp), spin_dot_down(:Bp)])
            prod = eps_up * eps_dn

            result = contract_metrics(prod)
            @test result isa Tensor
            @test result.name == :delta_spin_dot
        end
    end

    @testset "Dotted delta trace: delta^{A'}_{A'} = 2" begin
        reg = _spin_metric_registry()
        with_registry(reg) do
            delta = Tensor(:delta_spin_dot, [spin_dot_up(:Ap), spin_dot_down(:Ap)])
            result = contract_metrics(delta)
            @test result == TScalar(2 // 1)
        end
    end

    @testset "No cross-contraction: undotted eps vs dotted index" begin
        reg = _spin_metric_registry()
        with_registry(reg) do
            # eps^{AB} (undotted) should NOT contract with phi_{A'} (dotted)
            eps = Tensor(:eps_spin, [spin_up(:A), spin_up(:B)])
            phi = Tensor(:phi, [spin_dot_down(:A)])
            prod = eps * phi

            # The index 'A' in eps is SL2C, in phi is SL2C_dot; no contraction
            dp = dummy_pairs(prod)
            @test isempty(dp)

            # contract_metrics should not change anything
            result = contract_metrics(prod)
            @test result isa TProduct
        end
    end

    @testset "Metric raises spinor index: eps^{AB} psi_B -> psi^A" begin
        reg = _spin_metric_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:psi, manifold=:M4, rank=(0, 1),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}()))

            eps = Tensor(:eps_spin, [spin_up(:A), spin_up(:B)])
            psi = Tensor(:psi, [spin_down(:B)])
            prod = eps * psi

            result = contract_metrics(prod)
            @test result isa Tensor
            @test result.name == :psi
            @test length(result.indices) == 1
            @test result.indices[1].position == Up
            @test result.indices[1].name == :A
        end
    end

    @testset "Prerequisite check: bundles must exist" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
        end
        @test_throws ErrorException define_spin_metric!(reg; manifold=:M4)
    end

    @testset "Spacetime metric unaffected" begin
        reg = _spin_metric_registry()
        with_registry(reg) do
            # Spacetime metric should still work normally
            g_ab = Tensor(:g, [down(:a), down(:b)])
            g_up = Tensor(:g, [up(:a), up(:c)])
            prod = g_up * g_ab

            result = contract_metrics(prod)
            @test result isa Tensor
            # Should give delta^c_b or similar
            @test result.name == :δ
        end
    end

end
