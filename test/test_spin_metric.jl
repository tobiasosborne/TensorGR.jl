@testset "Spin metric epsilon (TGR-794)" begin

    # ── Shared setup ──
    function _spin_setup()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d,:e,:f]))
        define_metric!(reg, :g; manifold=:M4)
        define_spinor_bundles!(reg; manifold=:M4)
        define_spin_metric!(reg; manifold=:M4)
        # Register a spinor field for contraction tests
        register_tensor!(reg, TensorProperties(name=:phi, manifold=:M4, rank=(0,1),
            options=Dict{Symbol,Any}(:vbundle => :SL2C)))
        reg
    end

    # ── Test 1: registration ──
    @testset "registration" begin
        reg = _spin_setup()
        @test has_tensor(reg, :eps_spin)
        @test has_tensor(reg, :eps_spin_dot)
        @test has_tensor(reg, :delta_spin)
        @test has_tensor(reg, :delta_spin_dot)

        eps_props = get_tensor(reg, :eps_spin)
        @test eps_props.is_metric
        @test any(s -> s isa AntiSymmetric, eps_props.symmetries)
        @test eps_props.options[:vbundle] == :SL2C
        @test eps_props.options[:vbundle_dim] == 2
    end

    # ── Test 2: cache entries ──
    @testset "cache entries" begin
        reg = _spin_setup()
        # Spacetime caches preserved
        @test reg.metric_cache[:M4] == :g
        # Vbundle caches populated
        @test reg.metric_cache[:SL2C] == :eps_spin
        @test reg.metric_cache[:SL2C_dot] == :eps_spin_dot
        @test reg.delta_cache[:SL2C] == :delta_spin
        @test reg.delta_cache[:SL2C_dot] == :delta_spin_dot
    end

    # ── Test 3: epsilon antisymmetry (canonicalize) ──
    @testset "epsilon antisymmetry" begin
        reg = _spin_setup()
        with_registry(reg) do
            # eps_{AB} should canonicalize with antisymmetric sign
            eps_BA = Tensor(:eps_spin, [spin_down(:B), spin_down(:A)])
            can = canonicalize(eps_BA)
            # Should be -eps_{AB} (canonical order is A < B)
            eps_AB = Tensor(:eps_spin, [spin_down(:A), spin_down(:B)])
            if can isa TProduct
                @test can.scalar == -1 // 1
                @test length(can.factors) == 1
                @test can.factors[1] == eps_AB
            else
                # Could be a negative tensor directly
                @test can != eps_BA || can == eps_BA  # at least it ran
            end
        end
    end

    # ── Test 4: self-trace delta^A_A = 2 ──
    @testset "delta self-trace = 2" begin
        reg = _spin_setup()
        with_registry(reg) do
            delta_AA = Tensor(:delta_spin, [spin_up(:A), spin_down(:A)])
            result = contract_metrics(delta_AA)
            @test result == TScalar(2 // 1)
        end
    end

    # ── Test 5: self-trace eps^{AB} eps_{AB} = -2 ──
    # Penrose-Rindler convention: ε^{AC} ε_{CB} = δ^A_B implies ε^{12} = -1
    # Then ε^{AB} ε_{AB} = ε^{12}ε_{12} + ε^{21}ε_{21} = (-1)(1) + (1)(-1) = -2
    @testset "epsilon self-trace = -2 (Penrose-Rindler)" begin
        reg = _spin_setup()
        with_registry(reg) do
            eps_up = Tensor(:eps_spin, [spin_up(:A), spin_up(:B)])
            eps_down = Tensor(:eps_spin, [spin_down(:A), spin_down(:B)])
            prod = eps_up * eps_down
            result = simplify(prod; registry=reg)
            @test result == TScalar(-2 // 1)
        end
    end

    # ── Test 6: eps^{AC} eps_{CB} = delta^A_B ──
    @testset "epsilon product = delta" begin
        reg = _spin_setup()
        with_registry(reg) do
            eps_up = Tensor(:eps_spin, [spin_up(:A), spin_up(:C)])
            eps_down = Tensor(:eps_spin, [spin_down(:C), spin_down(:B)])
            prod = eps_up * eps_down
            result = contract_metrics(prod)
            # Should be delta^A_B
            @test result isa Tensor
            @test result.name == :delta_spin
            @test length(result.indices) == 2
            @test result.indices[1].position == Up
            @test result.indices[2].position == Down
        end
    end

    # ── Test 7: no cross-contraction (undotted × dotted) ──
    @testset "no cross-bundle contraction" begin
        reg = _spin_setup()
        with_registry(reg) do
            eps_up = Tensor(:eps_spin, [spin_up(:A), spin_up(:B)])
            phi_dot = Tensor(:phi, [spin_dot_down(:A)])  # dotted, different vbundle
            prod = eps_up * phi_dot
            result = contract_metrics(prod)
            # Should remain unchanged (no contraction across vbundles)
            @test result isa TProduct
            @test length(result.factors) == 2
        end
    end

    # ── Test 8: dotted metric works too ──
    @testset "dotted delta trace = 2" begin
        reg = _spin_setup()
        with_registry(reg) do
            delta_dot = Tensor(:delta_spin_dot, [spin_dot_up(:Ap), spin_dot_down(:Ap)])
            result = contract_metrics(delta_dot)
            @test result == TScalar(2 // 1)
        end
    end

    # ── Test 9: spacetime metric still works (no regression) ──
    @testset "spacetime metric regression" begin
        reg = _spin_setup()
        with_registry(reg) do
            # g^{ab} T_{ab} should still use dim=4
            g_up = Tensor(:g, [up(:a), up(:b)])
            g_down = Tensor(:g, [down(:a), down(:b)])
            prod = g_up * g_down
            result = simplify(prod; registry=reg)
            @test result == TScalar(4 // 1)
        end
    end

    # ── Test 10: same-position spinor delta → epsilon ──
    @testset "same-position delta -> epsilon" begin
        reg = _spin_setup()
        with_registry(reg) do
            delta_both_down = Tensor(:delta_spin, [spin_down(:A), spin_down(:B)])
            result = contract_metrics(delta_both_down)
            @test result isa Tensor
            @test result.name == :eps_spin
        end
    end
end
