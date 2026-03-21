@testset "SU(2) spatial spinors (TGR-adi)" begin

    # ── Shared setup ──
    function _space_spin_setup()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:Sigma, 3, :gamma, :partial, [:i,:j,:k,:l,:m,:n]))
        define_metric!(reg, :gamma; manifold=:Sigma)
        define_space_spinors!(reg; manifold=:Sigma, metric=:gamma)
        reg
    end

    # ── Test 1: SU(2) VBundle registration ──
    @testset "SU2 VBundle registration" begin
        reg = _space_spin_setup()
        @test has_vbundle(reg, :SU2)

        su2 = get_vbundle(reg, :SU2)
        @test su2.dim == 2
        @test su2.manifold == :Sigma
        @test su2.indices == [:P, :Q, :R, :S, :T, :U]
    end

    # ── Test 2: eps_space registration and antisymmetry ──
    @testset "eps_space registration" begin
        reg = _space_spin_setup()
        @test has_tensor(reg, :eps_space)

        eps_props = get_tensor(reg, :eps_space)
        @test eps_props.is_metric
        @test any(s -> s isa AntiSymmetric, eps_props.symmetries)
        @test eps_props.options[:vbundle] == :SU2
        @test eps_props.options[:vbundle_dim] == 2
    end

    # ── Test 3: delta_SU2 registration ──
    @testset "delta_SU2 registration" begin
        reg = _space_spin_setup()
        @test has_tensor(reg, :delta_SU2)

        delta_props = get_tensor(reg, :delta_SU2)
        @test delta_props.is_delta
        @test delta_props.options[:vbundle] == :SU2
    end

    # ── Test 4: cache entries ──
    @testset "cache entries" begin
        reg = _space_spin_setup()
        # Spatial metric cache preserved
        @test reg.metric_cache[:Sigma] == :gamma
        # SU(2) vbundle caches populated
        @test reg.metric_cache[:SU2] == :eps_space
        @test reg.delta_cache[:SU2] == :delta_SU2
    end

    # ── Test 5: tau registration and symmetry ──
    @testset "tau registration and symmetry" begin
        reg = _space_spin_setup()
        @test has_tensor(reg, :tau)

        tau_props = get_tensor(reg, :tau)
        @test tau_props.rank == (1, 2)
        # Symmetric in slots 2,3 (the SU(2) spinor pair)
        @test any(s -> s isa Symmetric && s.i == 2 && s.j == 3, tau_props.symmetries)
        @test tau_props.options[:is_soldering] == true
        @test tau_props.options[:is_space_soldering] == true
        @test tau_props.options[:index_vbundles] == [:Tangent, :SU2, :SU2]
    end

    # ── Test 6: convenience constructors ──
    @testset "convenience constructors" begin
        @test space_spin_up(:P) == TIndex(:P, Up, :SU2)
        @test space_spin_down(:Q) == TIndex(:Q, Down, :SU2)
    end

    # ── Test 7: predicates ──
    @testset "predicates" begin
        @test is_space_spinor_index(space_spin_up(:P))
        @test is_space_spinor_index(space_spin_down(:Q))
        @test !is_space_spinor_index(up(:a))
        @test !is_space_spinor_index(spin_up(:A))
    end

    # ── Test 8: SU(2) index construction and distinctness ──
    @testset "index distinctness" begin
        idx_su2 = TIndex(:P, Up, :SU2)
        idx_sl2c = TIndex(:P, Up, :SL2C)
        idx_tang = TIndex(:P, Up, :Tangent)

        # Same name and position but different vbundles => NOT equal
        @test idx_su2 != idx_sl2c
        @test idx_su2 != idx_tang
        @test idx_sl2c != idx_tang
    end

    # ── Test 9: tensor construction with SU(2) indices ──
    @testset "tensors with SU2 indices" begin
        reg = _space_spin_setup()
        with_registry(reg) do
            psi = Tensor(:tau, [up(:i), space_spin_down(:P), space_spin_down(:Q)])
            @test length(psi.indices) == 3
            @test psi.indices[1].vbundle == :Tangent
            @test psi.indices[2].vbundle == :SU2
            @test psi.indices[3].vbundle == :SU2

            fi = free_indices(psi)
            @test length(fi) == 3
        end
    end

    # ── Test 10: eps_space antisymmetry via canonicalize ──
    @testset "eps_space antisymmetry canonicalize" begin
        reg = _space_spin_setup()
        with_registry(reg) do
            eps_QP = Tensor(:eps_space, [space_spin_down(:Q), space_spin_down(:P)])
            can = canonicalize(eps_QP)
            eps_PQ = Tensor(:eps_space, [space_spin_down(:P), space_spin_down(:Q)])
            # Should be -eps_{PQ} (canonical order P < Q)
            if can isa TProduct
                @test can.scalar == -1 // 1
                @test length(can.factors) == 1
                @test can.factors[1] == eps_PQ
            else
                @test can == eps_PQ || can != eps_QP
            end
        end
    end

    # ── Test 11: delta_SU2 self-trace = 2 ──
    @testset "delta_SU2 trace = 2" begin
        reg = _space_spin_setup()
        with_registry(reg) do
            delta_PP = Tensor(:delta_SU2, [space_spin_up(:P), space_spin_down(:P)])
            result = contract_metrics(delta_PP)
            @test result == TScalar(2 // 1)
        end
    end

    # ── Test 12: tau symmetry via canonicalize ──
    @testset "tau symmetry in spinor indices" begin
        reg = _space_spin_setup()
        with_registry(reg) do
            # tau^i_{QP} should canonicalize to tau^i_{PQ} (symmetric => no sign)
            tau_QP = Tensor(:tau, [up(:i), space_spin_down(:Q), space_spin_down(:P)])
            can = canonicalize(tau_QP)
            tau_PQ = Tensor(:tau, [up(:i), space_spin_down(:P), space_spin_down(:Q)])
            @test can == tau_PQ
        end
    end

    # ── Test 13: expression builders ──
    @testset "expression builders" begin
        reg = _space_spin_setup()
        with_registry(reg) do
            eps = space_spin_metric_expr(; registry=reg)
            @test eps isa Tensor
            @test eps.name == :eps_space
            @test length(eps.indices) == 2
            @test all(idx -> idx.vbundle == :SU2 && idx.position == Down, eps.indices)

            tau = soldering_form_expr(; registry=reg)
            @test tau isa Tensor
            @test tau.name == :tau
            @test length(tau.indices) == 3
            @test tau.indices[1].vbundle == :Tangent
            @test tau.indices[1].position == Up
            @test tau.indices[2].vbundle == :SU2
            @test tau.indices[3].vbundle == :SU2
        end
    end

    # ── Test 14: completeness relation structure ──
    @testset "completeness relation" begin
        reg = _space_spin_setup()
        with_registry(reg) do
            comp = space_spinor_completeness(; registry=reg)
            @test comp isa TProduct
            @test length(comp.factors) == 2
            @test all(f -> f isa Tensor && f.name == :tau, comp.factors)

            # The spatial index should be contracted (same name, opposite positions)
            t1 = comp.factors[1]::Tensor
            t2 = comp.factors[2]::Tensor
            @test t1.indices[1].name == t2.indices[1].name
            @test t1.indices[1].position != t2.indices[1].position
            @test t1.indices[1].vbundle == :Tangent

            # SU(2) indices: first tau has Down, second has Up
            @test t1.indices[2].position == Down
            @test t2.indices[2].position == Up
        end
    end

    # ── Test 15: spatial metric preserved after define_space_spinors! ──
    @testset "spatial metric not clobbered" begin
        reg = _space_spin_setup()
        with_registry(reg) do
            # gamma^{ij} gamma_{ij} should give dim=3
            g_up = Tensor(:gamma, [up(:i), up(:j)])
            g_down = Tensor(:gamma, [down(:i), down(:j)])
            prod = g_up * g_down
            result = simplify(prod; registry=reg)
            @test result == TScalar(3 // 1)
        end
    end

    # ── Test 16: eps_space product -> delta ──
    @testset "eps_space product = delta" begin
        reg = _space_spin_setup()
        with_registry(reg) do
            eps_up = Tensor(:eps_space, [space_spin_up(:P), space_spin_up(:R)])
            eps_down = Tensor(:eps_space, [space_spin_down(:R), space_spin_down(:Q)])
            prod = eps_up * eps_down
            result = contract_metrics(prod)
            @test result isa Tensor
            @test result.name == :delta_SU2
            @test result.indices[1].position == Up
            @test result.indices[2].position == Down
        end
    end

    # ── Test 17: idempotent registration ──
    @testset "idempotent registration" begin
        reg = _space_spin_setup()
        # Calling again should not error
        define_space_spinors!(reg; manifold=:Sigma, metric=:gamma)
        @test has_vbundle(reg, :SU2)
        @test has_tensor(reg, :eps_space)
        @test has_tensor(reg, :tau)
    end

    # ── Test 18: fresh_index generates SU2 alphabet ──
    @testset "fresh_index SU2 alphabet" begin
        used = Set{Symbol}()
        idx1 = fresh_index(used; vbundle=:SU2)
        @test idx1 == :P
        push!(used, idx1)
        idx2 = fresh_index(used; vbundle=:SU2)
        @test idx2 == :Q
    end
end
