@testset "Sen connection (TGR-3up)" begin

    # ── Shared setup ──
    function _sen_setup()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:Sigma, 3, :gamma, :partial, [:i,:j,:k,:l,:m,:n]))
        define_metric!(reg, :gamma; manifold=:Sigma)
        define_space_spinors!(reg; manifold=:Sigma, metric=:gamma)
        define_sen_connection!(reg; manifold=:Sigma, spatial_metric=:gamma)
        reg
    end

    # ── Test 1: define_sen_connection! registers connection tensor ──
    @testset "Gamma_sen registration" begin
        reg = _sen_setup()
        @test has_tensor(reg, :Gamma_sen)

        props = get_tensor(reg, :Gamma_sen)
        @test props.rank == (1, 2)
        @test isempty(props.symmetries)
        @test props.options[:is_connection] == true
        @test props.options[:connection_name] == :D_sen
        @test props.options[:index_vbundles] == [:SU2, :SU2, :Tangent]
    end

    # ── Test 2: define_sen_connection! registers curvature tensor ──
    @testset "F_sen registration" begin
        reg = _sen_setup()
        @test has_tensor(reg, :F_sen)

        props = get_tensor(reg, :F_sen)
        @test props.rank == (1, 3)
        @test any(s -> s isa AntiSymmetric && s.i == 1 && s.j == 2, props.symmetries)
        @test props.options[:is_curvature] == true
        @test props.options[:connection_name] == :D_sen
        @test props.options[:index_vbundles] == [:Tangent, :Tangent, :SU2, :SU2]
    end

    # ── Test 3: sen_connection_expr builds valid TensorExpr ──
    @testset "sen_connection_expr" begin
        reg = _sen_setup()
        with_registry(reg) do
            gamma_expr = sen_connection_expr(; registry=reg)
            @test gamma_expr isa Tensor
            @test gamma_expr.name == :Gamma_sen
            @test length(gamma_expr.indices) == 3
            # First index: Up SU2
            @test gamma_expr.indices[1].position == Up
            @test gamma_expr.indices[1].vbundle == :SU2
            # Second index: Down SU2
            @test gamma_expr.indices[2].position == Down
            @test gamma_expr.indices[2].vbundle == :SU2
            # Third index: Down Tangent
            @test gamma_expr.indices[3].position == Down
            @test gamma_expr.indices[3].vbundle == :Tangent
        end
    end

    # ── Test 4: sen_covd produces TDeriv structure ──
    @testset "sen_covd structure" begin
        reg = _sen_setup()
        with_registry(reg) do
            # Build a spinor psi_P
            register_tensor!(reg, TensorProperties(
                name=:psi, manifold=:Sigma, rank=(0, 1),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}(:index_vbundles => [:SU2])
            ))
            psi = Tensor(:psi, [space_spin_down(:P)])
            result = sen_covd(psi, :i; registry=reg)

            @test result isa TDeriv
            @test result.index == TIndex(:i, Down, :Tangent)
            @test result.covd == :D_sen
            @test result.arg == psi
        end
    end

    # ── Test 5: sen_covd_expr with auto-generated index ──
    @testset "sen_covd_expr auto index" begin
        reg = _sen_setup()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:phi, manifold=:Sigma, rank=(0, 1),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}(:index_vbundles => [:SU2])
            ))
            phi = Tensor(:phi, [space_spin_down(:P)])
            result = sen_covd_expr(phi; registry=reg)

            @test result isa TDeriv
            @test result.index.position == Down
            @test result.index.vbundle == :Tangent
            @test result.covd == :D_sen
            @test result.arg == phi
            # The generated index should not clash with P
            @test result.index.name != :P
        end
    end

    # ── Test 6: metricity D_i eps_{PQ} = 0 ──
    @testset "metricity: D_i eps_PQ = 0" begin
        reg = _sen_setup()
        with_registry(reg) do
            eps = Tensor(:eps_space, [space_spin_down(:P), space_spin_down(:Q)])
            deriv_eps = TDeriv(TIndex(:i, Down, :Tangent), eps, :D_sen)
            result = simplify(deriv_eps; registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

    # ── Test 7: compatibility D_i tau^j_{PQ} = 0 ──
    @testset "compatibility: D_i tau^j_PQ = 0" begin
        reg = _sen_setup()
        with_registry(reg) do
            tau = Tensor(:tau, [up(:j), space_spin_down(:P), space_spin_down(:Q)])
            deriv_tau = TDeriv(TIndex(:i, Down, :Tangent), tau, :D_sen)
            result = simplify(deriv_tau; registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

    # ── Test 8: curvature tensor index structure ──
    @testset "sen_curvature_expr index structure" begin
        reg = _sen_setup()
        with_registry(reg) do
            F = sen_curvature_expr(; registry=reg)
            @test F isa Tensor
            @test F.name == :F_sen
            @test length(F.indices) == 4

            # Spatial indices: Down Tangent (slots 1, 2)
            @test F.indices[1].position == Down
            @test F.indices[1].vbundle == :Tangent
            @test F.indices[2].position == Down
            @test F.indices[2].vbundle == :Tangent

            # Spinor indices: (Down SU2, Up SU2) in slots 3, 4
            @test F.indices[3].position == Down
            @test F.indices[3].vbundle == :SU2
            @test F.indices[4].position == Up
            @test F.indices[4].vbundle == :SU2

            # All four indices should be distinct
            names = [idx.name for idx in F.indices]
            @test length(unique(names)) == 4
        end
    end

    # ── Test 9: curvature antisymmetry in spatial indices ──
    @testset "curvature antisymmetry" begin
        reg = _sen_setup()
        with_registry(reg) do
            # F_{ji P}^Q should canonicalize to -F_{ij P}^Q
            F_ji = Tensor(:F_sen, [down(:j), down(:i),
                                    space_spin_down(:P), space_spin_up(:Q)])
            can = canonicalize(F_ji)
            F_ij = Tensor(:F_sen, [down(:i), down(:j),
                                    space_spin_down(:P), space_spin_up(:Q)])
            if can isa TProduct
                @test can.scalar == -1 // 1
                @test length(can.factors) == 1
                @test can.factors[1] == F_ij
            else
                # Could be -F_ij directly if xperm returns a product
                @test can != F_ji
            end
        end
    end

    # ── Test 10: idempotent registration ──
    @testset "idempotent registration" begin
        reg = _sen_setup()
        # Calling again should not error
        define_sen_connection!(reg; manifold=:Sigma, spatial_metric=:gamma)
        @test has_tensor(reg, :Gamma_sen)
        @test has_tensor(reg, :F_sen)
    end

    # ── Test 11: prerequisite checks ──
    @testset "prerequisite errors" begin
        reg = TensorRegistry()
        # No manifold
        @test_throws ErrorException define_sen_connection!(reg)

        register_manifold!(reg, ManifoldProperties(:Sigma, 3, :gamma, :partial, [:i,:j,:k,:l,:m,:n]))
        # No SU2 vbundle
        @test_throws ErrorException define_sen_connection!(reg; manifold=:Sigma)
    end

    # ── Test 12: nested Sen derivatives ──
    @testset "nested Sen derivatives" begin
        reg = _sen_setup()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:chi, manifold=:Sigma, rank=(0, 1),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}(:index_vbundles => [:SU2])
            ))
            chi = Tensor(:chi, [space_spin_down(:P)])
            # D_i D_j chi_P
            d1 = sen_covd(chi, :j; registry=reg)
            d2 = sen_covd(d1, :i; registry=reg)

            @test d2 isa TDeriv
            @test d2.covd == :D_sen
            @test d2.index == TIndex(:i, Down, :Tangent)
            @test d2.arg isa TDeriv
            @test d2.arg.covd == :D_sen
            @test d2.arg.index == TIndex(:j, Down, :Tangent)
        end
    end

end
