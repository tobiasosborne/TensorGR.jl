@testset "DHOST class I Lagrangian (Langlois & Noui 2016)" begin

    # Helper: standard 4D manifold registry for DHOST tests
    function _dhost_registry()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :d,
            [:a,:b,:c,:d,:e,:f,:m,:n,:p,:q,:r,:s,:t,:u,:v,:w]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0,2),
            symmetries=SymmetrySpec[Symmetric(1,2)],
            is_metric=true,
            options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M4, rank=(1,1),
            symmetries=SymmetrySpec[],
            is_delta=true,
            options=Dict{Symbol,Any}(:is_delta => true)))
        reg
    end

    # -- Type and registration ------------------------------------------------

    @testset "DHOSTTheory struct and define_dhost!" begin
        reg = _dhost_registry()
        with_registry(reg) do
            dht = define_dhost!(reg; manifold=:M4, metric=:g)

            @test dht isa DHOSTTheory
            @test dht.manifold == :M4
            @test dht.metric == :g
            @test dht.scalar_field == :phi
            @test dht.covd == :nabla
            @test length(dht.a) == 5
        end
    end

    @testset "define_dhost! registers all coefficient functions" begin
        reg = _dhost_registry()
        with_registry(reg) do
            dht = define_dhost!(reg; manifold=:M4, metric=:g)

            for name in [:f0, :f1, :a1, :a2, :a3, :a4, :a5, :phi]
                @test has_tensor(reg, name)
            end

            # All coefficient functions are rank-0
            for name in [:f0, :f1, :a1, :a2, :a3, :a4, :a5]
                props = get_tensor(reg, name)
                @test props.rank == (0, 0)
            end
        end
    end

    @testset "Custom scalar field name" begin
        reg = _dhost_registry()
        with_registry(reg) do
            dht = define_dhost!(reg; manifold=:M4, metric=:g,
                                scalar_field=:varphi)
            @test dht.scalar_field == :varphi
            @test has_tensor(reg, :varphi)
        end
    end

    # -- Individual Lagrangians are scalars (no free indices) -----------------

    @testset "L_1 through L_5 have no free indices" begin
        reg = _dhost_registry()
        with_registry(reg) do
            dht = define_dhost!(reg; manifold=:M4, metric=:g)

            @test isempty(free_indices(dhost_L1(dht; registry=reg)))
            @test isempty(free_indices(dhost_L2(dht; registry=reg)))
            @test isempty(free_indices(dhost_L3(dht; registry=reg)))
            @test isempty(free_indices(dhost_L4(dht; registry=reg)))
            @test isempty(free_indices(dhost_L5(dht; registry=reg)))
        end
    end

    # -- Full Lagrangian structure --------------------------------------------

    @testset "dhost_lagrangian is a scalar" begin
        reg = _dhost_registry()
        with_registry(reg) do
            dht = define_dhost!(reg; manifold=:M4, metric=:g)

            L = dhost_lagrangian(dht; registry=reg)
            @test isempty(free_indices(L))
            @test L isa TSum
        end
    end

    # -- Vanishing limit: all coefficients zero gives zero --------------------

    @testset "All coefficients zero gives zero Lagrangian" begin
        reg = _dhost_registry()
        with_registry(reg) do
            dht = define_dhost!(reg; manifold=:M4, metric=:g)

            for name in [:f0, :f1, :a1, :a2, :a3, :a4, :a5]
                set_vanishing!(reg, name)
            end

            L = dhost_lagrangian(dht; registry=reg)
            L_simplified = simplify(L; registry=reg)
            @test L_simplified == TScalar(0 // 1)
        end
    end

    # -- f0-only limit: L = f0(phi,X) ----------------------------------------

    @testset "Only f0 nonzero gives pure scalar" begin
        reg = _dhost_registry()
        with_registry(reg) do
            dht = define_dhost!(reg; manifold=:M4, metric=:g)

            for name in [:f1, :a1, :a2, :a3, :a4, :a5]
                set_vanishing!(reg, name)
            end

            L = dhost_lagrangian(dht; registry=reg)
            L_simplified = simplify(L; registry=reg)
            @test L_simplified == Tensor(:f0, TIndex[])
        end
    end

    # -- L1 = (Box phi)^2: structure test -------------------------------------

    @testset "L_1 contains box operator (nested derivatives)" begin
        reg = _dhost_registry()
        with_registry(reg) do
            dht = define_dhost!(reg; manifold=:M4, metric=:g)
            L1 = dhost_L1(dht; registry=reg)

            # L1 = g^{ab} d_a d_b phi * g^{cd} d_c d_d phi
            # Should be a TProduct with metric factors and nested TDerivs
            @test L1 isa TProduct
            metric_count = count(f -> f isa Tensor && f.name == :g, L1.factors)
            @test metric_count >= 2  # at least 2 inverse metrics from 2 box operators
        end
    end

    # -- L2 structure test: contracted second derivatives ---------------------

    @testset "L_2 structure: contracted nabla_a nabla_b phi" begin
        reg = _dhost_registry()
        with_registry(reg) do
            dht = define_dhost!(reg; manifold=:M4, metric=:g)
            L2 = dhost_L2(dht; registry=reg)

            @test L2 isa TProduct
            metric_count = count(f -> f isa Tensor && f.name == :g, L2.factors)
            # Should have 2 inverse metrics for contraction g^{ac} g^{bd}
            @test metric_count >= 2
        end
    end

    # -- L1 and L2 are distinct -----------------------------------------------

    @testset "L_1 and L_2 are algebraically distinct" begin
        reg = _dhost_registry()
        with_registry(reg) do
            dht = define_dhost!(reg; manifold=:M4, metric=:g)

            L1 = dhost_L1(dht; registry=reg)
            L2 = dhost_L2(dht; registry=reg)

            # L1 - L2 should NOT simplify to zero
            diff = L1 - L2
            diff_simplified = simplify(diff; registry=reg)
            @test diff_simplified != TScalar(0 // 1)
        end
    end

    # -- Horndeski subcase: G4X[(Box phi)^2 - (dd phi)^2] = a1*L1 + a2*L2
    #    with a1 = G4X, a2 = -G4X

    @testset "Horndeski L4 kinetic part from a1*L1 - a1*L2" begin
        reg = _dhost_registry()
        with_registry(reg) do
            dht = define_dhost!(reg; manifold=:M4, metric=:g)

            L1 = dhost_L1(dht; registry=reg)
            L2 = dhost_L2(dht; registry=reg)

            # (Box phi)^2 - (dd phi)^2 should have no free indices
            horn_combo = L1 - L2
            @test isempty(free_indices(horn_combo))
        end
    end

    # -- L5 is a perfect square -----------------------------------------------

    @testset "L_5 has correct tensor structure (4 first derivatives, 2 second derivatives)" begin
        reg = _dhost_registry()
        with_registry(reg) do
            dht = define_dhost!(reg; manifold=:M4, metric=:g)
            L5 = dhost_L5(dht; registry=reg)

            # L5 = [dphi^a dphi^b dd_{ab}]^2 has 4 metrics for raising,
            # 4 first derivatives, and 2 second-derivative chains
            @test L5 isa TProduct
            metric_count = count(f -> f isa Tensor && f.name == :g, L5.factors)
            @test metric_count == 4  # 4 inverse metrics to raise indices
        end
    end

    # -- Coefficient function naming ------------------------------------------

    @testset "ScalarTensorFunction names for DHOST coefficients" begin
        reg = _dhost_registry()
        with_registry(reg) do
            dht = define_dhost!(reg; manifold=:M4, metric=:g)

            @test g_tensor_name(dht.f0) == :f0
            @test g_tensor_name(dht.f1) == :f1
            for (i, ai) in enumerate(dht.a)
                @test g_tensor_name(ai) == Symbol("a$i")
            end
        end
    end

end
