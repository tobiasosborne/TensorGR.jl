@testset "Soldering form sigma^a_{AA'} (Penrose-Rindler Vol 1, Sec 3.1)" begin

    # Helper: full spinor registry with manifold + bundles + metrics + soldering
    function _soldering_registry()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_spinor_bundles!(reg; manifold=:M4)
            define_spin_metric!(reg; manifold=:M4)
            define_soldering_form!(reg; manifold=:M4, name=:sigma)
        end
        reg
    end

    @testset "define_soldering_form! registers tensor" begin
        reg = _soldering_registry()
        @test has_tensor(reg, :sigma)
        props = get_tensor(reg, :sigma)
        @test props.rank == (1, 2)
        @test isempty(props.symmetries)
        @test get(props.options, :is_soldering, false) == true
        @test get(props.options, :index_vbundles, nothing) == [:Tangent, :SL2C, :SL2C_dot]
    end

    @testset "sigma has correct mixed index structure" begin
        reg = _soldering_registry()
        with_registry(reg) do
            sig = Tensor(:sigma, [up(:a), spin_down(:A), spin_dot_down(:Ap)])
            @test sig.indices[1].vbundle == :Tangent
            @test sig.indices[1].position == Up
            @test sig.indices[2].vbundle == :SL2C
            @test sig.indices[2].position == Down
            @test sig.indices[3].vbundle == :SL2C_dot
            @test sig.indices[3].position == Down
            @test length(sig.indices) == 3
        end
    end

    @testset "Completeness: sigma^a_{AA'} sigma_a^{BB'} -> delta^B_A delta^{B'}_{A'}" begin
        reg = _soldering_registry()
        with_registry(reg) do
            # sigma^a_{AA'} sigma_a^{BB'}
            sig1 = Tensor(:sigma, [up(:a), spin_down(:A), spin_dot_down(:Ap)])
            sig2 = Tensor(:sigma, [down(:a), spin_up(:B), spin_dot_up(:Bp)])
            prod = sig1 * sig2

            result = simplify(prod; registry=reg)

            # Should be delta^B_A * delta^{B'}_{A'}
            @test result isa TProduct
            @test length(result.factors) == 2
            @test result.scalar == 1 // 1

            # Check that both factors are deltas
            names = Set(f.name for f in result.factors if f isa Tensor)
            @test :delta_spin in names
            @test :delta_spin_dot in names

            # Verify delta indices
            for f in result.factors
                f isa Tensor || continue
                @test length(f.indices) == 2
                up_idxs = filter(i -> i.position == Up, f.indices)
                dn_idxs = filter(i -> i.position == Down, f.indices)
                @test length(up_idxs) == 1
                @test length(dn_idxs) == 1
            end
        end
    end

    @testset "Metric reconstruction: sigma^a_{AA'} sigma^{b AA'} -> g^{ab}" begin
        reg = _soldering_registry()
        with_registry(reg) do
            # sigma^a_{AA'} sigma^b^{AA'}
            # sigma^a with SL2C Down, SL2C_dot Down
            # sigma^b with SL2C Up, SL2C_dot Up (raised via epsilon)
            sig1 = Tensor(:sigma, [up(:a), spin_down(:A), spin_dot_down(:Ap)])
            sig2 = Tensor(:sigma, [up(:b), spin_up(:A), spin_dot_up(:Ap)])
            prod = sig1 * sig2

            result = simplify(prod; registry=reg)

            # Should be g^{ab}
            @test result isa Tensor
            @test result.name == :g
            @test length(result.indices) == 2
            positions = Set(idx.position for idx in result.indices)
            @test Up in positions
            names = Set(idx.name for idx in result.indices)
            @test :a in names
            @test :b in names
        end
    end

    @testset "Vector conversion: to_spinor_indices inserts sigma" begin
        reg = _soldering_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:V, manifold=:M4, rank=(0, 1),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}()))

            V_a = Tensor(:V, [down(:a)])
            result = to_spinor_indices(V_a, reg)

            # Should be sigma^a_{AA'} V_a (a product with sigma and V)
            @test result isa TProduct
            @test length(result.factors) == 2

            # One factor should be sigma, the other V
            sigma_factors = filter(f -> f isa Tensor && f.name == :sigma, result.factors)
            v_factors = filter(f -> f isa Tensor && f.name == :V, result.factors)
            @test length(sigma_factors) == 1
            @test length(v_factors) == 1

            # sigma should have the matching tangent index Up (to contract with V's Down)
            sig = sigma_factors[1]
            @test sig.indices[1].name == :a
            @test sig.indices[1].position == Up
            @test sig.indices[1].vbundle == :Tangent
            @test sig.indices[2].vbundle == :SL2C
            @test sig.indices[3].vbundle == :SL2C_dot
        end
    end

    @testset "Vector conversion round-trip: V_a -> V_{AA'} -> V_a" begin
        reg = _soldering_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:V, manifold=:M4, rank=(0, 1),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}()))

            V_a = Tensor(:V, [down(:a)])

            # Forward: V_a -> sigma^a_{AA'} V_a
            spinor_form = to_spinor_indices(V_a, reg)
            # After simplification, the tangent index contracts away
            spinor_simplified = simplify(spinor_form; registry=reg)

            # Backward: insert sigma to convert back
            tensor_form = to_tensor_indices(spinor_simplified, reg)
            # After simplification, spinor indices contract away via completeness
            result = simplify(tensor_form; registry=reg)

            # Result should be V with a single tangent index
            @test result isa Tensor
            @test result.name == :V
            @test length(result.indices) == 1
            @test result.indices[1].vbundle == :Tangent
        end
    end

    @testset "Rank-2 tensor conversion: T_{ab} gets two sigmas" begin
        reg = _soldering_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:T, manifold=:M4, rank=(0, 2),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}()))

            T_ab = Tensor(:T, [down(:a), down(:b)])
            result = to_spinor_indices(T_ab, reg)

            # Should have 3 factors: sigma, sigma, T
            @test result isa TProduct
            sigma_count = count(f -> f isa Tensor && f.name == :sigma, result.factors)
            @test sigma_count == 2
            t_count = count(f -> f isa Tensor && f.name == :T, result.factors)
            @test t_count == 1
        end
    end

    @testset "Prerequisite: bundles must exist" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
        end
        @test_throws ErrorException define_soldering_form!(reg)
    end

    @testset "Prerequisite: spin metric must exist" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_spinor_bundles!(reg; manifold=:M4)
        end
        @test_throws ErrorException define_soldering_form!(reg)
    end

    @testset "to_spinor_indices: no tangent indices = no change" begin
        reg = _soldering_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:psi, manifold=:M4, rank=(0, 1),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}()))

            # A pure spinor expression should not change
            psi_A = Tensor(:psi, [spin_down(:A)])
            result = to_spinor_indices(psi_A, reg)
            @test result == psi_A
        end
    end

    @testset "to_tensor_indices: no spinor pairs = no change" begin
        reg = _soldering_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:V, manifold=:M4, rank=(0, 1),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}()))

            # A pure tensor expression should not change
            V_a = Tensor(:V, [down(:a)])
            result = to_tensor_indices(V_a, reg)
            @test result == V_a
        end
    end

    @testset "Soldering form does not re-register" begin
        reg = _soldering_registry()
        with_registry(reg) do
            # Calling define_soldering_form! again should not error
            # (it checks has_tensor and skips registration)
            define_soldering_form!(reg; manifold=:M4, name=:sigma)
            @test has_tensor(reg, :sigma)
        end
    end

    @testset "Completeness contract then simplify to scalar trace" begin
        reg = _soldering_registry()
        with_registry(reg) do
            # sigma^a_{AA'} sigma_a^{AA'} should give dim=4
            # because sigma^a_{AA'} sigma_a^{BA'} = delta^B_A delta^{B'}_{A'}
            # and then tracing B=A, B'=A' gives 2*2 = 4
            sig1 = Tensor(:sigma, [up(:a), spin_down(:A), spin_dot_down(:Ap)])
            sig2 = Tensor(:sigma, [down(:a), spin_up(:A), spin_dot_up(:Ap)])
            prod = sig1 * sig2

            result = simplify(prod; registry=reg)
            # delta^A_A * delta^{A'}_{A'} = 2 * 2 = 4
            @test result == TScalar(4 // 1)
        end
    end

end
