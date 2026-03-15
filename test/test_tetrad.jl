@testset "Tetrad (vierbein/vielbein) definition and contraction" begin

    @testset "define_tetrad! registers tetrad and inverse" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_frame_bundle!(reg; manifold=:M4, dim=4)
            tp = define_tetrad!(reg, :e; manifold=:M4, metric=:g)

            @test tp isa TetradProperties
            @test tp.name == :e
            @test tp.manifold == :M4
            @test tp.metric == :g
            @test tp.frame_metric == :eta
            @test tp.inverse_name == :E

            # Both tensors registered
            @test has_tensor(reg, :e)
            @test has_tensor(reg, :E)

            # Tetrad properties stored
            @test has_tetrad(reg, :e)
            @test get_tetrad(reg, :e) === tp
        end
    end

    @testset "Correct index structure" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_frame_bundle!(reg; manifold=:M4, dim=4)
            define_tetrad!(reg, :e; manifold=:M4, metric=:g)

            # Tetrad e^I_a: one Lorentz Up, one Tangent Down
            e_props = get_tensor(reg, :e)
            @test e_props.rank == (1, 1)
            @test e_props.options[:up_vbundle] == :Lorentz
            @test e_props.options[:down_vbundle] == :Tangent
            @test e_props.options[:is_tetrad] == true

            # Inverse tetrad E^a_I: one Tangent Up, one Lorentz Down
            E_props = get_tensor(reg, :E)
            @test E_props.rank == (1, 1)
            @test E_props.options[:up_vbundle] == :Tangent
            @test E_props.options[:down_vbundle] == :Lorentz
            @test E_props.options[:is_inverse_tetrad] == true
        end
    end

    @testset "Custom inverse name" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_frame_bundle!(reg; manifold=:M4, dim=4)
            tp = define_tetrad!(reg, :e; manifold=:M4, metric=:g, inverse_name=:einv)

            @test tp.inverse_name == :einv
            @test has_tensor(reg, :einv)
            @test !has_tensor(reg, :E)
        end
    end

    @testset "Custom frame metric" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_frame_bundle!(reg; manifold=:M4, dim=4)

            # Register a custom frame metric (e.g. null frame metric)
            register_tensor!(reg, TensorProperties(
                name=:eta_null, manifold=:M4, rank=(0, 2),
                symmetries=SymmetrySpec[Symmetric(1, 2)],
                is_metric=true,
                options=Dict{Symbol,Any}(:is_metric => true, :vbundle => :Lorentz)))

            tp = define_tetrad!(reg, :e; manifold=:M4, metric=:g,
                                frame_metric=:eta_null)
            @test tp.frame_metric == :eta_null
        end
    end

    @testset "Error: manifold not registered" begin
        reg = TensorRegistry()
        @test_throws ErrorException define_tetrad!(reg, :e; manifold=:M4, metric=:g)
    end

    @testset "Error: metric not registered" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_frame_bundle!(reg; manifold=:M4, dim=4)
            @test_throws ErrorException define_tetrad!(reg, :e; manifold=:M4,
                                                        metric=:nonexistent)
        end
    end

    @testset "Error: frame bundle not set up" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            # No define_frame_bundle! call
            @test_throws ErrorException define_tetrad!(reg, :e; manifold=:M4, metric=:g)
        end
    end

    @testset "Error: double registration" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_frame_bundle!(reg; manifold=:M4, dim=4)
            define_tetrad!(reg, :e; manifold=:M4, metric=:g)
            @test_throws ErrorException define_tetrad!(reg, :e; manifold=:M4, metric=:g)
        end
    end

    @testset "Building tetrad expressions with mixed indices" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_frame_bundle!(reg; manifold=:M4, dim=4)
            define_tetrad!(reg, :e; manifold=:M4, metric=:g)

            # e^I_a: Lorentz Up, Tangent Down
            e = Tensor(:e, [frame_up(:I), down(:a)])
            @test e.indices[1].vbundle == :Lorentz
            @test e.indices[1].position == Up
            @test e.indices[2].vbundle == :Tangent
            @test e.indices[2].position == Down

            # E^a_I: Tangent Up, Lorentz Down
            E = Tensor(:E, [up(:a), frame_down(:I)])
            @test E.indices[1].vbundle == :Tangent
            @test E.indices[1].position == Up
            @test E.indices[2].vbundle == :Lorentz
            @test E.indices[2].position == Down
        end
    end

    @testset "Completeness: e^I_a E^a_J -> delta^I_J" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_frame_bundle!(reg; manifold=:M4, dim=4)
            define_tetrad!(reg, :e; manifold=:M4, metric=:g)

            # e^I_a E^a_J (frame completeness)
            e = Tensor(:e, [frame_up(:I), down(:a)])
            E = Tensor(:E, [up(:a), frame_down(:J)])
            prod = e * E

            # Apply rules
            result = simplify(prod; registry=reg)

            # Should produce delta^I_J (frame delta)
            @test result isa Tensor
            @test result.name == :delta_frame
            @test length(result.indices) == 2
            # Check index structure: one Up Lorentz, one Down Lorentz
            up_idx = findfirst(i -> i.position == Up, result.indices)
            down_idx = findfirst(i -> i.position == Down, result.indices)
            @test up_idx !== nothing
            @test down_idx !== nothing
            @test result.indices[up_idx].vbundle == :Lorentz
            @test result.indices[down_idx].vbundle == :Lorentz
        end
    end

    @testset "Completeness: E^a_I e^I_b -> delta^a_b" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_frame_bundle!(reg; manifold=:M4, dim=4)
            define_tetrad!(reg, :e; manifold=:M4, metric=:g)

            # E^a_I e^I_b (spacetime completeness)
            E = Tensor(:E, [up(:a), frame_down(:I)])
            e = Tensor(:e, [frame_up(:I), down(:b)])
            prod = E * e

            # Apply rules
            result = simplify(prod; registry=reg)

            # Should produce delta^a_b (spacetime delta)
            @test result isa Tensor
            delta_name = get(reg.delta_cache, :M4, :delta)
            @test result.name == delta_name
            @test length(result.indices) == 2
            # Check index structure: one Up Tangent, one Down Tangent
            up_idx = findfirst(i -> i.position == Up, result.indices)
            down_idx = findfirst(i -> i.position == Down, result.indices)
            @test up_idx !== nothing
            @test down_idx !== nothing
        end
    end

    @testset "Metricity: e^I_a e^J_b eta_{IJ} -> g_{ab}" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_frame_bundle!(reg; manifold=:M4, dim=4)
            define_tetrad!(reg, :e; manifold=:M4, metric=:g)

            # e^I_a e^J_b eta_{IJ}
            e1 = Tensor(:e, [frame_up(:I), down(:a)])
            e2 = Tensor(:e, [frame_up(:J), down(:b)])
            eta = Tensor(:eta, [frame_down(:I), frame_down(:J)])
            prod = e1 * e2 * eta

            # Apply rules
            result = simplify(prod; registry=reg)

            # Should produce g_{ab}
            @test result isa Tensor
            @test result.name == :g
            @test length(result.indices) == 2
            # Both indices should be Down Tangent
            @test all(i -> i.position == Down, result.indices)
            @test all(i -> i.vbundle == :Tangent, result.indices)
        end
    end

    @testset "Completeness in larger product" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_frame_bundle!(reg; manifold=:M4, dim=4)
            define_tetrad!(reg, :e; manifold=:M4, metric=:g)

            # T^I * e^I_a * E^a_J should reduce via completeness
            # but e^I_a E^a_J -> delta^I_J, then delta^I_J * T^I -> T^J (via delta contraction)
            T = Tensor(:T, [frame_up(:I)])
            register_tensor!(reg, TensorProperties(
                name=:T, manifold=:M4, rank=(1, 0),
                options=Dict{Symbol,Any}(:vbundle => :Lorentz)))

            e = Tensor(:e, [frame_up(:K), down(:a)])
            E = Tensor(:E, [up(:a), frame_down(:J)])
            prod = e * E * T

            # The completeness rule should fire on e*E
            result = simplify(prod; registry=reg)

            # After completeness: delta^K_J * T^I -> need K=I for contraction
            # Actually the product is T^I e^K_a E^a_J
            # => T^I delta^K_J
            # The delta contraction would apply if K and J contracted elsewhere
            # Just verify the completeness part fires
            @test !(result isa TProduct && any(f -> f isa Tensor && f.name == :e, result isa TProduct ? result.factors : TensorExpr[]))
        end
    end

    @testset "Spacetime metric cache not corrupted" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_frame_bundle!(reg; manifold=:M4, dim=4)
            define_tetrad!(reg, :e; manifold=:M4, metric=:g)

            # After tetrad registration, spacetime caches should be intact
            @test reg.metric_cache[:M4] == :g
            @test reg.metric_cache[:Lorentz] == :eta
        end
    end

end
