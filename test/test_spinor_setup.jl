# Tests for define_spinor_structure! and @spinor_manifold one-liner.

@testset "Spinor structure setup (TGR-1rf)" begin

    @testset "define_spinor_structure! registers everything" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_spinor_structure!(reg; manifold=:M4, metric=:g)

            # VBundles
            @test has_vbundle(reg, :SL2C)
            @test has_vbundle(reg, :SL2C_dot)

            # Spin metrics
            @test has_tensor(reg, :eps_spin)
            @test has_tensor(reg, :eps_spin_dot)

            # Spin deltas
            @test has_tensor(reg, :delta_spin)
            @test has_tensor(reg, :delta_spin_dot)

            # Soldering form
            @test has_tensor(reg, :sigma)

            # Cache entries
            @test reg.metric_cache[:SL2C] == :eps_spin
            @test reg.metric_cache[:SL2C_dot] == :eps_spin_dot
            @test reg.delta_cache[:SL2C] == :delta_spin
            @test reg.delta_cache[:SL2C_dot] == :delta_spin_dot

            # Spacetime metric not clobbered
            @test reg.metric_cache[:M4] == :g
        end
    end

    @testset "@spinor_manifold macro works" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            @spinor_manifold M4 metric=g

            @test has_vbundle(reg, :SL2C)
            @test has_tensor(reg, :eps_spin)
            @test has_tensor(reg, :sigma)
        end
    end

    @testset "Error if manifold not registered" begin
        reg = TensorRegistry()
        @test_throws ErrorException define_spinor_structure!(reg; manifold=:M4, metric=:g)
    end

    @testset "Error if metric not registered" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
        end
        @test_throws ErrorException define_spinor_structure!(reg; manifold=:M4, metric=:h)
    end

    @testset "Sigma completeness works after one-liner setup" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_spinor_structure!(reg; manifold=:M4, metric=:g)

            sig1 = Tensor(:sigma, [up(:a), spin_down(:A), spin_dot_down(:Ap)])
            sig2 = Tensor(:sigma, [down(:a), spin_up(:B), spin_dot_up(:Bp)])
            result = simplify(sig1 * sig2; registry=reg)

            # Should be delta^B_A * delta^{B'}_{A'}
            @test result isa TProduct
            names = Set(f.name for f in result.factors if f isa Tensor)
            @test :delta_spin in names
            @test :delta_spin_dot in names
        end
    end

    @testset "Idempotent: calling twice does not error" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_spinor_structure!(reg; manifold=:M4, metric=:g)
            define_spinor_structure!(reg; manifold=:M4, metric=:g)
            @test has_tensor(reg, :sigma)
        end
    end

end
