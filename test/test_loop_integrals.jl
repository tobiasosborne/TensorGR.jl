@testset "Loop integral representation" begin

    @testset "PropagatorDenom construction" begin
        pd = PropagatorDenom(:q, 0)
        @test pd.momentum == :q
        @test pd.mass_sq == 0
        @test pd.power == 1

        pd2 = PropagatorDenom(:q, :m2, 2)
        @test pd2.mass_sq == :m2
        @test pd2.power == 2
    end

    @testset "PropagatorDenom display" begin
        pd = PropagatorDenom(:q, 0)
        s = sprint(show, pd)
        @test occursin("q²", s)

        pd2 = PropagatorDenom(:q, :m2, 2)
        s2 = sprint(show, pd2)
        @test occursin("m2", s2)
        @test occursin("^2", s2)
    end

    @testset "MomentumIntegral construction" begin
        mi = MomentumIntegral([:q], TScalar(1),
            [PropagatorDenom(:q, 0), PropagatorDenom(:q_minus_k, 0)])
        @test length(mi.loop_momenta) == 1
        @test length(mi.denominators) == 2
        @test mi.dim == 4
    end

    @testset "MomentumIntegral display" begin
        mi = massless_bubble(:k2)
        s = sprint(show, mi)
        @test occursin("1-loop", s)
        @test occursin("bubble", s)
    end

    @testset "ScalarIntegral construction" begin
        si = ScalarIntegral(:B0, [0, 0], [:k2], 4)
        @test si.topology == :B0
        @test si.dim == 4
    end

    @testset "integral_topology" begin
        @test integral_topology(0) == :scaleless
        @test integral_topology(1) == :tadpole
        @test integral_topology(2) == :bubble
        @test integral_topology(3) == :triangle
        @test integral_topology(4) == :box
        @test integral_topology(5) == :pentagon
        @test integral_topology(6) == :hexagon
    end

    @testset "integral_topology from MomentumIntegral" begin
        bubble = massless_bubble(:k2)
        @test integral_topology(bubble) == :bubble

        tri = massless_triangle()
        @test integral_topology(tri) == :triangle
    end

    @testset "pv_topology" begin
        tadpole = MomentumIntegral([:q], TScalar(1),
            [PropagatorDenom(:q, 0)])
        @test pv_topology(tadpole) == :A0

        bubble = massless_bubble(:k2)
        @test pv_topology(bubble) == :B0

        tri = massless_triangle()
        @test pv_topology(tri) == :C0
    end

    @testset "massless_bubble" begin
        b = massless_bubble(:k2)
        @test length(b.loop_momenta) == 1
        @test b.loop_momenta[1] == :q
        @test length(b.denominators) == 2
        @test b.denominators[1].mass_sq == 0
        @test b.denominators[2].mass_sq == 0
        @test b.dim == 4
    end

    @testset "massless_bubble: dim-reg" begin
        b = massless_bubble(:k2; dim=:d)
        @test b.dim == :d
    end

    @testset "massless_triangle" begin
        t = massless_triangle()
        @test length(t.denominators) == 3
        @test integral_topology(t) == :triangle
    end

    @testset "to_momentum_integral" begin
        denoms = [PropagatorDenom(:q, 0)]
        mi = to_momentum_integral(TScalar(1), denoms, [:q])
        @test mi isa MomentumIntegral
        @test integral_topology(mi) == :tadpole
    end

    @testset "n_loops" begin
        b = massless_bubble(:k2)
        @test TensorGR.n_loops(b) == 1

        two_loop = MomentumIntegral([:q1, :q2], TScalar(1),
            [PropagatorDenom(:q1, 0)])
        @test TensorGR.n_loops(two_loop) == 2
    end

    @testset "total_propagator_power" begin
        b = massless_bubble(:k2)
        @test total_propagator_power(b) == 2

        # Higher-power propagator
        mi = MomentumIntegral([:q], TScalar(1),
            [PropagatorDenom(:q, 0, 3)])
        @test total_propagator_power(mi) == 3
    end

    @testset "superficial_divergence" begin
        # Bubble in d=4: ω = 4*1 - 2*2 = 0 (log divergent)
        b = massless_bubble(:k2; dim=4)
        @test superficial_divergence(b) == 0

        # Triangle in d=4: ω = 4*1 - 2*3 = -2 (UV finite)
        t = massless_triangle(; dim=4)
        @test superficial_divergence(t) == -2

        # Tadpole in d=4: ω = 4*1 - 2*1 = 2 (quadratic divergence)
        td = MomentumIntegral([:q], TScalar(1),
            [PropagatorDenom(:q, 0)]; dim=4)
        @test superficial_divergence(td) == 2
    end

    @testset "dimreg_trace" begin
        @test dimreg_trace(4) == 4
        @test dimreg_trace(:d) == :d
    end

end
