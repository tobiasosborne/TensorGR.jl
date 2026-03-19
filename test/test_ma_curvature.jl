# Ground truth: Hehl et al, Phys. Rep. 258 (1995), Sec 3.

@testset "Metric-affine curvature" begin

    function _mac_reg()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            @covd D on=M4 metric=g
        end
        ac = define_affine_connection!(reg, :Gamma; manifold=:M4, metric=:g)
        reg, ac
    end

    @testset "define_ma_curvature! registration" begin
        reg, ac = _mac_reg()
        fs = define_ma_curvature!(reg, ac)

        @test fs isa MAFieldStrength
        @test has_tensor(reg, fs.riemann_name)
        @test has_tensor(reg, fs.ricci_name)
        @test has_tensor(reg, fs.scalar_name)
    end

    @testset "MA Riemann: antisymmetric in last two only" begin
        reg, ac = _mac_reg()
        fs = define_ma_curvature!(reg, ac)
        props = get_tensor(reg, fs.riemann_name)

        @test props.rank == (1, 3)
        @test props.options[:is_metric_affine]
        @test props.options[:no_pair_symmetry]
        # Has antisymmetry in slots 3,4
        @test !isempty(props.symmetries)
    end

    @testset "MA Ricci: asymmetric" begin
        reg, ac = _mac_reg()
        fs = define_ma_curvature!(reg, ac)
        props = get_tensor(reg, fs.ricci_name)

        @test props.rank == (0, 2)
        @test props.options[:asymmetric]
        # NO symmetry (unlike Riemannian Ricci)
        @test isempty(props.symmetries)
    end

    @testset "MA Riemann decomposition structure" begin
        reg, ac = _mac_reg()
        with_registry(reg) do
            decomp = ma_riemann_decomposition(ac)
            @test decomp isa TSum
            # 5 terms: R(LC) + ∇N - ∇N + NN - NN
            @test length(decomp.terms) == 5

            # Free indices: a (Up), b, c, d (Down)
            free = free_indices(decomp)
            @test length(free) == 4
        end
    end

    @testset "Decomposition contains Riemannian curvature" begin
        reg, ac = _mac_reg()
        with_registry(reg) do
            decomp = ma_riemann_decomposition(ac)
            # First term should be R^a_{bcd}(LC) = standard Riem
            has_riem = any(t -> begin
                t isa Tensor && t.name == :Riem
            end, decomp.terms)
            @test has_riem
        end
    end

    @testset "Idempotent registration" begin
        reg, ac = _mac_reg()
        fs1 = define_ma_curvature!(reg, ac)
        fs2 = define_ma_curvature!(reg, ac)
        @test fs1.riemann_name == fs2.riemann_name
    end

    @testset "Display" begin
        reg, ac = _mac_reg()
        fs = define_ma_curvature!(reg, ac)
        s = sprint(show, fs)
        @test occursin("MAFieldStrength", s)
    end
end
