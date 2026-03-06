@testset "Phase 3: Perturbation Theory" begin

    @testset "3.1: Partitions" begin
        # Partitions of 4
        p4 = sorted_partitions(4)
        @test [4] in p4
        @test [3, 1] in p4
        @test [2, 2] in p4
        @test [2, 1, 1] in p4
        @test [1, 1, 1, 1] in p4
        @test length(p4) == 5

        # Partitions of 0
        @test sorted_partitions(0) == [Int[]]

        # Partitions of 1
        @test sorted_partitions(1) == [[1]]
    end

    @testset "3.1: Compositions" begin
        # All ways to write 3 as sum of 2 non-negative integers
        c32 = all_compositions(3, 2)
        @test length(c32) == 4  # (0,3), (1,2), (2,1), (3,0)
        @test [0, 3] in c32
        @test [3, 0] in c32

        # All ways to write 2 as sum of 3
        c23 = all_compositions(2, 3)
        @test length(c23) == 6  # (0,0,2), (0,1,1), (0,2,0), (1,0,1), (1,1,0), (2,0,0)
    end

    @testset "3.1: Multinomial" begin
        @test multinomial(4, [2, 2]) == 6
        @test multinomial(4, [1, 1, 1, 1]) == 24
        @test multinomial(3, [3]) == 1
        @test multinomial(3, [1, 2]) == 3
    end

    @testset "3.2: DefMetricPerturbation" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0,2),
            symmetries=Any[Symmetric(1,2)],
            options=Dict{Symbol,Any}(:is_metric => true)))

        with_registry(reg) do
            mp = define_metric_perturbation!(reg, :g, :h)
            @test mp.metric == :g
            @test mp.perturbation == :h
            @test has_tensor(reg, :h)
            props = get_tensor(reg, :h)
            @test props.rank == (0, 2)
            @test length(props.symmetries) == 1
        end
    end

    @testset "3.2: Perturb metric" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0,2),
            symmetries=Any[Symmetric(1,2)],
            options=Dict{Symbol,Any}(:is_metric => true)))

        with_registry(reg) do
            mp = define_metric_perturbation!(reg, :g, :h)

            g_ab = Tensor(:g, [down(:a), down(:b)])

            # Order 0: δ⁰(g) = g
            @test perturb(g_ab, mp, 0) == g_ab

            # Order 1: δ¹(g_{ab}) = h_{ab}
            result = perturb(g_ab, mp, 1)
            @test result == Tensor(:h, [down(:a), down(:b)])

            # Order 2: δ²(g_{ab}) = 0 (only first order perturbation of g itself)
            @test perturb(g_ab, mp, 2) == TScalar(0 // 1)
        end
    end

    @testset "3.2: Perturb product (Leibniz)" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0,2),
            symmetries=Any[Symmetric(1,2)],
            options=Dict{Symbol,Any}(:is_metric => true)))

        with_registry(reg) do
            mp = define_metric_perturbation!(reg, :g, :h)

            # Perturb g_{ab} * g_{cd} at order 1
            g_ab = Tensor(:g, [down(:a), down(:b)])
            g_cd = Tensor(:g, [down(:c), down(:d)])
            product = g_ab * g_cd
            result = perturb(product, mp, 1)

            # Should be h_{ab}*g_{cd} + g_{ab}*h_{cd}
            @test result isa TSum
            @test length(result.terms) == 2
        end
    end

    @testset "3.2: δ(g^{ab}) at order 1" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0,2),
            symmetries=Any[Symmetric(1,2)],
            options=Dict{Symbol,Any}(:is_metric => true)))

        with_registry(reg) do
            mp = define_metric_perturbation!(reg, :g, :h)

            # δ(g^{ab}) = -g^{ac} g^{bd} h_{cd}
            result = δinverse_metric(mp, up(:a), up(:b), 1)
            @test result isa TProduct
            @test result.scalar == -1 // 1
            @test length(result.factors) == 3
        end
    end

    @testset "3.6: Curved background perturbation" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)

            # Curved background registers Γ₀
            mp = define_metric_perturbation!(reg, :g, :h; curved=true)
            @test mp.curved == true
            @test mp.background_christoffel !== nothing
            @test has_tensor(reg, mp.background_christoffel)

            # δΓ on curved bg has extra ∂g₀ term vs flat
            mp_flat = define_metric_perturbation!(reg, :g, :h; curved=false)
            δΓ_flat = δchristoffel(mp_flat, up(:a), down(:b), down(:c), 1)
            δΓ_curved = δchristoffel(mp, up(:a), down(:b), down(:c), 1)
            # Curved has extra term from l=0 partition
            nf = δΓ_flat isa TSum ? length(δΓ_flat.terms) : 1
            nc = δΓ_curved isa TSum ? length(δΓ_curved.terms) : 1
            @test nc > nf

            # δR on curved bg has Γ₀ cross terms
            δR_flat = δriemann(mp_flat, up(:a), down(:b), down(:c), down(:d), 1)
            δR_curved = δriemann(mp, up(:a), down(:b), down(:c), down(:d), 1)
            nrf = δR_flat isa TSum ? length(δR_flat.terms) : 1
            nrc = δR_curved isa TSum ? length(δR_curved.terms) : 1
            @test nrc > nrf  # curved has extra Γ₀·δΓ cross terms

            # Flat bg behavior unchanged
            @test δΓ_flat isa TProduct  # single term on flat
        end
    end
end
