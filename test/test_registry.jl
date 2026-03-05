@testset "ManifoldProperties" begin
    mp = ManifoldProperties(:M4, 4, nothing, nothing, [:a, :b, :c, :d, :e, :f])
    @test mp.name == :M4
    @test mp.dim == 4
    @test mp.metric === nothing
    @test mp.derivative === nothing
    @test length(mp.indices) == 6

    mp2 = ManifoldProperties(:M4, 4, :g, :∇, [:a, :b, :c, :d])
    @test mp2.metric == :g
    @test mp2.derivative == :∇
end

@testset "TensorProperties" begin
    tp = TensorProperties(
        name=:R,
        manifold=:M4,
        rank=(0, 4),
        symmetries=Any[],
        dependencies=Symbol[],
        weight=0,
        options=Dict{Symbol,Any}()
    )
    @test tp.name == :R
    @test tp.manifold == :M4
    @test tp.rank == (0, 4)
    @test isempty(tp.symmetries)
    @test tp.weight == 0

    tp2 = TensorProperties(
        name=:g,
        manifold=:M4,
        rank=(0, 2),
        symmetries=Any[],
        dependencies=Symbol[],
        weight=0,
        options=Dict{Symbol,Any}(:is_metric => true)
    )
    @test tp2.options[:is_metric] == true
end

@testset "TensorRegistry basic operations" begin
    reg = TensorRegistry()

    # Register a manifold
    register_manifold!(reg, ManifoldProperties(:M4, 4, nothing, nothing, [:a,:b,:c,:d,:e,:f]))
    @test has_manifold(reg, :M4)
    @test !has_manifold(reg, :M3)

    mp = get_manifold(reg, :M4)
    @test mp.dim == 4

    # Register a tensor
    tp = TensorProperties(
        name=:g,
        manifold=:M4,
        rank=(0, 2),
        symmetries=Any[],
        dependencies=Symbol[],
        weight=0,
        options=Dict{Symbol,Any}(:is_metric => true)
    )
    register_tensor!(reg, tp)
    @test has_tensor(reg, :g)
    @test !has_tensor(reg, :R)

    gp = get_tensor(reg, :g)
    @test gp.rank == (0, 2)
    @test gp.options[:is_metric] == true
end

@testset "TensorRegistry duplicate handling" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, nothing, nothing, [:a,:b,:c,:d]))

    # Duplicate manifold registration should error
    @test_throws ErrorException register_manifold!(reg, ManifoldProperties(:M4, 3, nothing, nothing, [:a,:b,:c]))

    # Duplicate tensor registration should error
    tp = TensorProperties(name=:g, manifold=:M4, rank=(0,2), symmetries=Any[],
                          dependencies=Symbol[], weight=0, options=Dict{Symbol,Any}())
    register_tensor!(reg, tp)
    @test_throws ErrorException register_tensor!(reg, tp)
end

@testset "Global registry and with_registry" begin
    # Get the global registry
    reg = current_registry()
    @test reg isa TensorRegistry

    # with_registry should provide an isolated context
    isolated = TensorRegistry()
    register_manifold!(isolated, ManifoldProperties(:TestM, 3, nothing, nothing, [:x,:y,:z]))

    result = with_registry(isolated) do
        @test has_manifold(current_registry(), :TestM)
        current_registry()
    end
    @test result === isolated

    # After with_registry, the global registry should be restored
    @test !has_manifold(current_registry(), :TestM)
end

@testset "Registry lookup errors" begin
    reg = TensorRegistry()
    @test_throws KeyError get_manifold(reg, :nonexistent)
    @test_throws KeyError get_tensor(reg, :nonexistent)
end
