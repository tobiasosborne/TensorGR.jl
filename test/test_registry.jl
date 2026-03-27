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

# ── Symbolic manifold dimensions (session 15) ────────────────────────

@testset "Symbolic dimension manifolds" begin
    using TensorGR: ManifoldProperties, VBundleProperties, TensorRegistry,
                    register_manifold!, has_manifold, get_manifold,
                    define_vbundle!, has_vbundle, get_vbundle,
                    with_registry

    @testset "ManifoldProperties accepts symbolic dim" begin
        mp = ManifoldProperties(:Md, :d, :g, nothing, [:a, :b, :c])
        @test mp.dim === :d
        @test mp.name === :Md
    end

    @testset "register symbolic-dim manifold" begin
        reg = TensorRegistry()
        mp = ManifoldProperties(:Md, :d, :g, nothing, [:a, :b, :c])
        register_manifold!(reg, mp)
        @test has_manifold(reg, :Md)
        @test get_manifold(reg, :Md).dim === :d
    end

    @testset "VBundleProperties accepts symbolic dim" begin
        reg = TensorRegistry()
        mp = ManifoldProperties(:Md, :d, :g, nothing, [:a, :b, :c])
        register_manifold!(reg, mp)
        define_vbundle!(reg, :V; manifold=:Md, dim=:n, indices=[:I, :J, :K])
        @test has_vbundle(reg, :V)
        @test get_vbundle(reg, :V).dim === :n
    end

    @testset "mixed int and symbolic dims coexist" begin
        reg = TensorRegistry()
        with_registry(reg) do
            # Concrete dim
            mp4 = ManifoldProperties(:M4, 4, :g, nothing, [:a, :b, :c, :d])
            register_manifold!(reg, mp4)
            # Symbolic dim
            mpd = ManifoldProperties(:Md, :d, :h, nothing, [:p, :q, :r, :s])
            register_manifold!(reg, mpd)

            @test get_manifold(reg, :M4).dim == 4
            @test get_manifold(reg, :Md).dim === :d
        end
    end
end
