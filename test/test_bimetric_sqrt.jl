using Test
using TensorGR

@testset "Bimetric Matrix Square Root" begin
    # Helper: collect all Tensor names from an expression
    function _tensor_names(expr::TensorExpr)
        names = Set{Symbol}()
        _collect_names!(names, expr)
        names
    end
    _collect_names!(s::Set{Symbol}, t::Tensor) = push!(s, t.name)
    _collect_names!(::Set{Symbol}, ::TScalar) = nothing
    function _collect_names!(s::Set{Symbol}, p::TProduct)
        for f in p.factors; _collect_names!(s, f); end
    end
    function _collect_names!(s::Set{Symbol}, ts::TSum)
        for t in ts.terms; _collect_names!(s, t); end
    end
    function _collect_names!(s::Set{Symbol}, d::TDeriv)
        _collect_names!(s, d.arg)
    end

    # Helper: create a fresh bimetric setup
    function _sqrt_setup()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
        end
        bs = define_bimetric!(reg, :g, :f; manifold=:M4)
        reg, bs
    end

    @testset "sqrt_matrix_identity: well-formed expression" begin
        reg, bs = _sqrt_setup()
        with_registry(reg) do
            identity = TensorGR.sqrt_matrix_identity(bs; registry=reg)
            @test identity isa TensorExpr

            # Should contain S tensors (S_g_f)
            names = _tensor_names(identity)
            @test :S_g_f in names
            # Should contain metric g and/or f
            @test :g in names || :f in names
        end
    end

    @testset "sqrt_matrix_identity: free indices" begin
        reg, bs = _sqrt_setup()
        with_registry(reg) do
            identity = TensorGR.sqrt_matrix_identity(bs; registry=reg)
            fi = free_indices(identity)
            # The identity S^a_c S^c_b - g^{ac} f_{cb} has exactly 2 free indices
            @test length(fi) == 2
        end
    end

    @testset "sqrt_matrix_identity: index structure" begin
        reg, bs = _sqrt_setup()
        with_registry(reg) do
            identity = TensorGR.sqrt_matrix_identity(bs; registry=reg)
            fi = free_indices(identity)
            # Should have one Up and one Down free index
            ups = count(idx -> idx.position == Up, fi)
            downs = count(idx -> idx.position == Down, fi)
            @test ups == 1
            @test downs == 1
        end
    end

    @testset "sqrt_matrix_identity: is a sum (S^2 - g^{-1}f)" begin
        reg, bs = _sqrt_setup()
        with_registry(reg) do
            identity = TensorGR.sqrt_matrix_identity(bs; registry=reg)
            # S^2 - g^{-1}f should be a TSum with 2 terms
            @test identity isa TSum
            @test length(identity.terms) == 2
        end
    end

    @testset "cayley_hamilton_S: well-formed expression" begin
        reg, bs = _sqrt_setup()
        params = HassanRosenParams(m_sq=1, beta0=1, beta1=1, beta2=1, beta3=1, beta4=1)
        with_registry(reg) do
            ch = TensorGR.cayley_hamilton_S(bs, params; registry=reg)
            @test ch isa TensorExpr
        end
    end

    @testset "cayley_hamilton_S: free indices" begin
        reg, bs = _sqrt_setup()
        params = HassanRosenParams(m_sq=1, beta0=1, beta1=1, beta2=1, beta3=1, beta4=1)
        with_registry(reg) do
            ch = TensorGR.cayley_hamilton_S(bs, params; registry=reg)
            fi = free_indices(ch)
            # Matrix equation: 2 free indices (one up, one down)
            @test length(fi) == 2
            ups = count(idx -> idx.position == Up, fi)
            downs = count(idx -> idx.position == Down, fi)
            @test ups == 1
            @test downs == 1
        end
    end

    @testset "cayley_hamilton_S: contains S powers and e_n" begin
        reg, bs = _sqrt_setup()
        params = HassanRosenParams(m_sq=1, beta0=1, beta1=1, beta2=1, beta3=1, beta4=1)
        with_registry(reg) do
            ch = TensorGR.cayley_hamilton_S(bs, params; registry=reg)
            # Should be a TSum with 5 terms: S^4, -e1*S^3, +e2*S^2, -e3*S, +e4*I
            @test ch isa TSum
            @test length(ch.terms) == 5
        end
    end

    @testset "cayley_hamilton_S: with specific beta values" begin
        reg, bs = _sqrt_setup()
        # Only beta1 nonzero => e_n involve only Tr(S)-related terms
        params = HassanRosenParams(m_sq=1, beta0=0, beta1=1, beta2=0, beta3=0, beta4=0)
        with_registry(reg) do
            ch = TensorGR.cayley_hamilton_S(bs, params; registry=reg)
            @test ch isa TensorExpr
            fi = free_indices(ch)
            @test length(fi) == 2
        end
    end

    @testset "register_sqrt_rules!: creates rules" begin
        reg, bs = _sqrt_setup()
        with_registry(reg) do
            rules = TensorGR.register_sqrt_rules!(reg, bs)
            @test rules isa Vector{RewriteRule}
            @test length(rules) == 1
        end
    end

    @testset "register_sqrt_rules!: rules added to registry" begin
        reg, bs = _sqrt_setup()
        initial_rules = length(reg.rules)
        with_registry(reg) do
            TensorGR.register_sqrt_rules!(reg, bs)
        end
        @test length(reg.rules) == initial_rules + 1
    end

    @testset "sqrt_matrix_variation: well-formed expression" begin
        reg, bs = _sqrt_setup()
        with_registry(reg) do
            var = TensorGR.sqrt_matrix_variation(bs; registry=reg)
            @test var isa TensorExpr
        end
    end

    @testset "sqrt_matrix_variation: free indices" begin
        reg, bs = _sqrt_setup()
        with_registry(reg) do
            var = TensorGR.sqrt_matrix_variation(bs; registry=reg)
            fi = free_indices(var)
            # Matrix equation: 2 free indices (one up, one down)
            @test length(fi) == 2
            ups = count(idx -> idx.position == Up, fi)
            downs = count(idx -> idx.position == Down, fi)
            @test ups == 1
            @test downs == 1
        end
    end

    @testset "sqrt_matrix_variation: contains S, deltaS, delta_g, delta_f" begin
        reg, bs = _sqrt_setup()
        with_registry(reg) do
            var = TensorGR.sqrt_matrix_variation(bs; registry=reg)
            names = _tensor_names(var)
            @test :S_g_f in names
            @test :deltaS_g_f in names
        end
    end

    @testset "sqrt_matrix_variation: registers delta tensors" begin
        reg, bs = _sqrt_setup()
        with_registry(reg) do
            TensorGR.sqrt_matrix_variation(bs; registry=reg)
            @test has_tensor(reg, :deltaS_g_f)
            @test get_tensor(reg, :deltaS_g_f).rank == (1, 1)
        end
    end

    @testset "sqrt_matrix_variation: is a sum (4 terms)" begin
        reg, bs = _sqrt_setup()
        with_registry(reg) do
            var = TensorGR.sqrt_matrix_variation(bs; registry=reg)
            # S*dS + dS*S - g^{-1}*df + g^{-1}*dg*g^{-1}*f = 4 terms
            @test var isa TSum
            @test length(var.terms) == 4
        end
    end

    @testset "_S_power_chain: n=0 gives delta" begin
        used = Set{Symbol}([:a, :b])
        chain = TensorGR._S_power_chain(:S_g_f, 0, :a, :b, used)
        @test chain isa Tensor
        @test chain.name == :δ
        @test chain.indices == [up(:a), down(:b)]
    end

    @testset "_S_power_chain: n=1 gives S" begin
        used = Set{Symbol}([:a, :b])
        chain = TensorGR._S_power_chain(:S_g_f, 1, :a, :b, used)
        @test chain isa Tensor
        @test chain.name == :S_g_f
        @test chain.indices == [up(:a), down(:b)]
    end

    @testset "_S_power_chain: n=2 is contracted product" begin
        used = Set{Symbol}([:a, :b])
        chain = TensorGR._S_power_chain(:S_g_f, 2, :a, :b, used)
        @test chain isa TProduct
        @test length(chain.factors) == 2
        # Both factors should be S_g_f
        @test all(f -> f isa Tensor && f.name == :S_g_f, chain.factors)
        # First factor upper index = :a, last factor lower index = :b
        @test chain.factors[1].indices[1] == up(:a)
        @test chain.factors[2].indices[2] == down(:b)
        # Internal contraction: down of first == up of second
        @test chain.factors[1].indices[2].name == chain.factors[2].indices[1].name
        @test chain.factors[1].indices[2].position == Down
        @test chain.factors[2].indices[1].position == Up
    end

    @testset "_S_power_chain: n=4 has 4 factors" begin
        used = Set{Symbol}([:a, :b])
        chain = TensorGR._S_power_chain(:S_g_f, 4, :a, :b, used)
        @test chain isa TProduct
        @test length(chain.factors) == 4
    end
end
