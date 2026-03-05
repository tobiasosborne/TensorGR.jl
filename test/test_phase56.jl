@testset "Phase 5: Curvature Conversions" begin

    @testset "Riemann-Weyl decomposition structure" begin
        a, b, c, d = down(:a), down(:b), down(:c), down(:d)

        # riemann_to_weyl should return an expression with Weyl + Ricci terms
        result = riemann_to_weyl(a, b, c, d, :g; dim=4)
        @test result isa TSum

        # weyl_to_riemann should return Riemann - Ricci terms
        result2 = weyl_to_riemann(a, b, c, d, :g; dim=4)
        @test result2 isa TSum
    end

    @testset "Einstein-Ricci conversion" begin
        a, b = down(:a), down(:b)

        # ricci_to_einstein: R_{ab} = G_{ab} + (1/2) g_{ab} R
        result = ricci_to_einstein(a, b, :g)
        @test result isa TSum

        # einstein_to_ricci: G_{ab} = R_{ab} - (1/2) g_{ab} R
        result2 = einstein_to_ricci(a, b, :g)
        @test result2 isa TSum
    end
end

@testset "Phase 6: Exterior Calculus" begin

    @testset "6.1: Define forms" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :∂, [:a,:b,:c,:d,:e,:f]))

        with_registry(reg) do
            define_form!(reg, :ω; manifold=:M4, degree=2)
            @test has_tensor(reg, :ω)
            props = get_tensor(reg, :ω)
            @test props.rank == (0, 2)
            @test form_degree(reg, :ω) == 2
            # Should have antisymmetry
            @test length(props.symmetries) == 1
            @test props.symmetries[1] isa AntiSymmetric
        end
    end

    @testset "6.1: Define 0-form and 1-form" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :∂, [:a,:b,:c,:d]))

        with_registry(reg) do
            define_form!(reg, :f; manifold=:M4, degree=0)
            @test get_tensor(reg, :f).rank == (0, 0)

            define_form!(reg, :A; manifold=:M4, degree=1)
            @test get_tensor(reg, :A).rank == (0, 1)
            @test form_degree(reg, :A) == 1
        end
    end

    @testset "6.2: Wedge product" begin
        α = Tensor(:α, [down(:a)])
        β = Tensor(:β, [down(:b)])

        # α ∧ β for two 1-forms: coefficient = 2!/(1!1!) = 2
        result = wedge(α, β, 1, 1)
        @test result isa TProduct
        @test result.scalar == 2 // 1
    end

    @testset "6.2: Interior product" begin
        v = Tensor(:v, [up(:a)])
        ω = Tensor(:ω, [down(:b), down(:c)])

        result = interior_product(v, ω)
        # Should be v^{dummy} ω_{dummy, c}
        @test result isa TProduct
    end

    @testset "6.3: Exterior derivative" begin
        α = Tensor(:α, [down(:b)])
        result = exterior_d(α, 1, down(:a))
        @test result isa TDeriv
        @test result.index == down(:a)
        @test result.arg == α
    end
end
