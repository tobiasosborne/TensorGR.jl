@testset "Product Manifolds" begin

    # Helper: register a manifold with metric (no δ conflict for multi-manifold setups)
    function _setup_factor!(reg, name, dim, metric, indices)
        register_manifold!(reg, ManifoldProperties(name, dim, metric, :∂, indices))
        register_tensor!(reg, TensorProperties(
            name=metric, manifold=name, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)], is_metric=true))
    end

    @testset "define_product_manifold! registration" begin
        reg = TensorRegistry()
        with_registry(reg) do
            _setup_factor!(reg, :M1, 3, :g1, [:a,:b,:c,:d,:e,:f])
            _setup_factor!(reg, :M2, 4, :g2, [:i,:j,:k,:l,:m,:n])

            pp = define_product_manifold!(reg, :M; factors=[:M1, :M2])

            @test pp isa ProductManifoldProperties
            @test pp.name == :M
            @test pp.factors == [:M1, :M2]
            @test pp.factor_metrics == [:g1, :g2]
            @test pp.factor_dims == [3, 4]
            @test has_product_manifold(reg, :M)
            @test !has_product_manifold(reg, :Nope)
            @test get_product_manifold(reg, :M) === pp
        end
    end

    @testset "factor curvature tensors registered" begin
        reg = TensorRegistry()
        with_registry(reg) do
            _setup_factor!(reg, :M1, 3, :g1, [:a,:b,:c,:d,:e,:f])
            _setup_factor!(reg, :S2, 2, :g2, [:p,:q,:r,:s])

            define_product_manifold!(reg, :M; factors=[:M1, :S2])

            for name in [:Riem_g1, :Ric_g1, :RicScalar_g1, :Ein_g1, :Weyl_g1]
                @test has_tensor(reg, name)
            end
            for name in [:Riem_g2, :Ric_g2, :RicScalar_g2, :Ein_g2, :Weyl_g2]
                @test has_tensor(reg, name)
            end

            rp = get_tensor(reg, :Riem_g1)
            @test rp.manifold == :M1
            @test rp.rank == (0, 4)

            rp2 = get_tensor(reg, :Ric_g2)
            @test rp2.manifold == :S2
            @test rp2.rank == (0, 2)

            rs = get_tensor(reg, :RicScalar_g1)
            @test rs.rank == (0, 0)
        end
    end

    @testset "error handling" begin
        reg = TensorRegistry()
        with_registry(reg) do
            _setup_factor!(reg, :M1, 2, :g1, [:a,:b,:c,:d])

            # Missing factor
            @test_throws ErrorException define_product_manifold!(
                reg, :M; factors=[:M1, :Missing])

            # Single factor
            @test_throws ErrorException define_product_manifold!(
                reg, :M; factors=[:M1])

            # Duplicate registration
            _setup_factor!(reg, :M2, 2, :g2, [:p,:q,:r,:s])
            define_product_manifold!(reg, :M; factors=[:M1, :M2])
            @test_throws ErrorException define_product_manifold!(
                reg, :M; factors=[:M1, :M2])

            # Invalid factor in decomposition
            @test_throws ErrorException product_ricci(:M, :Nope)
        end
    end

    @testset "block-diagonal metric" begin
        reg = TensorRegistry()
        with_registry(reg) do
            _setup_factor!(reg, :M1, 2, :g1, [:a,:b,:c,:d])
            _setup_factor!(reg, :M2, 3, :g2, [:i,:j,:k,:l,:m,:n])

            define_product_manifold!(reg, :M; factors=[:M1, :M2])

            g = product_metric(:M)
            @test g isa TSum
            @test length(g.terms) == 2

            names = sort([t.name for t in g.terms])
            @test names == [:g1, :g2]

            g1_term = first(t for t in g.terms if t.name == :g1)
            @test g1_term.indices == [down(:a), down(:b)]

            g2_term = first(t for t in g.terms if t.name == :g2)
            @test g2_term.indices == [down(:i), down(:j)]
        end
    end

    @testset "scalar curvature: R = R₁ + R₂" begin
        reg = TensorRegistry()
        with_registry(reg) do
            _setup_factor!(reg, :M1, 2, :g1, [:a,:b,:c,:d])
            _setup_factor!(reg, :S2, 2, :g2, [:p,:q,:r,:s])

            define_product_manifold!(reg, :M; factors=[:M1, :S2])

            R = product_scalar_curvature(:M)
            @test R isa TSum
            @test length(R.terms) == 2

            names = sort([t.name for t in R.terms])
            @test names == [:RicScalar_g1, :RicScalar_g2]

            for t in R.terms
                @test isempty(t.indices)
            end
        end
    end

    @testset "Ricci decomposition" begin
        reg = TensorRegistry()
        with_registry(reg) do
            _setup_factor!(reg, :M1, 3, :g1, [:a,:b,:c,:d,:e,:f])
            _setup_factor!(reg, :M2, 4, :g2, [:i,:j,:k,:l,:m,:n])

            define_product_manifold!(reg, :M; factors=[:M1, :M2])

            ric1 = product_ricci(:M, :M1)
            @test ric1 isa Tensor
            @test ric1.name == :Ric_g1
            @test ric1.indices == [down(:a), down(:b)]

            ric2 = product_ricci(:M, :M2)
            @test ric2 isa Tensor
            @test ric2.name == :Ric_g2
            @test ric2.indices == [down(:i), down(:j)]
        end
    end

    @testset "Riemann decomposition" begin
        reg = TensorRegistry()
        with_registry(reg) do
            _setup_factor!(reg, :M1, 4, :g1, [:a,:b,:c,:d,:e,:f])
            _setup_factor!(reg, :M2, 4, :g2, [:i,:j,:k,:l,:m,:n])

            define_product_manifold!(reg, :M; factors=[:M1, :M2])

            riem1 = product_riemann(:M, :M1)
            @test riem1 isa Tensor
            @test riem1.name == :Riem_g1
            @test length(riem1.indices) == 4
            @test all(idx -> idx.position == Down, riem1.indices)
            @test riem1.indices == [down(:a), down(:b), down(:c), down(:d)]

            riem2 = product_riemann(:M, :M2)
            @test riem2.name == :Riem_g2
            @test riem2.indices == [down(:i), down(:j), down(:k), down(:l)]
        end
    end

    @testset "Einstein with cross-scalar terms (two factors)" begin
        reg = TensorRegistry()
        with_registry(reg) do
            _setup_factor!(reg, :M1, 2, :g1, [:a,:b,:c,:d])
            _setup_factor!(reg, :S2, 2, :g2, [:p,:q,:r,:s])

            define_product_manifold!(reg, :M; factors=[:M1, :S2])

            # M1 sector: G_{ab} = Ein_g1_{ab} - ½ RicScalar_g2 · g1_{ab}
            G1 = product_einstein(:M, :M1)
            @test G1 isa TSum
            @test length(G1.terms) == 2

            # S2 sector: G_{pq} = Ein_g2_{pq} - ½ RicScalar_g1 · g2_{pq}
            G2 = product_einstein(:M, :S2)
            @test G2 isa TSum
            @test length(G2.terms) == 2
        end
    end

    @testset "three-factor product" begin
        reg = TensorRegistry()
        with_registry(reg) do
            _setup_factor!(reg, :M1, 2, :g1, [:a,:b,:c,:d])
            _setup_factor!(reg, :M2, 3, :g2, [:i,:j,:k,:l,:m,:n])
            _setup_factor!(reg, :M3, 4, :g3, [:p,:q,:r,:s,:t,:u])

            pp = define_product_manifold!(reg, :M; factors=[:M1, :M2, :M3])
            @test pp.factor_dims == [2, 3, 4]
            @test sum(pp.factor_dims) == 9

            R = product_scalar_curvature(:M)
            @test R isa TSum
            @test length(R.terms) == 3

            # Einstein for M1: cross terms from BOTH M2 and M3
            G1 = product_einstein(:M, :M1)
            @test G1 isa TSum
            @test length(G1.terms) == 2

            G2 = product_einstein(:M, :M2)
            @test G2 isa TSum
        end
    end

    @testset "product_einstein_equations returns all sectors" begin
        reg = TensorRegistry()
        with_registry(reg) do
            _setup_factor!(reg, :M1, 3, :g1, [:a,:b,:c,:d,:e,:f])
            _setup_factor!(reg, :M2, 4, :g2, [:i,:j,:k,:l,:m,:n])

            define_product_manifold!(reg, :M; factors=[:M1, :M2])

            eqs = product_einstein_equations(:M)
            @test eqs isa Dict{Symbol, <:TensorExpr}
            @test haskey(eqs, :M1)
            @test haskey(eqs, :M2)
            @test length(eqs) == 2
        end
    end

    @testset "physics: 2D Einstein vanishes → pure cosmological constant" begin
        # In 2D, G_{ab} = 0 identically.
        # For M = M₁² × M₂², setting Ein_g1 = 0:
        #   G_{ab} = 0 - ½ R₂ g₁_{ab} = -½ R₂ g₁_{ab}
        # The curvature of the other factor acts as a cosmological constant!
        reg = TensorRegistry()
        with_registry(reg) do
            _setup_factor!(reg, :M1, 2, :g1, [:a,:b,:c,:d])
            _setup_factor!(reg, :S2, 2, :g2, [:p,:q,:r,:s])

            define_product_manifold!(reg, :M; factors=[:M1, :S2])
            set_vanishing!(reg, :Ein_g1)

            G1 = product_einstein(:M, :M1)
            result = simplify(G1; registry=reg)

            # Should reduce to -½ RicScalar_g2 · g1_{ab}
            @test result isa TProduct
            @test result.scalar == -1 // 2

            factor_names = sort([f.name for f in result.factors if f isa Tensor])
            @test :RicScalar_g2 in factor_names
            @test :g1 in factor_names
        end
    end

    @testset "idempotent: shared factor across products" begin
        reg = TensorRegistry()
        with_registry(reg) do
            _setup_factor!(reg, :M1, 2, :g1, [:a,:b,:c,:d])
            _setup_factor!(reg, :M2, 3, :g2, [:i,:j,:k,:l,:m,:n])
            _setup_factor!(reg, :M3, 2, :g3, [:p,:q,:r,:s])

            define_product_manifold!(reg, :P1; factors=[:M1, :M2])
            @test has_tensor(reg, :Riem_g1)

            # M1 appears in another product — no double-registration error
            define_product_manifold!(reg, :P2; factors=[:M1, :M3])
            @test has_tensor(reg, :Riem_g1)
            @test has_tensor(reg, :Riem_g3)
        end
    end

end
