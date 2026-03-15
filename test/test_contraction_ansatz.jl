using TensorGR: contraction_ansatz, free_indices

@testset "contraction_ansatz: single metric -> dimension scalar" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d,:e,:f]))
    define_metric!(reg, :g; manifold=:M4)

    with_registry(reg) do
        result = contraction_ansatz([:g], :g; registry=reg)
        # g^{ab} g_{ab} = delta^a_a = dim = 4, a single scalar term
        # After simplification this is just a number, so 1 term with coefficient
        @test result isa TensorExpr
        terms = result isa TSum ? result.terms : [result]
        @test length(terms) == 1
        # Each term should be a scalar (no free indices)
        for t in terms
            @test isempty(free_indices(t))
        end
    end
end

@testset "contraction_ansatz: Ric x Ric -> 2 invariants" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d,:e,:f]))
    define_metric!(reg, :g; manifold=:M4)

    with_registry(reg) do
        result = contraction_ansatz([:Ric, :Ric], :g; registry=reg)
        # Two independent quadratic Ricci invariants: R_{ab}R^{ab} and R^2
        @test result isa TSum
        @test length(result.terms) == 2
        for t in result.terms
            @test isempty(free_indices(t))
        end
    end
end

@testset "contraction_ansatz: coefficients are distinct symbols" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d,:e,:f]))
    define_metric!(reg, :g; manifold=:M4)

    with_registry(reg) do
        result = contraction_ansatz([:Ric, :Ric], :g; registry=reg)
        @test result isa TSum
        # Extract coefficient symbols from each term
        coeffs = Symbol[]
        for t in result.terms
            # Each term is TScalar(ci) * contraction, i.e. a TProduct
            @test t isa TProduct
            scalar_factors = filter(f -> f isa TScalar && f.val isa Symbol, t.factors)
            @test length(scalar_factors) >= 1
            push!(coeffs, scalar_factors[1].val)
        end
        # All coefficients should be distinct
        @test length(unique(coeffs)) == length(coeffs)
    end
end

@testset "contraction_ansatz: custom coeff_prefix" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d,:e,:f]))
    define_metric!(reg, :g; manifold=:M4)

    with_registry(reg) do
        result = contraction_ansatz([:Ric, :Ric], :g; registry=reg, coeff_prefix=:alpha)
        @test result isa TSum
        # Check that coefficient names start with "alpha"
        for t in result.terms
            @test t isa TProduct
            scalar_factors = filter(f -> f isa TScalar && f.val isa Symbol, t.factors)
            @test !isempty(scalar_factors)
            @test startswith(string(scalar_factors[1].val), "alpha")
        end
    end
end

@testset "contraction_ansatz: all terms are scalars" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d,:e,:f]))
    define_metric!(reg, :g; manifold=:M4)
    register_tensor!(reg, TensorProperties(name=:T, manifold=:M4, rank=(0,4),
        symmetries=SymmetrySpec[]))

    with_registry(reg) do
        result = contraction_ansatz([:T], :g; registry=reg)
        terms = result isa TSum ? result.terms : [result]
        @test length(terms) >= 1
        for t in terms
            @test isempty(free_indices(t))
        end
    end
end
