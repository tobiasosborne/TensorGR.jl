#= Tests for invariant_lagrangian: most general diffeomorphism-invariant
   Lagrangian at a given derivative order.

   Ground truth:
     - Nutma (2014), arXiv:1308.3493, Sec 5
     - Fulling, King, Wybourne & Cummins (1992), CQG 9, 1151, Table 1
     - Gauss-Bonnet: K - 4 Ric^2 + R^2 = 0 in d=4

   Key invariant counts:
     - Order 0: 1 (cosmological constant Lambda)
     - Order 2: 1 (Ricci scalar R)
     - Order 4, generic d: 3 (R^2, Ric^2, Kretschner)
     - Order 4, d=4: 2 (Gauss-Bonnet removes Kretschner)
=#

@testset "Invariant Lagrangian (TGR-99d.3)" begin

    # Shared registry builder
    function lagrangian_registry(; dim::Int=4)
        reg = TensorRegistry()
        with_registry(reg) do
            if dim == 4
                @manifold M4 dim=4 metric=g
                define_curvature_tensors!(reg, :M4, :g)
            elseif dim == 5
                @manifold M5 dim=5 metric=g
                define_curvature_tensors!(reg, :M5, :g)
            else
                # Generic setup via direct registration
                register_manifold!(reg, ManifoldProperties(Symbol(:M, dim), dim, :g, :partial,
                    [:a,:b,:c,:d,:e,:f,:g1,:h]))
                define_metric!(reg, :g; manifold=Symbol(:M, dim))
            end
        end
        reg
    end

    # ------------------------------------------------------------------
    # 1. Order 0: cosmological constant
    # ------------------------------------------------------------------
    @testset "Order 0: cosmological constant Lambda" begin
        reg = lagrangian_registry()
        with_registry(reg) do
            L0 = invariant_lagrangian(0; registry=reg)

            # Must be a TScalar with symbolic value :Lambda
            @test L0 isa TScalar
            @test L0.val === :Lambda

            # Scalar => no free indices
            @test isempty(free_indices(L0))
        end
    end

    # ------------------------------------------------------------------
    # 2. Order 2: Ricci scalar with undetermined coefficient
    # ------------------------------------------------------------------
    @testset "Order 2: 1 invariant (Ricci scalar)" begin
        reg = lagrangian_registry()
        with_registry(reg) do
            L2 = invariant_lagrangian(2; registry=reg)

            # Must be a product of TScalar(:c1) and RicScalar
            @test L2 isa TProduct

            # Must contain coefficient c1
            has_c1 = false
            for f in L2.factors
                if f isa TScalar && f.val === :c1
                    has_c1 = true
                end
            end
            @test has_c1

            # Must contain RicScalar
            has_ric_scalar = false
            for f in L2.factors
                if f isa Tensor && f.name == :RicScalar
                    has_ric_scalar = true
                end
            end
            @test has_ric_scalar

            # Must be scalar (no free indices)
            @test isempty(free_indices(L2))
        end
    end

    # ------------------------------------------------------------------
    # 3. Order 4, generic dimension: 3 independent invariants
    # ------------------------------------------------------------------
    @testset "Order 4, generic dim: 3 invariants (R^2, Ric^2, K)" begin
        reg = lagrangian_registry()
        with_registry(reg) do
            L4 = invariant_lagrangian(4; registry=reg)

            # Must be a TSum with 3 terms
            @test L4 isa TSum
            @test length(L4.terms) == 3

            # All terms must be scalar (no free indices)
            for t in L4.terms
                @test isempty(free_indices(t))
            end

            # Collect symbolic coefficients
            coeffs = Symbol[]
            for t in L4.terms
                @test t isa TProduct
                for f in t.factors
                    if f isa TScalar && f.val isa Symbol
                        push!(coeffs, f.val)
                    end
                end
            end
            # Must have 3 distinct coefficients
            @test length(unique(coeffs)) == 3
            @test Set(coeffs) == Set([:c1, :c2, :c3])
        end
    end

    # ------------------------------------------------------------------
    # 4. Order 4, d=4: 2 independent invariants (Gauss-Bonnet)
    # ------------------------------------------------------------------
    @testset "Order 4, d=4: 2 invariants (Gauss-Bonnet removes K)" begin
        reg = lagrangian_registry(dim=4)
        with_registry(reg) do
            L4 = invariant_lagrangian(4; dim=4, registry=reg)

            # Must be a TSum with 2 terms
            @test L4 isa TSum
            @test length(L4.terms) == 2

            # All terms must be scalar (no free indices)
            for t in L4.terms
                @test isempty(free_indices(t))
            end

            # Collect symbolic coefficients
            coeffs = Symbol[]
            for t in L4.terms
                @test t isa TProduct
                for f in t.factors
                    if f isa TScalar && f.val isa Symbol
                        push!(coeffs, f.val)
                    end
                end
            end
            # Must have 2 distinct coefficients
            @test length(unique(coeffs)) == 2
            @test Set(coeffs) == Set([:c1, :c2])

            # Verify no Riem tensor appears (Kretschner eliminated)
            has_riem = false
            walk(L4) do node
                if node isa Tensor && node.name == :Riem
                    has_riem = true
                end
                node
            end
            @test !has_riem
        end
    end

    # ------------------------------------------------------------------
    # 5. Order 4, d=5: 3 invariants (no Gauss-Bonnet reduction)
    # ------------------------------------------------------------------
    @testset "Order 4, d=5: 3 invariants (no DDI reduction)" begin
        reg = lagrangian_registry(dim=5)
        with_registry(reg) do
            L4 = invariant_lagrangian(4; dim=5, registry=reg)

            # d=5: Gauss-Bonnet does not reduce, all 3 remain
            @test L4 isa TSum
            @test length(L4.terms) == 3

            # All terms must be scalar
            for t in L4.terms
                @test isempty(free_indices(t))
            end
        end
    end

    # ------------------------------------------------------------------
    # 6. Invalid orders throw errors
    # ------------------------------------------------------------------
    @testset "Invalid orders" begin
        reg = lagrangian_registry()
        with_registry(reg) do
            # Negative order
            @test_throws ArgumentError invariant_lagrangian(-1; registry=reg)

            # Odd order (no invariants in metric gravity)
            @test_throws ArgumentError invariant_lagrangian(1; registry=reg)
            @test_throws ArgumentError invariant_lagrangian(3; registry=reg)
            @test_throws ArgumentError invariant_lagrangian(5; registry=reg)

            # Order 6+ not yet implemented
            @test_throws ArgumentError invariant_lagrangian(6; registry=reg)
        end
    end

    # ------------------------------------------------------------------
    # 7. Coefficients are symbolic TScalars
    # ------------------------------------------------------------------
    @testset "Coefficients are symbolic TScalars" begin
        reg = lagrangian_registry()
        with_registry(reg) do
            # Order 0: Lambda is a Symbol
            L0 = invariant_lagrangian(0; registry=reg)
            @test L0 isa TScalar
            @test L0.val isa Symbol

            # Order 2: c1 is a Symbol
            L2 = invariant_lagrangian(2; registry=reg)
            has_symbol_coeff = false
            walk(L2) do node
                if node isa TScalar && node.val isa Symbol
                    has_symbol_coeff = true
                end
                node
            end
            @test has_symbol_coeff

            # Order 4: c1, c2, c3 are Symbols
            L4 = invariant_lagrangian(4; registry=reg)
            symbol_coeffs = Symbol[]
            walk(L4) do node
                if node isa TScalar && node.val isa Symbol
                    push!(symbol_coeffs, node.val)
                end
                node
            end
            @test length(unique(symbol_coeffs)) == 3
        end
    end

    # ------------------------------------------------------------------
    # 8. Invariant content verification: order 4 terms
    # ------------------------------------------------------------------
    @testset "Order 4 invariant content" begin
        reg = lagrangian_registry()
        with_registry(reg) do
            L4 = invariant_lagrangian(4; registry=reg)

            # Check each term contains the expected tensor content
            has_R_sq = false        # RicScalar * RicScalar
            has_Ric_sq = false      # Ric * Ric
            has_Kretschner = false  # Riem * Riem

            for t in L4.terms
                str = string(t)
                if count("RicScalar", str) >= 2
                    has_R_sq = true
                end
                if occursin("Ric", str) && !occursin("RicScalar", str) && !occursin("Riem", str)
                    has_Ric_sq = true
                end
                if count("Riem", str) >= 2
                    has_Kretschner = true
                end
            end

            @test has_R_sq
            @test has_Ric_sq
            @test has_Kretschner
        end
    end

    # ------------------------------------------------------------------
    # 9. dim=nothing gives generic-dimension result (same as no DDI)
    # ------------------------------------------------------------------
    @testset "dim=nothing gives generic-dimension basis" begin
        reg = lagrangian_registry()
        with_registry(reg) do
            L4_nothing = invariant_lagrangian(4; dim=nothing, registry=reg)
            L4_default = invariant_lagrangian(4; registry=reg)

            # Both should produce 3 terms
            @test L4_nothing isa TSum
            @test L4_default isa TSum
            @test length(L4_nothing.terms) == 3
            @test length(L4_default.terms) == 3
        end
    end

end
