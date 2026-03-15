@testset "Phase Space: EOM Extraction" begin

    # ── Helper: set up a standard 4D GR registry ──
    function make_gr_registry()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :D,
            [:a, :b, :c, :d, :e, :f, :m, :n, :p, :q, :r, :s]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0, 2),
            symmetries=Any[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
        define_curvature_tensors!(reg, :M4, :g)
        define_covd!(reg, :D; manifold=:M4, metric=:g)
        reg
    end

    @testset "LagrangianDensity construction" begin
        reg = make_gr_registry()
        with_registry(reg) do
            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            @test L.expr == R
            @test L.fields == [:g]
            @test L.metric == :g
            @test L.covd == :D
            @test L.dim == 4
        end
    end

    @testset "EOMResult construction" begin
        reg = make_gr_registry()
        with_registry(reg) do
            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            G_ab = einstein_expr(down(:a), down(:b), :g)
            result = EOMResult(G_ab, nothing, L, :g)
            @test result.eom == G_ab
            @test result.theta === nothing
            @test result.field == :g
            @test result.lagrangian === L
        end
    end

    @testset "EH Lagrangian: EOM = Einstein tensor" begin
        reg = make_gr_registry()
        with_registry(reg) do
            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            result = eom_extract(L, :g; registry=reg)

            # The EOM should be G_{ab} = Ric_{ab} - (1/2) g_{ab} R
            eom = result.eom
            expected = einstein_expr(down(:a), down(:b), :g)
            @test eom == expected

            # Theta not yet implemented
            @test result.theta === nothing
            @test result.field == :g
        end
    end

    @testset "EH via extract_eom_and_theta convenience" begin
        reg = make_gr_registry()
        with_registry(reg) do
            R = Tensor(:RicScalar, TIndex[])
            eom, theta = extract_eom_and_theta(R, :g, :g; covd=:D, dim=4, registry=reg)

            expected = einstein_expr(down(:a), down(:b), :g)
            @test eom == expected
            @test theta === nothing
        end
    end

    @testset "Multi-field EOM extraction" begin
        reg = make_gr_registry()
        with_registry(reg) do
            # Register a scalar field phi
            register_tensor!(reg, TensorProperties(
                name=:phi, manifold=:M4, rank=(0, 0),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}()))

            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g, :phi], :g, :D, 4)
            results = eom_extract(L; registry=reg)

            @test length(results) == 2
            @test results[1].field == :g
            @test results[2].field == :phi
        end
    end

    @testset "Scalar field EOM via variational_derivative" begin
        reg = make_gr_registry()
        with_registry(reg) do
            # Register scalar field
            register_tensor!(reg, TensorProperties(
                name=:phi, manifold=:M4, rank=(0, 0),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}()))

            # L = (1/2) g^{ab} (partial_a phi)(partial_b phi)
            # => L = (1/2) (partial_a phi)(partial^a phi)
            phi = Tensor(:phi, TIndex[])
            dphi_a = TDeriv(down(:a), phi)
            dphi_up = TDeriv(up(:a), phi)
            L_scalar_expr = (1 // 2) * dphi_a * dphi_up

            L = LagrangianDensity(L_scalar_expr, [:phi], :g, :D, 4)
            result = eom_extract(L, :phi; registry=reg)

            # The EOM should be the Euler-Lagrange equation
            # For L = (1/2) partial_a(phi) partial^a(phi),
            # E = -partial_a(partial^a phi) = -Box phi
            eom = result.eom

            # The variational derivative should produce
            # -partial_a(partial^a phi) (the negative d'Alembertian)
            # The exact form depends on expand_derivatives, but it should
            # contain a second derivative term
            @test !(eom isa TScalar && eom.val == 0 // 1)
            @test result.theta === nothing
        end
    end

    @testset "General metric Lagrangian: Ric_{ab}Ric^{ab}" begin
        reg = make_gr_registry()
        with_registry(reg) do
            # L = Ric_{ab} Ric^{ab} (not a recognized special case)
            Ric_down = Tensor(:Ric, [down(:c), down(:d)])
            Ric_up = Tensor(:Ric, [up(:c), up(:d)])
            L_Ric2 = Ric_down * Ric_up

            L = LagrangianDensity(L_Ric2, [:g], :g, :D, 4)
            result = eom_extract(L, :g; registry=reg)

            # The general metric_variation path should produce something non-zero
            eom = result.eom
            @test !(eom isa TScalar && eom.val == 0 // 1)
        end
    end

    @testset "EOMResult stores Lagrangian reference" begin
        reg = make_gr_registry()
        with_registry(reg) do
            R = Tensor(:RicScalar, TIndex[])
            L = LagrangianDensity(R, [:g], :g, :D, 4)
            result = eom_extract(L, :g; registry=reg)

            @test result.lagrangian === L
            @test result.lagrangian.dim == 4
            @test result.lagrangian.metric == :g
        end
    end

    @testset "Non-metric field falls through to variational_derivative" begin
        reg = make_gr_registry()
        with_registry(reg) do
            # Register a vector field A_a
            register_tensor!(reg, TensorProperties(
                name=:A, manifold=:M4, rank=(0, 1),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}()))

            # Simple Lagrangian: A_a A^a (mass term)
            A_down = Tensor(:A, [down(:c)])
            A_up = Tensor(:A, [up(:c)])
            L_mass = A_down * A_up

            L = LagrangianDensity(L_mass, [:A], :g, :D, 4)
            result = eom_extract(L, :A; registry=reg)

            # variational_derivative of A_c A^c w.r.t. A
            # should give A^c + A_c (or 2*A depending on convention)
            eom = result.eom
            @test !(eom isa TScalar && eom.val == 0 // 1)
        end
    end

    @testset "_is_ricci_scalar helper" begin
        R = Tensor(:RicScalar, TIndex[])
        @test TensorGR._is_ricci_scalar(R)
        @test !TensorGR._is_ricci_scalar(Tensor(:Ric, [down(:a), down(:b)]))
        @test !TensorGR._is_ricci_scalar(TScalar(1 // 1))
        @test !TensorGR._is_ricci_scalar(Tensor(:RicScalar, [down(:a)]))
    end

end
