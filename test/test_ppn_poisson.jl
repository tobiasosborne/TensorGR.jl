using Test
using TensorGR
using TensorGR: PoissonEquation, PoissonSolution,
                define_poisson_equation, poisson_solve,
                standard_ppn_potentials, identify_ppn_potential

@testset "PPN Poisson Equation Solver" begin

    # Helper: set up a PPN-ready registry
    function _ppn_poisson_reg()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d,:e,:f,:g,:h,:i,:j,:k,:l]))
        reg
    end

    # ─────────────────────────────────────────────────────────────
    # PoissonEquation struct
    # ─────────────────────────────────────────────────────────────

    @testset "PoissonEquation construction" begin
        reg = _ppn_poisson_reg()
        with_registry(reg) do
            define_ppn_potentials!(reg; manifold=:M4)

            source = TScalar(:rho)
            eq = PoissonEquation(:U, source, 2, :scalar)
            @test eq isa PoissonEquation
            @test eq.potential == :U
            @test eq.order == 2
            @test eq.equation_type == :scalar
            @test eq.source isa TScalar
        end
    end

    @testset "PoissonEquation validation" begin
        # Invalid equation type
        @test_throws ErrorException PoissonEquation(:U, TScalar(:rho), 2, :bogus)
        # Negative order
        @test_throws ErrorException PoissonEquation(:U, TScalar(:rho), -1, :scalar)
    end

    @testset "PoissonEquation display" begin
        eq = PoissonEquation(:U, TScalar(:rho), 2, :scalar)
        s = sprint(show, eq)
        @test occursin("PoissonEquation", s)
        @test occursin("U", s)
        @test occursin("order=2", s)
    end

    # ─────────────────────────────────────────────────────────────
    # PoissonSolution struct
    # ─────────────────────────────────────────────────────────────

    @testset "PoissonSolution construction and display" begin
        eq = PoissonEquation(:U, TScalar(:rho), 2, :scalar)
        sol = PoissonSolution(eq, :newtonian, TScalar(:integral_placeholder))
        @test sol isa PoissonSolution
        @test sol.equation === eq
        @test sol.green_function == :newtonian
        @test sol.integral_form isa TensorExpr

        s = sprint(show, sol)
        @test occursin("PoissonSolution", s)
        @test occursin("U", s)
        @test occursin("newtonian", s)
    end

    # ─────────────────────────────────────────────────────────────
    # define_poisson_equation
    # ─────────────────────────────────────────────────────────────

    @testset "define_poisson_equation" begin
        reg = _ppn_poisson_reg()
        with_registry(reg) do
            define_ppn_potentials!(reg; manifold=:M4)

            source = tproduct(-1 // 1, TensorExpr[
                TScalar(Symbol("4piG")), TScalar(:rho)
            ])
            eq = define_poisson_equation(:U, source; order=2, type=:scalar, registry=reg)
            @test eq isa PoissonEquation
            @test eq.potential == :U
            @test eq.order == 2
        end
    end

    @testset "define_poisson_equation rejects unregistered potential" begin
        reg = _ppn_poisson_reg()
        with_registry(reg) do
            # Don't register PPN potentials
            source = TScalar(:rho)
            @test_throws ErrorException define_poisson_equation(
                :U, source; order=2, type=:scalar, registry=reg)
        end
    end

    # ─────────────────────────────────────────────────────────────
    # standard_ppn_potentials
    # ─────────────────────────────────────────────────────────────

    @testset "standard_ppn_potentials returns 6 equations" begin
        reg = _ppn_poisson_reg()
        with_registry(reg) do
            define_ppn_potentials!(reg; manifold=:M4)
            eqs = standard_ppn_potentials(; registry=reg)
            @test eqs isa Vector{PoissonEquation}
            @test length(eqs) >= 6
        end
    end

    @testset "standard_ppn_potentials: correct orders" begin
        reg = _ppn_poisson_reg()
        with_registry(reg) do
            define_ppn_potentials!(reg; manifold=:M4)
            eqs = standard_ppn_potentials(; registry=reg)

            # Build a lookup by potential name
            by_name = Dict(eq.potential => eq for eq in eqs)

            # U is order 2 (Newtonian)
            @test haskey(by_name, :U)
            @test by_name[:U].order == 2
            @test by_name[:U].equation_type == :scalar

            # V_ppn is order 3 (gravitomagnetic)
            @test haskey(by_name, :V_ppn)
            @test by_name[:V_ppn].order == 3
            @test by_name[:V_ppn].equation_type == :vector

            # Phi_1..Phi_4 are all order 4
            for k in 1:4
                pname = Symbol(:Phi_, k)
                @test haskey(by_name, pname)
                @test by_name[pname].order == 4
                @test by_name[pname].equation_type == :scalar
            end
        end
    end

    @testset "standard_ppn_potentials: consistent with PPN_ORDER_TABLE" begin
        reg = _ppn_poisson_reg()
        with_registry(reg) do
            define_ppn_potentials!(reg; manifold=:M4)
            eqs = standard_ppn_potentials(; registry=reg)

            for eq in eqs
                if haskey(PPN_ORDER_TABLE, eq.potential)
                    @test eq.order == PPN_ORDER_TABLE[eq.potential]
                end
            end
        end
    end

    # ─────────────────────────────────────────────────────────────
    # poisson_solve
    # ─────────────────────────────────────────────────────────────

    @testset "poisson_solve returns PoissonSolution" begin
        reg = _ppn_poisson_reg()
        with_registry(reg) do
            define_ppn_potentials!(reg; manifold=:M4)
            eqs = standard_ppn_potentials(; registry=reg)

            for eq in eqs
                sol = poisson_solve(eq; registry=reg)
                @test sol isa PoissonSolution
                @test sol.green_function == :newtonian
                @test sol.equation === eq
                @test sol.integral_form isa TensorExpr
            end
        end
    end

    @testset "poisson_solve: U solution structure" begin
        reg = _ppn_poisson_reg()
        with_registry(reg) do
            define_ppn_potentials!(reg; manifold=:M4)
            eqs = standard_ppn_potentials(; registry=reg)
            u_eq = first(eq for eq in eqs if eq.potential == :U)
            sol = poisson_solve(u_eq; registry=reg)

            # The integral form should be a TProduct with the (-1/4pi) x source structure
            @test sol.integral_form isa TProduct
            # The Green's function is Newtonian (1/|x-x'|)
            @test sol.green_function == :newtonian
            # The solution references the original equation
            @test sol.equation.potential == :U
            @test sol.equation.order == 2
        end
    end

    # ─────────────────────────────────────────────────────────────
    # identify_ppn_potential
    # ─────────────────────────────────────────────────────────────

    @testset "identify_ppn_potential: known sources" begin
        reg = _ppn_poisson_reg()
        with_registry(reg) do
            define_ppn_potentials!(reg; manifold=:M4)

            # U: source contains :rho
            source_U = tproduct(-1 // 1, TensorExpr[
                TScalar(Symbol("4piG")), TScalar(:rho)
            ])
            @test identify_ppn_potential(source_U; registry=reg) == :U

            # V_ppn: source contains :rho_vi
            source_V = tproduct(-1 // 1, TensorExpr[
                TScalar(Symbol("4piG")), TScalar(:rho_vi)
            ])
            @test identify_ppn_potential(source_V; registry=reg) == :V_ppn

            # Phi_1: source contains :rho_v2
            source_P1 = tproduct(-1 // 1, TensorExpr[
                TScalar(Symbol("4piG")), TScalar(:rho_v2)
            ])
            @test identify_ppn_potential(source_P1; registry=reg) == :Phi_1

            # Phi_2: source contains :rho_U
            source_P2 = tproduct(-1 // 1, TensorExpr[
                TScalar(Symbol("4piG")), TScalar(:rho_U)
            ])
            @test identify_ppn_potential(source_P2; registry=reg) == :Phi_2

            # Phi_3: source contains :rho_Pi
            source_P3 = tproduct(-1 // 1, TensorExpr[
                TScalar(Symbol("4piG")), TScalar(:rho_Pi)
            ])
            @test identify_ppn_potential(source_P3; registry=reg) == :Phi_3

            # Phi_4: source contains :p
            source_P4 = tproduct(-1 // 1, TensorExpr[
                TScalar(Symbol("4piG")), TScalar(:p)
            ])
            @test identify_ppn_potential(source_P4; registry=reg) == :Phi_4
        end
    end

    @testset "identify_ppn_potential: unknown source returns nothing" begin
        reg = _ppn_poisson_reg()
        with_registry(reg) do
            define_ppn_potentials!(reg; manifold=:M4)

            # Unknown source
            source_unknown = tproduct(-1 // 1, TensorExpr[
                TScalar(Symbol("4piG")), TScalar(:unknown_matter)
            ])
            @test identify_ppn_potential(source_unknown; registry=reg) === nothing

            # Non-symbol TScalar
            source_numeric = TScalar(42 // 1)
            @test identify_ppn_potential(source_numeric; registry=reg) === nothing

            # Pure tensor (no TScalar matter content)
            source_tensor = Tensor(:U, TIndex[])
            @test identify_ppn_potential(source_tensor; registry=reg) === nothing
        end
    end

    @testset "identify_ppn_potential: bare TScalar source" begin
        reg = _ppn_poisson_reg()
        with_registry(reg) do
            define_ppn_potentials!(reg; manifold=:M4)

            # Bare :rho as source (not wrapped in product)
            @test identify_ppn_potential(TScalar(:rho); registry=reg) == :U
            @test identify_ppn_potential(TScalar(:p); registry=reg) == :Phi_4
        end
    end

    # ─────────────────────────────────────────────────────────────
    # Round-trip: standard_ppn_potentials -> identify
    # ─────────────────────────────────────────────────────────────

    @testset "round-trip: identify matches standard potentials" begin
        reg = _ppn_poisson_reg()
        with_registry(reg) do
            define_ppn_potentials!(reg; manifold=:M4)
            eqs = standard_ppn_potentials(; registry=reg)

            for eq in eqs
                identified = identify_ppn_potential(eq.source; registry=reg)
                @test identified == eq.potential
            end
        end
    end

    # ─────────────────────────────────────────────────────────────
    # Edge cases
    # ─────────────────────────────────────────────────────────────

    @testset "equation types: vector and scalar" begin
        reg = _ppn_poisson_reg()
        with_registry(reg) do
            define_ppn_potentials!(reg; manifold=:M4)
            eqs = standard_ppn_potentials(; registry=reg)

            scalar_eqs = filter(eq -> eq.equation_type == :scalar, eqs)
            vector_eqs = filter(eq -> eq.equation_type == :vector, eqs)

            # 5 scalar equations (U, Phi_1..4), 1 vector (V_ppn)
            @test length(scalar_eqs) == 5
            @test length(vector_eqs) == 1
            @test vector_eqs[1].potential == :V_ppn
        end
    end

    @testset "PoissonEquation supports tensor type" begin
        # Tensor type is valid (for future superpotential equations)
        eq = PoissonEquation(:U_ppn, TScalar(:rho_ij), 2, :tensor)
        @test eq.equation_type == :tensor
    end

end
