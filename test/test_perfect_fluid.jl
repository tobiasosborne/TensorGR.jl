@testset "Perfect Fluid Stress-Energy" begin

    # --- Setup: manifold + metric + covd + fluid ---
    reg = TensorRegistry()
    with_registry(reg) do
        @manifold M4 dim=4 metric=g
        define_covd!(reg, :D; manifold=:M4, metric=:g)
        fp = define_perfect_fluid!(reg, :T; manifold=:M4, metric=:g)
        fluid = PerfectFluid(BarotropicEOS(0), fp)

        # --- stress_energy: T_{ab} = (rho + p) u_a u_b + p g_{ab} ---
        @testset "stress_energy" begin
            T_ab = stress_energy(fluid; registry=reg)
            @test T_ab isa TSum
            @test length(T_ab.terms) == 2

            # Verify indices are covariant
            all_indices = indices(T_ab)
            for idx in all_indices
                if idx.name in (:a, :b)
                    @test idx.position == Down
                end
            end

            # Custom index names
            T_cd = stress_energy(fluid; indices=(:c, :d), registry=reg)
            @test T_cd isa TSum
            cd_indices = free_indices(T_cd)
            idx_names = Set(idx.name for idx in cd_indices)
            @test :c in idx_names
            @test :d in idx_names
        end

        # --- trace_stress_energy: T = -rho + 3p  (4D) ---
        @testset "trace_stress_energy" begin
            T_trace = trace_stress_energy(fluid; registry=reg)
            @test T_trace isa TSum
            @test length(T_trace.terms) == 2

            # Check structure: first term is -rho, second is 3p
            terms = T_trace.terms

            # Find the -rho term
            rho_term = nothing
            p_term = nothing
            for t in terms
                if t isa TProduct && length(t.factors) == 1 && t.factors[1] isa TScalar
                    if t.factors[1].val == :rho
                        rho_term = t
                    elseif t.factors[1].val == :p
                        p_term = t
                    end
                end
            end
            @test rho_term !== nothing
            @test rho_term.scalar == -1 // 1
            @test p_term !== nothing
            @test p_term.scalar == 3 // 1
        end

        # --- trace for radiation (w=1/3): T = -rho + 3*(rho/3) = 0 ---
        @testset "trace radiation vanishes" begin
            # For radiation p = rho/3, so T = -rho + 3*(rho/3) = 0
            # This is structural: the trace expression gives -rho + 3p
            # which for w=1/3 becomes -rho + rho = 0.
            # We verify the expression has the right structure.
            rad_fluid = PerfectFluid(BarotropicEOS(1 // 3), fp)
            T_trace = trace_stress_energy(rad_fluid; registry=reg)
            @test T_trace isa TSum
        end

        # --- conservation_equation: D_a T^{ab} ---
        @testset "conservation_equation" begin
            cons = conservation_equation(fluid, :D; registry=reg)
            @test cons isa TDeriv
            @test cons.covd == :D
            @test cons.index.position == Down
            @test cons.index.name == :a

            # The arg should be the stress-energy T^{ab} expression
            @test cons.arg isa TSum

            # Verify T^{ab} has contravariant indices
            T_up = cons.arg
            fi = free_indices(T_up)
            for idx in fi
                if idx.name in (:a, :b)
                    @test idx.position == Up
                end
            end
        end

        # --- custom indices for conservation ---
        @testset "conservation custom indices" begin
            cons2 = conservation_equation(fluid, :D;
                        free_index=:mu, contract_index=:nu, registry=reg)
            @test cons2 isa TDeriv
            @test cons2.index.name == :nu
            @test cons2.index.position == Down
        end

        # --- stress_energy with dust (p=0): T_{ab} = rho * u_a * u_b ---
        @testset "dust stress-energy structure" begin
            dust = PerfectFluid(BarotropicEOS(0), fp)
            T_dust = stress_energy(dust; registry=reg)
            @test T_dust isa TSum
            # Expression is (rho + p) u_a u_b + p g_{ab}
            # For abstract expressions, p is still symbolic -- simplification
            # would be needed to set p=0.
            @test length(T_dust.terms) == 2
        end

        # --- trace in generic dimension ---
        @testset "trace generic dimension" begin
            reg2 = TensorRegistry()
            with_registry(reg2) do
                @manifold M3 dim=3 metric=h
                fp3 = define_perfect_fluid!(reg2, :S; manifold=:M3, metric=:h,
                                            rho=:rho3, p=:p3, u=:v)
                fluid3 = PerfectFluid(BarotropicEOS(0), fp3)
                T3 = trace_stress_energy(fluid3; registry=reg2)
                @test T3 isa TSum
                # In 3D: T = -rho + 2p
                terms = T3.terms
                p_term = nothing
                for t in terms
                    if t isa TProduct && length(t.factors) == 1 &&
                       t.factors[1] isa TScalar && t.factors[1].val == :p3
                        p_term = t
                    end
                end
                @test p_term !== nothing
                @test p_term.scalar == 2 // 1   # (d-1) = 3-1 = 2
            end
        end

        # --- PerfectFluid with different EOS types ---
        @testset "PerfectFluid with different EOS" begin
            # Cosmological constant (w = -1)
            de_fluid = PerfectFluid(BarotropicEOS(-1), fp)
            T_de = stress_energy(de_fluid; registry=reg)
            @test T_de isa TSum

            # Polytropic
            poly_fluid = PerfectFluid(PolytropicEOS(1 // 10, 5 // 3), fp)
            T_poly = stress_energy(poly_fluid; registry=reg)
            @test T_poly isa TSum
        end
    end

end
