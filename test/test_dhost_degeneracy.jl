@testset "DHOST degeneracy conditions (Langlois & Noui 2016)" begin

    # Helper: standard 4D manifold registry for DHOST tests
    function _dhost_deg_registry()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :d,
            [:a,:b,:c,:d,:e,:f,:m,:n,:p,:q,:r,:s,:t,:u,:v,:w]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0,2),
            symmetries=SymmetrySpec[Symmetric(1,2)],
            is_metric=true,
            options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M4, rank=(1,1),
            symmetries=SymmetrySpec[],
            is_delta=true,
            options=Dict{Symbol,Any}(:is_delta => true)))
        reg
    end

    # -----------------------------------------------------------------------
    # Test 1: Horndeski as DHOST satisfies degeneracy (Class Ia)
    # -----------------------------------------------------------------------
    @testset "Horndeski satisfies degeneracy conditions (Class Ia)" begin
        reg = _dhost_deg_registry()
        with_registry(reg) do
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            dht = horndeski_as_dhost(ht; registry=reg)

            # degeneracy_conditions should return all zeros
            conds = degeneracy_conditions(dht; registry=reg)
            @test length(conds) == 3
            @test all(c -> c == 0, conds)  # G4_X + (-G4_X) = 0 via _sym_add cancellation

            # is_degenerate should be true
            @test is_degenerate(dht; registry=reg)

            # Should be classified as Class Ia
            @test dhost_class(dht; registry=reg) == :class_Ia
        end
    end

    # -----------------------------------------------------------------------
    # Test 2: Horndeski reduces to HorndeskiTheory via reduce_to_horndeski
    # -----------------------------------------------------------------------
    @testset "Horndeski DHOST reduces back to HorndeskiTheory" begin
        reg = _dhost_deg_registry()
        with_registry(reg) do
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            dht = horndeski_as_dhost(ht; registry=reg)

            ht2 = reduce_to_horndeski(dht; registry=reg)
            @test ht2 !== nothing
            @test ht2 isa HorndeskiTheory
            if ht2 !== nothing
                @test ht2.manifold == :M4
                @test ht2.metric == :g
            end
        end
    end

    # -----------------------------------------------------------------------
    # Test 3: Generic random a_i does NOT satisfy degeneracy
    # -----------------------------------------------------------------------
    @testset "Generic DHOST is not degenerate" begin
        reg = _dhost_deg_registry()
        with_registry(reg) do
            dht = define_dhost!(reg; manifold=:M4, metric=:g)

            # Random coefficients that don't satisfy the algebraic conditions
            vals = Dict{Symbol,Float64}(
                :a1 => 1.0,
                :a2 => 2.0,
                :a3 => 0.5,
                :a4 => -0.3,
                :a5 => 0.7
            )

            @test !is_degenerate(dht; registry=reg, values=vals)
            @test dhost_class(dht; registry=reg, values=vals) == :not_degenerate
        end
    end

    # -----------------------------------------------------------------------
    # Test 4: Numerical Horndeski coefficients satisfy degeneracy
    # -----------------------------------------------------------------------
    @testset "Numerical Horndeski coefficients are degenerate" begin
        reg = _dhost_deg_registry()
        with_registry(reg) do
            dht = define_dhost!(reg; manifold=:M4, metric=:g)

            # Horndeski: a_1 = G4X, a_2 = -G4X, a_3 = a_4 = a_5 = 0
            G4X = 3.7  # arbitrary nonzero value
            vals = Dict{Symbol,Float64}(
                :a1 => G4X,
                :a2 => -G4X,
                :a3 => 0.0,
                :a4 => 0.0,
                :a5 => 0.0
            )

            @test is_degenerate(dht; registry=reg, values=vals)
            @test dhost_class(dht; registry=reg, values=vals) == :class_Ia
        end
    end

    # -----------------------------------------------------------------------
    # Test 5: DOF count = 3 for degenerate, 4 for non-degenerate
    # -----------------------------------------------------------------------
    @testset "DOF count: 3 for degenerate, 4 for non-degenerate" begin
        reg = _dhost_deg_registry()
        with_registry(reg) do
            dht = define_dhost!(reg; manifold=:M4, metric=:g)

            # Degenerate (Horndeski-like)
            vals_deg = Dict{Symbol,Float64}(
                :a1 => 2.0, :a2 => -2.0,
                :a3 => 0.0, :a4 => 0.0, :a5 => 0.0
            )
            @test dhost_dof_count(dht; registry=reg, values=vals_deg) == 3

            # Non-degenerate (random)
            vals_nondeg = Dict{Symbol,Float64}(
                :a1 => 1.0, :a2 => 2.0,
                :a3 => 0.5, :a4 => -0.3, :a5 => 0.7
            )
            @test dhost_dof_count(dht; registry=reg, values=vals_nondeg) == 4
        end
    end

    # -----------------------------------------------------------------------
    # Test 6: Class Ib -- general degenerate (non-Horndeski)
    # -----------------------------------------------------------------------
    @testset "Class Ib: general degenerate DHOST" begin
        reg = _dhost_deg_registry()
        with_registry(reg) do
            dht = define_dhost!(reg; manifold=:M4, metric=:g)

            # Construct coefficients satisfying the three degeneracy conditions:
            # C1: 2 a_2(a_1 + a_2) + 3 a_3^2 = 0
            # C2: a_2 a_4 + a_3(a_2 + a_3) = 0
            # C3: 2 a_2^2 a_5 + a_3^2(a_2 + a_3) = 0
            #
            # Choose a_2 = 1, a_3 = 1 (nonzero).
            # C1: 2(a_1 + 1) + 3 = 0 => a_1 = -5/2
            # C2: a_4 + (1 + 1) = 0 => a_4 = -2
            # C3: 2 a_5 + (1 + 1) = 0 => a_5 = -1
            vals = Dict{Symbol,Float64}(
                :a1 => -2.5,
                :a2 => 1.0,
                :a3 => 1.0,
                :a4 => -2.0,
                :a5 => -1.0
            )

            @test is_degenerate(dht; registry=reg, values=vals)
            @test dhost_class(dht; registry=reg, values=vals) == :class_Ib
            @test dhost_dof_count(dht; registry=reg, values=vals) == 3
        end
    end

    # -----------------------------------------------------------------------
    # Test 7: Class II classification (a_2 = 0 in our convention)
    # -----------------------------------------------------------------------
    @testset "Class II: a_2 = 0, a_1 != 0" begin
        reg = _dhost_deg_registry()
        with_registry(reg) do
            dht = define_dhost!(reg; manifold=:M4, metric=:g)

            vals = Dict{Symbol,Float64}(
                :a1 => 1.0, :a2 => 0.0,
                :a3 => 0.5, :a4 => 0.1, :a5 => 0.0
            )

            @test dhost_class(dht; registry=reg, values=vals) == :class_II
        end
    end

    # -----------------------------------------------------------------------
    # Test 8: Class III classification (a_1 = a_2 = 0)
    # -----------------------------------------------------------------------
    @testset "Class III: a_1 = a_2 = 0" begin
        reg = _dhost_deg_registry()
        with_registry(reg) do
            dht = define_dhost!(reg; manifold=:M4, metric=:g)

            vals = Dict{Symbol,Float64}(
                :a1 => 0.0, :a2 => 0.0,
                :a3 => 1.0, :a4 => 0.5, :a5 => 0.2
            )

            @test dhost_class(dht; registry=reg, values=vals) == :class_III
        end
    end

    # -----------------------------------------------------------------------
    # Test 9: Structural Class Ia via vanishing registry rules
    # -----------------------------------------------------------------------
    @testset "Structural classification via set_vanishing!" begin
        reg = _dhost_deg_registry()
        with_registry(reg) do
            ht = define_horndeski!(reg; manifold=:M4, metric=:g)
            dht = horndeski_as_dhost(ht; registry=reg)

            # horndeski_as_dhost sets a_3, a_4, a_5 to vanish
            @test dhost_class(dht; registry=reg) == :class_Ia
            @test dhost_dof_count(dht; registry=reg) == 3
        end
    end

    # -----------------------------------------------------------------------
    # Test 10: Non-degenerate cannot reduce to Horndeski
    # -----------------------------------------------------------------------
    @testset "Non-degenerate DHOST does not reduce to Horndeski" begin
        reg = _dhost_deg_registry()
        with_registry(reg) do
            dht = define_dhost!(reg; manifold=:M4, metric=:g)

            # Generic DHOST with no vanishing rules -- not degenerate structurally
            # (the conditions involve symbolic a_i that are not zero)
            # is_degenerate will be false for generic symbolic coefficients
            result = reduce_to_horndeski(dht; registry=reg)
            @test result === nothing
        end
    end

    # -----------------------------------------------------------------------
    # Test 11: Verify degeneracy conditions are correct polynomials
    # -----------------------------------------------------------------------
    @testset "Degeneracy conditions have correct algebraic structure" begin
        reg = _dhost_deg_registry()
        with_registry(reg) do
            dht = define_dhost!(reg; manifold=:M4, metric=:g)
            conds = degeneracy_conditions(dht; registry=reg)

            @test length(conds) == 3

            # For Horndeski (a_1 = alpha, a_2 = -alpha, rest = 0):
            for alpha in [1.0, 5.0, 0.1, 100.0]
                vals = Dict{Symbol,Float64}(
                    :a1 => alpha, :a2 => -alpha,
                    :a3 => 0.0, :a4 => 0.0, :a5 => 0.0
                )
                for c in conds
                    @test abs(sym_eval(c, vals)) < 1e-10
                end
            end
        end
    end

    # -----------------------------------------------------------------------
    # Test 12: Beyond-Horndeski numerical coefficients
    # -----------------------------------------------------------------------
    @testset "Beyond-Horndeski numerical DHOST embedding" begin
        reg = _dhost_deg_registry()
        with_registry(reg) do
            dht = define_dhost!(reg; manifold=:M4, metric=:g)

            # Beyond-Horndeski modifies the DHOST coefficients relative to
            # pure Horndeski. With the F_4 term, the DHOST coefficients
            # (in the full theory with f*R coupling) satisfy degeneracy.
            # In the f=0 sector, the specific bH values are:
            # a_1 = G4X + 2X F4, a_2 = -(G4X + 2X F4), a_3 = a_4 = a_5 = 0
            # This still has a_1 + a_2 = 0 and a_3=a_4=a_5=0.
            G4X = 1.0
            F4_val = 0.5
            X_val = 0.3
            vals = Dict{Symbol,Float64}(
                :a1 => G4X + 2*X_val*F4_val,
                :a2 => -(G4X + 2*X_val*F4_val),
                :a3 => 0.0,
                :a4 => 0.0,
                :a5 => 0.0
            )

            @test is_degenerate(dht; registry=reg, values=vals)
            @test dhost_class(dht; registry=reg, values=vals) == :class_Ia
            @test dhost_dof_count(dht; registry=reg, values=vals) == 3
        end
    end

    # -----------------------------------------------------------------------
    # Test 13: Perturbation of degenerate coefficients breaks degeneracy
    # -----------------------------------------------------------------------
    @testset "Small perturbation of degenerate coefficients breaks degeneracy" begin
        reg = _dhost_deg_registry()
        with_registry(reg) do
            dht = define_dhost!(reg; manifold=:M4, metric=:g)

            # Start with degenerate Horndeski
            vals_deg = Dict{Symbol,Float64}(
                :a1 => 2.0, :a2 => -2.0,
                :a3 => 0.0, :a4 => 0.0, :a5 => 0.0
            )
            @test is_degenerate(dht; registry=reg, values=vals_deg)

            # Perturb a_3 slightly
            vals_pert = Dict{Symbol,Float64}(
                :a1 => 2.0, :a2 => -2.0,
                :a3 => 0.1, :a4 => 0.0, :a5 => 0.0
            )
            @test !is_degenerate(dht; registry=reg, values=vals_pert)
        end
    end

    # -----------------------------------------------------------------------
    # Test 14: Class Ib with second parametric family
    # -----------------------------------------------------------------------
    @testset "Class Ib: second parametric degenerate family" begin
        reg = _dhost_deg_registry()
        with_registry(reg) do
            dht = define_dhost!(reg; manifold=:M4, metric=:g)

            # Choose a_2 = 2, a_3 = 2
            # C1: 2*2*(a1+2) + 3*4 = 0 => 4(a1+2) + 12 = 0 => a1 = -5
            # C2: 2*a4 + 2*(2+2) = 0 => a4 = -4
            # C3: 2*4*a5 + 4*(2+2) = 0 => 8*a5 + 16 = 0 => a5 = -2
            vals = Dict{Symbol,Float64}(
                :a1 => -5.0,
                :a2 => 2.0,
                :a3 => 2.0,
                :a4 => -4.0,
                :a5 => -2.0
            )

            @test is_degenerate(dht; registry=reg, values=vals)
            @test dhost_class(dht; registry=reg, values=vals) == :class_Ib
        end
    end

end
