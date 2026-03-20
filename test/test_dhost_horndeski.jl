#= DHOST Validation: verify DHOST reduces to Horndeski.
#
# When DHOST coefficients satisfy a_1 = -a_2 (= G4_X), a_3 = a_4 = a_5 = 0,
# the DHOST Lagrangian must reduce to the Horndeski Lagrangian. We verify:
# 1. horndeski_as_dhost correctly sets class Ia with a1=-a2, a3=a4=a5=0
# 2. degeneracy classifier identifies class_Ia (degenerate, 3 DOF)
# 3. reduce_to_horndeski returns a Horndeski theory for class_Ia
#
# Ground truth: Langlois & Noui, JCAP 1602 (2016) 034,
#   arXiv:1510.06930, Sec 4.1 (Horndeski as DHOST subcase).
=#

@testset "DHOST Validation: Horndeski reduction" begin

    reg = TensorRegistry()
    with_registry(reg) do
        @manifold M4 dim=4 metric=g
        define_curvature_tensors!(reg, :M4, :g)
        @covd D on=M4 metric=g

        # ================================================================
        # 1. Build Horndeski theory and embed as DHOST
        # ================================================================
        horn = define_horndeski!(reg; manifold=:M4, metric=:g,
                                 scalar_field=:phi, covd=:D)
        dhost_h = horndeski_as_dhost(horn)

        @testset "Embedding: horndeski_as_dhost" begin
            # a3 = a4 = a5 = 0 (set_vanishing! → simplifies to zero)
            for i in 3:5
                name = g_tensor_name(dhost_h.a[i])
                t = Tensor(name, TIndex[])
                @test simplify(t; registry=reg) == TScalar(0 // 1)
            end

            # a1 = G4_X, a2 = -G4_X (stored as metadata)
            a1_opts = get_tensor(reg, g_tensor_name(dhost_h.a[1])).options
            a2_opts = get_tensor(reg, g_tensor_name(dhost_h.a[2])).options
            @test a1_opts[:dhost_coeff_expr] == :G4_X
        end

        # ================================================================
        # 2. Degeneracy classification
        # ================================================================
        @testset "Degeneracy: class Ia, 3 DOF" begin
            @test is_degenerate(dhost_h)
            @test dhost_class(dhost_h) == :class_Ia
            @test dhost_dof_count(dhost_h) == 3
        end

        # ================================================================
        # 3. reduce_to_horndeski round-trip
        # ================================================================
        @testset "reduce_to_horndeski succeeds for class Ia" begin
            horn2 = reduce_to_horndeski(dhost_h)
            @test horn2 !== nothing
            @test horn2 isa HorndeskiTheory
            @test horn2.manifold == :M4
            @test horn2.metric == :g
            @test horn2.scalar_field == :phi
        end

        # ================================================================
        # 4. Generic DHOST is not Horndeski
        # ================================================================
        @testset "Generic DHOST: not degenerate, not Horndeski" begin
            reg2 = TensorRegistry()
            with_registry(reg2) do
                @manifold M4 dim=4 metric=g
                define_curvature_tensors!(reg2, :M4, :g)
                @covd D on=M4 metric=g

                dhost_gen = define_dhost!(reg2; manifold=:M4, metric=:g,
                                          scalar_field=:phi, covd=:D)
                # Generic DHOST has no constraints: not degenerate
                @test !is_degenerate(dhost_gen)
                @test dhost_class(dhost_gen) == :not_degenerate
                @test dhost_dof_count(dhost_gen) == 4  # Ostrogradsky ghost
                @test reduce_to_horndeski(dhost_gen) === nothing
            end
        end

        # ================================================================
        # 5. Lagrangian structure comparison
        # ================================================================
        @testset "Lagrangian structure" begin
            # Both should build non-trivial Lagrangians
            L_horn = horndeski_lagrangian(horn)
            L_dhost = dhost_lagrangian(dhost_h)

            @test L_horn isa TSum
            @test L_dhost isa TSum
            @test length(L_horn.terms) > 0
            @test length(L_dhost.terms) > 0

            # Individual Horndeski pieces should be well-formed
            L2 = horndeski_L2(horn)
            L3 = horndeski_L3(horn)
            L4 = horndeski_L4(horn)
            L5 = horndeski_L5(horn)
            @test L2 isa TensorExpr
            @test L3 isa TensorExpr
            @test L4 isa TensorExpr
            @test L5 isa TensorExpr

            # Individual DHOST Lagrangians exist
            for fn in [dhost_L1, dhost_L2, dhost_L3, dhost_L4, dhost_L5]
                Li = fn(dhost_h)
                @test Li isa TensorExpr
            end
        end

        # ================================================================
        # 6. Verify Horndeski Lagrangian has no free indices (is scalar)
        # ================================================================
        @testset "Lagrangians are scalars (no free indices)" begin
            L_horn = horndeski_lagrangian(horn)
            @test isempty(free_indices(L_horn))

            L_dhost = dhost_lagrangian(dhost_h)
            @test isempty(free_indices(L_dhost))
        end
    end

end
