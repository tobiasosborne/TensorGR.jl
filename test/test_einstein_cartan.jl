# Validation: Einstein-Cartan theory (Q=0, T!=0)
#
# Einstein-Cartan theory is general relativity with torsion but metric
# compatibility: Q_{abc} = 0, T^a_{bc} != 0.  The connection is
#
#   Gamma^a_{bc} = {a; bc} + K^a_{bc}
#
# where {a; bc} is the Levi-Civita connection and K^a_{bc} is the
# contortion (built from torsion).  The disformation vanishes (L=0)
# because non-metricity vanishes.
#
# Ground truth: Cartan (1922); Kibble (1961); Sciama (1964);
#               Hehl, McCrea, Mielke & Ne'eman, Phys. Rep. 258 (1995).

using Test
using TensorGR
using TensorGR: free_indices, contortion_from_torsion,
                disformation_from_nonmetricity,
                einstein_cartan_action, pgt_action, PGTParams,
                torsion_quadratic

@testset "Einstein-Cartan theory (Q=0, T!=0)" begin

    # Shared setup: 4D manifold with general affine connection,
    # metric compatibility imposed but torsion left nonzero.
    function _ec_reg()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
        end
        ac = define_affine_connection!(reg, :Gamma; manifold=:M4, metric=:g)
        set_metric_compatible!(reg, ac)
        # Do NOT call set_torsion_free! — torsion remains nonzero
        reg, ac
    end

    # ── Test 1: Non-metricity vanishes (metric compatibility) ──────────

    @testset "non-metricity vanishes (metric compatibility)" begin
        reg, ac = _ec_reg()
        @test is_metric_compatible(ac, reg)
    end

    # ── Test 2: Torsion is NOT zero ────────────────────────────────────

    @testset "torsion is NOT zero" begin
        reg, ac = _ec_reg()
        @test !is_torsion_free(ac, reg)
    end

    # ── Test 3: Non-metricity tensor simplifies to zero ────────────────

    @testset "non-metricity tensor simplifies to zero" begin
        reg, ac = _ec_reg()
        with_registry(reg) do
            Q_expr = Tensor(ac.nonmetricity_name, [down(:a), down(:b), down(:c)])
            result = simplify(Q_expr; registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

    # ── Test 4: Torsion tensor does NOT simplify to zero ───────────────

    @testset "torsion tensor does NOT simplify to zero" begin
        reg, ac = _ec_reg()
        with_registry(reg) do
            T_expr = Tensor(ac.torsion_name, [up(:a), down(:b), down(:c)])
            result = simplify(T_expr; registry=reg)
            @test result != TScalar(0 // 1)
        end
    end

    # ── Test 5: Disformation vanishes when Q=0 ─────────────────────────
    # L^a_{bc} = (1/2)(Q^a_{bc} - Q_b^a_c - Q_c^a_b) = 0 when Q=0.

    @testset "disformation vanishes when Q=0" begin
        reg, ac = _ec_reg()
        with_registry(reg) do
            L_expr = disformation_from_nonmetricity(ac.nonmetricity_name; registry=reg)
            result = simplify(L_expr; registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

    # ── Test 6: Contortion does NOT vanish when T!=0 ───────────────────
    # K^a_{bc} = (1/2)(T^a_{bc} + T_b^a_c + T_c^a_b) != 0.

    @testset "contortion does NOT vanish when T!=0" begin
        reg, ac = _ec_reg()
        with_registry(reg) do
            K_expr = contortion_from_torsion(ac.torsion_name; registry=reg)
            result = simplify(K_expr; registry=reg)
            @test result != TScalar(0 // 1)
        end
    end

    # ── Test 7: Distortion = contortion only (N = K + 0) ──────────────
    # When Q=0, disformation L=0, so distortion N^a_{bc} = K^a_{bc}.
    # The distortion N = K + L, and L simplifies to zero.

    @testset "distortion = contortion only (L=0)" begin
        reg, ac = _ec_reg()
        with_registry(reg) do
            dd = decompose_distortion!(reg, ac)

            # Both contortion and disformation are registered
            @test has_tensor(reg, dd.contortion_name)
            @test has_tensor(reg, dd.disformation_name)

            # Disformation (from Q) vanishes
            L_expr = disformation_from_nonmetricity(ac.nonmetricity_name; registry=reg)
            L_result = simplify(L_expr; registry=reg)
            @test L_result == TScalar(0 // 1)

            # Contortion (from T) does NOT vanish
            K_expr = contortion_from_torsion(ac.torsion_name; registry=reg)
            K_result = simplify(K_expr; registry=reg)
            @test K_result != TScalar(0 // 1)
        end
    end

    # ── Test 8: Einstein-Cartan action produces a valid scalar ─────────
    # L_EC = R (the Ricci scalar of the full torsioned connection).

    @testset "Einstein-Cartan action is a valid scalar" begin
        reg, ac = _ec_reg()
        with_registry(reg) do
            fs = define_ma_curvature!(reg, ac)
            ec = einstein_cartan_action(ac, fs; registry=reg)

            # Must be a Tensor (the Ricci scalar)
            @test ec isa Tensor
            @test ec.name == fs.scalar_name
            # Must have no free indices (it's a scalar)
            @test isempty(ec.indices)
            @test isempty(free_indices(ec))
        end
    end

    # ── Test 9: EC action = PGT with a₀=1, all others zero ────────────
    # Einstein-Cartan is the special case of PGT where only the EH term
    # survives: a₀=1, t₁=t₂=t₃=0, r₁=...=r₆=0.

    @testset "EC action consistent with PGT (a0=1, rest=0)" begin
        reg, ac = _ec_reg()
        with_registry(reg) do
            fs = define_ma_curvature!(reg, ac)

            # EC action
            ec = einstein_cartan_action(ac, fs; registry=reg)

            # PGT with EC parameters
            ec_params = PGTParams(; a0=1, t1=0, t2=0, t3=0,
                                    r1=0, r2=0, r3=0, r4=0, r5=0, r6=0)
            pgt = pgt_action(ac, fs, ec_params; registry=reg)

            # Both should be the same Ricci scalar tensor
            @test ec isa Tensor
            @test pgt isa Tensor
            @test ec.name == pgt.name
            @test ec.indices == pgt.indices
        end
    end

    # ── Test 10: Torsion quadratic invariants are well-formed ──────────
    # In EC theory, torsion is nonzero, so the quadratic invariants
    # I₁, I₂, I₃ should be nontrivial expressions (not zero).

    @testset "torsion quadratic invariants are nontrivial" begin
        reg, ac = _ec_reg()
        with_registry(reg) do
            tq = torsion_quadratic(ac; registry=reg)

            # All three invariants should be TensorExprs (not zero scalars)
            @test tq.I1 isa TensorExpr
            @test tq.I2 isa TensorExpr
            @test tq.I3 isa TensorExpr

            # They should be scalars: no free indices
            @test isempty(free_indices(tq.I1))
            @test isempty(free_indices(tq.I2))
            @test isempty(free_indices(tq.I3))
        end
    end

    # ── Test 11: MA curvature tensors have correct properties ──────────
    # The MA Riemann is R^a_{bcd} with antisymmetry in c,d only.
    # The MA Ricci R_{ab} is NOT symmetric (asymmetric in general).

    @testset "MA curvature tensors have correct properties" begin
        reg, ac = _ec_reg()
        with_registry(reg) do
            fs = define_ma_curvature!(reg, ac)

            # Riemann: rank (1,3), antisymmetric in last 2, no pair symmetry
            riem_tp = get_tensor(reg, fs.riemann_name)
            @test riem_tp.rank == (1, 3)
            @test any(s -> s isa AntiSymmetric, riem_tp.symmetries)
            @test get(riem_tp.options, :no_pair_symmetry, false)

            # Ricci: rank (0,2), no symmetry (asymmetric in general)
            ric_tp = get_tensor(reg, fs.ricci_name)
            @test ric_tp.rank == (0, 2)
            @test isempty(ric_tp.symmetries)
            @test get(ric_tp.options, :asymmetric, false)

            # Scalar: rank (0,0)
            scal_tp = get_tensor(reg, fs.scalar_name)
            @test scal_tp.rank == (0, 0)
        end
    end

    # ── Test 12: Torsion decomposition registers all 3 pieces ──────────
    # T^a_{bc} = vector + axial + tensor (irreducible decomposition).
    # In d=4: 24 = 4 + 4 + 16 components.

    @testset "torsion decomposition registers irreducible pieces" begin
        reg, ac = _ec_reg()
        with_registry(reg) do
            td = decompose_torsion!(reg, ac)

            # All three pieces registered
            @test has_tensor(reg, td.vector)
            @test has_tensor(reg, td.axial)
            @test has_tensor(reg, td.tensor)

            # Vector: rank (0,1)
            vec_tp = get_tensor(reg, td.vector)
            @test vec_tp.rank == (0, 1)

            # Axial: rank (1,0)
            ax_tp = get_tensor(reg, td.axial)
            @test ax_tp.rank == (1, 0)

            # Tensor: rank (1,2), antisymmetric in lower indices
            ten_tp = get_tensor(reg, td.tensor)
            @test ten_tp.rank == (1, 2)
            @test any(s -> s isa AntiSymmetric, ten_tp.symmetries)
        end
    end

end
