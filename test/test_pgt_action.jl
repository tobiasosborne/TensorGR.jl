# Ground truth: Blagojevic & Hehl, "Gauge Theories of Gravitation" (2013), Ch 5;
#               Hehl, McCrea, Mielke & Ne'eman, Phys. Rep. 258 (1995), Sec 5.

using Test
using TensorGR
using TensorGR: torsion_quadratic, curvature_quadratic_ma, pgt_action,
                einstein_cartan_action, PGTParams, free_indices

@testset "Poincare Gauge Theory Action" begin

    function _pgt_reg()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
        end
        ac = define_affine_connection!(reg, :Gamma; manifold=:M4, metric=:g)
        fs = define_ma_curvature!(reg, ac)
        reg, ac, fs
    end

    # ── PGTParams ────────────────────────────────────────────────────────

    @testset "PGTParams constructor" begin
        # Default: a0=1, all others zero
        p = PGTParams()
        @test p.a0 == 1
        @test p.t == (0, 0, 0)
        @test p.r == (0, 0, 0, 0, 0, 0)

        # Keyword constructor
        p2 = PGTParams(a0=2, t1=:alpha, t2=3, r1=1//2, r5=:beta)
        @test p2.a0 == 2
        @test p2.t == (:alpha, 3, 0)
        @test p2.r == (1//2, 0, 0, 0, :beta, 0)
    end

    @testset "PGTParams display" begin
        p = PGTParams()
        s = sprint(show, p)
        @test occursin("PGTParams", s)
    end

    # ── Torsion quadratic invariants ─────────────────────────────────────

    @testset "torsion_quadratic: returns 3 invariants" begin
        reg, ac, fs = _pgt_reg()
        with_registry(reg) do
            tq = torsion_quadratic(ac; registry=reg)

            @test hasproperty(tq, :I1)
            @test hasproperty(tq, :I2)
            @test hasproperty(tq, :I3)
        end
    end

    @testset "torsion_quadratic: all are scalars (no free indices)" begin
        reg, ac, fs = _pgt_reg()
        with_registry(reg) do
            tq = torsion_quadratic(ac; registry=reg)

            @test isempty(free_indices(tq.I1))
            @test isempty(free_indices(tq.I2))
            @test isempty(free_indices(tq.I3))
        end
    end

    @testset "torsion_quadratic: contains torsion tensors" begin
        reg, ac, fs = _pgt_reg()
        with_registry(reg) do
            tq = torsion_quadratic(ac; registry=reg)
            T_name = ac.torsion_name

            # Check each invariant contains torsion factors
            for inv in (tq.I1, tq.I2, tq.I3)
                if inv isa TProduct
                    names = [f.name for f in inv.factors if f isa Tensor]
                    @test T_name in names
                elseif inv isa Tensor
                    @test inv.name == T_name
                end
            end
        end
    end

    @testset "torsion_quadratic: I1 and I2 have correct structure" begin
        reg, ac, fs = _pgt_reg()
        with_registry(reg) do
            tq = torsion_quadratic(ac; registry=reg)

            # I1 and I2 are TProducts with 5 factors (3 metrics + 2 torsions)
            @test tq.I1 isa TProduct
            @test tq.I2 isa TProduct

            # Count tensor factors
            n_factors_I1 = length(tq.I1.factors)
            n_factors_I2 = length(tq.I2.factors)
            @test n_factors_I1 == 5  # 3 metrics + 2 torsions
            @test n_factors_I2 == 5
        end
    end

    @testset "torsion_quadratic: I3 is trace-squared" begin
        reg, ac, fs = _pgt_reg()
        with_registry(reg) do
            tq = torsion_quadratic(ac; registry=reg)

            # I3 = T^a_{ba} T^c_{dc} g^{bd} — 3 factors (2 self-contracting torsions + 1 metric)
            @test tq.I3 isa TProduct
            @test length(tq.I3.factors) == 3  # 2 torsions + 1 metric
        end
    end

    # ── Curvature quadratic invariants ───────────────────────────────────

    @testset "curvature_quadratic_ma: returns 6 invariants" begin
        reg, ac, fs = _pgt_reg()
        with_registry(reg) do
            cq = curvature_quadratic_ma(ac, fs; registry=reg)

            @test hasproperty(cq, :J1)
            @test hasproperty(cq, :J2)
            @test hasproperty(cq, :J3)
            @test hasproperty(cq, :J4)
            @test hasproperty(cq, :J5)
            @test hasproperty(cq, :J6)
        end
    end

    @testset "curvature_quadratic_ma: all are scalars" begin
        reg, ac, fs = _pgt_reg()
        with_registry(reg) do
            cq = curvature_quadratic_ma(ac, fs; registry=reg)

            @test isempty(free_indices(cq.J1))
            @test isempty(free_indices(cq.J2))
            @test isempty(free_indices(cq.J3))
            @test isempty(free_indices(cq.J4))
            @test isempty(free_indices(cq.J5))
            @test isempty(free_indices(cq.J6))
        end
    end

    @testset "curvature_quadratic_ma: J1, J2 contain MA Riemann" begin
        reg, ac, fs = _pgt_reg()
        with_registry(reg) do
            cq = curvature_quadratic_ma(ac, fs; registry=reg)
            Riem = fs.riemann_name

            for inv in (cq.J1, cq.J2)
                @test inv isa TProduct
                names = [f.name for f in inv.factors if f isa Tensor]
                @test Riem in names
            end
        end
    end

    @testset "curvature_quadratic_ma: J3, J4 contain MA Ricci" begin
        reg, ac, fs = _pgt_reg()
        with_registry(reg) do
            cq = curvature_quadratic_ma(ac, fs; registry=reg)
            Ric = fs.ricci_name

            for inv in (cq.J3, cq.J4)
                @test inv isa TProduct
                names = [f.name for f in inv.factors if f isa Tensor]
                @test Ric in names
            end
        end
    end

    @testset "curvature_quadratic_ma: J5 is scalar squared" begin
        reg, ac, fs = _pgt_reg()
        with_registry(reg) do
            cq = curvature_quadratic_ma(ac, fs; registry=reg)
            R = fs.scalar_name

            @test cq.J5 isa TProduct
            names = [f.name for f in cq.J5.factors if f isa Tensor]
            @test count(==(R), names) == 2
        end
    end

    @testset "curvature_quadratic_ma: J6 is antisymmetric Ricci squared" begin
        reg, ac, fs = _pgt_reg()
        with_registry(reg) do
            cq = curvature_quadratic_ma(ac, fs; registry=reg)

            # J6 = (1/2)(J3 - J4), a TSum of two TProducts
            @test cq.J6 isa TSum
            @test length(cq.J6.terms) == 2
        end
    end

    @testset "curvature_quadratic_ma: J1 factor count" begin
        reg, ac, fs = _pgt_reg()
        with_registry(reg) do
            cq = curvature_quadratic_ma(ac, fs; registry=reg)

            # J1 = g_.. g^.. g^.. g^.. R^._{...} R^._{...} → 6 factors
            @test cq.J1 isa TProduct
            @test length(cq.J1.factors) == 6  # 4 metrics + 2 Riemanns
        end
    end

    # ── PGT action ───────────────────────────────────────────────────────

    @testset "pgt_action: returns scalar" begin
        reg, ac, fs = _pgt_reg()
        with_registry(reg) do
            params = PGTParams(a0=1, t1=:t1, r1=:r1)
            L = pgt_action(ac, fs, params; registry=reg)

            @test isempty(free_indices(L))
        end
    end

    @testset "pgt_action: only a0 gives scalar R" begin
        reg, ac, fs = _pgt_reg()
        with_registry(reg) do
            params = PGTParams(a0=1)
            L = pgt_action(ac, fs, params; registry=reg)

            # With only a0=1, should reduce to just the Ricci scalar
            @test L isa Tensor
            @test L.name == fs.scalar_name
        end
    end

    @testset "pgt_action: a0=2 gives 2*R" begin
        reg, ac, fs = _pgt_reg()
        with_registry(reg) do
            params = PGTParams(a0=2)
            L = pgt_action(ac, fs, params; registry=reg)

            # 2*R is a TProduct with scalar 2
            @test L isa TProduct
            @test L.scalar == 2 // 1
        end
    end

    @testset "pgt_action: all zeros gives zero" begin
        reg, ac, fs = _pgt_reg()
        with_registry(reg) do
            params = PGTParams(a0=0)
            L = pgt_action(ac, fs, params; registry=reg)

            @test L isa TScalar
            @test L.val == 0
        end
    end

    @testset "pgt_action: torsion term included" begin
        reg, ac, fs = _pgt_reg()
        with_registry(reg) do
            params = PGTParams(a0=0, t1=1)
            L = pgt_action(ac, fs, params; registry=reg)

            @test isempty(free_indices(L))
            # Should contain torsion tensor
            all_tensors = TensorExpr[]
            TensorGR.walk(t -> (push!(all_tensors, t); t), L)
            names = [t.name for t in all_tensors if t isa Tensor]
            @test ac.torsion_name in names
        end
    end

    @testset "pgt_action: curvature term included" begin
        reg, ac, fs = _pgt_reg()
        with_registry(reg) do
            params = PGTParams(a0=0, r3=1)
            L = pgt_action(ac, fs, params; registry=reg)

            @test isempty(free_indices(L))
            # Should contain MA Ricci tensor
            all_tensors = TensorExpr[]
            TensorGR.walk(t -> (push!(all_tensors, t); t), L)
            names = [t.name for t in all_tensors if t isa Tensor]
            @test fs.ricci_name in names
        end
    end

    @testset "pgt_action: full action with symbolic couplings" begin
        reg, ac, fs = _pgt_reg()
        with_registry(reg) do
            params = PGTParams(a0=:kappa, t1=:t1, t2=:t2, t3=:t3,
                               r1=:r1, r2=:r2, r3=:r3, r4=:r4, r5=:r5, r6=:r6)
            L = pgt_action(ac, fs, params; registry=reg)

            # Full action is a TSum with 10 terms (1 EH + 3 torsion + 6 curvature)
            @test L isa TSum
            @test length(L.terms) == 10
        end
    end

    # ── Einstein-Cartan action ───────────────────────────────────────────

    @testset "einstein_cartan_action: returns Ricci scalar" begin
        reg, ac, fs = _pgt_reg()
        with_registry(reg) do
            ec = einstein_cartan_action(ac, fs; registry=reg)

            @test ec isa Tensor
            @test ec.name == fs.scalar_name
            @test isempty(ec.indices)
        end
    end

    @testset "einstein_cartan_action equals pgt_action(a0=1)" begin
        reg, ac, fs = _pgt_reg()
        with_registry(reg) do
            ec = einstein_cartan_action(ac, fs; registry=reg)
            pgt_ec = pgt_action(ac, fs, PGTParams(a0=1); registry=reg)

            # Both should be the same Ricci scalar
            @test ec isa Tensor
            @test pgt_ec isa Tensor
            @test ec.name == pgt_ec.name
        end
    end
end
