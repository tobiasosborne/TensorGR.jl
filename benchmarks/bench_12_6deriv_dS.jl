# ============================================================================
# Benchmark 12: 6-Derivative Gravity on de Sitter
#
# Compute δ²[I_i] for 6 independent cubic curvature invariants on a maximally
# symmetric (de Sitter) background. Tests the full pipeline: perturbation
# expansion, metric contraction, curvature contraction, canonicalization,
# and background rule application.
#
# Invariants:
#   I1 = R^3,  I2 = R Ric^2,  I3 = Ric^3,
#   I4 = R Riem^2,  I5 = Ric Riem^2,  I6 = Riem^3 (Goroff-Sagnotti)
#
# Exercises: curved-background perturbation (order 2), Leibniz rule,
#            large-expression simplification, optional parallelism
# ============================================================================

using TensorGR, Test
include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "ground_truth.jl"))

# ── Ground truth: pinned term counts ─────────────────────────────────────────

const SIXDERIV_RAW_TERMS = Dict(
    1 => 6, 2 => 13, 3 => 18, 4 => 24, 5 => 31, 6 => 39
)
const SIXDERIV_SIMPLIFIED_TERMS = Dict(
    1 => 324, 2 => 1042, 3 => 1144, 4 => 1344, 5 => 1202, 6 => 1488
)

# ── Setup ────────────────────────────────────────────────────────────────────

function _setup_6deriv()
    reg = TensorRegistry()
    with_registry(reg) do
        @manifold M4 dim=4 metric=g
        define_curvature_tensors!(reg, :M4, :g)
        maximally_symmetric_background!(reg, :M4; metric=:g, cosmological_constant=:Λ)
        @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
        mp = define_metric_perturbation!(reg, :g, :h; curved=true)
        return reg, mp
    end
end

# Invariant builders (all-down Riemann + explicit g^{..} for contractions)

function _build_I1(reg)
    R1 = Tensor(:RicScalar, TIndex[])
    R2 = Tensor(:RicScalar, TIndex[])
    R3 = Tensor(:RicScalar, TIndex[])
    R1 * R2 * R3
end

function _build_I2(reg)
    R = Tensor(:RicScalar, TIndex[])
    Ric1 = Tensor(:Ric, [down(:a), down(:b)])
    Ric2 = Tensor(:Ric, [down(:c), down(:d)])
    R * Ric1 * Ric2 * Tensor(:g, [up(:a), up(:c)]) * Tensor(:g, [up(:b), up(:d)])
end

function _build_I3(reg)
    Ric1 = Tensor(:Ric, [down(:a), down(:e)])
    Ric2 = Tensor(:Ric, [down(:b), down(:f)])
    Ric3 = Tensor(:Ric, [down(:c), down(:g)])
    Ric1 * Ric2 * Ric3 *
        Tensor(:g, [up(:e), up(:b)]) * Tensor(:g, [up(:f), up(:c)]) * Tensor(:g, [up(:g), up(:a)])
end

function _build_I4(reg)
    R = Tensor(:RicScalar, TIndex[])
    Riem1 = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
    Riem2 = Tensor(:Riem, [down(:e), down(:f), down(:g), down(:h)])
    R * Riem1 * Riem2 *
        Tensor(:g, [up(:a), up(:e)]) * Tensor(:g, [up(:b), up(:f)]) *
        Tensor(:g, [up(:c), up(:g)]) * Tensor(:g, [up(:d), up(:h)])
end

function _build_I5(reg)
    Ric = Tensor(:Ric, [down(:p), down(:q)])
    Riem1 = Tensor(:Riem, [down(:a), down(:c), down(:d), down(:e)])
    Riem2 = Tensor(:Riem, [down(:b), down(:f), down(:g), down(:h)])
    Ric * Riem1 * Riem2 *
        Tensor(:g, [up(:p), up(:a)]) * Tensor(:g, [up(:q), up(:b)]) *
        Tensor(:g, [up(:c), up(:f)]) * Tensor(:g, [up(:d), up(:g)]) * Tensor(:g, [up(:e), up(:h)])
end

function _build_I6(reg)
    Riem1 = Tensor(:Riem, [down(:a), down(:b), down(:i), down(:j)])
    Riem2 = Tensor(:Riem, [down(:c), down(:d), down(:k), down(:l)])
    Riem3 = Tensor(:Riem, [down(:e), down(:f), down(:m), down(:n)])
    Riem1 * Riem2 * Riem3 *
        Tensor(:g, [up(:i), up(:c)]) * Tensor(:g, [up(:j), up(:d)]) *
        Tensor(:g, [up(:k), up(:e)]) * Tensor(:g, [up(:l), up(:f)]) *
        Tensor(:g, [up(:m), up(:a)]) * Tensor(:g, [up(:n), up(:b)])
end

const _BUILDERS_6D = [_build_I1, _build_I2, _build_I3, _build_I4, _build_I5, _build_I6]
const _NAMES_6D = ["R^3", "R*Ric^2", "Ric^3", "R*Riem^2", "Ric*Riem^2", "Riem^3"]

# ── Tests ────────────────────────────────────────────────────────────────────

@testset "Bench 12: 6-Derivative Gravity (de Sitter)" begin
    reg, mp = _setup_6deriv()
    with_registry(reg) do

        # ── 12.1: Raw perturbation term counts ─────────────────────────
        @testset "Raw term counts (δ² expansion)" begin
            for i in 1:6
                expr = _BUILDERS_6D[i](reg)
                raw = expand_perturbation(expr, mp, 2)
                @test count_terms(raw) == SIXDERIV_RAW_TERMS[i]
            end
        end

        # ── 12.2: Simplified term counts (I1-I3, faster) ──────────────
        @testset "Simplified terms: $(_NAMES_6D[i])" for i in 1:3
            expr = _BUILDERS_6D[i](reg)
            raw = expand_perturbation(expr, mp, 2)
            tc = timed_compute() do
                simplify(raw; registry=reg)
            end
            @test count_terms(tc.result) == SIXDERIV_SIMPLIFIED_TERMS[i]
        end

        # ── 12.3: I4-I6 (heavier, verify term counts) ─────────────────
        @testset "Simplified terms: $(_NAMES_6D[i])" for i in 4:6
            expr = _BUILDERS_6D[i](reg)
            raw = expand_perturbation(expr, mp, 2)
            tc = timed_compute() do
                simplify(raw; registry=reg)
            end
            @test count_terms(tc.result) == SIXDERIV_SIMPLIFIED_TERMS[i]
        end

        # ── 12.4: Parallel consistency (I1) ────────────────────────────
        if Threads.nthreads() > 1
            @testset "Parallel == serial (I1)" begin
                expr = _build_I1(reg)
                raw = expand_perturbation(expr, mp, 2)
                serial = simplify(raw; registry=reg, parallel=false)
                par = simplify(raw; registry=reg, parallel=true)
                @test count_terms(serial) == count_terms(par)
                diff = simplify(tsum(TensorExpr[serial, tproduct(-1 // 1, TensorExpr[par])]);
                               registry=reg)
                @test diff == TScalar(0 // 1)
            end
        end

        # ── 12.5: Gauss-Bonnet structure ───────────────────────────────
        @testset "Gauss-Bonnet E4 raw expansion" begin
            # E₄ = Riem² - 4 Ric² + R² is topological; δ²[E₄] is a total
            # derivative, not identically zero as a tensor expression.
            Riem1 = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            Riem2 = Tensor(:Riem, [down(:e), down(:f), down(:g), down(:h)])
            kretschner = Riem1 * Riem2 *
                Tensor(:g, [up(:a), up(:e)]) * Tensor(:g, [up(:b), up(:f)]) *
                Tensor(:g, [up(:c), up(:g)]) * Tensor(:g, [up(:d), up(:h)])
            Ric1 = Tensor(:Ric, [down(:i), down(:j)])
            Ric2 = Tensor(:Ric, [down(:k), down(:l)])
            ricci_sq = Ric1 * Ric2 * Tensor(:g, [up(:i), up(:k)]) * Tensor(:g, [up(:j), up(:l)])
            R = Tensor(:RicScalar, TIndex[])
            E4 = kretschner + tproduct(-4 // 1, TensorExpr[ricci_sq]) + R * R

            E4_raw = expand_perturbation(E4, mp, 2)
            @test count_terms(E4_raw) == 28
            @test E4_raw != TScalar(0 // 1)
        end

        # ── 12.6: Background values (dS identity checks) ──────────────
        @testset "Background curvature values on dS" begin
            # R = 4Λ on dS (dim=4)
            R = Tensor(:RicScalar, TIndex[])
            R_bg = simplify(R; registry=reg)
            @test R_bg isa TProduct  # 4*Λ
        end

        # ── 12.7: Covariant output path (∇h, no Γ₀g) ────────────────
        @testset "Covariant output: R³ raw terms" begin
            # Setup covariant perturbation (separate mp to avoid conflict)
            mp_cov = define_metric_perturbation!(reg, :g, :h;
                curved=true, covariant_output=true)

            # R³ raw terms should match non-covariant (same partition count)
            expr = _build_I1(reg)
            raw = expand_perturbation(expr, mp_cov, 2)
            @test count_terms(raw) == 6

            # Verify no background Christoffel in output (covariant path)
            @test mp_cov.background_christoffel === nothing
            @test mp_cov.covd_name === :∇g
        end
    end
end
