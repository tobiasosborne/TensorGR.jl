# ============================================================================
# 6-Derivative Gravity on de Sitter: Cubic Curvature Invariants
#
# Compute delta^2[I_i] for 6 independent cubic curvature monomials on a
# maximally symmetric (de Sitter) background, with optional parallelism.
#
# Invariants:
#   I1 = R^3
#   I2 = R * R_{ab} R^{ab}
#   I3 = R_a^b R_b^c R_c^a
#   I4 = R * R_{abcd} R^{abcd}
#   I5 = R^{ab} R_{acde} R_b^{cde}
#   I6 = R_{ab}^{cd} R_{cd}^{ef} R_{ef}^{ab}   (Goroff-Sagnotti)
#
# Usage:
#   julia --project=. examples/11_6deriv_gravity_dS.jl
#   julia -t8 --project=. examples/11_6deriv_gravity_dS.jl   # parallel
# ============================================================================

using TensorGR

# ── Registry setup ──────────────────────────────────────────────────────────

function setup_dS_registry()
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

# ── Invariant builders ──────────────────────────────────────────────────────
# Convention: use all-down Riemann + explicit g^{..} for contractions,
# and Ric with both down indices + g^{..} for raising.

function build_I1(reg)
    # I1 = R^3
    R1 = Tensor(:RicScalar, TIndex[])
    R2 = Tensor(:RicScalar, TIndex[])
    R3 = Tensor(:RicScalar, TIndex[])
    R1 * R2 * R3
end

function build_I2(reg)
    # I2 = R * R_{ab} R^{ab} = R * R_{ab} R_{cd} g^{ac} g^{bd}
    R = Tensor(:RicScalar, TIndex[])
    Ric1 = Tensor(:Ric, [down(:a), down(:b)])
    Ric2 = Tensor(:Ric, [down(:c), down(:d)])
    g_ac = Tensor(:g, [up(:a), up(:c)])
    g_bd = Tensor(:g, [up(:b), up(:d)])
    R * Ric1 * Ric2 * g_ac * g_bd
end

function build_I3(reg)
    # I3 = R_a^b R_b^c R_c^a = R_{ae} R_{bf} R_{cg} g^{eb} g^{fc} g^{ga}
    Ric1 = Tensor(:Ric, [down(:a), down(:e)])
    Ric2 = Tensor(:Ric, [down(:b), down(:f)])
    Ric3 = Tensor(:Ric, [down(:c), down(:g)])
    g1 = Tensor(:g, [up(:e), up(:b)])
    g2 = Tensor(:g, [up(:f), up(:c)])
    g3 = Tensor(:g, [up(:g), up(:a)])
    Ric1 * Ric2 * Ric3 * g1 * g2 * g3
end

function build_I4(reg)
    # I4 = R * R_{abcd} R^{abcd}
    #    = R * R_{abcd} R_{efgh} g^{ae} g^{bf} g^{cg} g^{dh}
    R = Tensor(:RicScalar, TIndex[])
    Riem1 = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
    Riem2 = Tensor(:Riem, [down(:e), down(:f), down(:g), down(:h)])
    g1 = Tensor(:g, [up(:a), up(:e)])
    g2 = Tensor(:g, [up(:b), up(:f)])
    g3 = Tensor(:g, [up(:c), up(:g)])
    g4 = Tensor(:g, [up(:d), up(:h)])
    R * Riem1 * Riem2 * g1 * g2 * g3 * g4
end

function build_I5(reg)
    # I5 = R^{ab} R_{acde} R_b^{cde}
    #    = R_{pq} R_{acde} R_{bfgh} g^{pa} g^{qb} g^{cf} g^{dg} g^{eh}
    Ric = Tensor(:Ric, [down(:p), down(:q)])
    Riem1 = Tensor(:Riem, [down(:a), down(:c), down(:d), down(:e)])
    Riem2 = Tensor(:Riem, [down(:b), down(:f), down(:g), down(:h)])
    g_pa = Tensor(:g, [up(:p), up(:a)])
    g_qb = Tensor(:g, [up(:q), up(:b)])
    g_cf = Tensor(:g, [up(:c), up(:f)])
    g_dg = Tensor(:g, [up(:d), up(:g)])
    g_eh = Tensor(:g, [up(:e), up(:h)])
    Ric * Riem1 * Riem2 * g_pa * g_qb * g_cf * g_dg * g_eh
end

function build_I6(reg)
    # I6 = R_{ab}^{cd} R_{cd}^{ef} R_{ef}^{ab}  (Goroff-Sagnotti)
    #    = R_{abij} R_{cdkl} R_{efmn}
    #      g^{ic} g^{jd} g^{ke} g^{lf} g^{ma} g^{nb}
    Riem1 = Tensor(:Riem, [down(:a), down(:b), down(:i), down(:j)])
    Riem2 = Tensor(:Riem, [down(:c), down(:d), down(:k), down(:l)])
    Riem3 = Tensor(:Riem, [down(:e), down(:f), down(:m), down(:n)])
    g1 = Tensor(:g, [up(:i), up(:c)])
    g2 = Tensor(:g, [up(:j), up(:d)])
    g3 = Tensor(:g, [up(:k), up(:e)])
    g4 = Tensor(:g, [up(:l), up(:f)])
    g5 = Tensor(:g, [up(:m), up(:a)])
    g6 = Tensor(:g, [up(:n), up(:b)])
    Riem1 * Riem2 * Riem3 * g1 * g2 * g3 * g4 * g5 * g6
end

# ── Computation ─────────────────────────────────────────────────────────────

const INVARIANT_NAMES = ["R^3", "R*Ric^2", "Ric^3", "R*Riem^2", "Ric*Riem^2", "Riem^3"]
const INVARIANT_BUILDERS = [build_I1, build_I2, build_I3, build_I4, build_I5, build_I6]

function run_6deriv_gravity(; parallel::Bool=Threads.nthreads() > 1,
                              invariants::AbstractVector{Int}=1:6)
    reg, mp = setup_dS_registry()

    use_parallel = parallel && Threads.nthreads() > 1
    nthreads = Threads.nthreads()

    println("=" ^ 72)
    println("6-DERIVATIVE GRAVITY ON DE SITTER ($(length(invariants)) invariants)")
    println("  Threads: $nthreads, parallel simplify: $use_parallel")
    println("=" ^ 72)

    results = Vector{Any}(undef, length(invariants))
    total_time = @elapsed begin
        if use_parallel && length(invariants) > 1
            # Level 1: parallel across invariants
            @sync for (slot, idx) in enumerate(invariants)
                let slot=slot, idx=idx
                    Threads.@spawn begin
                        with_registry(reg) do
                            t = @elapsed begin
                                expr = INVARIANT_BUILDERS[idx](reg)
                                raw = expand_perturbation(expr, mp, 2)
                                simplified = simplify(raw; registry=reg, parallel=true)
                                results[slot] = (name=INVARIANT_NAMES[idx],
                                                 raw_terms=count_terms(raw),
                                                 simplified_terms=count_terms(simplified),
                                                 result=simplified,
                                                 time=0.0)
                            end
                            results[slot] = (; results[slot]..., time=t)
                        end
                    end
                end
            end
        else
            # Serial fallback
            for (slot, idx) in enumerate(invariants)
                with_registry(reg) do
                    t = @elapsed begin
                        expr = INVARIANT_BUILDERS[idx](reg)
                        raw = expand_perturbation(expr, mp, 2)
                        simplified = simplify(raw; registry=reg, parallel=use_parallel)
                        results[slot] = (name=INVARIANT_NAMES[idx],
                                         raw_terms=count_terms(raw),
                                         simplified_terms=count_terms(simplified),
                                         result=simplified,
                                         time=0.0)
                    end
                    results[slot] = (; results[slot]..., time=t)
                end
            end
        end
    end

    # Report
    println()
    for r in results
        println("  $(rpad(r.name, 12))  raw=$(lpad(r.raw_terms, 6))  simplified=$(lpad(r.simplified_terms, 5))  $(round(r.time, digits=2))s")
    end
    println()
    println("  Total: $(round(total_time, digits=2))s")
    println("=" ^ 72)

    return results, reg, mp
end

# Helper used by benchmark
count_terms(e::TSum) = length(e.terms)
count_terms(e::TScalar) = e.val == 0 ? 0 : 1
count_terms(::TensorExpr) = 1

# ── Main ────────────────────────────────────────────────────────────────────

"""Build E4 = Riem^2 - 4 Ric^2 + R^2 using all-down indices + explicit metrics."""
function build_gauss_bonnet(reg)
    # Kretschner: R_{abcd} R_{efgh} g^{ae} g^{bf} g^{cg} g^{dh}
    Riem1 = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
    Riem2 = Tensor(:Riem, [down(:e), down(:f), down(:g), down(:h)])
    kretschner = Riem1 * Riem2 *
        Tensor(:g, [up(:a), up(:e)]) * Tensor(:g, [up(:b), up(:f)]) *
        Tensor(:g, [up(:c), up(:g)]) * Tensor(:g, [up(:d), up(:h)])

    # Ricci squared: R_{ij} R_{kl} g^{ik} g^{jl}
    Ric1 = Tensor(:Ric, [down(:i), down(:j)])
    Ric2 = Tensor(:Ric, [down(:k), down(:l)])
    ricci_sq = Ric1 * Ric2 * Tensor(:g, [up(:i), up(:k)]) * Tensor(:g, [up(:j), up(:l)])

    # R^2
    R = Tensor(:RicScalar, TIndex[])
    scalar_sq = R * R

    kretschner + tproduct(-4 // 1, TensorExpr[ricci_sq]) + scalar_sq
end

if abspath(PROGRAM_FILE) == @__FILE__
    results, reg, mp = run_6deriv_gravity()

    # Gauss-Bonnet verification
    println("\n--- Gauss-Bonnet Verification ---")
    with_registry(reg) do
        E4 = build_gauss_bonnet(reg)
        E4_pert = expand_perturbation(E4, mp, 2)
        n_raw = count_terms(E4_pert)
        E4_simplified = simplify(E4_pert; registry=reg, parallel=Threads.nthreads()>1)
        println("  delta^2[E4]: raw=$n_raw, simplified=$(count_terms(E4_simplified))")
        println("  ", E4_simplified == TScalar(0//1) ? "ZERO (topological, as expected)" : "non-zero")
    end
end
