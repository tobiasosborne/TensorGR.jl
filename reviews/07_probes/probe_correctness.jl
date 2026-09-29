# Reproduction probes for reviews/07_julia_best_practices_review.md
#
# Run from the repo root (requires deps/libxperm.so; see review §3.1 — build.jl is broken,
# so compile by hand: gcc -shared -fPIC -O2 -o deps/libxperm.so deps/xperm.c):
#
#     julia --project reviews/07_probes/probe_correctness.jl
#
# Each line prints the label, the observed result, and what the correct result is.

using TensorGR
include(joinpath(@__DIR__, "setup.jl"))

function probe(label, f)
    print(rpad(label, 62), " => ")
    try
        println(f())
    catch e
        println("THROWS ", typeof(e), ": ", first(sprint(showerror, e), 110))
    end
end

reg = setup()
with_registry(reg) do
    S = Tensor(:S, [down(:c), down(:d)])
    DbDaS = TDeriv(down(:b), TDeriv(down(:a), S, :D), :D)
    V = Tensor(:V, [up(:e)])

    println("── §1a covariant derivatives treated as commuting ──")
    probe("A1 canonicalize D_b D_a S_cd            [correct: unchanged]",
          () -> canonicalize(TProduct(1//1, TensorExpr[DbDaS])))
    probe("A2 simplify V^e(D_bD_aS_cd - D_aD_bS_cd) [correct: Riemann terms]",
          () -> simplify(V * DbDaS - V * DaDbS_(S); registry=reg))
    probe("A3 same with ∂ (control)                 [correct: 0]",
          () -> simplify(TDeriv(down(:b), TDeriv(down(:a), S)) -
                         TDeriv(down(:a), TDeriv(down(:b), S)); registry=reg))

    println("── §1c display renders ∇ as ∂ ──")
    probe("C1 to_latex(D_a D_b S_cd)                [correct: \\nabla or D]",
          () -> to_latex(DbDaS))
    probe("C2 to_unicode(D_a D_b S_cd)",               () -> to_unicode(DbDaS))

    println("── §1b free-index positions not preserved ──")
    x = Tensor(:S, [up(:b), down(:a)])
    probe("B1 free_indices(S^b_a) vs canonicalize    [correct: equal]",
          () -> (free_indices(x), free_indices(canonicalize(x))))
    probe("B2 simplify A^b_a + A_a^b (A antisym)     [correct: 0]",
          () -> simplify(Tensor(:A, [up(:b), down(:a)]) + Tensor(:A, [down(:a), up(:b)]); registry=reg))

    println("── §1d no well-formedness validation ──")
    probe("D1 simplify V^a + V_a                     [correct: error]",
          () -> simplify(Tensor(:V, [up(:a)]) + Tensor(:V, [down(:a)]); registry=reg))
    probe("D2 simplify S^b_a - S^a_b                 [correct: error]",
          () -> simplify(Tensor(:S, [up(:b), down(:a)]) - Tensor(:S, [up(:a), down(:b)]); registry=reg))

    println("── §2.1 node types not handled by core passes ──")
    define_parameter!(reg, :t)
    probe("E1 simplify(TParamDeriv)",
          () -> simplify(TParamDeriv([:t], Tensor(:V, [up(:a)])); registry=reg))
    probe("E2 simplify(Y_{2,1} + Y_{2,1})",
          () -> simplify(ScalarHarmonic(2, 1) + ScalarHarmonic(2, 1); registry=reg))

    println("── §2.5 coefficient / scalar layer ──")
    probe("F1 (1//3)^20 * ((1//3)^20 * V)",           () -> (1//3)^20 * ((1//3)^20 * V))
    probe("F2 0.1 * V",                               () -> 0.1 * V)
    probe("F3 im * V",                                () -> im * V)
    probe("F4 TScalar(:x)^2 == simplify(x*x)          [correct: true]",
          () -> (TScalar(:x)^2) == simplify(TScalar(:x) * TScalar(:x); registry=reg))

    println("── §2.6 multiple zero representations ──")
    Z = TensorGR.ZERO
    probe("G1 ZERO == TSum([]), ZERO == 0*V (raw)     [correct: true, true]",
          () -> (Z == TSum(TensorExpr[]), Z == TProduct(0//1, TensorExpr[V])))
end

println("── §2.3 objectid-keyed global cache ──")
for _ in 1:50
    register_ddi_rules!(setup(); dim=4, order=2)
end
println(rpad("H1 _DDI_REGISTERED entries after 50 throwaway registries", 62), " => ",
        length(TensorGR._DDI_REGISTERED), "   [correct: 0 retained]")
