# ============================================================================
# TensorGR.jl — Getting Started
#
# Basic setup: define a manifold, metric, tensors, and use simplification
# to verify fundamental tensor identities.
# ============================================================================

using TensorGR

# --- Set up a 4D Lorentzian manifold ---
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g

    # (1) Metric contraction: g^{ab} g_{bc} = δ^a_c
    product = Tensor(:g, [up(:a), up(:b)]) * Tensor(:g, [down(:b), down(:c)])
    result = simplify(product)
    println("g^{ab} g_{bc} = ", to_unicode(result))
    @assert result == Tensor(:δ, [up(:a), down(:c)])

    # (2) Trace of Kronecker delta: δ^a_a = dim = 4
    delta_trace = simplify(Tensor(:δ, [up(:a), down(:a)]))
    println("δ^a_a       = ", to_unicode(delta_trace))
    @assert delta_trace == TScalar(4 // 1)

    # (3) Define curvature tensors and check Riemann antisymmetry
    define_curvature_tensors!(reg, :M4, :g)

    R1 = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
    R2 = Tensor(:Riem, [down(:b), down(:a), down(:c), down(:d)])
    antisym = simplify(R1 + R2)
    println("R_{abcd} + R_{bacd} = ", to_unicode(antisym))
    @assert antisym == TScalar(0 // 1)

    # (4) Pair symmetry: R_{abcd} = R_{cdab}
    R3 = Tensor(:Riem, [down(:c), down(:d), down(:a), down(:b)])
    pair_sym = simplify(R1 - R3)
    println("R_{abcd} - R_{cdab} = ", to_unicode(pair_sym))
    @assert pair_sym == TScalar(0 // 1)

    # (5) Symmetric tensor: T_{ab} + T_{ba} = 2 T_{ab}
    @define_tensor T on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    T1 = Tensor(:T, [down(:a), down(:b)])
    T2 = Tensor(:T, [down(:b), down(:a)])
    sym_result = simplify(T1 + T2)
    println("T_{ab} + T_{ba} = ", to_unicode(sym_result))
    @assert sym_result.scalar == 2 // 1

    # (6) LaTeX output
    expr = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
    println("\nLaTeX: ", to_latex(expr))

    println("\nAll checks passed!")
end
