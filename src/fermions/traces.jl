#= Gamma matrix trace identities.
#
# Tr(I) = 4 (in d=4)
# Tr(γ^a) = 0
# Tr(γ^a γ^b) = 4 g^{ab}
# Tr(γ^a γ^b γ^c) = 0
# Tr(γ^a γ^b γ^c γ^d) = 4(g^{ab}g^{cd} - g^{ac}g^{bd} + g^{ad}g^{bc})
# Tr(odd number of gammas) = 0
#
# Ground truth: Peskin & Schroeder (1995) Appendix A, Eqs A.21-A.28.
=#

"""
    gamma_chain_trace(gammas::Vector{GammaMatrix};
                       metric::Symbol=:g, dim::Int=4) -> TensorExpr

Evaluate the trace of a chain of gamma matrices.

Uses the standard trace identities:
- Tr(I) = d_s (spinor dimension, = 4 in d=4)
- Tr(γ^a₁...γ^aₙ) = 0 for odd n
- Tr(γ^a γ^b) = d_s g^{ab}
- Tr(γ^a γ^b γ^c γ^d) = d_s (g^{ab}g^{cd} - g^{ac}g^{bd} + g^{ad}g^{bc})

For n ≥ 6, uses the recursive relation:
    Tr(γ^a₁...γ^aₙ) = Σᵢ (-1)^{i+1} g^{a₁aᵢ} Tr(γ^a₂...γ̂^aᵢ...γ^aₙ)

Ground truth: Peskin & Schroeder (1995) Eqs A.21-A.28.
"""
function gamma_chain_trace(gammas::Vector{GammaMatrix};
                            metric::Symbol=:g, dim::Int=4)
    n = length(gammas)
    spinor_dim = dim  # Tr(I) in d dimensions

    # Empty trace
    n == 0 && return TScalar(Rational{Int}(spinor_dim))

    # Odd traces vanish
    n % 2 == 1 && return TScalar(0 // 1)

    # Tr(γ^a γ^b) = d_s g^{ab}
    if n == 2
        a, b = gammas[1].index, gammas[2].index
        return tproduct(Rational{Int}(spinor_dim),
                        TensorExpr[Tensor(metric, [a, b])])
    end

    # Tr(γ^a γ^b γ^c γ^d) = d_s (g^{ab}g^{cd} - g^{ac}g^{bd} + g^{ad}g^{bc})
    if n == 4
        a, b, c, d = [g.index for g in gammas]
        g_ab = Tensor(metric, [a, b])
        g_cd = Tensor(metric, [c, d])
        g_ac = Tensor(metric, [a, c])
        g_bd = Tensor(metric, [b, d])
        g_ad = Tensor(metric, [a, d])
        g_bc = Tensor(metric, [b, c])

        return tproduct(Rational{Int}(spinor_dim),
                        TensorExpr[g_ab * g_cd - g_ac * g_bd + g_ad * g_bc])
    end

    # n ≥ 6: recursive relation
    # Tr(γ^a₁ γ^a₂ ... γ^aₙ) = Σᵢ₌₂ⁿ (-1)^i g^{a₁aᵢ} Tr(γ^a₂...γ̂^aᵢ...γ^aₙ)
    a1 = gammas[1].index
    terms = TensorExpr[]

    for i in 2:n
        sign = (-1)^i  # (-1)^i gives alternating +/- starting from i=2: +, -, +, ...
        ai = gammas[i].index
        g_metric = Tensor(metric, [a1, ai])

        # Sub-chain: gammas[2:n] with gammas[i] removed
        sub = GammaMatrix[gammas[j] for j in 2:n if j != i]
        sub_trace = gamma_chain_trace(sub; metric=metric, dim=dim)

        if sign == 1
            push!(terms, g_metric * sub_trace)
        else
            push!(terms, tproduct(-1 // 1, TensorExpr[g_metric, sub_trace]))
        end
    end

    tsum(terms)
end

"""
    trace_identity_2(a::TIndex, b::TIndex; metric::Symbol=:g, dim::Int=4) -> TensorExpr

Tr(γ^a γ^b) = d_s g^{ab}.
"""
function trace_identity_2(a::TIndex, b::TIndex; metric::Symbol=:g, dim::Int=4)
    tproduct(Rational{Int}(dim), TensorExpr[Tensor(metric, [a, b])])
end

"""
    trace_identity_4(a::TIndex, b::TIndex, c::TIndex, d::TIndex;
                      metric::Symbol=:g, dim::Int=4) -> TensorExpr

Tr(γ^a γ^b γ^c γ^d) = d_s (g^{ab}g^{cd} - g^{ac}g^{bd} + g^{ad}g^{bc}).
"""
function trace_identity_4(a::TIndex, b::TIndex, c::TIndex, d::TIndex;
                            metric::Symbol=:g, dim::Int=4)
    g_ab = Tensor(metric, [a, b])
    g_cd = Tensor(metric, [c, d])
    g_ac = Tensor(metric, [a, c])
    g_bd = Tensor(metric, [b, d])
    g_ad = Tensor(metric, [a, d])
    g_bc = Tensor(metric, [b, c])

    tproduct(Rational{Int}(dim),
             TensorExpr[g_ab * g_cd - g_ac * g_bd + g_ad * g_bc])
end

"""
    slash(v::Tensor; registry::TensorRegistry=current_registry()) -> TensorExpr

Feynman slash notation: v̸ = γ^a v_a (contraction of vector with gamma matrix).

Returns a TProduct of GammaMatrix and the vector.
"""
function slash(v::Tensor; registry::TensorRegistry=current_registry())
    length(v.indices) == 1 || error("slash requires a vector (rank 1), got rank $(length(v.indices))")
    idx = v.indices[1]
    # Contract gamma with the vector
    used = Set{Symbol}([idx.name])
    d = fresh_index(used)
    gamma = GammaMatrix(up(d))
    v_renamed = Tensor(v.name, [down(d)])
    tproduct(1 // 1, TensorExpr[gamma, v_renamed])
end
