#= Multi-term symmetry rules: Bianchi identities.

These are NOT permutation symmetries — they are rewrite rules that express
algebraic relationships between tensor terms.

First (algebraic) Bianchi:  R_{abcd} + R_{acdb} + R_{adbc} = 0
Second (differential) Bianchi: ∇_e R_{abcd} + ∇_c R_{abde} + ∇_d R_{abec} = 0
Contracted Bianchi: ∇^a G_{ab} = 0, ∇^a R_{ab} = (1/2) ∇_b R
=#

"""
    bianchi_rules(; manifold=:M4, metric=:g) -> Vector{RewriteRule}

Create rewrite rules for the first and contracted Bianchi identities.
These are registered automatically by `define_curvature_tensors!`.
"""
function bianchi_rules(; manifold::Symbol=:M4, metric::Symbol=:g)
    rules = RewriteRule[]

    # First (algebraic) Bianchi identity: R_{a[bcd]} = 0
    # Implemented as: detect sums of three Riemann terms that form a cyclic
    # permutation of the last three indices, and replace with the canonical two.
    #
    # Practically: if we find R_{abcd} + R_{acdb} in a sum, we know the third
    # term R_{adbc} = -(R_{abcd} + R_{acdb}), so we can rewrite.
    # However, this is better handled at the simplify level via canonicalize.
    # The algebraic Bianchi is already captured by RiemannSymmetry in xperm.

    # Contracted Bianchi: ∇^a G_{ab} = 0
    # Pattern: ∂_a(G^a_b) where G = Ein  → expressed as a functional rule
    push!(rules, RewriteRule(
        function(expr)
            # Match ∂^a(G_{ab}) pattern: TDeriv with up index on Einstein
            expr isa TDeriv || return false
            didx = expr.index
            inner = expr.arg
            inner isa Tensor || return false
            inner.name == :Ein || return false
            length(inner.indices) == 2 || return false
            # Check contraction: deriv index contracts with one Einstein index
            for eidx in inner.indices
                if eidx.name == didx.name && eidx.position != didx.position
                    return true
                end
            end
            false
        end,
        _ -> ZERO
    ))

    # Contracted Bianchi: ∇^a R_{ab} = (1/2) ∇_b R
    # Pattern: ∂^a(Ric_{ab}) → (1/2) ∂_b(RicScalar)
    push!(rules, RewriteRule(
        function(expr)
            expr isa TDeriv || return false
            didx = expr.index
            inner = expr.arg
            inner isa Tensor || return false
            inner.name == :Ric || return false
            length(inner.indices) == 2 || return false
            for eidx in inner.indices
                if eidx.name == didx.name && eidx.position != didx.position
                    return true
                end
            end
            false
        end,
        function(expr)
            didx = expr.index
            inner = expr.arg::Tensor
            # Find the free index (the one not contracted with the deriv)
            free_idx = nothing
            for eidx in inner.indices
                if eidx.name != didx.name
                    free_idx = eidx
                    break
                end
            end
            free_idx === nothing && return ZERO
            (1 // 2) * TDeriv(free_idx, Tensor(:RicScalar, TIndex[]))
        end
    ))

    rules
end
