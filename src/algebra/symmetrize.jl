#= Symmetrization and antisymmetrization of tensor expressions.

Given a tensor expression and a list of index names, produce the
symmetrized or antisymmetrized expression:

  T_{(ab)} = (1/n!) Σ_σ T_{σ(a)σ(b)}
  T_{[ab]} = (1/n!) Σ_σ sign(σ) T_{σ(a)σ(b)}
=#

"""
    _permutations_with_sign(n) -> Vector{Tuple{Vector{Int}, Int}}

Return all permutations of `1:n` together with their signs (+1 or -1).
Uses Heap's algorithm to generate permutations, tracking sign via parity.
"""
function _permutations_with_sign(n::Int)
    n == 0 && return [(Int[], 1)]
    result = Tuple{Vector{Int}, Int}[]
    a = collect(1:n)
    c = ones(Int, n)
    push!(result, (copy(a), 1))
    sign = 1
    i = 1
    while i <= n
        if c[i] < i
            if isodd(i)
                a[1], a[i] = a[i], a[1]
            else
                a[c[i]], a[i] = a[i], a[c[i]]
            end
            sign = -sign
            push!(result, (copy(a), sign))
            c[i] += 1
            i = 1
        else
            c[i] = 1
            i += 1
        end
    end
    result
end

"""
    _apply_index_permutation(expr::TensorExpr, idxs::Vector{Symbol}, perm::Vector{Int}) -> TensorExpr

Apply a permutation to the named indices in `expr`. For a permutation `perm`,
index `idxs[k]` is replaced by `idxs[perm[k]]`.

Uses a two-phase renaming through temporary symbols to avoid collisions.
"""
function _apply_index_permutation(expr::TensorExpr, idxs::Vector{Symbol}, perm::Vector{Int})
    n = length(idxs)
    # Phase 1: rename each index to a temporary name
    tmps = [Symbol("__sym_tmp_", k) for k in 1:n]
    result = expr
    for k in 1:n
        result = rename_dummy(result, idxs[k], tmps[k])
    end
    # Phase 2: rename temporaries to the permuted targets
    for k in 1:n
        result = rename_dummy(result, tmps[k], idxs[perm[k]])
    end
    result
end

"""
    symmetrize(expr::TensorExpr, idxs::Vector{Symbol}) -> TensorExpr

Symmetrize `expr` over the indices named by `idxs`:

    T_{(a₁...aₙ)} = (1/n!) Σ_σ T_{σ(a₁)...σ(aₙ)}

The indices must all be free indices of `expr`. The result is a `TSum`
(or simplified form) with `n!` terms, each weighted by `1/n!`.
"""
function symmetrize(expr::TensorExpr, idxs::Vector{Symbol})
    n = length(idxs)
    n <= 1 && return expr

    perms = _permutations_with_sign(n)
    nfact = factorial(n)
    coeff = 1 // nfact

    terms = TensorExpr[]
    for (perm, _sign) in perms
        permuted = _apply_index_permutation(expr, idxs, perm)
        push!(terms, tproduct(coeff, TensorExpr[permuted]))
    end

    tsum(terms)
end

"""
    antisymmetrize(expr::TensorExpr, idxs::Vector{Symbol}) -> TensorExpr

Antisymmetrize `expr` over the indices named by `idxs`:

    T_{[a₁...aₙ]} = (1/n!) Σ_σ sign(σ) T_{σ(a₁)...σ(aₙ)}

The indices must all be free indices of `expr`. The result is a `TSum`
(or simplified form) with `n!` terms, each weighted by `sign(σ)/n!`.
"""
function antisymmetrize(expr::TensorExpr, idxs::Vector{Symbol})
    n = length(idxs)
    n <= 1 && return expr

    perms = _permutations_with_sign(n)
    nfact = factorial(n)

    terms = TensorExpr[]
    for (perm, sgn) in perms
        coeff = sgn // nfact
        permuted = _apply_index_permutation(expr, idxs, perm)
        push!(terms, tproduct(coeff, TensorExpr[permuted]))
    end

    tsum(terms)
end

"""
    impose_symmetry(expr::TensorExpr, sym, idxs::Vector{Symbol}) -> TensorExpr

Project an expression onto a symmetry subspace.
"""
function impose_symmetry(expr::TensorExpr, sym::Symbol, idxs::Vector{Symbol})
    if sym == :symmetric
        return symmetrize(expr, idxs)
    elseif sym == :antisymmetric
        return antisymmetrize(expr, idxs)
    else
        error("Unknown symmetry type: $sym. Use :symmetric or :antisymmetric")
    end
end

function impose_symmetry(expr::TensorExpr, sym::Symmetric, idxs::Vector{Symbol})
    symmetrize(expr, [idxs[sym.i], idxs[sym.j]])
end

function impose_symmetry(expr::TensorExpr, sym::AntiSymmetric, idxs::Vector{Symbol})
    antisymmetrize(expr, [idxs[sym.i], idxs[sym.j]])
end
