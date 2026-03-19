# Irreducible spinor decomposition.
#
# Any spinor decomposes into totally symmetric parts plus trace terms
# (epsilon contractions). For SL(2,C):
#
#   Rank-2: phi_{AB} = phi_{(AB)} + (1/2) epsilon_{AB} phi^C_C
#   Rank-3: phi_{ABC} = phi_{(ABC)} + epsilon_{A(B} chi_{C)}
#           where chi_C = (1/3) phi^D_{DC}
#
# The totally symmetric part is irreducible under SL(2,C).
#
# Reference: Penrose & Rindler Vol 1 (1984), Section 3.3.

"""
    irreducible_decompose(expr::TensorExpr;
                          registry::TensorRegistry=current_registry()) -> TensorExpr

Decompose a spinor expression into its irreducible (totally symmetric) part
plus trace terms involving epsilon contractions.

For a rank-2 spinor `phi_{AB}`:
  `phi_{AB} = phi_{(AB)} + (1/2) epsilon_{AB} phi^C_C`

The symmetric part is computed via `symmetrize` and the trace part via
contraction with the spin metric.

Currently supports rank-2 undotted and dotted spinors. Higher ranks
return the input unchanged (to be extended).

# Reference
Penrose & Rindler Vol 1 (1984), Section 3.3.
"""
function irreducible_decompose(expr::TensorExpr;
                               registry::TensorRegistry=current_registry())
    # Only meaningful for single tensors with spinor indices
    expr isa Tensor || return expr

    free = free_indices(expr)
    spinor_free = filter(idx -> is_spinor_index(idx), free)
    length(spinor_free) < 2 && return expr

    # Separate by vbundle type
    undotted = filter(idx -> idx.vbundle == :SL2C, spinor_free)
    dotted = filter(idx -> idx.vbundle == :SL2C_dot, spinor_free)

    # Rank-2 undotted decomposition: phi_{AB} = phi_{(AB)} + (1/2) eps_{AB} tr
    if length(undotted) == 2 && length(dotted) == 0
        return _decompose_rank2(expr, undotted[1], undotted[2], :SL2C; registry=registry)
    end

    # Rank-2 dotted decomposition: same structure with dotted indices
    if length(undotted) == 0 && length(dotted) == 2
        return _decompose_rank2(expr, dotted[1], dotted[2], :SL2C_dot; registry=registry)
    end

    # Higher ranks: return unchanged for now
    expr
end

"""
Decompose a rank-2 spinor into symmetric + trace parts.
  phi_{AB} = phi_{(AB)} + (1/2) epsilon_{AB} phi^C_C
"""
function _decompose_rank2(expr::TensorExpr, idx1::TIndex, idx2::TIndex,
                          vbundle::Symbol;
                          registry::TensorRegistry=current_registry())
    # Symmetric part: phi_{(AB)} = (1/2)(phi_{AB} + phi_{BA})
    sym_part = symmetrize(expr, [idx1.name, idx2.name])

    # Trace: phi^C_C -- contract idx1 and idx2 by raising one
    used = Set{Symbol}(idx.name for idx in indices(expr))
    c_name = fresh_index(used; vbundle=vbundle)

    # Build the traced expression: phi^C_C
    # For a rank-2 tensor phi_{AB}, replace indices to form phi^C_C
    # Rebuild the tensor with contracted dummy indices
    new_indices = map(expr.indices) do idx
        if idx.name == idx1.name && idx.vbundle == vbundle
            TIndex(c_name, Up, vbundle)
        elseif idx.name == idx2.name && idx.vbundle == vbundle
            TIndex(c_name, Down, vbundle)
        else
            idx
        end
    end
    traced = Tensor(expr.name, new_indices)

    # Get the epsilon metric name
    eps_name = get(registry.metric_cache, vbundle, nothing)
    eps_name === nothing && return expr  # no spin metric, can't decompose

    # Build: (1/2) epsilon_{AB} trace
    eps = Tensor(eps_name, [
        TIndex(idx1.name, Down, vbundle),
        TIndex(idx2.name, Down, vbundle)
    ])
    trace_part = tproduct(1 // 2, TensorExpr[eps, traced])

    tsum(TensorExpr[sym_part, trace_part])
end
