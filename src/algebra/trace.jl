#= Trace operations on abstract tensor expressions.

trace(expr, idx1, idx2) — contract two indices
make_traceless(expr, metric, idx1, idx2) — decompose T = T_TF + (1/d)g·tr(T)
=#

"""
    trace(expr::TensorExpr, idx1::Symbol, idx2::Symbol; metric::Symbol=:g) -> TensorExpr

Contract indices `idx1` and `idx2` by inserting a metric tensor.
Assumes one is up and one is down; if both are the same position,
inserts a metric to raise/lower.
"""
function abstract_trace(expr::TensorExpr, idx1::Symbol, idx2::Symbol;
                        metric::Symbol=:g)
    all_idxs = indices(expr)

    # Find the positions of idx1 and idx2
    pos1 = nothing
    pos2 = nothing
    for idx in all_idxs
        if idx.name == idx1
            pos1 = idx.position
        elseif idx.name == idx2
            pos2 = idx.position
        end
    end

    pos1 === nothing && error("Index $idx1 not found in expression")
    pos2 === nothing && error("Index $idx2 not found in expression")

    if pos1 != pos2
        # Already one up, one down — just identify the names for contraction
        # Rename idx2 to idx1 to force contraction
        return rename_dummy(expr, idx2, idx1)
    else
        # Same position: need metric to change one
        # If both down: insert g^{idx1, idx2} and contract
        # If both up: insert g_{idx1, idx2} and contract
        used = Set(idx.name for idx in all_idxs)
        d = fresh_index(used)

        if pos1 == Down
            # g^{d, idx2} * expr, then rename idx2→d in expr
            g = Tensor(metric, [up(d), up(idx2)])
            expr_renamed = rename_dummy(expr, idx2, d)
            return g * expr_renamed
        else
            g = Tensor(metric, [down(d), down(idx2)])
            expr_renamed = rename_dummy(expr, idx2, d)
            return g * expr_renamed
        end
    end
end

"""
    make_traceless(expr, metric, idx1, idx2; dim=4) -> TensorExpr

Decompose a rank-2 tensor into trace-free part + trace:
  T_{ab} = T^TF_{ab} + (1/d) g_{ab} T^c_c

Returns the trace-free part: T^TF_{ab} = T_{ab} - (1/d) g_{ab} g^{cd} T_{cd}
"""
function make_traceless(expr::TensorExpr, metric::Symbol,
                        idx1::Symbol, idx2::Symbol; dim::Int=4)
    # Compute trace: g^{cd} T_{cd} (with fresh dummies)
    used = Set{Symbol}()
    for idx in indices(expr)
        push!(used, idx.name)
    end
    c = fresh_index(used)
    push!(used, c)
    d = fresh_index(used)

    # Find positions of idx1, idx2
    all_idxs = indices(expr)
    pos1 = Down
    pos2 = Down
    for idx in all_idxs
        if idx.name == idx1
            pos1 = idx.position
        end
        if idx.name == idx2
            pos2 = idx.position
        end
    end

    # Build trace term
    g_inv = Tensor(metric, [up(c), up(d)])
    expr_cd = rename_dummy(rename_dummy(expr, idx1, c), idx2, d)
    trace_val = g_inv * expr_cd

    # Build metric term
    g_term = Tensor(metric, [TIndex(idx1, pos1), TIndex(idx2, pos2)])

    # T^TF = T - (1/d) g tr(T)
    expr - (1 // dim) * g_term * trace_val
end
