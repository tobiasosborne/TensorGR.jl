#= Young tableaux symmetrization.

Implements Young projectors for decomposing tensors into irreducible
representations of the symmetric group.
=#

"""
    YoungTableau

A Young tableau specified by rows of index slot positions.
Example: [[1,2],[3]] for a 2-row tableau with slots 1,2 in row 1 and slot 3 in row 2.
"""
struct YoungTableau
    rows::Vector{Vector{Int}}
end

"""
    young_shape(yt::YoungTableau) -> Vector{Int}

Return the shape (partition) of the Young tableau.
"""
young_shape(yt::YoungTableau) = [length(r) for r in yt.rows]

"""
    young_columns(yt::YoungTableau) -> Vector{Vector{Int}}

Return the columns of the Young tableau.
"""
function young_columns(yt::YoungTableau)
    ncols = maximum(length(r) for r in yt.rows; init=0)
    cols = Vector{Int}[]
    for j in 1:ncols
        col = Int[]
        for r in yt.rows
            j <= length(r) && push!(col, r[j])
        end
        push!(cols, col)
    end
    cols
end

"""
    young_symmetrize(expr::TensorExpr, yt::YoungTableau, idxs::Vector{Symbol}) -> TensorExpr

Apply the Young symmetrizer to an expression:
1. Symmetrize over each row
2. Antisymmetrize over each column

The `idxs` maps slot positions to index names: slot i uses idxs[i].
"""
function young_symmetrize(expr::TensorExpr, yt::YoungTableau, idxs::Vector{Symbol})
    result = expr

    # Step 1: Symmetrize over rows
    for row in yt.rows
        length(row) <= 1 && continue
        row_idxs = [idxs[i] for i in row]
        result = symmetrize(result, row_idxs)
    end

    # Step 2: Antisymmetrize over columns
    for col in young_columns(yt)
        length(col) <= 1 && continue
        col_idxs = [idxs[i] for i in col]
        result = antisymmetrize(result, col_idxs)
    end

    result
end

"""
    young_project(expr::TensorExpr, yt::YoungTableau, idxs::Vector{Symbol};
                  normalize::Bool=true) -> TensorExpr

Apply the Young projector (column antisymmetrize then row symmetrize).
With normalize=true, scales by the appropriate factor.
"""
function young_project(expr::TensorExpr, yt::YoungTableau, idxs::Vector{Symbol};
                       normalize::Bool=true)
    result = expr

    # Column antisymmetrize first
    for col in young_columns(yt)
        length(col) <= 1 && continue
        col_idxs = [idxs[i] for i in col]
        result = antisymmetrize(result, col_idxs)
    end

    # Then row symmetrize
    for row in yt.rows
        length(row) <= 1 && continue
        row_idxs = [idxs[i] for i in row]
        result = symmetrize(result, row_idxs)
    end

    if normalize
        # Normalization factor: (dim of irrep) / n!
        n = sum(length(r) for r in yt.rows)
        shape = young_shape(yt)
        hook = _hook_length_product(shape)
        dim_irrep = factorial(n) ÷ hook
        coeff = dim_irrep // factorial(n)
        result = tproduct(coeff, TensorExpr[result])
    end

    result
end

"""Compute the product of hook lengths for a partition."""
function _hook_length_product(shape::Vector{Int})
    prod_hooks = 1
    for i in eachindex(shape)
        for j in 1:shape[i]
            # Hook length at (i,j): arm length + leg length + 1
            arm = shape[i] - j  # cells to the right
            leg = count(k -> k >= j, shape[i+1:end])  # cells below
            prod_hooks *= arm + leg + 1
        end
    end
    prod_hooks
end
