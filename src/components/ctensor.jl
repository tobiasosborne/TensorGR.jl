#= CTensor: component tensor arrays.

A CTensor holds the numerical (or symbolic) components of a tensor
in a specific coordinate chart. It provides arithmetic operations
and index contraction.
=#

"""
    CTensor(data, chart, index_positions)

A component tensor: `data` is an Array of numbers/expressions,
`chart` identifies the coordinate chart, and `index_positions`
records whether each slot is Up or Down.
"""
struct CTensor{T, N}
    data::Array{T, N}
    chart::Symbol
    positions::Vector{IndexPosition}
end

function CTensor(data::Array{T, N}, chart::Symbol) where {T, N}
    CTensor{T, N}(data, chart, fill(Down, N))
end

Base.size(ct::CTensor) = size(ct.data)
Base.getindex(ct::CTensor, I...) = ct.data[I...]
Base.ndims(ct::CTensor) = ndims(ct.data)

function Base.show(io::IO, ct::CTensor)
    print(io, "CTensor(", ct.chart, ", ", ct.positions, ", ", size(ct.data), ")")
end

# Arithmetic
function Base.:+(a::CTensor{T,N}, b::CTensor{S,N}) where {T,S,N}
    a.chart == b.chart || error("Charts must match for addition")
    a.positions == b.positions || error("Index positions must match")
    CTensor(a.data .+ b.data, a.chart, a.positions)
end

function Base.:-(a::CTensor{T,N}, b::CTensor{S,N}) where {T,S,N}
    a.chart == b.chart || error("Charts must match")
    a.positions == b.positions || error("Index positions must match")
    CTensor(a.data .- b.data, a.chart, a.positions)
end

function Base.:*(s::Number, ct::CTensor)
    CTensor(s .* ct.data, ct.chart, ct.positions)
end
Base.:*(ct::CTensor, s::Number) = s * ct

"""
    ctensor_contract(a::CTensor, b::CTensor, idx_a::Int, idx_b::Int) -> CTensor

Contract index `idx_a` of `a` with index `idx_b` of `b`.
Requires opposite positions (one Up, one Down).
"""
function ctensor_contract(a::CTensor, b::CTensor, idx_a::Int, idx_b::Int)
    a.chart == b.chart || error("Charts must match")
    a.positions[idx_a] != b.positions[idx_b] || error("Contracted indices must have opposite positions")

    dim = size(a.data, idx_a)
    @assert dim == size(b.data, idx_b)

    # Result dimensions
    na = ndims(a)
    nb = ndims(b)
    result_positions = vcat(
        [a.positions[i] for i in 1:na if i != idx_a],
        [b.positions[i] for i in 1:nb if i != idx_b]
    )

    result_dims = vcat(
        [size(a.data, i) for i in 1:na if i != idx_a],
        [size(b.data, i) for i in 1:nb if i != idx_b]
    )

    if isempty(result_dims)
        # Scalar result
        s = sum(a.data[CartesianIndex(ntuple(i -> i == idx_a ? k : 1, na))] *
                b.data[CartesianIndex(ntuple(i -> i == idx_b ? k : 1, nb))]
                for k in 1:dim)
        return CTensor(fill(s), a.chart, IndexPosition[])
    end

    result = zeros(promote_type(eltype(a.data), eltype(b.data)), result_dims...)
    # General contraction via Einstein summation
    for k in 1:dim
        # Slice a at index idx_a = k
        a_slice = selectdim(a.data, idx_a, k)
        b_slice = selectdim(b.data, idx_b, k)
        # Outer product and accumulate
        result .+= reshape(a_slice, (size(a_slice)..., ntuple(_ -> 1, nb - 1)...)) .*
                   reshape(b_slice, (ntuple(_ -> 1, na - 1)..., size(b_slice)...))
    end

    CTensor(result, a.chart, result_positions)
end

"""
    ctensor_inverse(ct::CTensor{T,2}) -> CTensor where T

Compute the matrix inverse of a rank-2 CTensor.
"""
function ctensor_inverse(ct::CTensor{T,2}) where T
    @assert size(ct.data, 1) == size(ct.data, 2) "Must be square"
    inv_data = inv(ct.data)
    # Flip positions: Up↔Down
    inv_pos = [p == Up ? Down : Up for p in ct.positions]
    CTensor(inv_data, ct.chart, inv_pos)
end

"""
    ctensor_det(ct::CTensor{T,2}) -> T where T

Compute the determinant of a rank-2 CTensor.
"""
function ctensor_det(ct::CTensor{T,2}) where T
    @assert size(ct.data, 1) == size(ct.data, 2) "Must be square"
    det(ct.data)
end

# Import det from LinearAlgebra when available, otherwise use a simple implementation
using LinearAlgebra: det, inv

"""
    ctensor_trace(ct::CTensor, idx1::Int, idx2::Int) -> CTensor

Trace over indices `idx1` and `idx2`.
"""
function ctensor_trace(ct::CTensor, idx1::Int, idx2::Int)
    ct.positions[idx1] != ct.positions[idx2] || error("Traced indices must have opposite positions")
    dim = size(ct.data, idx1)
    @assert dim == size(ct.data, idx2)

    # Sum over the diagonal
    remaining = [i for i in 1:ndims(ct) if i != idx1 && i != idx2]
    result_positions = ct.positions[remaining]
    result_dims = [size(ct.data, i) for i in remaining]

    if isempty(result_dims)
        s = sum(ct.data[ntuple(i -> i == idx1 || i == idx2 ? k : 1, ndims(ct))...]
                for k in 1:dim)
        return CTensor(fill(s), ct.chart, IndexPosition[])
    end

    result = zeros(eltype(ct.data), result_dims...)
    for k in 1:dim
        slice = selectdim(selectdim(ct.data, max(idx1, idx2), k), min(idx1, idx2), k)
        result .+= slice
    end

    CTensor(result, ct.chart, result_positions)
end

"""
    basis_change(ct::CTensor, jacobian::Matrix) -> CTensor

Transform a CTensor from one basis to another using a Jacobian matrix.
Handles rank-1 and rank-2 tensors. For Up indices, multiply by J;
for Down indices, multiply by J^{-1 T}.
"""
function basis_change(ct::CTensor{T,1}, jacobian::Matrix) where T
    dim = size(ct.data, 1)
    @assert size(jacobian) == (dim, dim)
    if ct.positions[1] == Up
        CTensor(jacobian * ct.data, ct.chart, ct.positions)
    else
        CTensor(inv(jacobian)' * ct.data, ct.chart, ct.positions)
    end
end

function basis_change(ct::CTensor{T,2}, jacobian::Matrix) where T
    dim = size(ct.data, 1)
    @assert size(jacobian) == (dim, dim)
    J = jacobian
    Jinv = inv(jacobian)

    result = zeros(promote_type(T, eltype(J)), dim, dim)
    for a in 1:dim, b in 1:dim
        for c in 1:dim, d in 1:dim
            transform_a = ct.positions[1] == Up ? J[a, c] : Jinv[c, a]
            transform_b = ct.positions[2] == Up ? J[b, d] : Jinv[d, b]
            result[a, b] += transform_a * transform_b * ct.data[c, d]
        end
    end
    CTensor(result, ct.chart, ct.positions)
end
