"""
    Perm(data::Vector{Int32})

A permutation in Images notation. `data[i]` is the image of point `i`.
Points are 1-indexed (matching xperm.c convention).
"""
struct Perm
    data::Vector{Int32}
end

degree(p::Perm) = length(p.data)

function perm_identity(n::Int)
    Perm(Int32.(1:n))
end

function perm_is_identity(p::Perm)
    for i in eachindex(p.data)
        p.data[i] != i && return false
    end
    true
end

function perm_compose(p1::Perm, p2::Perm)
    n = degree(p1)
    @assert degree(p2) == n
    result = Vector{Int32}(undef, n)
    for i in 1:n
        result[i] = p2.data[p1.data[i]]
    end
    Perm(result)
end

function perm_inverse(p::Perm)
    n = degree(p)
    result = Vector{Int32}(undef, n)
    for i in 1:n
        result[p.data[i]] = Int32(i)
    end
    Perm(result)
end
