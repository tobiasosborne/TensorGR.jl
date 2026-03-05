#= Partition combinatorics for perturbation theory.

Integer partitions, compositions, and multinomial coefficients
needed for the xPert-style perturbation algorithm.
=#

"""
    sorted_partitions(n::Int) -> Vector{Vector{Int}}

All partitions of integer n, each in non-increasing order.
"""
function sorted_partitions(n::Int)
    n <= 0 && return [Int[]]
    result = Vector{Vector{Int}}()
    _partition_helper!(result, Int[], n, n)
    result
end

function _partition_helper!(result, current, remaining, max_val)
    if remaining == 0
        push!(result, copy(current))
        return
    end
    for k in min(remaining, max_val):-1:1
        push!(current, k)
        _partition_helper!(result, current, remaining - k, k)
        pop!(current)
    end
end

"""
    all_compositions(m::Int, n::Int) -> Vector{Vector{Int}}

All ordered compositions of m into n non-negative parts.
"""
function all_compositions(m::Int, n::Int)
    n == 0 && return m == 0 ? [Int[]] : Vector{Int}[]
    n == 1 && return [[m]]
    result = Vector{Vector{Int}}()
    for k in 0:m
        for rest in all_compositions(m - k, n - 1)
            push!(result, vcat([k], rest))
        end
    end
    result
end

"""
    multinomial(n::Int, ks::Vector{Int}) -> Int

Multinomial coefficient n! / (k1! k2! ... km!).
"""
function multinomial(n::Int, ks::Vector{Int})
    @assert sum(ks) == n
    result = factorial(n)
    for k in ks
        result = div(result, factorial(k))
    end
    result
end
