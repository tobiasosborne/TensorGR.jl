#= Weyl scalars: Newman-Penrose projection of the Weyl tensor.
#
# Given the all-down Weyl tensor C_{abcd} and a null tetrad (l, n, m, mbar),
# compute the five complex Weyl scalars Psi0...Psi4.
#
# References:
#   Newman & Penrose, J. Math. Phys. 3, 566 (1962), Eqs. (4.2a-e)
#   Chandrasekhar, "The Mathematical Theory of Black Holes" (1983), Ch 1
#   Stephani et al., "Exact Solutions" (2003), Ch 4, Eq. (3.59)
=#

"""
    _contract4(C::Array{T,4}, v1, v2, v3, v4) where T

Contract a rank-4 all-down tensor with four contravariant vectors:
    result = C_{abcd} v1^a v2^b v3^c v4^d
"""
function _contract4(C::Array{<:Any,4}, v1, v2, v3, v4)
    dim = size(C, 1)
    s = C[1,1,1,1] * v1[1] * v2[1] * v3[1] * v4[1]
    first = true
    @inbounds for a in 1:dim, b in 1:dim, c in 1:dim, d in 1:dim
        if first; first = false; continue; end
        s += C[a, b, c, d] * v1[a] * v2[b] * v3[c] * v4[d]
    end
    s
end

"""
    weyl_scalars(Weyl::Array{T,4}, l, n, m, mbar) where T
        -> NamedTuple{(:Psi0, :Psi1, :Psi2, :Psi3, :Psi4)}

Compute the five Newman-Penrose Weyl scalars from the all-down Weyl tensor
C_{abcd} and null tetrad vectors (l^a, n^a, m^a, m_bar^a).

The Weyl tensor must be the all-down form C_{abcd} (as returned by `metric_weyl`).
The tetrad vectors are contravariant (upper index).

Definitions (Newman & Penrose 1962):
    Psi0 = C_{abcd} l^a m^b   l^c m^d
    Psi1 = C_{abcd} l^a n^b   l^c m^d
    Psi2 = C_{abcd} l^a m^b   mbar^c n^d
    Psi3 = C_{abcd} l^a n^b   mbar^c n^d
    Psi4 = C_{abcd} n^a mbar^b n^c mbar^d
"""
function weyl_scalars(Weyl::Array{<:Any,4}, l, n, m, mbar)
    dim = size(Weyl, 1)
    @assert length(l) == length(n) == length(m) == length(mbar) == dim

    Psi0 = _contract4(Weyl, l, m,    l,    m)
    Psi1 = _contract4(Weyl, l, n,    l,    m)
    Psi2 = _contract4(Weyl, l, m,    mbar, n)
    Psi3 = _contract4(Weyl, l, n,    mbar, n)
    Psi4 = _contract4(Weyl, n, mbar, n,    mbar)

    (Psi0=Psi0, Psi1=Psi1, Psi2=Psi2, Psi3=Psi3, Psi4=Psi4)
end

"""
    validate_null_tetrad(l, n, m, mbar, g::Matrix; atol=1e-10) -> Bool

Check that the tetrad satisfies the Newman-Penrose normalization:
    l . n = -1,  m . mbar = +1
    l . l = n . n = m . m = mbar . mbar = 0
    l . m = l . mbar = n . m = n . mbar = 0

Uses metric g_{ab} to lower indices: v . w = g_{ab} v^a w^b.
"""
function validate_null_tetrad(l, n, m, mbar, g::Matrix; atol=1e-10)
    dot(v, w) = sum(g[a, b] * v[a] * w[b] for a in axes(g, 1), b in axes(g, 2))

    isapprox(dot(l, n), -1; atol) &&
    isapprox(dot(m, mbar), 1; atol) &&
    isapprox(dot(l, l), 0; atol) &&
    isapprox(dot(n, n), 0; atol) &&
    isapprox(dot(m, m), 0; atol) &&
    isapprox(dot(mbar, mbar), 0; atol) &&
    isapprox(dot(l, m), 0; atol) &&
    isapprox(dot(l, mbar), 0; atol) &&
    isapprox(dot(n, m), 0; atol) &&
    isapprox(dot(n, mbar), 0; atol)
end

"""
    null_tetrad_from_metric(g::CTensor) -> (l, n, m, mbar)

Construct a standard null tetrad from a diagonal metric with Lorentzian
signature (-,+,+,+). Uses the coordinate-aligned construction:

    l^a = (1/sqrt(2)) (e_0^a + e_1^a)
    n^a = (1/sqrt(2)) (e_0^a - e_1^a)
    m^a = (1/sqrt(2)) (e_2^a + i e_3^a)
    mbar^a = (1/sqrt(2)) (e_2^a - i e_3^a)

where e_mu^a = delta_mu^a / sqrt(|g_{mu mu}|) is the orthonormal frame.

Returns vectors of ComplexF64.
"""
function null_tetrad_from_metric(g::CTensor{T,2}) where T
    dim = size(g.data, 1)
    dim == 4 || error("null_tetrad_from_metric requires dim=4, got $dim")

    gd = g.data
    # Verify diagonal
    for i in 1:4, j in 1:4
        i == j && continue
        abs(gd[i, j]) < 1e-14 || error("null_tetrad_from_metric requires diagonal metric")
    end

    # Build orthonormal frame e_mu^a = delta^a_mu / sqrt(|g_{mu mu}|)
    e = [zeros(ComplexF64, 4) for _ in 1:4]
    for mu in 1:4
        e[mu][mu] = 1.0 / sqrt(abs(gd[mu, mu]))
    end

    inv_sqrt2 = 1.0 / sqrt(2.0)
    l    = inv_sqrt2 .* (e[1] .+ e[2])
    n    = inv_sqrt2 .* (e[1] .- e[2])
    m    = inv_sqrt2 .* (e[3] .+ im .* e[4])
    mbar = inv_sqrt2 .* (e[3] .- im .* e[4])

    (l, n, m, mbar)
end
