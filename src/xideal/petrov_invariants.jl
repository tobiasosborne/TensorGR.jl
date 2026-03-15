#= Petrov invariants I, J from the Weyl tensor.
#
# The two complex scalar invariants that encode the algebraic (Petrov) type
# of the Weyl tensor, independent of the choice of null tetrad.
#
# From Weyl scalars (NP formalism):
#   I = Psi0*Psi4 - 4*Psi1*Psi3 + 3*Psi2^2
#   J = det([Psi0 Psi1 Psi2; Psi1 Psi2 Psi3; Psi2 Psi3 Psi4])
#
# Classification: I^3 = 27 J^2  iff algebraically special (types II, D, III, N, O).
#
# References:
#   Stephani et al., "Exact Solutions" (2003), Eqs. (4.19)-(4.20)
#   Chandrasekhar, "The Mathematical Theory of Black Holes" (1983), Ch 1
#   Penrose & Rindler, "Spinors and Space-time" Vol 1 (1984), Sec 8.7
=#

"""
    petrov_invariants(psi::NamedTuple) -> NamedTuple{(:I, :J)}

Compute Petrov invariants from Newman-Penrose Weyl scalars.
Expects a NamedTuple with fields Psi0, Psi1, Psi2, Psi3, Psi4
(as returned by `weyl_scalars`).

    I = Psi0*Psi4 - 4*Psi1*Psi3 + 3*Psi2^2
    J = det([Psi0 Psi1 Psi2; Psi1 Psi2 Psi3; Psi2 Psi3 Psi4])
"""
function petrov_invariants(psi::NamedTuple{(:Psi0, :Psi1, :Psi2, :Psi3, :Psi4)})
    P0, P1, P2, P3, P4 = psi.Psi0, psi.Psi1, psi.Psi2, psi.Psi3, psi.Psi4
    Ival = P0 * P4 - 4 * P1 * P3 + 3 * P2^2
    # J = det of the symmetric 3x3 matrix Q_{ij} = Psi_{i+j}
    Jval = P0 * (P2 * P4 - P3^2) - P1 * (P1 * P4 - P2 * P3) + P2 * (P1 * P3 - P2^2)
    (I=Ival, J=Jval)
end

"""
    petrov_invariants(Weyl::Array{T,4}, g::Matrix; dim::Int=4) where T
        -> NamedTuple{(:I, :J)}

Compute Petrov invariants I and J from the all-down Weyl tensor C_{abcd}
and the metric g_{ab}.

Constructs a null tetrad from the diagonal metric, computes the NP Weyl
scalars, and returns the standard Petrov invariants. Requires dim=4 and
a diagonal metric.

For the raw tensor contraction invariants (Kretschmann-type, without
self-dual projection), see `weyl_contraction_invariants`.
"""
function petrov_invariants(Weyl::Array{T,4}, g::Matrix; dim::Int=4) where T
    dim == 4 || error("petrov_invariants from tensor components requires dim=4")
    g_ct = CTensor(Float64.(g), :_petrov)
    l, n, m, mbar = null_tetrad_from_metric(g_ct)
    W = T === Any ? Float64.(Weyl) : Weyl
    psi = weyl_scalars(W, l, n, m, mbar)
    petrov_invariants(psi)
end

"""
    petrov_invariants(Weyl::CTensor{T,4}, g::CTensor{S,2}) where {T,S}
        -> NamedTuple{(:I, :J)}

Convenience method: compute Petrov invariants directly from component tensors.
`Weyl` must be all-down C_{abcd}; `g` must be the diagonal metric g_{ab}.
Constructs a null tetrad internally and routes through the NP scalar formula.
"""
function petrov_invariants(Weyl::CTensor{T,4}, g::CTensor{S,2}) where {T,S}
    l, n, m, mbar = null_tetrad_from_metric(g)
    W = T === Any ? Float64.(Weyl.data) : Weyl.data
    psi = weyl_scalars(W, l, n, m, mbar)
    petrov_invariants(psi)
end

"""
    weyl_contraction_invariants(Weyl::Array{T,4}, ginv::Matrix; dim::Int=4) where T
        -> NamedTuple{(:I2, :I3)}

Compute the real Weyl contraction invariants from the all-down Weyl tensor
and inverse metric:

    I2 = (1/2) C^{ab}_{cd} C^{cd}_{ab}   (= Kretschmann/2 for vacuum)
    I3 = (1/6) C^{ab}_{cd} C^{cd}_{ef} C^{ef}_{ab}

These are the real invariants of the full Weyl tensor (not the complex
self-dual Petrov invariants). For Petrov classification, use `petrov_invariants`.
"""
function weyl_contraction_invariants(Weyl::Array{T,4}, ginv::Matrix; dim::Int=4) where T
    W = T === Any ? Float64.(Weyl) : Weyl
    S = promote_type(eltype(W), eltype(ginv))

    # Raise first two indices: C^{ab}_{cd} = g^{ae} g^{bf} C_{efcd}
    Cup = Array{S}(undef, dim, dim, dim, dim)
    @inbounds for a in 1:dim, b in 1:dim, c in 1:dim, d in 1:dim
        s = zero(S)
        for e in 1:dim, f in 1:dim
            s += ginv[a, e] * ginv[b, f] * W[e, f, c, d]
        end
        Cup[a, b, c, d] = s
    end

    # I2 = (1/2) C^{ab}_{cd} C^{cd}_{ab}
    I2val = zero(S)
    @inbounds for a in 1:dim, b in 1:dim, c in 1:dim, d in 1:dim
        I2val += Cup[a, b, c, d] * Cup[c, d, a, b]
    end
    I2val /= 2

    # I3 = (1/6) C^{ab}_{cd} C^{cd}_{ef} C^{ef}_{ab}
    # Factor as QQ then trace
    QQ = Array{S}(undef, dim, dim, dim, dim)
    @inbounds for a in 1:dim, b in 1:dim, e in 1:dim, f in 1:dim
        s = zero(S)
        for c in 1:dim, d in 1:dim
            s += Cup[a, b, c, d] * Cup[c, d, e, f]
        end
        QQ[a, b, e, f] = s
    end
    I3val = zero(S)
    @inbounds for a in 1:dim, b in 1:dim, e in 1:dim, f in 1:dim
        I3val += QQ[a, b, e, f] * Cup[e, f, a, b]
    end
    I3val /= 6

    (I2=I2val, I3=I3val)
end

"""
    is_algebraically_special(I, J; atol=1e-10) -> Bool

Test whether a spacetime is algebraically special (Petrov types II, D, III, N, O)
by checking the criterion I^3 = 27 J^2.

Returns `true` if |I^3 - 27 J^2| <= atol * max(|I^3|, |27 J^2|, 1).
"""
function is_algebraically_special(I, J; atol=1e-10)
    I3 = I^3
    J2_27 = 27 * J^2
    scale = max(abs(I3), abs(J2_27), one(real(typeof(I3))))
    abs(I3 - J2_27) <= atol * scale
end
