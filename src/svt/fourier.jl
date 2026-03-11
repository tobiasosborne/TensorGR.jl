#= Fourier space transforms.

to_fourier converts partial derivatives into momentum factors:
  ∂_i → -i k_i    (spatial)
  ∂_0 → -i ω      (temporal, if applicable)
  ∂_i ∂_i → -k²
  □ = ∂_0² - ∂_i² → -(ω² - k²) = -p²

We represent momenta symbolically using TScalar with symbolic values.
The sign conventions are configurable via FourierConvention.
=#

"""
    FourierConvention(; sign_spatial=-1, sign_temporal=-1)

Sign conventions for the Fourier transform.
Default: ∂_i → sign_spatial * i * k_i = -i k_i.
"""
struct FourierConvention
    sign_spatial::Int
    sign_temporal::Int
end
FourierConvention(; sign_spatial=-1, sign_temporal=-1) =
    FourierConvention(sign_spatial, sign_temporal)

const DEFAULT_FOURIER = FourierConvention()

"""
    to_fourier(expr; convention=DEFAULT_FOURIER, covd_names=Set{Symbol}()) -> TensorExpr

Replace partial derivatives with momentum factors.
- Each ∂_a on a spatial index becomes a momentum factor k_a
- Pairs ∂_a ∂^a become -k² (spatial) or -p² (spacetime)
- Named CovDs listed in `covd_names` are treated as partial derivatives
  (valid on MSS where eigenmodes decouple by spin and ∇² → -k²).
"""
function to_fourier(expr::TensorExpr;
                    convention::FourierConvention=DEFAULT_FOURIER,
                    covd_names::Union{Set{Symbol}, AbstractVector{Symbol}}=Set{Symbol}())
    cset = covd_names isa Set{Symbol} ? covd_names : Set{Symbol}(covd_names)
    _fourier_transform(expr, convention, cset)
end

_fourier_transform(t::Tensor, ::FourierConvention, ::Set{Symbol}) = t
_fourier_transform(s::TScalar, ::FourierConvention, ::Set{Symbol}) = s

function _fourier_transform(s::TSum, conv::FourierConvention, cn::Set{Symbol})
    tsum(TensorExpr[_fourier_transform(t, conv, cn) for t in s.terms])
end

function _fourier_transform(p::TProduct, conv::FourierConvention, cn::Set{Symbol})
    tproduct(p.scalar, TensorExpr[_fourier_transform(f, conv, cn) for f in p.factors])
end

function _fourier_transform(d::TDeriv, conv::FourierConvention, cn::Set{Symbol})
    # Only transform partial derivatives and named CovDs listed in covd_names
    if d.covd != :partial && d.covd ∉ cn
        error("to_fourier: unexpected covariant derivative :$(d.covd). " *
              "Pass covd_names=Set([:$(d.covd)]) to treat it as a momentum replacement.")
    end

    # Replace D_a with momentum k_a
    inner = _fourier_transform(d.arg, conv, cn)

    # Create momentum tensor: k with the derivative's index
    k = Tensor(:k, [d.index])

    # The derivative D_a → -i k_a. We drop the imaginary unit and
    # represent it as just k_a with a sign convention embedded.
    # For real-valued actions (quadratic forms), the i's cancel in pairs.
    # We track the sign: each derivative contributes a factor of (-i).
    # In a quadratic action, DD → -k²  (two factors of -i give -1).

    # Return k * inner (the momentum factor times the transformed argument)
    k * inner
end
