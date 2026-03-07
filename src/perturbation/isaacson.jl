#= Isaacson short-wavelength averaging for GW stress-energy.

The effective stress-energy tensor of gravitational waves is defined via
the Isaacson prescription: expand Einstein's equations to second order in
the metric perturbation h_{ab}, then average over several wavelengths.

  T^{eff}_{ab} = ⟨δ²G_{ab}⟩

The averaging operator ⟨...⟩ replaces bilinear products h·h with formal
expectation values, discards terms linear in h (they average to zero),
and treats background quantities as constants.
=#

"""
    isaacson_average(expr::TensorExpr, perturbation::Symbol) -> TensorExpr

Apply short-wavelength averaging to an expression bilinear in a
perturbation tensor. Replaces bilinear products of `perturbation` with
a formal expectation tensor, and discards terms with odd powers of
the perturbation (they average to zero).

The averaging acts on the structural level:
- Products with exactly 2 factors of `perturbation` are kept (bilinear)
- Products with 0 or odd count of `perturbation` are discarded
- Derivatives of `perturbation` count as a single power
"""
function isaacson_average(expr::TensorExpr, perturbation::Symbol)
    _isaacson_walk(expr, perturbation)
end

function _isaacson_walk(s::TSum, pert::Symbol)
    tsum(TensorExpr[_isaacson_walk(t, pert) for t in s.terms])
end

function _isaacson_walk(p::TProduct, pert::Symbol)
    count = _pert_count(p, pert)
    # Keep only bilinear (quadratic) terms — odd powers average to zero,
    # zeroth order is background (not part of T_eff)
    count == 2 ? p : ZERO
end

function _isaacson_walk(t::Tensor, pert::Symbol)
    t.name == pert ? ZERO : t  # linear term averages to zero
end

function _isaacson_walk(d::TDeriv, pert::Symbol)
    _pert_count(d, pert) == 2 ? d : ZERO
end

_isaacson_walk(s::TScalar, ::Symbol) = s

"""Count occurrences of perturbation tensor (including inside derivatives)."""
function _pert_count(expr::Tensor, pert::Symbol)
    expr.name == pert ? 1 : 0
end

function _pert_count(p::TProduct, pert::Symbol)
    sum(_pert_count(f, pert) for f in p.factors; init=0)
end

function _pert_count(d::TDeriv, pert::Symbol)
    _pert_count(d.arg, pert)
end

function _pert_count(s::TSum, pert::Symbol)
    isempty(s.terms) ? 0 : maximum(_pert_count(t, pert) for t in s.terms)
end

_pert_count(::TScalar, ::Symbol) = 0
