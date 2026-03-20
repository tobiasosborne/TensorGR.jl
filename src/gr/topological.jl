#= Topological density constructors.

Pontryagin density (Chern-Pontryagin): ★(R∧R) = ε^{abcd} R_{abef} R_{cd}^{ef}
Euler density (Gauss-Bonnet): ε^{abcd} ε^{efgh} R_{abef} R_{cdgh} / 4
                              = R^2 - 4 Ric^2 + Riem^2  (in 4D)

Lovelock Lagrangians (arbitrary order):
  L_p = (1/2^p) δ^{a1 b1 ⋯ ap bp}_{c1 d1 ⋯ cp dp} R^{c1 d1}_{a1 b1} ⋯ R^{cp dp}_{ap bp}

  L_0 = 1
  L_1 = R
  L_2 = R² - 4 Ric² + Riem²  (Gauss-Bonnet)
  L_3 = cubic Lovelock (8 terms)

The Euler density in d dimensions is E_d = L_{d/2} for even d, zero for odd d.

References:
  - Lovelock (1971), J. Math. Phys. 12, 498
  - Padmanabhan, Rep. Prog. Phys. 73 (2010) 046901
  - Casalino et al. (2020), arXiv:2003.07068
=#

"""
    pontryagin_density(metric::Symbol; registry=current_registry()) -> TensorExpr

Construct the Pontryagin (Chern-Pontryagin) density in 4D:
`★(R∧R) = ε^{abcd} R_{ab}^{ef} R_{cdef}`

This is a pseudoscalar, a total derivative in 4D.
"""
function pontryagin_density(metric::Symbol;
                             registry::TensorRegistry=current_registry())
    with_registry(registry) do
        reg = registry
        mprops = get_tensor(reg, metric)
        eps_name = Symbol(:ε, metric)

        used = Set{Symbol}()
        a = fresh_index(used); push!(used, a)
        b = fresh_index(used); push!(used, b)
        c = fresh_index(used); push!(used, c)
        d = fresh_index(used); push!(used, d)
        e = fresh_index(used); push!(used, e)
        f = fresh_index(used); push!(used, f)

        eps = Tensor(eps_name, [up(a), up(b), up(c), up(d)])
        R1 = Tensor(:Riem, [down(a), down(b), up(e), up(f)])
        R2 = Tensor(:Riem, [down(c), down(d), down(e), down(f)])

        eps * R1 * R2
    end
end

"""
    euler_density(metric::Symbol; dim::Int=4, registry=current_registry()) -> TensorExpr

Construct the Euler density (generalized Gauss-Bonnet) in `dim` dimensions.

The Euler density E_d is the Lovelock Lagrangian of order d/2:
`E_d = L_{d/2} = (1/2^{d/2}) δ^{a1 b1 ⋯}_{c1 d1 ⋯} R^{c1 d1}_{a1 b1} ⋯`

Special cases:
- d odd: returns `TScalar(0//1)` (Euler density vanishes in odd dimensions)
- d=2: `E₂ = R` (Ricci scalar)
- d=4: `E₄ = R² - 4 R_{ab}R^{ab} + R_{abcd}R^{abcd}` (Gauss-Bonnet, fast path)
- d=6: cubic Lovelock `L₃` (8-term expression)
- d≥8: general Lovelock `L_{d/2}` via generalized Kronecker delta

References:
- Lovelock (1971), J. Math. Phys. 12, 498
- Padmanabhan, Rep. Prog. Phys. 73 (2010) 046901
"""
function euler_density(metric::Symbol; dim::Int=4,
                        registry::TensorRegistry=current_registry())
    dim < 1 && error("euler_density: dim must be positive (got $dim)")

    # Odd dimensions: Euler density vanishes identically
    isodd(dim) && return TScalar(0 // 1)

    # d=2: Euler density is the Ricci scalar
    dim == 2 && return Tensor(:RicScalar, TIndex[])

    # d=4: fast path (existing well-tested Gauss-Bonnet expression)
    if dim == 4
        return with_registry(registry) do
            _euler_density_4d()
        end
    end

    # General even d >= 6: Euler density = Lovelock Lagrangian of order d/2
    lovelock_lagrangian(dim ÷ 2, metric; dim=dim, registry=registry)
end

"""Build the d=4 Euler density (Gauss-Bonnet) as Riem² - 4 Ric² + R²."""
function _euler_density_4d()
    used = Set{Symbol}()

    # Kretschner: R_{abcd} R^{abcd}
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used); push!(used, d)
    Riem_down = Tensor(:Riem, [down(a), down(b), down(c), down(d)])
    Riem_up = Tensor(:Riem, [up(a), up(b), up(c), up(d)])
    kretschner = Riem_down * Riem_up

    # Ricci squared: R_{ab} R^{ab}
    e = fresh_index(used); push!(used, e)
    f = fresh_index(used); push!(used, f)
    Ric_down = Tensor(:Ric, [down(e), down(f)])
    Ric_up = Tensor(:Ric, [up(e), up(f)])
    ricci_sq = Ric_down * Ric_up

    # Scalar squared: R²
    R = Tensor(:RicScalar, TIndex[])
    scalar_sq = R * R

    # E₄ = Riem² - 4 Ric² + R²
    kretschner + tproduct(-4 // 1, TensorExpr[ricci_sq]) + scalar_sq
end

"""
    chern_simons_action(scalar_field::Tensor, metric::Symbol;
                         registry=current_registry()) -> TensorExpr

Construct the Chern-Simons gravitational coupling:
`S_CS = ϑ · ★(R∧R) = ϑ · ε^{abcd} R_{ab}^{ef} R_{cdef}`

where `ϑ` is the scalar field (axion/dilaton).
"""
function chern_simons_action(scalar_field::Tensor, metric::Symbol;
                              registry::TensorRegistry=current_registry())
    with_registry(registry) do
        scalar_field * pontryagin_density(metric; registry=registry)
    end
end

# ── Lovelock Lagrangians ─────────────────────────────────────────────

"""
    lovelock_lagrangian(order::Int, metric::Symbol;
                         dim::Int=0, registry=current_registry()) -> TensorExpr

Construct the order-`order` Lovelock Lagrangian:

`L_p = (1/2^p) δ^{a₁b₁ ⋯ aₚbₚ}_{c₁d₁ ⋯ cₚdₚ} R^{c₁d₁}_{a₁b₁} ⋯ R^{cₚdₚ}_{aₚbₚ}`

Special cases:
- `order=0`: returns `TScalar(1//1)` (cosmological constant term)
- `order=1`: returns `Tensor(:RicScalar, TIndex[])` (Einstein-Hilbert = R)
- `order=2`: returns `R² - 4 Ric² + Riem²` (Gauss-Bonnet)
- `order≥3`: general Lovelock via generalized Kronecker delta expansion

The `dim` keyword specifies the manifold dimension (used only for the DDI
vanishing check: L_p = 0 when 2p > dim). If `dim=0` (default), it is
looked up from the registry.

Note: for large `order` (≥ 4), the generalized delta expansion produces
`(2p)!` terms before simplification. This is exact but computationally
expensive. Use `simplify` to reduce the result.

References:
- Lovelock (1971), J. Math. Phys. 12, 498
- Casalino et al. (2020), arXiv:2003.07068, Eqs. (4)-(8)

# Examples
```julia
reg = TensorRegistry()
@manifold M dim=6 metric=g registry=reg
define_curvature_tensors!(reg, :M, :g)

L1 = lovelock_lagrangian(1, :g; registry=reg)  # R
L2 = lovelock_lagrangian(2, :g; registry=reg)  # Gauss-Bonnet
L3 = lovelock_lagrangian(3, :g; dim=6, registry=reg)  # cubic Lovelock
```
"""
function lovelock_lagrangian(order::Int, metric::Symbol;
                              dim::Int=0,
                              registry::TensorRegistry=current_registry())
    order < 0 && error("lovelock_lagrangian: order must be non-negative (got $order)")

    # order=0: cosmological constant term L_0 = 1
    order == 0 && return TScalar(1 // 1)

    # order=1: L_1 = R (Ricci scalar)
    order == 1 && return Tensor(:RicScalar, TIndex[])

    # Determine dimension for DDI vanishing check
    actual_dim = dim > 0 ? dim : _lovelock_lookup_dim(registry)

    # L_p vanishes when 2p > dim (DDI: generalized delta of order 2p vanishes)
    2 * order > actual_dim && return TScalar(0 // 1)

    # order=2: fast path for Gauss-Bonnet
    if order == 2
        return with_registry(registry) do
            _euler_density_4d()
        end
    end

    # General order >= 3: build from generalized Kronecker delta
    with_registry(registry) do
        _build_lovelock_from_delta(order, registry)
    end
end

"""
    _build_lovelock_from_delta(p::Int, registry::TensorRegistry) -> TensorExpr

Build the Lovelock Lagrangian of order p using the generalized Kronecker delta:

`L_p = (1/2^p) δ^{a₁b₁ ⋯ aₚbₚ}_{c₁d₁ ⋯ cₚdₚ} R^{c₁d₁}_{a₁b₁} ⋯ R^{cₚdₚ}_{aₚbₚ}`

The generalized delta is expanded via the Leibniz formula (sum over permutations
of S_{2p} with signs), producing a sum of signed products of ordinary Kronecker
deltas. Each term is multiplied by the product of p Riemann tensors. The delta
contractions select specific index pairings of the Riemanns, yielding all
possible Riemann contraction structures with the correct combinatorial weights.

The overall factor is 1/2^p from the Lovelock definition.
"""
function _build_lovelock_from_delta(p::Int, registry::TensorRegistry)
    used = Set{Symbol}()

    # Generate 2p upper index names (delta upper = Riemann lower indices)
    up_names = Symbol[]
    for _ in 1:(2p)
        s = fresh_index(used)
        push!(used, s)
        push!(up_names, s)
    end

    # Generate 2p lower index names (delta lower = Riemann upper indices)
    down_names = Symbol[]
    for _ in 1:(2p)
        s = fresh_index(used)
        push!(used, s)
        push!(down_names, s)
    end

    # Delta tensor name from registry
    delta_name = isempty(registry.delta_cache) ? :delta :
                 first(values(registry.delta_cache))

    # Expand generalized delta via Leibniz formula:
    # delta^{up_1...up_{2p}}_{dn_1...dn_{2p}} =
    #   sum_{sigma in S_{2p}} sign(sigma) * prod_{i=1}^{2p} delta^{up_i}_{dn_{sigma(i)}}
    perms = _permutations_with_sign(2p)
    delta_terms = TensorExpr[]
    for (perm, sgn) in perms
        factors = TensorExpr[]
        for i in 1:(2p)
            push!(factors, Tensor(delta_name, [up(up_names[i]), down(down_names[perm[i]])]))
        end
        push!(delta_terms, tproduct(Rational{Int}(sgn), factors))
    end
    gdelta = tsum(delta_terms)

    # Build the product of p Riemann tensors:
    # R^{dn₁dn₂}_{up₁up₂} * R^{dn₃dn₄}_{up₃up₄} * ⋯ * R^{dn_{2p-1}dn_{2p}}_{up_{2p-1}up_{2p}}
    riem_factors = TensorExpr[]
    for i in 1:p
        a_i = up_names[2i - 1]
        b_i = up_names[2i]
        c_i = down_names[2i - 1]
        d_i = down_names[2i]
        push!(riem_factors, Tensor(:Riem, [up(c_i), up(d_i), down(a_i), down(b_i)]))
    end
    riem_product = tproduct(1 // 1, riem_factors)

    # L_p = (1/2^p) * gdelta * riem_product
    coeff = 1 // (2^p)
    tproduct(coeff, TensorExpr[gdelta, riem_product])
end

"""Look up manifold dimension from registry (first registered manifold)."""
function _lovelock_lookup_dim(reg::TensorRegistry)
    isempty(reg.manifolds) && error("lovelock_lagrangian: no manifolds registered; cannot determine dimension")
    first(values(reg.manifolds)).dim
end
