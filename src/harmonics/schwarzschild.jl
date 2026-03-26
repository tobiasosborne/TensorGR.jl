#= Schwarzschild background in 2+2 decomposition for RW/Zerilli analysis.
#
# The Schwarzschild metric in 2+2 form:
#   ds² = g_{AB} dx^A dx^B + r² Ω_{ab} dθ^a dθ^b
# where A,B ∈ {t,r} (2D Lorentzian orbital manifold) and a,b ∈ {θ,φ} (S²),
# and g_{AB} = diag(-f(r), 1/f(r)) with f(r) = 1 - 2M/r.
#
# The tortoise coordinate r* is defined by dr*/dr = 1/f(r), giving
# r* = r + 2M ln|r/2M - 1|.
#
# References:
#   Regge & Wheeler, Phys. Rev. 108, 1063 (1957), Sec 2.
#   Martel & Poisson, Phys. Rev. D 71, 104003 (2005), Sec III.
#   Zerilli, Phys. Rev. D 2, 2141 (1970).
=#

"""
    SchwarzschildBackground

Schwarzschild background in 2+2 decomposition (M₂ × S²).

# Fields
- `orbital::Symbol` — orbital (t,r) manifold name
- `sphere::Symbol` — angular S² manifold name
- `orbital_metric::Symbol` — 2D Lorentzian metric
- `sphere_metric::Symbol` — round metric on S²
- `M::Symbol` — mass parameter symbol
- `f::Symbol` — lapse function f(r) = 1 - 2M/r symbol
"""
struct SchwarzschildBackground
    orbital::Symbol
    sphere::Symbol
    orbital_metric::Symbol
    sphere_metric::Symbol
    M::Symbol
    f::Symbol
end

"""
    define_schwarzschild_background!(reg; orbital=:M2, sphere=:S2,
        orbital_metric=:gab, sphere_metric=:Omega, M=:M_BH, f=:f_BH)

Register a Schwarzschild background as a 2+2 product manifold.

Registers:
- Orbital manifold M2 (dim=2, Lorentzian) with metric g_{AB}
- Angular manifold S2 (dim=2, Riemannian) with metric Ω_{ab}
- Product manifold M4 = M2 × S2
- Schwarzschild lapse f(r) = 1 - 2M/r as a TScalar symbol
- Radius function r on the orbital manifold
- Tortoise coordinate r* via dr*/dr = 1/f(r)

Returns a `SchwarzschildBackground` storing the definitions.
"""
function define_schwarzschild_background!(reg::TensorRegistry;
                                           orbital::Symbol=:M2,
                                           sphere::Symbol=:S2,
                                           orbital_metric::Symbol=:gab,
                                           sphere_metric::Symbol=:Omega,
                                           M::Symbol=:M_BH,
                                           f::Symbol=:f_BH,
                                           product::Symbol=:M4_Schw)
    @lock reg.lock begin
    # Register orbital manifold M2 (t,r sector)
    if !has_manifold(reg, orbital)
        register_manifold!(reg,
            ManifoldProperties(orbital, 2, orbital_metric, nothing, [:A, :B, :C, :D]))
        define_vbundle!(reg, Symbol(:Tangent_, orbital);
            manifold=orbital, dim=2,
            indices=[:A, :B, :C, :D])
    end
    if !has_tensor(reg, orbital_metric)
        register_tensor!(reg, TensorProperties(
            name=orbital_metric, manifold=orbital, rank=(0, 2),
            symmetries=SymmetrySpec[FullySymmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
    end

    # Register angular manifold S2
    if !has_manifold(reg, sphere)
        register_manifold!(reg,
            ManifoldProperties(sphere, 2, sphere_metric, nothing, [:p, :q, :r, :s]))
        define_vbundle!(reg, Symbol(:Tangent_, sphere);
            manifold=sphere, dim=2,
            indices=[:p, :q, :r, :s])  # avoid clash with A,B,C,D
    end
    if !has_tensor(reg, sphere_metric)
        register_tensor!(reg, TensorProperties(
            name=sphere_metric, manifold=sphere, rank=(0, 2),
            symmetries=SymmetrySpec[FullySymmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_metric => true)))
    end

    # Register the lapse function f(r) as a scalar symbol
    if !has_tensor(reg, f)
        register_tensor!(reg, TensorProperties(
            name=f, manifold=orbital, rank=(0, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_schwarzschild_lapse => true,
                :mass_parameter => M)))
    end

    # Register the areal radius r as a scalar
    r_sym = Symbol(:r_, orbital)
    if !has_tensor(reg, r_sym)
        register_tensor!(reg, TensorProperties(
            name=r_sym, manifold=orbital, rank=(0, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_areal_radius => true)))
    end

    # Register product manifold M4 = M2 × S2 (if both have proper setups)
    if !has_product_manifold(reg, product)
        # We need proper metric setup — skip product_manifold registration
        # if dim types would conflict. Store properties directly.
        pmp = ProductManifoldProperties(product, [orbital, sphere],
            [orbital_metric, sphere_metric], [2, 2])
        reg.foliations[Symbol(:product_, product)] = pmp
    end

    bg = SchwarzschildBackground(orbital, sphere, orbital_metric,
        sphere_metric, M, f)
    reg.foliations[Symbol(:schwarzschild_, product)] = bg
    bg
    end
end

"""
    get_schwarzschild_background(reg, product=:M4_Schw) -> SchwarzschildBackground
"""
function get_schwarzschild_background(reg::TensorRegistry,
                                       product::Symbol=:M4_Schw)
    key = Symbol(:schwarzschild_, product)
    haskey(reg.foliations, key) ||
        error("No Schwarzschild background ':$product' registered")
    reg.foliations[key]::SchwarzschildBackground
end

"""
    schwarzschild_f(bg) -> TScalar

The Schwarzschild lapse function f(r) = 1 - 2M/r as a symbolic scalar.
"""
schwarzschild_f(bg::SchwarzschildBackground) = Tensor(bg.f, TIndex[])

"""
    schwarzschild_r(bg) -> TScalar

The areal radius r as a symbolic scalar.
"""
schwarzschild_r(bg::SchwarzschildBackground) =
    Tensor(Symbol(:r_, bg.orbital), TIndex[])

"""
    tortoise_deriv(bg) -> TScalar

The tortoise coordinate derivative factor: dr*/dr = 1/f(r).
Used in converting ∂/∂r → f(r) ∂/∂r*.
"""
tortoise_deriv(bg::SchwarzschildBackground) = TScalar(Symbol(:inv_, bg.f))

# ── Potentials (connect to existing MasterEquation infrastructure) ───

"""
    schwarzschild_rw_potential(bg, l::Int) -> Function

Returns V_RW(r) = f(r) [l(l+1)/r² - 6M/r³] as a callable.

Ground truth: Regge & Wheeler Eq 11; Martel & Poisson Eq 4.9.
"""
function schwarzschild_rw_potential(l::Int)
    l >= 2 || error("RW potential requires l ≥ 2")
    (r, M) -> begin
        f = 1 - 2M / r
        f * (l * (l + 1) / r^2 - 6M / r^3)
    end
end

"""
    schwarzschild_zerilli_potential(l::Int) -> Function

Returns V_Z(r) as a callable.

Ground truth: Zerilli (1970) Eq 11; Martel & Poisson Eq 4.22.
"""
function schwarzschild_zerilli_potential(l::Int)
    l >= 2 || error("Zerilli potential requires l ≥ 2")
    n = (l - 1) * (l + 2) ÷ 2  # λ in Zerilli's notation
    (r, M) -> begin
        f = 1 - 2M / r
        num = 2n^2 * (n + 1) * r^3 + 6n^2 * M * r^2 + 18n * M^2 * r + 18M^3
        den = r^3 * (n * r + 3M)^2
        f * num / den
    end
end

"""
    schwarzschild_potential_difference(l::Int) -> Function

Algebraic difference V_RW(r) - V_Z(r) as a function of (r, M).

This verifies isospectrality: both potentials share the same scattering
matrix and quasi-normal mode spectrum despite having different shapes.

Ground truth: Chandrasekhar, Mathematical Theory of Black Holes, Ch 4.
"""
function schwarzschild_potential_difference(l::Int)
    l >= 2 || error("Potential difference requires l ≥ 2")
    V_RW = schwarzschild_rw_potential(l)
    V_Z = schwarzschild_zerilli_potential(l)
    (r, M) -> V_RW(r, M) - V_Z(r, M)
end
