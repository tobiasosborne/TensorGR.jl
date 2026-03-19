# Linearized bimetric perturbation theory.
#
# Hassan-Rosen bimetric gravity: two dynamical metrics g_{ab} and f_{ab}
# coupled through the interaction potential V = m² Σ β_n e_n(√(g⁻¹f)).
#
# Proportional background: ḡ_{ab} = background metric, f̄_{ab} = c² ḡ_{ab}
# where c is the background ratio.
#
# Perturbations: g = ḡ + δg, f = c²ḡ + δf
#
# The linearized field equations couple δg and δf through a mass matrix.
# Diagonalization yields:
#   - Massless mode: γ_{ab} ∝ δg_{ab} + c² δf_{ab}  (standard GR graviton)
#   - Massive mode:  χ_{ab} ∝ δg_{ab} - δf_{ab}       (Fierz-Pauli massive graviton)
#
# Ground truth: Hassan & Rosen, JHEP 02 (2012) 126, arXiv:1109.3515, Sec 3;
#              Torsello et al, CQG 37, 025013 (2020), arXiv:1904.10464, Sec 4.

"""
    BimetricPerturbation

Linearized perturbation of a bimetric setup about proportional backgrounds.

# Fields
- `setup::BimetricSetup`      -- the bimetric setup
- `params::HassanRosenParams`  -- HR potential parameters
- `background_ratio::Any`      -- c such that f̄ = c² ḡ
- `delta_g::Symbol`            -- perturbation of g
- `delta_f::Symbol`            -- perturbation of f
- `massless_mode::Symbol`      -- massless combination
- `massive_mode::Symbol`       -- massive combination
"""
struct BimetricPerturbation
    setup::BimetricSetup
    params::HassanRosenParams
    background_ratio::Any
    delta_g::Symbol
    delta_f::Symbol
    massless_mode::Symbol
    massive_mode::Symbol
end

function Base.show(io::IO, bp::BimetricPerturbation)
    print(io, "BimetricPerturbation(c=$(bp.background_ratio), ",
          "massless=:$(bp.massless_mode), massive=:$(bp.massive_mode))")
end

"""
    define_bimetric_perturbation!(reg::TensorRegistry, bs::BimetricSetup,
                                   params::HassanRosenParams;
                                   background_ratio=:c) -> BimetricPerturbation

Register the linearized perturbation fields for bimetric gravity about
a proportional background: f̄_{ab} = c² ḡ_{ab}.

Creates:
- δg_{ab}: perturbation of first metric (symmetric rank-2)
- δf_{ab}: perturbation of second metric (symmetric rank-2)
- γ_{ab}: massless mode ∝ δg + c² δf (propagates as standard graviton)
- χ_{ab}: massive mode ∝ δg - δf (Fierz-Pauli massive spin-2)

Ground truth: Hassan & Rosen (2012) Sec 3; Torsello et al (2020) Sec 4.
"""
function define_bimetric_perturbation!(reg::TensorRegistry, bs::BimetricSetup,
                                        params::HassanRosenParams;
                                        background_ratio=:c)
    manifold = bs.manifold

    # Perturbation fields
    dg = Symbol(:delta_, bs.metric_g)
    df = Symbol(:delta_, bs.metric_f)
    massless = :gamma_bim
    massive = :chi_bim

    for (name, desc) in [(dg, "perturbation of g"),
                          (df, "perturbation of f"),
                          (massless, "massless mass eigenstate"),
                          (massive, "massive mass eigenstate")]
        if !has_tensor(reg, name)
            register_tensor!(reg, TensorProperties(
                name=name, manifold=manifold, rank=(0, 2),
                symmetries=SymmetrySpec[Symmetric(1, 2)],
                options=Dict{Symbol,Any}(
                    :is_bimetric_perturbation => true,
                    :description => desc)))
        end
    end

    BimetricPerturbation(bs, params, background_ratio, dg, df, massless, massive)
end

"""
    fierz_pauli_mass_squared(params::HassanRosenParams, c;
                              dim::Int=4) -> Any

Compute the Fierz-Pauli mass squared for the massive spin-2 mode
in bimetric gravity about proportional backgrounds.

For proportional backgrounds f̄ = c² ḡ, the mass of the massive mode is:

    m²_FP = m² × (β₁ + 2c β₂ + c² β₃) / (1 + c²)

where β_n are the Hassan-Rosen parameters and c is the background ratio.

Ground truth: Hassan & Rosen (2012) Eq 3.9;
             de Rham, Living Rev. Rel. 17 (2014) Sec 8.3.
"""
function fierz_pauli_mass_squared(params::HassanRosenParams, c; dim::Int=4)
    β = params.beta  # (β₀, β₁, β₂, β₃, β₄)
    m2 = params.m_sq

    # m²_FP = m² × (β₁ + 2c β₂ + c² β₃) / (1 + c²)
    numerator = β[2] + 2 * c * β[3] + c^2 * β[4]   # β₁ + 2c β₂ + c² β₃
    denominator = 1 + c^2

    if m2 isa Number && numerator isa Number && denominator isa Number
        return m2 * numerator // denominator
    end
    :($m2 * $numerator / $denominator)
end

"""
    bimetric_mass_matrix(params::HassanRosenParams, c) -> Matrix{Any}

Return the 2×2 mass matrix for the bimetric system (δg, δf).

The linearized field equations have the form:
    (G[g]_ab + M²_{gg} δg_{ab} + M²_{gf} δf_{ab} = 0)
    (G[f]_ab + M²_{fg} δg_{ab} + M²_{ff} δf_{ab} = 0)

The mass matrix M² is:
    M²_{gg} = m²(β₁ + 2c β₂ + c² β₃)
    M²_{gf} = -m²(β₁ + 2c β₂ + c² β₃)
    M²_{fg} = -m²(β₁ + 2c β₂ + c² β₃) / c²
    M²_{ff} = m²(β₁ + 2c β₂ + c² β₃) / c²

Note: M² has rank 1 (one zero eigenvalue = massless mode).

Ground truth: Hassan & Rosen (2012) Sec 3; Torsello et al (2020) Sec 4.
"""
function bimetric_mass_matrix(params::HassanRosenParams, c)
    m2 = params.m_sq
    β = params.beta

    # Effective mass parameter
    μ2 = β[2] + 2 * c * β[3] + c^2 * β[4]   # β₁ + 2c β₂ + c² β₃

    if m2 isa Number && μ2 isa Number
        M_gg = m2 * μ2
        M_gf = -m2 * μ2
        M_fg = -m2 * μ2 / c^2
        M_ff = m2 * μ2 / c^2
    else
        M_gg = :($m2 * $μ2)
        M_gf = :(-$m2 * $μ2)
        M_fg = :(-$m2 * $μ2 / $c^2)
        M_ff = :($m2 * $μ2 / $c^2)
    end

    Any[M_gg M_gf; M_fg M_ff]
end

"""
    bimetric_mass_eigenvalues(params::HassanRosenParams, c) -> NamedTuple

Compute the mass eigenvalues of the bimetric system.

Returns (massless=0, massive=m²_FP) where m²_FP is the Fierz-Pauli mass.

The eigenvectors are:
- Massless: γ = (1/(1+c²))(δg + c² δf)  — standard graviton
- Massive:  χ = (1/(1+c²))(δg - δf)  — massive spin-2

Ground truth: Hassan & Rosen (2012) Sec 3.
"""
function bimetric_mass_eigenvalues(params::HassanRosenParams, c)
    m2_FP = fierz_pauli_mass_squared(params, c)
    (massless = 0, massive = m2_FP)
end

"""
    bimetric_mass_eigenstates(bp::BimetricPerturbation) -> NamedTuple

Compute the mass eigenstates (massless and massive modes) as linear combinations
of the bimetric perturbation fields δg and δf.

For proportional backgrounds f̄ = c² ḡ:
- Massless: γ_{ab} = (1/(1+c²))(δg_{ab} + c² δf_{ab})  — standard graviton
- Massive:  χ_{ab} = (1/(1+c²))(δg_{ab} - δf_{ab})   — Fierz-Pauli massive spin-2

Returns `(massless=TensorExpr, massive=TensorExpr, m2_FP=value)`.

Ground truth: Hassan & Rosen (2012) Sec 3; Torsello et al (2020) Sec 4.

The inverse transformation is: δg = γ + c²χ, δf = γ - χ.
"""
function bimetric_mass_eigenstates(bp::BimetricPerturbation)
    c = bp.background_ratio

    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used)

    dg = Tensor(bp.delta_g, [down(a), down(b)])
    df = Tensor(bp.delta_f, [down(a), down(b)])

    # Normalization: 1/(1+c²)
    if c isa Number
        norm = 1 // (1 + c^2)
        c2 = c isa Integer ? c^2 // 1 : c^2
    else
        # Symbolic c: embed as TScalar factors
        norm = 1 // 1
        c2 = c
    end

    if c isa Number
        # Massless: (1/(1+c²)) * (δg + c² δf)
        #   = norm * δg + norm*c² * δf
        massless = tsum(TensorExpr[
            tproduct(norm, TensorExpr[dg]),
            tproduct(norm * c2, TensorExpr[df])
        ])

        # Massive: (1/(1+c²)) * (δg - δf)
        #   = norm * δg - norm * δf
        massive = tsum(TensorExpr[
            tproduct(norm, TensorExpr[dg]),
            tproduct(-norm, TensorExpr[df])
        ])
    else
        # Symbolic c: use TScalar wrappers
        c_sq = TScalar(:($c^2))
        one_plus_c2 = TScalar(:(1 + $c^2))
        inv_norm = TScalar(:(1 / (1 + $c^2)))

        massless = tsum(TensorExpr[
            tproduct(1 // 1, TensorExpr[inv_norm, dg]),
            tproduct(1 // 1, TensorExpr[inv_norm, c_sq, df])
        ])

        massive = tsum(TensorExpr[
            tproduct(1 // 1, TensorExpr[inv_norm, dg]),
            tproduct(-1 // 1, TensorExpr[inv_norm, df])
        ])
    end

    m2_FP = fierz_pauli_mass_squared(bp.params, c)

    (massless = massless, massive = massive, m2_FP = m2_FP)
end

"""
    bimetric_inverse_transform(bp::BimetricPerturbation) -> NamedTuple

Compute the inverse transformation from mass eigenstates back to metric perturbations.

Given eigenstates γ (massless) and χ (massive):
- δg_{ab} = γ_{ab} + c² χ_{ab}
- δf_{ab} = γ_{ab} - χ_{ab}

This is the exact inverse of `bimetric_mass_eigenstates`.

Returns `(delta_g=TensorExpr, delta_f=TensorExpr)`.

Ground truth: Hassan & Rosen (2012) Sec 3.
"""
function bimetric_inverse_transform(bp::BimetricPerturbation)
    c = bp.background_ratio

    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used)

    gamma = Tensor(bp.massless_mode, [down(a), down(b)])
    chi = Tensor(bp.massive_mode, [down(a), down(b)])

    if c isa Number
        c2 = c isa Integer ? c^2 // 1 : c^2

        # δg = γ + c² χ
        delta_g = tsum(TensorExpr[
            tproduct(1 // 1, TensorExpr[gamma]),
            tproduct(c2, TensorExpr[chi])
        ])

        # δf = γ - χ
        delta_f = tsum(TensorExpr[
            tproduct(1 // 1, TensorExpr[gamma]),
            tproduct(-1 // 1, TensorExpr[chi])
        ])
    else
        c_sq = TScalar(:($c^2))

        delta_g = tsum(TensorExpr[
            tproduct(1 // 1, TensorExpr[gamma]),
            tproduct(1 // 1, TensorExpr[c_sq, chi])
        ])

        delta_f = tsum(TensorExpr[
            tproduct(1 // 1, TensorExpr[gamma]),
            tproduct(-1 // 1, TensorExpr[chi])
        ])
    end

    (delta_g = delta_g, delta_f = delta_f)
end

"""
    higuchi_bound(Lambda) -> Any

The Higuchi bound for massive spin-2 on de Sitter: m² >= 2Lambda/3.
Returns 2Lambda/3 (the minimum mass squared for a healthy massive graviton).

Below this bound, the helicity-0 mode of the massive graviton becomes a ghost
(negative kinetic energy). At the exact bound m² = 2Lambda/3, the theory has
a partially massless gauge symmetry.

Ground truth: Higuchi, Nucl. Phys. B 282, 397 (1987);
             Hassan, Schmidt-May & von Strauss, PLB 715, 335 (2012).
"""
function higuchi_bound(Lambda)
    if Lambda isa Number
        return 2 * Lambda // 3
    end
    :(2 * $Lambda / 3)
end

"""
    higuchi_coefficient(m2_FP, Lambda) -> Any

The helicity-0 kinetic coefficient, proportional to (m² - 2Lambda/3).
Positive = healthy, zero = partially massless, negative = ghost.

Ground truth: Higuchi (1987); Hassan, Schmidt-May & von Strauss (2012).
"""
function higuchi_coefficient(m2_FP, Lambda)
    if m2_FP isa Number && Lambda isa Number
        return m2_FP - 2 * Lambda // 3
    end
    :($m2_FP - 2 * $Lambda / 3)
end

"""
    is_higuchi_healthy(m2_FP, Lambda) -> Bool

Check whether the Fierz-Pauli mass satisfies the Higuchi bound (m² >= 2Lambda/3).

Ground truth: Higuchi (1987); Hassan, Schmidt-May & von Strauss (2012).
"""
function is_higuchi_healthy(m2_FP, Lambda)
    m2_FP isa Number && Lambda isa Number || error("Both arguments must be numeric")
    return m2_FP >= 2 * Lambda // 3
end
