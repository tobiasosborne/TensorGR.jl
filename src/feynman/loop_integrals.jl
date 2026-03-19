#= Loop integral representation with d-dimensional regularization.
#
# Represents one-loop momentum integrals from Feynman diagram contraction:
#   I = int d^d q / (2pi)^d  N(q, k) / [q^2 (q-k1)^2 ...]
#
# Supports:
# - PropagatorDenom: single denominator factor
# - MomentumIntegral: complete loop integral with numerator/denominators
# - ScalarIntegral: Passarino-Veltman scalar master integrals (A0, B0, C0, D0)
# - Topology identification and dim-reg metric contraction
#
# Ground truth:
# - Passarino & Veltman, Nucl. Phys. B 160 (1979) 151
# - Leibbrandt, Rev. Mod. Phys. 47 (1975) 849
# - Goldberger & Rothstein, hep-th/0409156
=#

# ────────────────────────────────────────────────────────────────────
# Types
# ────────────────────────────────────────────────────────────────────

"""
    PropagatorDenom

A single propagator denominator factor: 1 / (p² - m²)^power.

# Fields
- `momentum::Symbol` -- momentum label (e.g., :q, :q_minus_k)
- `mass_sq::Any`     -- mass² (0 for graviton, :m2 for massive)
- `power::Int`       -- exponent (1 = standard, >1 from IBP)
"""
struct PropagatorDenom
    momentum::Symbol
    mass_sq::Any
    power::Int
end

PropagatorDenom(momentum::Symbol, mass_sq=0) =
    PropagatorDenom(momentum, mass_sq, 1)

function Base.show(io::IO, pd::PropagatorDenom)
    m_str = pd.mass_sq == 0 ? "" : " - $(pd.mass_sq)"
    pow_str = pd.power == 1 ? "" : "^$(pd.power)"
    print(io, "($(pd.momentum)²$(m_str))$(pow_str)")
end

"""
    MomentumIntegral

A loop integral with tensor-valued numerator and propagator denominators.

# Fields
- `loop_momenta::Vector{Symbol}`      -- integration variables ([:q] for 1-loop)
- `numerator::TensorExpr`             -- tensor-valued numerator N(q, k)
- `denominators::Vector{PropagatorDenom}` -- propagator factors
- `dim::Any`                           -- spacetime dimension (4, or :d for dim-reg)
- `prefactor::Any`                     -- overall normalization (1/(2π)^d etc.)
"""
struct MomentumIntegral
    loop_momenta::Vector{Symbol}
    numerator::TensorExpr
    denominators::Vector{PropagatorDenom}
    dim::Any
    prefactor::Any
end

function MomentumIntegral(loop_momenta::Vector{Symbol}, numerator::TensorExpr,
                          denominators::Vector{PropagatorDenom};
                          dim=4, prefactor=1)
    MomentumIntegral(loop_momenta, numerator, denominators, dim, prefactor)
end

function Base.show(io::IO, mi::MomentumIntegral)
    nl = length(mi.loop_momenta)
    nd = length(mi.denominators)
    top = integral_topology(mi)
    print(io, "MomentumIntegral($(nl)-loop, $(nd) props, topology=:$(top), d=$(mi.dim))")
end

"""
    ScalarIntegral

A scalar master integral in the Passarino-Veltman basis.

Topologies: :A0 (tadpole, 1 prop), :B0 (bubble, 2 props),
:C0 (triangle, 3 props), :D0 (box, 4 props).

# Fields
- `topology::Symbol`          -- :A0, :B0, :C0, :D0
- `masses::Vector{Any}`       -- internal masses² [m1², m2², ...]
- `external_inv::Vector{Any}` -- external momentum invariants [k1², (k1+k2)², ...]
- `dim::Any`                  -- spacetime dimension
"""
struct ScalarIntegral
    topology::Symbol
    masses::Vector{Any}
    external_inv::Vector{Any}
    dim::Any
end

function Base.show(io::IO, si::ScalarIntegral)
    print(io, si.topology, "(masses=", si.masses, ", ext=", si.external_inv,
          ", d=", si.dim, ")")
end

# ────────────────────────────────────────────────────────────────────
# Topology identification
# ────────────────────────────────────────────────────────────────────

"""
    integral_topology(integral::MomentumIntegral) -> Symbol

Identify the integral topology from the number of propagator denominators.

Returns :tadpole (1), :bubble (2), :triangle (3), :box (4),
or :pentagon (5), :hexagon (6), :N_gon for higher.

Ground truth: standard QFT nomenclature.
"""
function integral_topology(integral::MomentumIntegral)
    n = length(integral.denominators)
    n == 0 && return :scaleless
    n == 1 && return :tadpole
    n == 2 && return :bubble
    n == 3 && return :triangle
    n == 4 && return :box
    n == 5 && return :pentagon
    n == 6 && return :hexagon
    Symbol(n, :_gon)
end

"""
    integral_topology(n_props::Int) -> Symbol

Identify topology from propagator count directly.
"""
function integral_topology(n_props::Int)
    n_props == 0 && return :scaleless
    n_props == 1 && return :tadpole
    n_props == 2 && return :bubble
    n_props == 3 && return :triangle
    n_props == 4 && return :box
    n_props == 5 && return :pentagon
    n_props == 6 && return :hexagon
    Symbol(n_props, :_gon)
end

"""
    pv_topology(integral::MomentumIntegral) -> Symbol

Return the Passarino-Veltman master integral name.
:A0 (tadpole), :B0 (bubble), :C0 (triangle), :D0 (box).
"""
function pv_topology(integral::MomentumIntegral)
    n = length(integral.denominators)
    n == 1 && return :A0
    n == 2 && return :B0
    n == 3 && return :C0
    n == 4 && return :D0
    Symbol(Char('A' - 1 + n), :0)
end

# ────────────────────────────────────────────────────────────────────
# Construction helpers
# ────────────────────────────────────────────────────────────────────

"""
    to_momentum_integral(numerator::TensorExpr, denominators::Vector{PropagatorDenom},
                         loop_momenta::Vector{Symbol}; dim=4) -> MomentumIntegral

Construct a MomentumIntegral from a numerator and denominator factors.
"""
function to_momentum_integral(numerator::TensorExpr,
                               denominators::Vector{PropagatorDenom},
                               loop_momenta::Vector{Symbol};
                               dim=4, prefactor=1)
    MomentumIntegral(loop_momenta, numerator, denominators; dim=dim, prefactor=prefactor)
end

"""
    massless_bubble(k_sq::Any; loop_mom::Symbol=:q, dim=4) -> MomentumIntegral

Construct a standard massless bubble integral: ∫ d^d q / [q² (q-k)²].

Ground truth: Passarino & Veltman 1979, Eq A.3.
"""
function massless_bubble(k_sq; loop_mom::Symbol=:q, dim=4)
    denoms = [
        PropagatorDenom(loop_mom, 0),
        PropagatorDenom(Symbol(loop_mom, :_minus_k), 0)
    ]
    MomentumIntegral([loop_mom], TScalar(1 // 1), denoms; dim=dim)
end

"""
    massless_triangle(; loop_mom::Symbol=:q, dim=4) -> MomentumIntegral

Construct a standard massless triangle integral: ∫ d^d q / [q² (q-k1)² (q-k2)²].
"""
function massless_triangle(; loop_mom::Symbol=:q, dim=4)
    denoms = [
        PropagatorDenom(loop_mom, 0),
        PropagatorDenom(Symbol(loop_mom, :_minus_k1), 0),
        PropagatorDenom(Symbol(loop_mom, :_minus_k2), 0)
    ]
    MomentumIntegral([loop_mom], TScalar(1 // 1), denoms; dim=dim)
end

# ────────────────────────────────────────────────────────────────────
# Dim-reg metric contraction
# ────────────────────────────────────────────────────────────────────

"""
    dimreg_trace(dim) -> Any

The d-dimensional metric trace: η^{ab}η_{ab} = d.

In dim-reg, d = 4 - 2ε. Returns the symbolic or numeric dimension.
"""
dimreg_trace(dim::Int) = dim
dimreg_trace(dim) = dim  # symbolic: returns :d or expression

"""
    n_loops(integral::MomentumIntegral) -> Int

Number of independent loop momenta.
"""
n_loops(integral::MomentumIntegral) = length(integral.loop_momenta)

"""
    total_propagator_power(integral::MomentumIntegral) -> Int

Sum of all propagator powers (for superficial degree of divergence).
"""
function total_propagator_power(integral::MomentumIntegral)
    sum(d.power for d in integral.denominators; init=0)
end

"""
    superficial_divergence(integral::MomentumIntegral) -> Any

Superficial degree of divergence: ω = d·L - 2·P + N_num
where L = loops, P = total propagator power, N_num = numerator momentum power.

For ω ≥ 0 the integral is UV-divergent and requires regularization.
"""
function superficial_divergence(integral::MomentumIntegral)
    L = n_loops(integral)
    P = total_propagator_power(integral)
    # Numerator power is not easily extractable from TensorExpr,
    # so we return the scalar part only (assuming no numerator momenta)
    dim = integral.dim
    if dim isa Int
        return dim * L - 2 * P
    end
    # Symbolic: return expression
    Expr(:call, :-, Expr(:call, :*, dim, L), 2 * P)
end
