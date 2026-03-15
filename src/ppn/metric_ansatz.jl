#= PPN (Parametrized Post-Newtonian) metric ansatz.

Implements the standard PPN metric parameterization from:
  Will, "Theory and Experiment in Gravitational Physics" (2018), Eqs 4.1--4.3.

The PPN metric in the standard gauge (isotropic coordinates) is:

  g_{00} = -1 + 2U - 2*beta*U^2 + 2*xi*Phi_W
           + (2*gamma + 2 + alpha3 + zeta1 - 2*xi)*Phi_1
           + 2*(3*gamma - 2*beta + 1 + zeta2 + xi)*Phi_2
           + 2*(1 + zeta3)*Phi_3
           + 2*(3*gamma + 3*zeta4 - 2*xi)*Phi_4
           - (zeta1 - 2*xi)*A

  g_{0i} = -1/2*(4*gamma + 3 + alpha1 - alpha2 + zeta1 - 2*xi)*V_i
           - 1/2*(1 + alpha2 - zeta1 + 2*xi)*W_i

  g_{ij} = (1 + 2*gamma*U)*delta_{ij}

The 10 PPN parameters are: gamma, beta, xi, alpha1, alpha2, alpha3,
zeta1, zeta2, zeta3, zeta4.

GR values: gamma = 1, beta = 1, all others = 0.
=#

# ─────────────────────────────────────────────────────────────
# PPNParameters: container for the 10 PPN parameters
# ─────────────────────────────────────────────────────────────

"""
    PPNParameters

Container for the 10 standard PPN parameters that characterize
deviations from general relativity in the weak-field, slow-motion
limit. Ground truth: Will (2018), Table 4.1.

# Fields
- `gamma::Any`  -- light deflection, Shapiro delay (GR: 1)
- `beta::Any`   -- perihelion precession (GR: 1)
- `xi::Any`     -- preferred-location (Whitehead) effects (GR: 0)
- `alpha1::Any` -- preferred-frame effects (GR: 0)
- `alpha2::Any` -- preferred-frame effects (GR: 0)
- `alpha3::Any` -- preferred-frame effects (GR: 0)
- `zeta1::Any`  -- conservation law violation (GR: 0)
- `zeta2::Any`  -- conservation law violation (GR: 0)
- `zeta3::Any`  -- conservation law violation (GR: 0)
- `zeta4::Any`  -- conservation law violation (GR: 0)

# Example
```julia
params = PPNParameters(1, 1, 0, 0, 0, 0, 0, 0, 0, 0)  # GR
ppn_gr()  # convenience constructor for GR values
```
"""
struct PPNParameters
    gamma::Any    # light deflection, Shapiro delay (GR: 1)
    beta::Any     # perihelion precession (GR: 1)
    xi::Any       # preferred-location effects (GR: 0)
    alpha1::Any   # preferred-frame effects (GR: 0)
    alpha2::Any
    alpha3::Any
    zeta1::Any    # conservation law violation (GR: 0)
    zeta2::Any
    zeta3::Any
    zeta4::Any
end

"""
    ppn_gr() -> PPNParameters

Return PPN parameters corresponding to general relativity:
gamma = 1, beta = 1, all others = 0.
"""
ppn_gr() = PPNParameters(1, 1, 0, 0, 0, 0, 0, 0, 0, 0)

"""
    is_gr(params::PPNParameters) -> Bool

Check whether PPN parameters match GR values (gamma=1, beta=1, rest=0).
"""
function is_gr(params::PPNParameters)
    params.gamma == 1 && params.beta == 1 &&
    params.xi == 0 && params.alpha1 == 0 && params.alpha2 == 0 &&
    params.alpha3 == 0 && params.zeta1 == 0 && params.zeta2 == 0 &&
    params.zeta3 == 0 && params.zeta4 == 0
end

function Base.show(io::IO, p::PPNParameters)
    print(io, "PPNParameters(gamma=", p.gamma, ", beta=", p.beta,
          ", xi=", p.xi, ", alpha1=", p.alpha1, ", alpha2=", p.alpha2,
          ", alpha3=", p.alpha3, ", zeta1=", p.zeta1, ", zeta2=", p.zeta2,
          ", zeta3=", p.zeta3, ", zeta4=", p.zeta4, ")")
end

# ─────────────────────────────────────────────────────────────
# PPN potentials
# ─────────────────────────────────────────────────────────────

"""
    define_ppn_potentials!(reg; manifold=:M4)

Register the standard PPN potentials as tensors in the registry.

Potentials registered (Will 2018, Sec 4.1):
- `U`         -- Newtonian potential (scalar, rank (0,0))
- `U_ppn`     -- superpotential U_{ij} (symmetric rank-2, (0,2))
- `Phi_W`     -- Whitehead potential (scalar, (0,0))
- `Phi_1`...`Phi_4` -- post-Newtonian potentials (scalars, (0,0))
- `A_ppn`     -- vector-potential scalar piece (scalar, (0,0))
- `V_ppn`     -- gravito-magnetic vector V_i (vector, (0,1))
- `W_ppn`     -- gravito-magnetic vector W_i (vector, (0,1))

Returns nothing. Potentials are accessed by their registered names.

# Example
```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_ppn_potentials!(reg; manifold=:M4)
end
```
"""
function define_ppn_potentials!(reg::TensorRegistry; manifold::Symbol=:M4)
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")

    # U: Newtonian gravitational potential (scalar)
    if !has_tensor(reg, :U)
        register_tensor!(reg, TensorProperties(
            name=:U, manifold=manifold, rank=(0, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_ppn_potential => true,
                                     :ppn_type => :newtonian)))
    end

    # U_{ij}: superpotential (symmetric rank-2 covariant tensor)
    if !has_tensor(reg, :U_ppn)
        register_tensor!(reg, TensorProperties(
            name=:U_ppn, manifold=manifold, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_ppn_potential => true,
                                     :ppn_type => :superpotential)))
    end

    # Phi_W: Whitehead potential (scalar)
    if !has_tensor(reg, :Phi_W)
        register_tensor!(reg, TensorProperties(
            name=:Phi_W, manifold=manifold, rank=(0, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_ppn_potential => true,
                                     :ppn_type => :whitehead)))
    end

    # Phi_1..Phi_4: post-Newtonian potentials (scalars)
    for k in 1:4
        pname = Symbol(:Phi_, k)
        if !has_tensor(reg, pname)
            register_tensor!(reg, TensorProperties(
                name=pname, manifold=manifold, rank=(0, 0),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}(:is_ppn_potential => true,
                                         :ppn_type => Symbol(:phi, k))))
        end
    end

    # A_ppn: vector potential scalar piece (scalar)
    if !has_tensor(reg, :A_ppn)
        register_tensor!(reg, TensorProperties(
            name=:A_ppn, manifold=manifold, rank=(0, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_ppn_potential => true,
                                     :ppn_type => :vector_scalar)))
    end

    # V_i: gravito-magnetic vector potential (covariant vector)
    if !has_tensor(reg, :V_ppn)
        register_tensor!(reg, TensorProperties(
            name=:V_ppn, manifold=manifold, rank=(0, 1),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_ppn_potential => true,
                                     :ppn_type => :gravimagnetic_V)))
    end

    # W_i: gravito-magnetic vector potential (covariant vector)
    if !has_tensor(reg, :W_ppn)
        register_tensor!(reg, TensorProperties(
            name=:W_ppn, manifold=manifold, rank=(0, 1),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_ppn_potential => true,
                                     :ppn_type => :gravimagnetic_W)))
    end

    nothing
end

# ─────────────────────────────────────────────────────────────
# Scalar coefficient builder helper
# ─────────────────────────────────────────────────────────────

# Build a TScalar from a numeric or symbolic value.
# Returns nothing for zero coefficients (caller should skip the term).
function _ppn_coeff(val)
    if val isa Integer || val isa Rational
        val == 0 && return nothing
        return TScalar(Rational{Int}(val))
    end
    # Symbolic: wrap as-is
    TScalar(val)
end

# Build a term: coeff * potential_expr
# Returns nothing if coeff is zero.
function _ppn_term(coeff, potential_expr::TensorExpr)
    if coeff isa Integer || coeff isa Rational
        coeff == 0 && return nothing
        return tproduct(Rational{Int}(coeff), TensorExpr[potential_expr])
    end
    # Symbolic coefficient: wrap in TScalar
    tproduct(1 // 1, TensorExpr[TScalar(coeff), potential_expr])
end

# ─────────────────────────────────────────────────────────────
# PPN metric ansatz builder
# ─────────────────────────────────────────────────────────────

"""
    ppn_metric_ansatz(params::PPNParameters, reg; order=2) -> Dict{Tuple{Symbol,Symbol}, TensorExpr}

Build the PPN metric components as abstract tensor expressions.
Returns a Dict mapping component-type keys to TensorExpr values:

- `(:time, :time)`     => g_{00} expression
- `(:time, :space)`    => g_{0i} expression (carries one free spatial index)
- `(:space, :space)`   => g_{ij} expression (carries two free spatial indices or delta)

Ground truth: Will (2018), Eqs 4.1--4.3.

At `order=1` (1PN), only the O(v^2) metric corrections are included:
  g_{00} = -1 + 2U,  g_{0i} = 0,  g_{ij} = (1 + 2*gamma*U)*delta_{ij}

At `order=2` (default, 2PN / full PPN), the complete PPN metric is returned.

# Arguments
- `params::PPNParameters` -- the 10 PPN parameters
- `reg::TensorRegistry`   -- registry with manifold and PPN potentials defined
- `order::Int=2`           -- PPN order (1 or 2)

# Example
```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_ppn_potentials!(reg)
    metric = ppn_metric_ansatz(ppn_gr(), reg)
    g00 = metric[(:time, :time)]
    g0i = metric[(:time, :space)]
    gij = metric[(:space, :space)]
end
```
"""
function ppn_metric_ansatz(params::PPNParameters, reg::TensorRegistry;
                           order::Int=2)
    order in (1, 2) || error("PPN order must be 1 or 2, got $order")

    # Ensure PPN potentials are registered
    has_tensor(reg, :U) || error("PPN potentials not registered. Call define_ppn_potentials!(reg) first.")

    result = Dict{Tuple{Symbol,Symbol}, TensorExpr}()

    # ── g_{00} ──
    # At 1PN: g_{00} = -1 + 2U
    # At 2PN: g_{00} = -1 + 2U - 2*beta*U^2 + 2*xi*Phi_W
    #           + (2*gamma + 2 + alpha3 + zeta1 - 2*xi)*Phi_1
    #           + 2*(3*gamma - 2*beta + 1 + zeta2 + xi)*Phi_2
    #           + 2*(1 + zeta3)*Phi_3
    #           + 2*(3*gamma + 3*zeta4 - 2*xi)*Phi_4
    #           - (zeta1 - 2*xi)*A_ppn
    result[(:time, :time)] = _build_g00(params, order)

    # ── g_{0i} ──
    # At 1PN: g_{0i} = 0
    # At 2PN: g_{0i} = -1/2*(4*gamma + 3 + alpha1 - alpha2 + zeta1 - 2*xi)*V_i
    #                  - 1/2*(1 + alpha2 - zeta1 + 2*xi)*W_i
    result[(:time, :space)] = _build_g0i(params, order)

    # ── g_{ij} ──
    # g_{ij} = (1 + 2*gamma*U)*delta_{ij}
    result[(:space, :space)] = _build_gij(params, order)

    result
end

# ── g_{00} builder ──

function _build_g00(params::PPNParameters, order::Int)
    U = Tensor(:U, TIndex[])

    # Leading term: -1
    terms = TensorExpr[TScalar(-1 // 1)]

    # 1PN: +2U
    push!(terms, tproduct(2 // 1, TensorExpr[U]))

    if order >= 2
        # -2*beta*U^2
        _add_g00_usq!(terms, params)

        # +2*xi*Phi_W
        _add_g00_scalar_potential!(terms, params.xi, 2, :Phi_W)

        # (2*gamma + 2 + alpha3 + zeta1 - 2*xi)*Phi_1
        _add_g00_phi1!(terms, params)

        # 2*(3*gamma - 2*beta + 1 + zeta2 + xi)*Phi_2
        _add_g00_phi2!(terms, params)

        # 2*(1 + zeta3)*Phi_3
        _add_g00_phi3!(terms, params)

        # 2*(3*gamma + 3*zeta4 - 2*xi)*Phi_4
        _add_g00_phi4!(terms, params)

        # -(zeta1 - 2*xi)*A_ppn
        _add_g00_a!(terms, params)
    end

    length(terms) == 1 ? terms[1] : TSum(terms)
end

function _add_g00_usq!(terms, params)
    U = Tensor(:U, TIndex[])
    # -2*beta*U^2  => coefficient is -2*beta, factors are [U, U]
    coeff_beta = params.beta
    if coeff_beta isa Integer || coeff_beta isa Rational
        c = Rational{Int}(-2 * coeff_beta)
        c == 0 && return
        push!(terms, tproduct(c, TensorExpr[U, U]))
    else
        # Symbolic beta
        push!(terms, tproduct(-2 // 1, TensorExpr[TScalar(coeff_beta), U, U]))
    end
end

function _add_g00_scalar_potential!(terms, param_val, multiplier, pot_name::Symbol)
    pot = Tensor(pot_name, TIndex[])
    if param_val isa Integer || param_val isa Rational
        c = Rational{Int}(multiplier * param_val)
        c == 0 && return
        push!(terms, tproduct(c, TensorExpr[pot]))
    else
        push!(terms, tproduct(Rational{Int}(multiplier), TensorExpr[TScalar(param_val), pot]))
    end
end

function _add_g00_phi1!(terms, params)
    # Coefficient: 2*gamma + 2 + alpha3 + zeta1 - 2*xi
    pot = Tensor(:Phi_1, TIndex[])
    g, a3, z1, xi = params.gamma, params.alpha3, params.zeta1, params.xi
    if all(v -> v isa Integer || v isa Rational, (g, a3, z1, xi))
        c = Rational{Int}(2*g + 2 + a3 + z1 - 2*xi)
        c == 0 && return
        push!(terms, tproduct(c, TensorExpr[pot]))
    else
        # Build symbolic sum as TScalar
        push!(terms, tproduct(1 // 1, TensorExpr[
            _symbolic_ppn_sum([(2, g), (2, 1), (1, a3), (1, z1), (-2, xi)]),
            pot]))
    end
end

function _add_g00_phi2!(terms, params)
    # Coefficient: 2*(3*gamma - 2*beta + 1 + zeta2 + xi)
    pot = Tensor(:Phi_2, TIndex[])
    g, b, z2, xi = params.gamma, params.beta, params.zeta2, params.xi
    if all(v -> v isa Integer || v isa Rational, (g, b, z2, xi))
        c = Rational{Int}(2 * (3*g - 2*b + 1 + z2 + xi))
        c == 0 && return
        push!(terms, tproduct(c, TensorExpr[pot]))
    else
        push!(terms, tproduct(2 // 1, TensorExpr[
            _symbolic_ppn_sum([(3, g), (-2, b), (1, 1), (1, z2), (1, xi)]),
            pot]))
    end
end

function _add_g00_phi3!(terms, params)
    # Coefficient: 2*(1 + zeta3)
    pot = Tensor(:Phi_3, TIndex[])
    z3 = params.zeta3
    if z3 isa Integer || z3 isa Rational
        c = Rational{Int}(2 * (1 + z3))
        c == 0 && return
        push!(terms, tproduct(c, TensorExpr[pot]))
    else
        push!(terms, tproduct(2 // 1, TensorExpr[
            _symbolic_ppn_sum([(1, 1), (1, z3)]),
            pot]))
    end
end

function _add_g00_phi4!(terms, params)
    # Coefficient: 2*(3*gamma + 3*zeta4 - 2*xi)
    pot = Tensor(:Phi_4, TIndex[])
    g, z4, xi = params.gamma, params.zeta4, params.xi
    if all(v -> v isa Integer || v isa Rational, (g, z4, xi))
        c = Rational{Int}(2 * (3*g + 3*z4 - 2*xi))
        c == 0 && return
        push!(terms, tproduct(c, TensorExpr[pot]))
    else
        push!(terms, tproduct(2 // 1, TensorExpr[
            _symbolic_ppn_sum([(3, g), (3, z4), (-2, xi)]),
            pot]))
    end
end

function _add_g00_a!(terms, params)
    # Coefficient: -(zeta1 - 2*xi)
    pot = Tensor(:A_ppn, TIndex[])
    z1, xi = params.zeta1, params.xi
    if all(v -> v isa Integer || v isa Rational, (z1, xi))
        c = Rational{Int}(-(z1 - 2*xi))
        c == 0 && return
        push!(terms, tproduct(c, TensorExpr[pot]))
    else
        push!(terms, tproduct(-1 // 1, TensorExpr[
            _symbolic_ppn_sum([(1, z1), (-2, xi)]),
            pot]))
    end
end

# ── g_{0i} builder ──

function _build_g0i(params::PPNParameters, order::Int)
    if order < 2
        return TScalar(0 // 1)
    end

    # g_{0i} = -1/2*(4*gamma + 3 + alpha1 - alpha2 + zeta1 - 2*xi)*V_i
    #          - 1/2*(1 + alpha2 - zeta1 + 2*xi)*W_i

    # Use index :i for the spatial index
    idx_i = down(:i)
    V = Tensor(:V_ppn, [idx_i])
    W = Tensor(:W_ppn, [idx_i])

    terms = TensorExpr[]

    g, a1, a2, z1, xi = params.gamma, params.alpha1, params.alpha2,
                         params.zeta1, params.xi

    # V_i coefficient: -1/2*(4*gamma + 3 + alpha1 - alpha2 + zeta1 - 2*xi)
    if all(v -> v isa Integer || v isa Rational, (g, a1, a2, z1, xi))
        cv = Rational{Int}(4*g + 3 + a1 - a2 + z1 - 2*xi)
        if cv != 0
            push!(terms, tproduct(-1 // 2 * cv, TensorExpr[V]))
        end
    else
        push!(terms, tproduct(-1 // 2, TensorExpr[
            _symbolic_ppn_sum([(4, g), (3, 1), (1, a1), (-1, a2), (1, z1), (-2, xi)]),
            V]))
    end

    # W_i coefficient: -1/2*(1 + alpha2 - zeta1 + 2*xi)
    if all(v -> v isa Integer || v isa Rational, (a2, z1, xi))
        cw = Rational{Int}(1 + a2 - z1 + 2*xi)
        if cw != 0
            push!(terms, tproduct(-1 // 2 * cw, TensorExpr[W]))
        end
    else
        push!(terms, tproduct(-1 // 2, TensorExpr[
            _symbolic_ppn_sum([(1, 1), (1, a2), (-1, z1), (2, xi)]),
            W]))
    end

    isempty(terms) && return TScalar(0 // 1)
    length(terms) == 1 ? terms[1] : TSum(terms)
end

# ── g_{ij} builder ──

function _build_gij(params::PPNParameters, order::Int)
    # g_{ij} = (1 + 2*gamma*U)*delta_{ij}
    # We express this as: delta_{ij} + 2*gamma*U*delta_{ij}

    idx_i = down(:i)
    idx_j = down(:j)
    delta_ij = Tensor(:delta, [idx_i, idx_j])
    U = Tensor(:U, TIndex[])

    terms = TensorExpr[]

    # delta_{ij}
    push!(terms, delta_ij)

    # 2*gamma*U*delta_{ij}
    g = params.gamma
    if g isa Integer || g isa Rational
        c = Rational{Int}(2 * g)
        if c != 0
            push!(terms, tproduct(c, TensorExpr[U, delta_ij]))
        end
    else
        push!(terms, tproduct(2 // 1, TensorExpr[TScalar(g), U, delta_ij]))
    end

    length(terms) == 1 ? terms[1] : TSum(terms)
end

# ─────────────────────────────────────────────────────────────
# Helper: build symbolic sum as TScalar for non-numeric params
# ─────────────────────────────────────────────────────────────

# _symbolic_ppn_sum builds a TScalar or TSum from a list of (coefficient, value) pairs.
# For numeric values, evaluates directly. For symbolic, builds expression.
function _symbolic_ppn_sum(pairs)
    # Try numeric evaluation first
    all_numeric = all(p -> (p[2] isa Integer || p[2] isa Rational), pairs)
    if all_numeric
        val = sum(Rational{Int}(c * v) for (c, v) in pairs)
        return TScalar(val)
    end

    # Symbolic: build sum of terms
    parts = TensorExpr[]
    for (c, v) in pairs
        if v isa Integer || v isa Rational
            cv = Rational{Int}(c * v)
            cv == 0 && continue
            push!(parts, TScalar(cv))
        elseif c == 1
            push!(parts, TScalar(v))
        elseif c == -1
            push!(parts, tproduct(-1 // 1, TensorExpr[TScalar(v)]))
        else
            push!(parts, tproduct(Rational{Int}(c), TensorExpr[TScalar(v)]))
        end
    end
    isempty(parts) && return TScalar(0 // 1)
    length(parts) == 1 ? parts[1] : TSum(parts)
end
