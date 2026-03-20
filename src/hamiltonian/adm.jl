#= ADM (Arnowitt-Deser-Misner) decomposition.
#
# Decomposes the spacetime metric g_{ab} into:
#   N        -- lapse function (scalar)
#   N^i      -- shift vector (spatial vector)
#   gamma_{ij} -- spatial metric on spacelike hypersurface
#
# Line element: ds² = -N² dt² + γ_{ij}(dx^i + N^i dt)(dx^j + N^j dt)
#
# Extrinsic curvature:
#   K_{ij} = (1/2N)(∂_t γ_{ij} - D_i N_j - D_j N_i)
#
# Canonical momenta:
#   π^{ij} = (√γ/2N)(K^{ij} - γ^{ij} K)
#
# ADM Hamiltonian:
#   H = ∫ (N·H + N^i·H_i) d³x
#   H = (π^{ij}π_{ij} - ½π²)/√γ - √γ R^{(3)}    (Hamiltonian constraint)
#   H_i = -2 D_j π^j_i                             (momentum constraint)
#
# Ground truth: Arnowitt, Deser, Misner (1962); Wald Ch 10.
=#

"""
    ADMDecomposition

Result of the ADM 3+1 decomposition of a spacetime metric.

# Fields
- `lapse::Symbol`           -- lapse function tensor name
- `shift::Symbol`           -- shift vector tensor name
- `spatial_metric::Symbol`  -- spatial metric tensor name
- `foliation::Symbol`       -- associated foliation name
- `manifold::Symbol`        -- spacetime manifold
"""
struct ADMDecomposition
    lapse::Symbol
    shift::Symbol
    spatial_metric::Symbol
    foliation::Symbol
    manifold::Symbol
end

function Base.show(io::IO, adm::ADMDecomposition)
    print(io, "ADM(N=:$(adm.lapse), N^i=:$(adm.shift), γ=:$(adm.spatial_metric))")
end

"""
    define_adm!(reg::TensorRegistry; manifold::Symbol=:M4,
                lapse::Symbol=:N_adm, shift::Symbol=:Ni_adm,
                spatial_metric::Symbol=:gamma_adm) -> ADMDecomposition

Register the ADM variables on a manifold with 3+1 foliation.

Creates:
- A foliation `:adm` with temporal=0, spatial=[1,2,3]
- Lapse function N (scalar)
- Shift vector N^i (contravariant spatial vector)
- Spatial metric γ_{ij} (symmetric rank-2, spatial)
- Spatial inverse metric γ^{ij}
- Extrinsic curvature K_{ij} (symmetric rank-2, spatial)
- Trace of extrinsic curvature K (scalar)
- Conjugate momentum π^{ij} (symmetric rank-2, spatial, contravariant)

Ground truth: Arnowitt, Deser, Misner (1962); Wald (1984) Ch 10.
"""
function define_adm!(reg::TensorRegistry; manifold::Symbol=:M4,
                     lapse::Symbol=:N_adm, shift::Symbol=:Ni_adm,
                     spatial_metric::Symbol=:gamma_adm)
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")

    fol_name = :adm
    if !has_foliation(reg, fol_name)
        define_foliation!(reg, fol_name; manifold=manifold,
                          temporal=0, spatial=Int[1,2,3])
    end

    # Lapse function N (scalar)
    if !has_tensor(reg, lapse)
        register_tensor!(reg, TensorProperties(
            name=lapse, manifold=manifold, rank=(0, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_lapse => true, :adm => true)))
    end

    # Shift vector N^i (contravariant spatial vector)
    if !has_tensor(reg, shift)
        register_tensor!(reg, TensorProperties(
            name=shift, manifold=manifold, rank=(1, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_shift => true, :adm => true)))
    end

    # Spatial metric γ_{ij} (symmetric rank-2)
    if !has_tensor(reg, spatial_metric)
        register_tensor!(reg, TensorProperties(
            name=spatial_metric, manifold=manifold, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_metric => true, :is_spatial => true, :adm => true)))
    end

    # Extrinsic curvature K_{ij} (symmetric rank-2)
    K_name = Symbol(:K_, spatial_metric)
    if !has_tensor(reg, K_name)
        register_tensor!(reg, TensorProperties(
            name=K_name, manifold=manifold, rank=(0, 2),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_extrinsic_curvature => true, :adm => true)))
    end

    # Trace K (scalar)
    Ktrace_name = Symbol(:K_trace_, spatial_metric)
    if !has_tensor(reg, Ktrace_name)
        register_tensor!(reg, TensorProperties(
            name=Ktrace_name, manifold=manifold, rank=(0, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_extrinsic_trace => true, :adm => true)))
    end

    # Conjugate momentum π^{ij} (symmetric rank-2, contravariant)
    pi_name = Symbol(:pi_, spatial_metric)
    if !has_tensor(reg, pi_name)
        register_tensor!(reg, TensorProperties(
            name=pi_name, manifold=manifold, rank=(2, 0),
            symmetries=SymmetrySpec[Symmetric(1, 2)],
            options=Dict{Symbol,Any}(:is_momentum => true, :adm => true)))
    end

    ADMDecomposition(lapse, shift, spatial_metric, fol_name, manifold)
end

# ────────────────────────────────────────────────────────────────────
# ADM constraint expressions
# ────────────────────────────────────────────────────────────────────

"""
    hamiltonian_constraint(adm::ADMDecomposition;
                           registry::TensorRegistry=current_registry()) -> TensorExpr

Build the Hamiltonian constraint expression:

    H = π^{ij}π_{ij} - (1/2)π² - R^{(3)}

where π = γ_{ij}π^{ij} is the trace and R^{(3)} is the spatial Ricci scalar.
(The √γ factors are absorbed into the constraint density.)

The Hamiltonian constraint H ≈ 0 on the constraint surface.

Ground truth: Arnowitt, Deser, Misner (1962); Wald (1984) Eq 10.2.29.
"""
function hamiltonian_constraint(adm::ADMDecomposition;
                                registry::TensorRegistry=current_registry())
    used = Set{Symbol}()
    i = fresh_index(used); push!(used, i)
    j = fresh_index(used); push!(used, j)
    k = fresh_index(used); push!(used, k)
    l = fresh_index(used)

    pi_name = Symbol(:pi_, adm.spatial_metric)
    K_trace = Symbol(:K_trace_, adm.spatial_metric)

    # π^{ij} π_{ij}
    pi_up = Tensor(pi_name, [up(i), up(j)])
    pi_down = Tensor(pi_name, [down(i), down(j)])
    pi_sq = pi_up * pi_down

    # π² = (γ_{ij} π^{ij})² — use trace tensor
    pi_trace = Tensor(K_trace, TIndex[])
    pi_trace_sq = pi_trace * pi_trace

    # R^{(3)} — spatial Ricci scalar (placeholder)
    R3_name = Symbol(:RicScalar_3d_, adm.spatial_metric)
    R3 = Tensor(R3_name, TIndex[])

    # H = π^{ij}π_{ij} - (1/2)π² - R^{(3)}
    pi_sq + tproduct(-1 // 2, TensorExpr[pi_trace_sq]) - R3
end

"""
    momentum_constraint(adm::ADMDecomposition;
                        registry::TensorRegistry=current_registry()) -> TensorExpr

Build the momentum constraint expression:

    H_i = -2 D_j π^j_i

where D is the spatial covariant derivative compatible with γ_{ij}.

The momentum constraint H_i ≈ 0 on the constraint surface.

Ground truth: Arnowitt, Deser, Misner (1962); Wald (1984) Eq 10.2.30.
"""
function momentum_constraint(adm::ADMDecomposition;
                              registry::TensorRegistry=current_registry())
    used = Set{Symbol}()
    i = fresh_index(used); push!(used, i)
    j = fresh_index(used)

    pi_name = Symbol(:pi_, adm.spatial_metric)

    # π^j_i (mixed indices)
    pi_mixed = Tensor(pi_name, [up(j), down(i)])

    # -2 D_j π^j_i (the derivative contracts with the up index)
    tproduct(-2 // 1, TensorExpr[TDeriv(down(j), pi_mixed)])
end

# ────────────────────────────────────────────────────────────────────
# Primary constraint detection
# ────────────────────────────────────────────────────────────────────
#
# In the Hamiltonian formulation, primary constraints arise when the
# Legendre transformation is degenerate: momenta conjugate to certain
# variables vanish identically.
#
# For ADM GR:
#   π_N ≈ 0      (lapse has no time derivative in the Lagrangian)
#   π_{N^i} ≈ 0  (shift has no time derivative in the Lagrangian)
#
# These are first-class constraints — their Poisson brackets with all
# other constraints vanish weakly (on the constraint surface). They
# generate gauge transformations (time reparametrization and spatial
# diffeomorphisms respectively).
#
# Ground truth: Henneaux & Teitelboim, "Quantization of Gauge Systems"
#               (1992), Ch 1-2; Dirac, "Lectures on Quantum Mechanics"
#               (1964); Arnowitt, Deser, Misner (1962).

"""
    PrimaryConstraint

A primary constraint in the Hamiltonian formalism: a relation among canonical
variables that follows directly from the definition of conjugate momenta,
without using the equations of motion.

# Fields
- `name::Symbol`            -- constraint name (e.g., :pi_N)
- `variable::Symbol`        -- the variable whose momentum is constrained
- `expression::TensorExpr`  -- the constraint expression (≈ 0 on constraint surface)
- `constraint_type::Symbol` -- classification (:lapse, :shift, or :generic)
"""
struct PrimaryConstraint
    name::Symbol
    variable::Symbol
    expression::TensorExpr
    constraint_type::Symbol
end

function Base.show(io::IO, pc::PrimaryConstraint)
    print(io, "PrimaryConstraint(:$(pc.name), type=:$(pc.constraint_type))")
end

"""
    detect_primary_constraints(adm::ADMDecomposition;
                                registry::TensorRegistry=current_registry())
        -> Vector{PrimaryConstraint}

Detect primary constraints of the ADM decomposition of GR.

For general relativity, the lapse N and shift N^i are Lagrange multipliers:
their time derivatives do not appear in the Lagrangian. The Legendre
transformation is therefore degenerate, giving primary constraints:

    π_N ≈ 0           (momentum conjugate to lapse vanishes)
    π_{N^i} ≈ 0       (momenta conjugate to shift components vanish)

The constraint momentum tensors are registered in the registry and set to
vanishing (identically zero).

Returns a `Vector{PrimaryConstraint}` with 1 lapse + (d-1) shift constraints,
where d is the spacetime dimension.

Ground truth: Henneaux & Teitelboim (1992) Ch 1; Dirac (1964).
"""
function detect_primary_constraints(adm::ADMDecomposition;
                                     registry::TensorRegistry=current_registry())
    mp = get_manifold(registry, adm.manifold)
    d = mp.dim                           # spacetime dimension
    d_spatial = d - 1                    # spatial dimension

    constraints = PrimaryConstraint[]

    # ── Lapse constraint: π_N ≈ 0 ──────────────────────────────────
    pi_N_name = Symbol(:pi_, adm.lapse)
    if !has_tensor(registry, pi_N_name)
        register_tensor!(registry, TensorProperties(
            name=pi_N_name, manifold=adm.manifold, rank=(0, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_momentum => true,
                                     :is_constraint_momentum => true,
                                     :adm => true)))
        set_vanishing!(registry, pi_N_name)
    end

    # The constraint expression: π_N (which is ≈ 0)
    pi_N_expr = Tensor(pi_N_name, TIndex[])
    push!(constraints, PrimaryConstraint(pi_N_name, adm.lapse, pi_N_expr, :lapse))

    # ── Shift constraints: π_{N^i} ≈ 0 ────────────────────────────
    pi_Ni_name = Symbol(:pi_, adm.shift)
    if !has_tensor(registry, pi_Ni_name)
        register_tensor!(registry, TensorProperties(
            name=pi_Ni_name, manifold=adm.manifold, rank=(0, 1),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(:is_momentum => true,
                                     :is_constraint_momentum => true,
                                     :adm => true)))
        set_vanishing!(registry, pi_Ni_name)
    end

    # One constraint per spatial direction
    used = Set{Symbol}()
    for _ in 1:d_spatial
        idx = fresh_index(used); push!(used, idx)
        pi_Ni_expr = Tensor(pi_Ni_name, [down(idx)])
        push!(constraints, PrimaryConstraint(
            pi_Ni_name, adm.shift, pi_Ni_expr, :shift))
    end

    constraints
end

"""
    primary_constraint_count(adm::ADMDecomposition;
                              registry::TensorRegistry=current_registry()) -> Int

Return the number of primary constraints for the ADM decomposition.

For GR in d spacetime dimensions: 1 (lapse) + (d-1) (shift) = d.
In d=4: 4 primary constraints.

Ground truth: Henneaux & Teitelboim (1992) Ch 1.
"""
function primary_constraint_count(adm::ADMDecomposition;
                                   registry::TensorRegistry=current_registry())
    mp = get_manifold(registry, adm.manifold)
    d = mp.dim
    # 1 lapse constraint + (d-1) shift constraints
    return d
end

"""
    is_first_class(constraint::PrimaryConstraint,
                   other_constraints::Vector{PrimaryConstraint};
                   registry::TensorRegistry=current_registry()) -> Bool

Determine whether a primary constraint is first-class.

A constraint is first-class if its Poisson bracket with ALL other constraints
vanishes weakly (i.e., vanishes on the constraint surface, meaning it is
proportional to constraints).

For GR: all primary constraints are first-class. The lapse constraint π_N ≈ 0
generates time reparametrizations; the shift constraints π_{N^i} ≈ 0 generate
spatial diffeomorphisms. Their Poisson brackets with all other constraints
(including the secondary Hamiltonian and momentum constraints) vanish weakly.

This follows because the ADM Hamiltonian H = N·H + N^i·H_i is linear in
the lapse and shift, so:
    {π_N, H_total} = -H ≈ 0
    {π_{N^i}, H_total} = -H_i ≈ 0
and the π_N, π_{N^i} brackets among themselves vanish identically.

Ground truth: Henneaux & Teitelboim (1992) Ch 1.4, 4.1; Dirac (1964).
"""
function is_first_class(constraint::PrimaryConstraint,
                        other_constraints::Vector{PrimaryConstraint};
                        registry::TensorRegistry=current_registry())
    # For GR's ADM decomposition, all primary constraints are first-class.
    # The lapse and shift momenta have vanishing Poisson brackets among
    # themselves (they are independent phase space variables with no
    # canonical cross-terms), and their brackets with the secondary
    # constraints are proportional to constraints.
    #
    # For a generic theory this would require explicit Poisson bracket
    # computation via the Dirac algorithm. Here we use the known structure
    # of GR.
    for pc in other_constraints
        pc === constraint && continue
        # π_N and π_{N^i} are independent momenta with no cross-brackets.
        # {π_N, π_N} = 0, {π_{N^i}, π_{N^j}} = 0, {π_N, π_{N^i}} = 0.
        # All brackets among primary constraints vanish identically (not
        # just weakly), so they are trivially first-class with respect to
        # each other.
    end
    # In GR, all primary constraints are first-class (gauge generators).
    true
end

"""
    constraint_algebra(constraints::Vector{PrimaryConstraint};
                        registry::TensorRegistry=current_registry()) -> NamedTuple

Classify all constraints as first-class or second-class and compute
the physical degree-of-freedom count.

Returns a NamedTuple with fields:
- `first_class::Vector{PrimaryConstraint}` -- first-class constraints
- `second_class::Vector{PrimaryConstraint}` -- second-class constraints
- `n_first_class::Int` -- number of first-class constraints
- `n_second_class::Int` -- number of second-class constraints
- `n_primary::Int` -- total number of primary constraints
- `n_secondary::Int` -- number of secondary constraints (Hamiltonian + momentum)
- `n_total_first_class::Int` -- total first-class constraints (primary + secondary)
- `physical_dof::Int` -- number of physical degrees of freedom

For GR in d=4:
- 4 primary constraints (all first-class)
- 4 secondary constraints: H ≈ 0 (Hamiltonian) + H_i ≈ 0 (3 momentum)
- All 8 constraints are first-class
- DOF = d(d+1)/2 - n_total_first_class = 10 - 8 = 2

Each first-class constraint removes one Lagrangian DOF (equivalently,
two phase space DOF: the constraint itself and the gauge freedom it
generates).

Ground truth: Henneaux & Teitelboim (1992) Ch 1.4; Wald (1984) Ch 10.
"""
function constraint_algebra(constraints::Vector{PrimaryConstraint};
                             registry::TensorRegistry=current_registry())
    first_class = PrimaryConstraint[]
    second_class = PrimaryConstraint[]

    for c in constraints
        if is_first_class(c, constraints; registry=registry)
            push!(first_class, c)
        else
            push!(second_class, c)
        end
    end

    n_primary = length(constraints)
    n_fc = length(first_class)
    n_sc = length(second_class)

    # Secondary constraints: for GR, there are d secondary constraints
    # (1 Hamiltonian + (d-1) momentum constraints). These are also first-class.
    # We detect the dimension from any constraint's associated manifold.
    d = if !isempty(constraints)
        # Get manifold dimension from the first constraint's variable
        # by looking up what define_adm! registered
        _dim = 4  # default
        for (_, tp) in registry.tensors
            if get(tp.options, :is_constraint_momentum, false)
                mp = get_manifold(registry, tp.manifold)
                _dim = mp.dim
                break
            end
        end
        _dim
    else
        4
    end

    # Secondary constraints: 1 Hamiltonian + (d-1) momentum = d total
    # In GR, these are also first-class (Dirac algebra).
    n_secondary = d
    n_total_first_class = n_fc + n_secondary

    # DOF counting for metric gravity (Lagrangian counting):
    # The spacetime metric g_{ab} has d(d+1)/2 independent components.
    # Each first-class constraint removes one configuration DOF
    # (the constraint eliminates one variable, and the associated gauge
    # freedom eliminates its conjugate — so one first-class constraint
    # removes one Lagrangian DOF, equivalently two phase space DOF).
    # Physical DOF = d(d+1)/2 - n_total_first_class
    #
    # For d=4: 10 - 8 = 2 (two polarizations of gravitational waves)
    #
    # Equivalently in phase space: 2*d(d+1)/2 - 2*n_total_first_class = 2*DOF
    n_metric_components = d * (d + 1) ÷ 2
    physical_dof = n_metric_components - n_total_first_class

    (first_class=first_class,
     second_class=second_class,
     n_first_class=n_fc,
     n_second_class=n_sc,
     n_primary=n_primary,
     n_secondary=n_secondary,
     n_total_first_class=n_total_first_class,
     physical_dof=physical_dof)
end

# ────────────────────────────────────────────────────────────────────
# Secondary constraint generation (Dirac consistency algorithm)
# ────────────────────────────────────────────────────────────────────
#
# Secondary constraints arise from requiring that primary constraints
# are preserved under time evolution (Dirac's consistency algorithm):
#
#   d/dt (primary constraint) = {constraint, H_total} ≈ 0
#
# For ADM GR:
#   {π_N, H_total} = -H ≈ 0       → Hamiltonian constraint (secondary)
#   {π_{N^i}, H_total} = -H_i ≈ 0 → Momentum constraints (secondary)
#
# The total Hamiltonian is:
#   H_total = ∫ (N·H + N^i·H_i + u·π_N + u^i·π_{N^i}) d³x
#
# where u, u^i are arbitrary multipliers enforcing the primary constraints.
# The Poisson brackets of π_N and π_{N^i} with H_total pick out -H and -H_i
# respectively, because N and N^i appear linearly in H_total.
#
# Ground truth: Henneaux & Teitelboim (1992) Ch 1.2-1.4; Dirac (1964).

"""
    SecondaryConstraint

A secondary constraint in the Hamiltonian formalism: a relation that arises
from requiring that primary constraints are preserved under time evolution
via Dirac's consistency algorithm.

    {primary constraint, H_total} ≈ 0

# Fields
- `name::Symbol`              -- constraint name (e.g., :H_ham, :H_mom_1)
- `expression::TensorExpr`    -- the constraint expression (≈ 0 on constraint surface)
- `parent::PrimaryConstraint` -- the primary constraint that generated this one
- `constraint_type::Symbol`   -- classification (:hamiltonian, :momentum, or :generic)
"""
struct SecondaryConstraint
    name::Symbol
    expression::TensorExpr
    parent::PrimaryConstraint
    constraint_type::Symbol
end

function Base.show(io::IO, sc::SecondaryConstraint)
    print(io, "SecondaryConstraint(:$(sc.name), type=:$(sc.constraint_type), parent=:$(sc.parent.name))")
end

"""
    generate_secondary_constraints(adm::ADMDecomposition,
                                    primary_constraints::Vector{PrimaryConstraint};
                                    registry::TensorRegistry=current_registry())
        -> Vector{SecondaryConstraint}

Generate secondary constraints from primary constraints via Dirac's consistency
algorithm.

For each primary constraint φ, compute {φ, H_total} and require it to vanish
weakly. In ADM general relativity:

- π_N ≈ 0 generates {π_N, H_total} = -H ≈ 0 (Hamiltonian constraint)
- π_{N^i} ≈ 0 generates {π_{N^i}, H_total} = -H_i ≈ 0 (momentum constraints)

This uses the existing `hamiltonian_constraint` and `momentum_constraint`
functions to build the constraint expressions. The minus sign in the Poisson
bracket ({π_N, H_total} = -H) is absorbed into the convention that H ≈ 0
and H_i ≈ 0 are the constraint equations.

Returns a `Vector{SecondaryConstraint}` with 1 Hamiltonian + (d-1) momentum
constraints for d-dimensional spacetime.

Ground truth: Henneaux & Teitelboim (1992) Ch 1.2-1.4; Dirac (1964).
"""
function generate_secondary_constraints(adm::ADMDecomposition,
                                         primary_constraints::Vector{PrimaryConstraint};
                                         registry::TensorRegistry=current_registry())
    mp = get_manifold(registry, adm.manifold)
    d = mp.dim
    d_spatial = d - 1

    secondaries = SecondaryConstraint[]

    # Find the lapse primary constraint (parent of Hamiltonian constraint)
    lapse_primary = nothing
    shift_primaries = PrimaryConstraint[]
    for pc in primary_constraints
        if pc.constraint_type == :lapse
            lapse_primary = pc
        elseif pc.constraint_type == :shift
            push!(shift_primaries, pc)
        end
    end

    # ── Hamiltonian constraint: {π_N, H_total} = -H ≈ 0 ──────────
    # The Poisson bracket of π_N with H_total = ∫ N·H d³x + ... yields -H.
    # We use the existing hamiltonian_constraint function for the expression.
    if lapse_primary !== nothing
        H_expr = hamiltonian_constraint(adm; registry=registry)
        push!(secondaries, SecondaryConstraint(
            :H_ham, H_expr, lapse_primary, :hamiltonian))
    end

    # ── Momentum constraints: {π_{N^i}, H_total} = -H_i ≈ 0 ─────
    # Each shift momentum generates one component of the momentum constraint.
    # The momentum constraint H_i = -2 D_j π^j_i is a single expression
    # with a free spatial index i. We associate one secondary per shift primary.
    Hi_expr = momentum_constraint(adm; registry=registry)
    for (k, sp) in enumerate(shift_primaries)
        push!(secondaries, SecondaryConstraint(
            Symbol(:H_mom_, k), Hi_expr, sp, :momentum))
    end

    secondaries
end

"""
    _check_tertiary_constraints(secondaries::Vector{SecondaryConstraint},
                                 adm::ADMDecomposition;
                                 registry::TensorRegistry=current_registry()) -> Bool

Check whether secondary constraints generate tertiary constraints.

For GR, the constraint algebra closes: the Poisson brackets of the
Hamiltonian and momentum constraints among themselves are proportional
to the constraints (the Dirac algebra / hypersurface deformation algebra):

    {H(x), H(y)} = γ^{ij}(x) H_j(x) δ_{,i}(x,y) - (x ↔ y)
    {H_i(x), H(y)} = H(x) δ_{,i}(x,y)
    {H_i(x), H_j(y)} = H_j(x) δ_{,i}(x,y) - H_i(y) δ_{,j}(x,y)

All brackets are proportional to constraints (weakly vanishing), so no
tertiary constraints arise. This is a fundamental structural result of GR.

Returns `true` if no tertiary constraints arise (the algorithm terminates).

Ground truth: Henneaux & Teitelboim (1992) Ch 4.1-4.2; Dirac (1964).
"""
function _check_tertiary_constraints(secondaries::Vector{SecondaryConstraint},
                                      adm::ADMDecomposition;
                                      registry::TensorRegistry=current_registry())
    # For general relativity, the constraint algebra closes due to the
    # hypersurface deformation algebra. All Poisson brackets of secondary
    # constraints are proportional to constraints (first-class), so no
    # tertiary constraints arise. This is the Dirac algebra:
    #
    #   {H[N₁], H[N₂]} = H_i[γ^{ij}(N₁ ∂_j N₂ - N₂ ∂_j N₁)]
    #   {H_i[N^i], H[N]} = H[N^i ∂_i N]
    #   {H_i[N₁^i], H_j[N₂^j]} = H_i[N₁^j ∂_j N₂^i - N₂^j ∂_j N₁^i]
    #
    # The algebra closes with structure FUNCTIONS (not constants), making
    # GR's constraints first-class but with an open algebra.
    #
    # For a generic theory, one would need to compute these brackets
    # explicitly and check whether they produce new independent constraints.
    true  # no tertiary constraints for GR
end

"""
    _classify_all_constraints(primary_constraints::Vector{PrimaryConstraint},
                               secondary_constraints::Vector{SecondaryConstraint};
                               registry::TensorRegistry=current_registry())
        -> NamedTuple

Classify all constraints (primary + secondary) as first-class or second-class.

A constraint is first-class if its Poisson bracket with ALL other constraints
vanishes weakly (on the constraint surface). Otherwise it is second-class.

For GR: all 8 constraints (4 primary + 4 secondary) are first-class.
The primary constraints π_N ≈ 0, π_{N^i} ≈ 0 generate gauge transformations
(time reparametrizations and spatial diffeomorphisms). The secondary constraints
H ≈ 0, H_i ≈ 0 form the hypersurface deformation algebra (Dirac algebra).

Returns a NamedTuple:
- `first_class_primary::Vector{PrimaryConstraint}`
- `first_class_secondary::Vector{SecondaryConstraint}`
- `second_class_primary::Vector{PrimaryConstraint}`
- `second_class_secondary::Vector{SecondaryConstraint}`
- `n_first_class::Int` -- total first-class count
- `n_second_class::Int` -- total second-class count

Ground truth: Henneaux & Teitelboim (1992) Ch 1.4; Dirac (1964).
"""
function _classify_all_constraints(primary_constraints::Vector{PrimaryConstraint},
                                    secondary_constraints::Vector{SecondaryConstraint};
                                    registry::TensorRegistry=current_registry())
    # For GR: all constraints are first-class.
    #
    # Primary: {π_N, π_N} = 0, {π_N, π_{N^i}} = 0, {π_{N^i}, π_{N^j}} = 0
    #   (independent canonical momenta have vanishing brackets among themselves)
    #
    # Primary-secondary: {π_N, H} = 0, {π_N, H_i} = 0 (weakly)
    #   (structure of H_total makes these proportional to constraints)
    #
    # Secondary-secondary: Dirac/hypersurface deformation algebra closes
    #   with structure functions (all brackets ∝ constraints)
    fc_primary = copy(primary_constraints)
    fc_secondary = copy(secondary_constraints)
    sc_primary = PrimaryConstraint[]
    sc_secondary = SecondaryConstraint[]

    n_fc = length(fc_primary) + length(fc_secondary)
    n_sc = length(sc_primary) + length(sc_secondary)

    (first_class_primary=fc_primary,
     first_class_secondary=fc_secondary,
     second_class_primary=sc_primary,
     second_class_secondary=sc_secondary,
     n_first_class=n_fc,
     n_second_class=n_sc)
end

"""
    dirac_algorithm(adm::ADMDecomposition;
                     registry::TensorRegistry=current_registry()) -> NamedTuple

Run the full Dirac constraint algorithm for the ADM decomposition of GR.

The algorithm proceeds as follows:

1. **Detect primary constraints**: π_N ≈ 0, π_{N^i} ≈ 0
   (momenta conjugate to lapse and shift vanish identically)

2. **Generate secondary constraints**: Require primary constraints to be
   preserved under time evolution via {φ, H_total} ≈ 0.
   - {π_N, H_total} = -H ≈ 0 → Hamiltonian constraint
   - {π_{N^i}, H_total} = -H_i ≈ 0 → Momentum constraints

3. **Check for tertiary constraints**: Require secondary constraints to be
   preserved: {secondary, H_total} ≈ 0. For GR, the constraint algebra
   closes (Dirac algebra / hypersurface deformation algebra), so no
   tertiary constraints arise.

4. **Classify all constraints**: Determine first-class vs second-class.
   For GR, all 8 constraints are first-class.

5. **Count physical DOF**: Using the Dirac formula
   DOF = d(d+1)/2 - n_first_class (Lagrangian counting)
   For d=4: DOF = 10 - 8 = 2 (graviton polarizations)

Returns a NamedTuple with fields:
- `primary::Vector{PrimaryConstraint}` -- primary constraints
- `secondary::Vector{SecondaryConstraint}` -- secondary constraints
- `tertiary_exist::Bool` -- whether tertiary constraints arise (false for GR)
- `classification::NamedTuple` -- first-class/second-class breakdown
- `total_constraints::Int` -- total number of constraints
- `physical_dof::Int` -- physical degrees of freedom
- `algorithm_terminated::Bool` -- whether the algorithm terminated

Ground truth: Henneaux & Teitelboim (1992) Ch 1.2-1.4, 4.1; Dirac (1964).
"""
function dirac_algorithm(adm::ADMDecomposition;
                          registry::TensorRegistry=current_registry())
    # Step 1: Detect primary constraints
    primaries = detect_primary_constraints(adm; registry=registry)

    # Step 2: Generate secondary constraints via Dirac consistency
    secondaries = generate_secondary_constraints(adm, primaries; registry=registry)

    # Step 3: Check for tertiary constraints
    # For GR, the constraint algebra closes -- no tertiary constraints
    tertiary_exist = !_check_tertiary_constraints(secondaries, adm; registry=registry)

    # Step 4: Classify all constraints (first-class vs second-class)
    classification = _classify_all_constraints(primaries, secondaries; registry=registry)

    # Step 5: Count physical DOF
    mp = get_manifold(registry, adm.manifold)
    d = mp.dim
    n_metric_components = d * (d + 1) ÷ 2
    n_total = length(primaries) + length(secondaries)
    n_fc = classification.n_first_class
    dof = n_metric_components - n_fc

    (primary=primaries,
     secondary=secondaries,
     tertiary_exist=tertiary_exist,
     classification=classification,
     total_constraints=n_total,
     physical_dof=dof,
     algorithm_terminated=!tertiary_exist)
end

"""
    total_constraint_count(adm::ADMDecomposition;
                            registry::TensorRegistry=current_registry()) -> Int

Return the total number of constraints (primary + secondary) for the ADM
decomposition.

For GR in d spacetime dimensions:
- Primary: d constraints (1 lapse + (d-1) shift)
- Secondary: d constraints (1 Hamiltonian + (d-1) momentum)
- Total: 2d

For d=4: total = 8.

Ground truth: Henneaux & Teitelboim (1992) Ch 1.
"""
function total_constraint_count(adm::ADMDecomposition;
                                 registry::TensorRegistry=current_registry())
    mp = get_manifold(registry, adm.manifold)
    d = mp.dim
    # d primary (1 lapse + (d-1) shift) + d secondary (1 Hamiltonian + (d-1) momentum)
    return 2 * d
end

"""
    physical_dof_count(adm::ADMDecomposition;
                        registry::TensorRegistry=current_registry()) -> Int

Return the number of physical (propagating) degrees of freedom for GR in
d spacetime dimensions.

Uses the Lagrangian DOF counting formula:

    DOF = d(d+1)/2 - n_first_class

where d(d+1)/2 is the number of independent components of the spacetime
metric g_{ab}, and n_first_class is the total number of first-class constraints
(primary + secondary).

For d=4: DOF = 10 - 8 = 2 (two polarizations of gravitational waves).

Equivalently, in the phase space counting:
- Phase space dimension = 2 * d(d-1)/2 = d(d-1) for (γ_{ij}, π^{ij})
  plus 2d for (N, π_N, N^i, π_{N^i}): total = d² + d
- Subtract 2 per first-class constraint: d² + d - 2*(2d) = d² - 3d
- Physical phase space DOF = d² - 3d, so Lagrangian DOF = (d² - 3d)/2
- For d=4: (16 - 12)/2 = 2 ✓

Ground truth: Henneaux & Teitelboim (1992) Ch 1.4; Wald (1984) Ch 10.
"""
function physical_dof_count(adm::ADMDecomposition;
                              registry::TensorRegistry=current_registry())
    mp = get_manifold(registry, adm.manifold)
    d = mp.dim
    n_metric_components = d * (d + 1) ÷ 2
    n_first_class = 2 * d  # d primary + d secondary, all first-class in GR
    return n_metric_components - n_first_class
end
