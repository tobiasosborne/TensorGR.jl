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
