#= Constraint classification (first-class vs second-class).
#
# Classifies constraints into first-class and second-class using the
# Dirac algorithm. First-class constraints have weakly vanishing Poisson
# brackets with ALL other constraints and generate gauge transformations.
# Second-class constraints have at least one non-vanishing bracket and
# reduce phase space; they must come in pairs.
#
# DOF formula (phase space counting):
#   DOF = (dim_phase_space - 2*n_first_class - n_second_class) / 2
#
# For 4D GR:
#   phase_space_dim = 20 (10 metric components x 2)
#   n_first_class = 8 (4 primary + 4 secondary, all first-class)
#   n_second_class = 0
#   DOF = (20 - 16 - 0) / 2 = 2 (graviton polarizations)
#
# Ground truth: Henneaux & Teitelboim (1992) Ch 1, Sec 1.3-1.4;
#               Dirac (1964) "Lectures on Quantum Mechanics".
=#

"""
    ConstraintClassification

Result of classifying constraints as first-class or second-class via
the Dirac algorithm.

# Fields
- `first_class`    -- constraints whose Poisson brackets with ALL other
                      constraints vanish weakly (generate gauge transformations)
- `second_class`   -- constraints with at least one non-vanishing bracket
                      (reduce phase space, must come in pairs)
- `n_first_class`  -- number of first-class constraints
- `n_second_class` -- number of second-class constraints
- `dof`            -- physical degrees of freedom via phase space counting:
                      DOF = (phase_space_dim - 2*n_first_class - n_second_class) / 2

Ground truth: Henneaux & Teitelboim (1992) Ch 1.3-1.4.
"""
struct ConstraintClassification
    first_class::Vector{Union{PrimaryConstraint, SecondaryConstraint}}
    second_class::Vector{Union{PrimaryConstraint, SecondaryConstraint}}
    n_first_class::Int
    n_second_class::Int
    dof::Int
end

function Base.show(io::IO, cc::ConstraintClassification)
    print(io, "ConstraintClassification(first_class=$(cc.n_first_class), ",
          "second_class=$(cc.n_second_class), dof=$(cc.dof))")
end

"""
    classify_constraints(primary_constraints::Vector{PrimaryConstraint},
                          secondary_constraints::Vector{SecondaryConstraint};
                          phase_space_dim::Union{Int,Nothing}=nothing,
                          registry::TensorRegistry=current_registry())
        -> ConstraintClassification

Classify all constraints (primary + secondary) into first-class and
second-class using the Dirac algorithm.

**First-class constraints**: All Poisson brackets with ALL other constraints
vanish weakly (on the constraint surface). These generate gauge transformations.

**Second-class constraints**: At least one Poisson bracket is non-vanishing
on the constraint surface. These reduce phase space and must come in pairs.

# Arguments
- `primary_constraints`   -- primary constraints from the Legendre transform
- `secondary_constraints` -- secondary constraints from the Dirac consistency algorithm
- `phase_space_dim`       -- dimension of the full phase space (default: auto-detect
                             from registry as 2 * d(d+1)/2 for metric gravity)
- `registry`              -- tensor registry

# DOF counting (phase space formula)
    DOF = (phase_space_dim - 2*n_first_class - n_second_class) / 2

For 4D GR: phase_space_dim = 20, n_first_class = 8, n_second_class = 0
    DOF = (20 - 16 - 0) / 2 = 2

# Implementation
For the general case, delegates to the existing internal `_classify_all_constraints`
which uses the known Dirac algebra structure. For GR, all constraints are
first-class because:
- Primary-primary: {π_N, π_{N^i}} = 0 (independent momenta)
- Primary-secondary: {π_N, H} ∝ constraints, {π_{N^i}, H_j} ∝ constraints
- Secondary-secondary: Dirac/hypersurface deformation algebra closes with
  structure functions (all brackets proportional to constraints)

Ground truth: Henneaux & Teitelboim (1992) Ch 1.3-1.4; Dirac (1964).
"""
function classify_constraints(primary_constraints::Vector{PrimaryConstraint},
                               secondary_constraints::Vector{SecondaryConstraint};
                               phase_space_dim::Union{Int,Nothing}=nothing,
                               registry::TensorRegistry=current_registry())
    # Delegate to existing internal classifier (handles GR and general cases)
    cls = _classify_all_constraints(primary_constraints, secondary_constraints;
                                     registry=registry)

    # Combine first-class constraints into a single list
    first_class = Union{PrimaryConstraint, SecondaryConstraint}[]
    for pc in cls.first_class_primary
        push!(first_class, pc)
    end
    for sc in cls.first_class_secondary
        push!(first_class, sc)
    end

    # Combine second-class constraints into a single list
    second_class = Union{PrimaryConstraint, SecondaryConstraint}[]
    for pc in cls.second_class_primary
        push!(second_class, pc)
    end
    for sc in cls.second_class_secondary
        push!(second_class, sc)
    end

    n_fc = length(first_class)
    n_sc = length(second_class)

    # Validate: second-class constraints must come in pairs (Dirac consistency)
    if n_sc > 0 && isodd(n_sc)
        error("Second-class constraints must come in pairs, got $n_sc. " *
              "This indicates an error in the constraint analysis.")
    end

    # Determine phase space dimension
    ps_dim = if phase_space_dim !== nothing
        phase_space_dim
    else
        _infer_phase_space_dim(primary_constraints, secondary_constraints, registry)
    end

    # DOF = (phase_space_dim - 2*n_first_class - n_second_class) / 2
    dof_numerator = ps_dim - 2 * n_fc - n_sc
    if dof_numerator < 0
        error("Negative DOF numerator ($dof_numerator): phase_space_dim=$ps_dim, " *
              "n_first_class=$n_fc, n_second_class=$n_sc. " *
              "Check constraint counts or phase space dimension.")
    end
    if isodd(dof_numerator)
        error("DOF numerator ($dof_numerator) is odd, cannot yield integer DOF. " *
              "Check phase_space_dim=$ps_dim, n_first_class=$n_fc, n_second_class=$n_sc.")
    end
    dof = dof_numerator ÷ 2

    ConstraintClassification(first_class, second_class, n_fc, n_sc, dof)
end

"""
    _infer_phase_space_dim(primary_constraints, secondary_constraints, registry)
        -> Int

Infer the phase space dimension from the registry. For metric gravity in
d spacetime dimensions, the configuration space has d(d+1)/2 independent
metric components, giving phase_space_dim = 2 * d(d+1)/2 = d(d+1).

This uses the same manifold-dimension lookup pattern as the existing
`constraint_algebra` function.
"""
function _infer_phase_space_dim(primary_constraints::Vector{PrimaryConstraint},
                                 secondary_constraints::Vector{SecondaryConstraint},
                                 registry::TensorRegistry)
    # Find the spacetime dimension from the registry
    d = 4  # default fallback
    for (_, tp) in registry.tensors
        if get(tp.options, :is_constraint_momentum, false) ||
           get(tp.options, :adm, false)
            mp = get_manifold(registry, tp.manifold)
            d = mp.dim
            break
        end
    end
    # Phase space dim = 2 * (number of config variables)
    # For metric gravity: d(d+1)/2 metric components => d(d+1) phase space
    d * (d + 1)
end

"""
    count_dof(classification::ConstraintClassification;
              phase_space_dim::Int) -> Int

Compute the physical degrees of freedom from a constraint classification
using the Dirac phase space formula:

    DOF = (phase_space_dim - 2*n_first_class - n_second_class) / 2

Each first-class constraint removes 2 phase space DOF (the constraint itself
plus the gauge freedom it generates). Each second-class constraint removes
1 phase space DOF.

# Examples
```julia
# GR in 4D: phase_space_dim = 20, 8 first-class, 0 second-class
count_dof(cls; phase_space_dim=20)  # => 2
```

Ground truth: Henneaux & Teitelboim (1992) Ch 1.4.
"""
function count_dof(classification::ConstraintClassification;
                   phase_space_dim::Int)
    n_fc = classification.n_first_class
    n_sc = classification.n_second_class

    dof_numerator = phase_space_dim - 2 * n_fc - n_sc
    if dof_numerator < 0
        error("Negative DOF numerator ($dof_numerator): phase_space_dim=$phase_space_dim, " *
              "n_first_class=$n_fc, n_second_class=$n_sc.")
    end
    if isodd(dof_numerator)
        error("DOF numerator ($dof_numerator) is odd. " *
              "Check phase_space_dim=$phase_space_dim, n_first_class=$n_fc, n_second_class=$n_sc.")
    end
    dof_numerator ÷ 2
end

"""
    gauge_generators(classification::ConstraintClassification)
        -> Vector{Union{PrimaryConstraint, SecondaryConstraint}}

Return the first-class constraints, which generate gauge transformations.

In the Dirac formalism, each first-class constraint generates an
infinitesimal gauge transformation via its Poisson bracket with the
canonical variables:

    δ_ε q = {q, ε·φ}  ,  δ_ε p = {p, ε·φ}

where ε is a gauge parameter and φ is a first-class constraint.

For GR in 4D, the 8 gauge generators correspond to:
- π_N ≈ 0       → time reparametrization freedom
- π_{N^i} ≈ 0   → spatial diffeomorphism freedom (3 generators)
- H ≈ 0         → on-shell time evolution (Hamiltonian constraint)
- H_i ≈ 0       → on-shell spatial diffeomorphisms (momentum constraints)

Ground truth: Henneaux & Teitelboim (1992) Ch 1.4, 3.1.
"""
function gauge_generators(classification::ConstraintClassification)
    classification.first_class
end
