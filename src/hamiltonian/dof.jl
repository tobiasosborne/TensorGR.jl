#= Degree of freedom counting via Dirac's constraint algorithm.
#
# Provides a rich DOFSummary struct that wraps the Dirac phase-space
# counting formula:
#
#   phase_dof = 2*N - 2*n_fc - n_sc
#   config_dof = phase_dof / 2
#
# where N = number of canonical (q,p) pairs, n_fc = first-class constraint
# count, n_sc = second-class constraint count.
#
# Each first-class constraint removes 2 phase-space DOF (constraint + gauge).
# Each second-class constraint removes 1 phase-space DOF.
#
# Ground truth: Henneaux & Teitelboim (1992) Sec 1.4, Eq (1.69);
#               Dirac (1964) "Lectures on Quantum Mechanics".
=#

"""
    DOFSummary

Summary of degree-of-freedom analysis from the Dirac constraint algorithm.

# Fields
- `config_dof::Int`       -- Configuration-space degrees of freedom
- `phase_dof::Int`        -- Phase-space DOF = 2 * config_dof
- `n_canonical_pairs::Int` -- Number of canonical (q,p) pairs
- `n_first_class::Int`    -- First-class constraint count
- `n_second_class::Int`   -- Second-class constraint count
- `n_gauge::Int`          -- Gauge freedom count (= n_first_class)
- `description::String`   -- Human-readable summary

The Dirac formula:
    phase_dof = 2*n_canonical_pairs - 2*n_first_class - n_second_class
    config_dof = phase_dof / 2

Ground truth: Henneaux & Teitelboim (1992) Sec 1.4.
"""
struct DOFSummary
    config_dof::Int
    phase_dof::Int
    n_canonical_pairs::Int
    n_first_class::Int
    n_second_class::Int
    n_gauge::Int
    description::String
end

function Base.show(io::IO, ds::DOFSummary)
    print(io, "DOFSummary(config_dof=$(ds.config_dof), phase_dof=$(ds.phase_dof), ",
          "n_canonical_pairs=$(ds.n_canonical_pairs), ",
          "n_first_class=$(ds.n_first_class), n_second_class=$(ds.n_second_class), ",
          "n_gauge=$(ds.n_gauge))")
end

"""
    dof_count(n_canonical_pairs::Int, n_first_class::Int,
              n_second_class::Int) -> DOFSummary

Count physical degrees of freedom using the Dirac formula:

    phase_dof = 2*n_canonical_pairs - 2*n_first_class - n_second_class
    config_dof = phase_dof / 2

Each first-class constraint removes 2 phase-space DOF (the constraint
equation itself plus the gauge freedom it generates). Each second-class
constraint removes exactly 1 phase-space DOF.

# Validation
- `n_second_class` must be even (second-class constraints come in pairs)
- The resulting phase_dof must be non-negative
- The resulting phase_dof must be even (for integer config_dof)

# Ground truth values
- GR (4D): dof_count(10, 8, 0) -> config_dof=2, phase_dof=4
- Maxwell (4D): dof_count(4, 2, 0) -> config_dof=2, phase_dof=4
- Proca (4D): dof_count(4, 0, 2) -> config_dof=3, phase_dof=6

Ground truth: Henneaux & Teitelboim (1992) Sec 1.4, Eq (1.69).
"""
function dof_count(n_canonical_pairs::Int, n_first_class::Int,
                   n_second_class::Int)
    # Validate inputs
    if n_canonical_pairs < 0
        error("n_canonical_pairs must be non-negative, got $n_canonical_pairs.")
    end
    if n_first_class < 0
        error("n_first_class must be non-negative, got $n_first_class.")
    end
    if n_second_class < 0
        error("n_second_class must be non-negative, got $n_second_class.")
    end

    # Second-class constraints must come in pairs (Dirac consistency)
    if isodd(n_second_class)
        error("Second-class constraints must come in pairs (even count), " *
              "got n_second_class=$n_second_class. " *
              "This indicates an error in the constraint analysis.")
    end

    # Dirac formula
    phase_dof = 2 * n_canonical_pairs - 2 * n_first_class - n_second_class

    if phase_dof < 0
        error("Negative phase-space DOF ($phase_dof): " *
              "n_canonical_pairs=$n_canonical_pairs, " *
              "n_first_class=$n_first_class, n_second_class=$n_second_class. " *
              "Constraint count exceeds phase space dimension.")
    end

    if isodd(phase_dof)
        error("Odd phase-space DOF ($phase_dof) cannot yield integer " *
              "configuration-space DOF. " *
              "n_canonical_pairs=$n_canonical_pairs, " *
              "n_first_class=$n_first_class, n_second_class=$n_second_class.")
    end

    config_dof = phase_dof ÷ 2

    desc = "DOF analysis: $n_canonical_pairs canonical pairs, " *
           "$n_first_class first-class constraints (gauge generators), " *
           "$n_second_class second-class constraints. " *
           "Physical DOF: $config_dof config-space ($phase_dof phase-space)."

    DOFSummary(config_dof, phase_dof, n_canonical_pairs,
               n_first_class, n_second_class, n_first_class, desc)
end

"""
    dof_from_classification(cc::ConstraintClassification;
                             n_canonical_pairs::Int) -> DOFSummary

Build a DOFSummary from an existing `ConstraintClassification`.

Extracts the first-class and second-class counts from `cc` and delegates
to `dof_count` for the Dirac formula.

# Arguments
- `cc`                -- constraint classification from `classify_constraints`
- `n_canonical_pairs` -- number of canonical (q,p) pairs in the full phase space

Ground truth: Henneaux & Teitelboim (1992) Sec 1.4.
"""
function dof_from_classification(cc::ConstraintClassification;
                                  n_canonical_pairs::Int)
    dof_count(n_canonical_pairs, cc.n_first_class, cc.n_second_class)
end

"""
    dof_summary(adm::ADMDecomposition;
                registry::TensorRegistry=current_registry()) -> DOFSummary

Run the full DOF counting pipeline for an ADM decomposition:

1. Detect primary constraints (degenerate Legendre transform)
2. Generate secondary constraints (Dirac consistency algorithm)
3. Classify constraints (first-class vs second-class)
4. Count physical DOF via the Dirac formula

For GR in d spacetime dimensions:
- n_canonical_pairs = d(d+1)/2 (independent metric components)
- n_first_class = 2d (d primary + d secondary, all first-class)
- n_second_class = 0
- config_dof = d(d+1)/2 - 2d/1 = ... for d=4: 10 - 8 = 2

Returns a complete DOFSummary.

Ground truth: Henneaux & Teitelboim (1992) Sec 1.4, Ch 4;
              Arnowitt, Deser, Misner (1962).
"""
function dof_summary(adm::ADMDecomposition;
                     registry::TensorRegistry=current_registry())
    # Step 1: Detect primary constraints
    primaries = detect_primary_constraints(adm; registry=registry)

    # Step 2: Generate secondary constraints
    secondaries = generate_secondary_constraints(adm, primaries; registry=registry)

    # Step 3: Classify all constraints
    cc = classify_constraints(primaries, secondaries; registry=registry)

    # Step 4: Determine canonical pairs from manifold dimension
    mp = get_manifold(registry, adm.manifold)
    d = mp.dim
    n_canonical_pairs = d * (d + 1) ÷ 2

    # Step 5: Build DOFSummary
    dof_from_classification(cc; n_canonical_pairs=n_canonical_pairs)
end
