using Test
using TensorGR

@testset "TensorGR.jl" begin
    include("test_types.jl")
    include("test_registry.jl")
    include("test_show.jl")
    include("test_display.jl")
    include("test_xperm.jl")
    include("test_indices.jl")
    include("test_walk.jl")
    include("test_escape_hatch.jl")
    include("test_arithmetic.jl")
    include("test_macros.jl")
    include("test_contraction.jl")
    include("test_canonicalize.jl")
    include("test_derivatives.jl")
    include("test_gr_objects.jl")
    include("test_linearize.jl")
    include("test_svt.jl")
    include("test_quadratic_action.jl")
    include("test_integration.jl")
    # Phase 0: Infrastructure
    include("test_rules.jl")
    include("test_scalar_algebra.jl")
    include("test_simplify.jl")
    # Phase 1: Full Canonicalization
    include("test_phase1.jl")
    # Phase 2: Covariant Derivatives
    include("test_phase2.jl")
    # Phase 3: Perturbation Theory
    include("test_phase3.jl")
    # Phase 4: Component Calculations
    include("test_phase4.jl")
    # Phase 5+6: Curvature Algebra + Exterior Calculus
    include("test_phase56.jl")
    # Curvature algebra conversions
    include("test_conversions.jl")
    # Phase 7: Advanced Features
    include("test_phase7.jl")
    # Phase 8: Hardening & Extensions
    include("test_phase8.jl")
    # Integration V2: Full pipeline
    include("test_integration_v2.jl")
    # Symmetrize / Antisymmetrize
    include("test_symmetrize.jl")
    # New features (streams A-K)
    include("test_new_features.jl")
    # VBundle support
    include("test_vbundle.jl")
    # LaTeX parser
    include("test_latex_parser.jl")
    # 3+1 Foliation
    include("test_foliation.jl")
    # CAS integration (Symbolics.jl) — requires Symbolics in test deps
    if isdefined(Main, :Symbolics) || try @eval(using Symbolics); true catch; false end
        include("test_cas_integration.jl")
        # Resolve simplify name collision (TensorGR vs Symbolics)
        import TensorGR: simplify
        # Symbolic components (Symbolics.jl)
        include("test_symbolic_components.jl")
        # Metric ansatz generators (Symbolics.jl)
        include("test_metric_ansatz.jl")
    else
        @info "Skipping CAS/symbolic component tests: Symbolics.jl not available"
    end
    # Tensor equation solver
    include("test_solve.jl")
    # Smooth mappings (pullback/pushforward)
    include("test_mapping.jl")
    # Product manifolds
    include("test_product_manifold.jl")
    # Six-derivative gravity spectrum
    include("test_6deriv_spectrum.jl")
    # Kernel extraction and spin projection (two-momentum-correct)
    include("test_kernel_extraction.jl")
    # Curvature invariant catalog
    include("test_invariants.jl")
    # Curvature syzygies (Gauss-Bonnet, dim-dependent identities)
    include("test_syzygies.jl")
    # All contractions (metric-based full contraction enumeration)
    include("test_all_contractions.jl")
    # Contraction ansatz (most general scalar from tensor products)
    include("test_contraction_ansatz.jl")
    # Contraction filtering by symmetry
    include("test_contraction_filtering.jl")
    # Regression and coverage tests
    include("test_regressions.jl")
    # Geodesic equation
    include("test_geodesics.jl")
    # Equation of state
    include("test_eos.jl")
    # Submanifold & boundary tests (GHY, Gauss-Codazzi, ibp_with_boundary)
    include("test_submanifold_boundary.jl")
    # TOV solver
    include("test_tov.jl")
    # dS-adapted spin projectors (MSS form factors)
    include("test_ds_spin_projectors.jl")
    # Scalar spherical harmonics
    include("test_scalar_harmonics.jl")
    # Clebsch-Gordan coefficients and harmonic products
    include("test_clebsch_gordan.jl")
    # Vector spherical harmonics
    include("test_vector_harmonics.jl")
    # Vector harmonic orthogonality (Martel & Poisson 2005, Eqs 2.13-2.15)
    include("test_harmonic_orthogonality.jl")
    # Tensor spherical harmonics (rank-2, even/odd parity)
    include("test_tensor_harmonics.jl")
    # Tensor harmonic orthogonality (Martel & Poisson 2005, Eqs 2.18-2.21)
    include("test_tensor_harmonic_orthogonality.jl")
    # Angular Laplacian on S^2
    include("test_laplacian_s2.jl")
    # Scalar field harmonic decomposition (Martel & Poisson 2005, Eq 2.4)
    include("test_decompose_scalar.jl")
    # Symmetric tensor harmonic decomposition (Martel & Poisson 2005, Sec III.A)
    include("test_decompose_tensor.jl")
    # Ground truth verification: 3j, CG, and Gaunt vs independent Racah formula
    include("test_ground_truth_3j.jl")
    # Trace-free enforcement
    include("test_tracefree.jl")
    # sort_covds_to_box enhancement
    include("test_sort_covds_to_box.jl")
    # Divergence-free enforcement
    include("test_divfree.jl")
    # sort_covds_to_div enhancement
    include("test_sort_covds_to_div.jl")
    # Constraints validation: Weyl trace-free and Bianchi identities
    include("test_constraints_validation.jl")
    # Algebra-valued differential forms
    include("test_algebra_forms.jl")
    # Gauge-covariant exterior derivative
    include("test_gauge_covd.jl")
    # Field strength, Yang-Mills EOM, instanton density
    include("test_field_strength.jl")
    # Chern-Simons 3-form (Nakahara Sec 11.5, Eq 11.106b)
    include("test_chern_simons.jl")
    # Ground-truth verification: Martel & Poisson (2005) spherical harmonics
    include("test_ground_truth_harmonics.jl")
    # symmetrize_covds
    include("test_symmetrize_covds.jl")
    # Ground truth verification: Weyl tensor identities (Wald 1984)
    include("test_ground_truth_weyl.jl")
    # Ground truth: contraction counts vs Nutma (2014), arXiv:1308.3493
    include("test_ground_truth_contractions.jl")
    # Ground truth: quadratic Riemann invariants vs Fulling et al. (1992), CQG 9:1151
    include("test_quadratic_riemann_invariants.jl")
    # Energy condition checker (xIdeal)
    include("test_energy_conditions.jl")
    # Weyl scalars (Newman-Penrose) and null tetrad
    include("test_weyl_scalars.jl")
    # Petrov invariants I, J from Weyl tensor
    include("test_petrov_invariants.jl")
    # Covariant phase space: EOM extraction
    include("test_phase_space_eom.jl")
end
