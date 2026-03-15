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
    # Perfect fluid stress-energy tensor
    include("test_perfect_fluid.jl")
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
    # Vector harmonic orthogonality (Martel & Poisson 2005, Eqs 3.3-3.5)
    include("test_harmonic_orthogonality.jl")
    # Tensor spherical harmonics (rank-2, even/odd parity)
    include("test_tensor_harmonics.jl")
    # Tensor harmonic orthogonality (Martel & Poisson 2005, Eqs 3.8-3.11)
    include("test_tensor_harmonic_orthogonality.jl")
    # Angular Laplacian on S^2
    include("test_laplacian_s2.jl")
    # Scalar field harmonic decomposition (Martel & Poisson 2005, Sec III)
    include("test_decompose_scalar.jl")
    # Symmetric tensor harmonic decomposition (Martel & Poisson 2005, Sec III.A)
    include("test_decompose_tensor.jl")
    # Vector field harmonic decomposition (Martel & Poisson 2005, Sec III)
    include("test_decompose_vector.jl")
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
    # Angular integrals for products of tensor spherical harmonics
    include("test_angular_integrals.jl")
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
    # Petrov classification (Types I, II, III, D, N, O)
    include("test_petrov_classify.jl")
    # Covariant phase space: EOM extraction
    include("test_phase_space_eom.jl")
    # Horndeski scalar-tensor theory (Kobayashi 2019, arXiv:1901.04778)
    include("test_horndeski.jl")
    # Horndeski field equations: metric + scalar EOMs (Kobayashi 2019, Eqs 2.5-2.7)
    include("test_horndeski_eom.jl")
    # Bellini-Sawicki alpha parametrization (EFT of dark energy)
    include("test_alpha_params.jl")
    # Validation: f(R) gravity as Horndeski subcase (Kobayashi 2019, Sec 2.3)
    include("test_fr_as_horndeski.jl")
    # Beyond-Horndeski (GLPV) extensions (Gleyzes et al 2015, Kobayashi 2019)
    include("test_beyond_horndeski.jl")
    # DHOST class I Lagrangian (Langlois & Noui 2016, arXiv:1510.06930)
    include("test_dhost.jl")
    # Quadratic action for scalar-tensor perturbations on FRW (Kobayashi 2019, Sec 5.2)
    include("test_quadratic_action_st.jl")
    # Segre classification of the Ricci tensor (xIdeal)
    include("test_segre.jl")
    # Covariant phase space: symplectic potential
    include("test_symplectic_potential.jl")
    # Covariant phase space: symplectic current (Iyer-Wald 1994, Eq 2.7)
    include("test_symplectic_current.jl")
    # Covariant phase space: divergence detection
    include("test_divergence_detect.jl")
    # Spinor bundles (SL2C / SL2C_dot conjugation metadata)
    include("test_spinor_bundles.jl")
    # Spinor dummy pair analysis (fresh_spinor_index, spinor_dummy_pairs, normalize_spinor_dummies)
    include("test_spinor_indices.jl")
    # Spinor metric epsilon_{AB} (antisymmetric, Penrose-Rindler Sec 2.5)
    include("test_spin_metric.jl")
    # Spinor display: dotted/primed notation (Penrose-Rindler Vol 1, Sec 2.5)
    include("test_spinor_display.jl")
    # Frame bundle (Lorentz VBundle for tetrad/frame indices)
    include("test_frame_bundle.jl")
    # See-saw contraction rule for antisymmetric spin metric (Penrose-Rindler Sec 2.5, Eq 2.5.24)
    include("test_seesaw_contraction.jl")
    # Spinor canonicalization: vbundle-aware xperm (TGR-6cn)
    include("test_spinor_canonicalize.jl")
    # Covariant phase space: Noether current
    include("test_noether_current.jl")
    # Schwarzschild RW/Zerilli decomposition (Martel & Poisson 2005, Secs IV-V)
    include("test_schwarzschild_decompose.jl")
    # Regge-Wheeler and Zerilli master equations (Martel & Poisson 2005, Eqs 4.25-4.26, 5.14-5.15)
    include("test_master_equations.jl")
    # Covariant phase space: Noether charge (potential extraction)
    include("test_noether_charge.jl")
    # Covariant phase space: First law / Hamiltonian variation / Wald entropy
    include("test_first_law.jl")
    # Generalized Kronecker delta and DDIs
    include("test_generalized_delta.jl")
    # CPS Validation: EH Noether charge = Komar integral (Iyer & Wald 1994; Komar 1959)
    include("test_cps_komar_validation.jl")
    # xIdeal validation: Schwarzschild = Petrov Type D, Segre [(1,1)(11)]
    include("test_xideal_schwarzschild.jl")
    # xIdeal validation: FRW = Petrov Type O (conformally flat)
    include("test_xideal_frw.jl")
    # Feynman diagram type hierarchy and builder API
    include("test_feynman_types.jl")
<<<<<<< HEAD
    # Graviton 3-point and 4-point vertices (DeWitt 1967; Sannan 1986)
    include("test_graviton_vertices.jl")
=======
    # Gauge-fixing action and Faddeev-Popov ghost sector
    include("test_gauge_fixing.jl")
>>>>>>> worktree-agent-a8d3e6a4
    # DDI rule generation (rank-2 tensor contractions)
    include("test_ddi_rules.jl")
<<<<<<< HEAD
    # Wald entropy: S = A/4 from Noether charge (Iyer & Wald 1994, Eq 4.1)
    include("test_wald_entropy.jl")
    # RInv: contraction permutation representation for scalar Riemann monomials
    include("test_rinv.jl")
=======
    # Riemann DDI generation (rank-4 tensor contractions, cubic identities)
    include("test_riemann_ddi.jl")
>>>>>>> worktree-agent-a3cc0fcc
end
