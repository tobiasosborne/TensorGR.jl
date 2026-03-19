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
    # Ground truth: contraction counts (Nutma 2014)
    include("test_ground_truth_contractions.jl")
    # Ground truth: quadratic Riemann invariants (Fulling et al. 1992)
    include("test_quadratic_riemann_invariants.jl")
    # symmetrize_covds: scalar optimization, covd tag preservation
    include("test_symmetrize_covds.jl")
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
    # sort_covds_to_box enhancement
    include("test_sort_covds_to_box.jl")
    # Spinor VBundle registration
    include("test_spinor_bundles.jl")
    # Spinor dummy pair analysis
    include("test_spinor_dummy_pairs.jl")
    # Spin metric epsilon
    include("test_spin_metric.jl")
    # Spinor display (dotted/primed notation)
    include("test_spinor_display.jl")
    # Spinor contraction (see-saw rule)
    include("test_spinor_contraction.jl")
    # Spinor canonicalization (xperm with spinor index symmetries)
    include("test_spinor_canonicalize.jl")
    # Soldering form sigma^a_{AA'} and tensor-spinor conversion
    include("test_soldering_form.jl")
    # Spinor structure one-liner setup
    include("test_spinor_setup.jl")
    # Frame bundle (Lorentz VBundle for tetrads)
    include("test_frame_bundle.jl")
    # xIdeal: Petrov classification
    include("test_petrov_invariants.jl")
    include("test_petrov_classify.jl")
    # xIdeal: Segre classification
    include("test_segre.jl")
    # xIdeal: Energy conditions
    include("test_energy_conditions.jl")
    # xIdeal: Validation (Schwarzschild, FRW, pp-wave)
    include("test_xideal_schwarzschild.jl")
    include("test_xideal_frw.jl")
    include("test_petrov_ppwave.jl")
    # Scalar-tensor gravity
    include("test_horndeski.jl")
    include("test_horndeski_eom.jl")
    include("test_beyond_horndeski.jl")
    include("test_dhost.jl")
    include("test_dhost_degeneracy.jl")
    include("test_eft_de.jl")
    include("test_fr_as_horndeski.jl")
    include("test_multi_horndeski.jl")
    # Spherical harmonics
    include("test_scalar_harmonics.jl")
    include("test_vector_harmonics.jl")
    include("test_tensor_harmonics.jl")
    include("test_clebsch_gordan.jl")
    include("test_harmonic_orthogonality.jl")
    include("test_tensor_harmonic_orthogonality.jl")
    include("test_ground_truth_harmonics.jl")
    # Covariant phase space
    include("test_phase_space_eom.jl")
    include("test_noether_current.jl")
    include("test_noether_charge.jl")
    include("test_symplectic_potential.jl")
    include("test_symplectic_current.jl")
    # DDI (dimensionally dependent identities)
    include("test_ddi_rules.jl")
    include("test_ddi_simplify.jl")
    include("test_riemann_ddi.jl")
    # Invar (RInv/DualRInv)
    include("test_rinv.jl")
    include("test_rinv_canonicalize.jl")
    include("test_dual_rinv.jl")
    # Feynman diagram types
    include("test_feynman_types.jl")
    # PPN metric ansatz
    include("test_ppn_metric.jl")
    # Algebra-valued differential forms
    include("test_algebra_forms.jl")
    # Feynman diagram contraction engine
    include("test_feynman_contraction.jl")
    # PPN-to-component bridge
    include("test_ppn_decompose.jl")
    # PPN velocity-order expansion
    include("test_ppn_velocity_order.jl")
    # Loop integral representation
    include("test_loop_integrals.jl")
    # Vector field spin projectors
    include("test_vector_spin_projectors.jl")
    # PN potential extraction
    include("test_pn_matching.jl")
    # Antisymmetric rank-2 spin projectors
    include("test_antisym2_spin.jl")
    # Bianchi cosmology
    include("test_bianchi.jl")
    # Cosmological gauge choices
    include("test_gauge_choices.jl")
    # Invar Level 1: Permutation symmetries
    include("test_simplify_levels.jl")
    # Rank-3 spin projectors
    include("test_rank3_spin.jl")
    # Scalar curvature spinor Lambda
    include("test_lambda_spinor.jl")
    # Invar Level 2: Bianchi identity
    include("test_invar_level2.jl")
    # ADM decomposition
    include("test_adm.jl")
    # Bimetric gravity
    include("test_bimetric.jl")
    # PPN field equations
    include("test_ppn_field_equations.jl")
    # PPN observables
    include("test_ppn_observables.jl")
    # Gamma matrices
    include("test_gamma_matrix.jl")
    # Affine connection
    include("test_affine_connection.jl")
    # Chain continuations: Poisson, HR potential, torsion, traces
    include("test_chain_continuations.jl")
    # xAct ground truth: paper-verified equations (Nutma, Brizuela, Barker, Hohmann)
    include("test_xact_ground_truth.jl")
    # Weyl and Ricci curvature spinors
    include("test_weyl_ricci_spinors.jl")
    # Newman-Penrose null tetrad
    include("test_null_tetrad.jl")
    # Spin covariant derivative
    include("test_spin_covd.jl")
    # Irreducible spinor decomposition
    include("test_irreducible_spinor.jl")
    # NP Weyl and Ricci scalars
    include("test_np_scalars.jl")
    # Spinor pipeline integration
    include("test_spinor_integration.jl")
    # Riemann spinor decomposition
    include("test_curvature_decomp.jl")
    # Spinor commutator identities
    include("test_spin_commutator.jl")
    # NP spin coefficients
    include("test_spin_coefficients.jl")
    # GHP spin/boost weights
    include("test_ghp_weights.jl")
    # NP commutator relations
    include("test_np_commutators.jl")
    # GHP derivative operators
    include("test_ghp_derivatives.jl")
    # Multi-field kernel extraction
    include("test_multi_field_kernel.jl")
    # Invar Level 3: Second Bianchi identity
    include("test_invar_level3.jl")
    # PPN scalar-tensor validation (Hohmann 2021)
    include("test_ppn_scalar_tensor.jl")
    # Moore-Penrose propagator (PSALTer)
    include("test_moore_penrose.jl")
    # Gamma^5 chirality matrix
    include("test_gamma5.jl")
    # Unitarity conditions (PSALTer)
    include("test_unitarity.jl")
    # Non-metricity decomposition
    include("test_nonmetricity.jl")
    # GHP commutator relations
    include("test_ghp_commutators.jl")
    # Source constraints from gauge invariance
    include("test_source_constraints.jl")
    # PSALTer validation: Maxwell theory
    include("test_maxwell_spectrum.jl")
    # PSALTer validation: Proca theory
    include("test_proca_spectrum.jl")
    # PSALTer validation: Fierz-Pauli unitarity
    include("test_fp_unitarity.jl")
    # Distortion tensor decomposition
    include("test_distortion.jl")
    # Fierz identities
    include("test_fierz.jl")
    # Metric-affine curvature
    include("test_ma_curvature.jl")
    # Charge conjugation matrix
    include("test_charge_conjugation.jl")
    # Bimetric linearized perturbation
    include("test_bimetric_linearize.jl")
end
