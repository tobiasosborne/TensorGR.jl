module TensorGR

# Types (must come first)
include("types.jl")
include("registry.jl")
include("show.jl")

# Layer 1: xperm FFI
include("xperm/permutations.jl")
include("xperm/wrapper.jl")

# Layer 1: AST primitives
include("ast/indices.jl")
include("ast/walk.jl")

# Escape hatch
include("escape_hatch.jl")

# Layer 1.5: Rules engine
include("rules.jl")

# Layer 2: Algebra + Symmetries
include("gr/symmetries.jl")
include("algebra/arithmetic.jl")
include("algebra/contraction.jl")
include("algebra/canonicalize.jl")
include("algebra/derivatives.jl")
include("algebra/simplify.jl")
include("algebra/ibp.jl")
include("algebra/collect_tensors.jl")
include("algebra/ansatz.jl")
include("algebra/all_contractions.jl")
include("algebra/trace.jl")
include("algebra/symmetrize.jl")
include("algebra/young.jl")
include("algebra/solve.jl")

# Layer 2.5: Spinor infrastructure
include("spinors/spinor_bundles.jl")
include("spinors/spin_metric.jl")
include("spinors/soldering_form.jl")
include("spinors/setup.jl")
include("spinors/curvature_spinors.jl")
include("spinors/irreducible.jl")
include("spinors/spin_covd.jl")
include("spinors/np.jl")
include("spinors/curvature_decomp.jl")
include("spinors/spin_commutator.jl")
include("spinors/ghp.jl")
include("spinors/np_equations.jl")
include("spinors/ghp_equations.jl")

# Layer 2.6: Tetrad/frame bundle
include("tetrads/frame_bundle.jl")

# Layer 2.6: Scalar algebra
include("scalar/algebra.jl")
include("scalar/functions.jl")

# Layer 3: GR objects
include("gr/curvature.jl")
include("gr/bianchi.jl")
include("gr/syzygies.jl")
include("gr/metric.jl")
include("gr/covd.jl")
include("gr/sort_covds.jl")
include("gr/box.jl")
include("gr/lie.jl")
include("gr/killing.jl")
include("gr/symmetry_ansatz.jl")
include("gr/metric_ansatz_gen.jl")
include("gr/matter.jl")
include("gr/hypersurface.jl")
include("gr/junction.jl")
include("gr/mapping.jl")
include("gr/product_manifold.jl")
include("gr/invariants.jl")

# Layer 4: Perturbation theory
include("perturbation/partitions.jl")
include("perturbation/linearize.jl")
include("perturbation/metric_perturbation.jl")
include("perturbation/expand.jl")
include("perturbation/gauge.jl")
include("perturbation/variation.jl")
include("perturbation/backgrounds.jl")
include("perturbation/isaacson.jl")

# Layer 4: SVT decomposition
include("svt/fourier.jl")
include("svt/decompose.jl")
include("svt/projectors.jl")
include("svt/gauge_choices.jl")

# Layer 4: 3+1 foliation
include("foliation/foliation.jl")
include("foliation/decompose.jl")
include("foliation/svt_rules.jl")
include("foliation/constraints.jl")
include("foliation/sectors.jl")
include("foliation/bianchi.jl")
include("foliation/bianchi_structure.jl")

# Layer 4: Quadratic action
include("action/quadratic_action.jl")
include("action/extract_quadratic.jl")
include("action/spin_projectors.jl")
include("action/vector_spin_projectors.jl")
include("action/antisym2_spin_projectors.jl")
include("action/rank3_spin_projectors.jl")
include("action/kernel_extraction.jl")
include("action/svt_quadratic.jl")

# Layer 4: CAS integration hooks (after QuadraticForm is defined)
include("scalar/simplify_cas.jl")

# Layer 4.5: Curvature conversions + Exterior calculus
include("gr/topological.jl")
include("gr/conversions.jl")
include("exterior/forms.jl")
include("exterior/operations.jl")
include("exterior/connection_forms.jl")
include("exterior/algebra_forms.jl")

# Layer 5: Component calculations
include("components/basis.jl")
include("components/ctensor.jl")
include("components/metric_compute.jl")
include("components/values.jl")
include("components/to_basis.jl")
include("components/symbolic_metric.jl")

# Layer 5.5: xIdeal — spacetime classification (Petrov, Segre, energy conditions)
include("xideal/weyl_scalars.jl")
include("xideal/petrov_invariants.jl")
include("xideal/petrov_classify.jl")
include("xideal/segre.jl")
include("xideal/energy_conditions.jl")

# Layer 5.5: DDI (dimensionally dependent identities)
include("algebra/generalized_delta.jl")
include("algebra/ddi_rules.jl")

# Layer 5.5: Invar (RInv/DualRInv canonical forms)
include("invariants/rinv.jl")
include("invariants/dual_rinv.jl")
include("invariants/simplify_levels.jl")

# Layer 5.5: Feynman rules (graviton vertices, propagators)
include("feynman/types.jl")
include("feynman/propagator.jl")
include("feynman/vertices.jl")
include("feynman/gauge_fixing.jl")
include("feynman/matter_vertices.jl")
include("feynman/contraction.jl")
include("feynman/loop_integrals.jl")
include("feynman/pn_matching.jl")

# Layer 5.5: PPN (Parametrized Post-Newtonian)
include("ppn/metric_ansatz.jl")
include("ppn/decompose.jl")
include("ppn/velocity_order.jl")
include("ppn/field_equations.jl")
include("ppn/observables.jl")

# Layer 5.5: Fermions (Clifford algebra)
include("fermions/gamma.jl")
include("fermions/traces.jl")
include("fermions/fierz.jl")

# Layer 5.5: Metric-affine gravity
include("metric_affine/connection.jl")
include("metric_affine/torsion.jl")
include("metric_affine/nonmetricity.jl")
include("metric_affine/distortion.jl")

# Layer 5.5: Hamiltonian analysis
include("hamiltonian/adm.jl")
include("hamiltonian/poisson.jl")

# Layer 5.5: Bimetric gravity
include("bimetric/registration.jl")
include("bimetric/potential.jl")

# Layer 5.5: Covariant phase space (Noether charge, symplectic form)
include("phase_space/divergence.jl")
include("phase_space/eom.jl")
include("phase_space/symplectic.jl")
include("phase_space/noether.jl")
include("phase_space/potential.jl")
include("phase_space/first_law.jl")

# Layer 5.5: Spherical harmonics (scalar, vector, tensor)
include("harmonics/scalar_harmonics.jl")
include("harmonics/clebsch_gordan.jl")
include("harmonics/vector_harmonics.jl")
include("harmonics/tensor_harmonics.jl")
include("harmonics/orthogonality.jl")
include("harmonics/angular_integrals.jl")
include("harmonics/laplacian.jl")
include("harmonics/decompose_scalar.jl")
include("harmonics/decompose_vector.jl")
include("harmonics/decompose_tensor.jl")

# Layer 5.5: Scalar-tensor gravity (Horndeski, DHOST, EFT-DE)
include("scalar_tensor/horndeski.jl")
include("scalar_tensor/horndeski_eom.jl")
include("scalar_tensor/alpha_params.jl")
include("scalar_tensor/beyond_horndeski.jl")
include("scalar_tensor/dhost.jl")
include("scalar_tensor/dhost_degeneracy.jl")
include("scalar_tensor/eft_de.jl")
include("scalar_tensor/multi_horndeski.jl")
include("scalar_tensor/quadratic_action_st.jl")

# Layer 5.5: Worldline / PN
include("worldline/worldline.jl")

# Layer 5.5: Matter / EOS
include("matter/eos.jl")

# Layer 5.5: Geodesics
include("geodesics/geodesic.jl")

# Layer 5.5: Solvers (TOV, etc.)
include("solvers/tov.jl")

# Macros
include("macros/tensor_macro.jl")
include("macros/definitions.jl")

# LaTeX parser
include("parser/latex_parser.jl")

# Exports: Types
export TensorExpr, Tensor, TProduct, TSum, TDeriv, TScalar
export TIndex, IndexPosition, Up, Down, up, down

# Exports: Registry
export TensorRegistry, TensorProperties, ManifoldProperties, VBundleProperties
export has_manifold, has_tensor, has_vbundle, get_manifold, get_tensor, get_vbundle
export register_manifold!, register_tensor!, register_rule!, get_rules
export unregister_tensor!, unregister_manifold!, unregister_covd!
export tex_alias!
export define_vbundle!, conjugate_vbundle
export define_spinor_bundles!
export spin_up, spin_down, spin_dot_up, spin_dot_down
export is_spinor_index, is_dotted, conjugate_index
export define_spin_metric!, spin_metric
export define_soldering_form!, to_spinor_indices, to_tensor_indices
export define_spinor_structure!, @spinor_manifold
export define_frame_bundle!, frame_up, frame_down, is_frame_index
export set_vanishing!
export current_registry, with_registry

# Exports: AST primitives
export indices, free_indices, dummy_pairs
export rename_dummy, rename_dummies, fresh_index, ensure_no_dummy_clash
export index_sort, same_dummies, split_index
export walk, substitute, children
export derivative_order, is_constant, is_sorted_covds
export dagger

# Exports: Display
export to_latex, to_unicode

# Exports: Escape hatch
export to_expr, from_expr, is_well_formed, validate

# Exports: Rules
export RewriteRule, apply_rules, apply_rules_fixpoint, @rule
export make_rule, folded_rule, is_pattern_variable

# Exports: Symmetries
export Symmetric, AntiSymmetric, PairSymmetric, RiemannSymmetry
export FullySymmetric, FullyAntiSymmetric, SymmetrySpec
export symmetry_generators

# Exports: Algebra
export tproduct, tsum, contract_metrics, canonicalize, expand_derivatives
export expand_products, distribute_derivs_over_sums, flatten_metric_derivs, collect_terms, ibp, ibp_product, fix_dummy_positions, normalize_field_positions
export simplify
export collect_tensors, remove_constants, remove_tensors, index_collect
export make_ansatz
export abstract_trace, make_traceless
export symmetrize, antisymmetrize, impose_symmetry
export YoungTableau, young_shape, young_symmetrize, young_project
export all_contractions, contraction_ansatz, filter_independent_contractions
export solve_tensors

# Exports: Scalar algebra
export scalar_expand, scalar_collect, scalar_subst, scalar_cancel
export define_scalar_function!, scalar_function_derivative

# Exports: GR
export define_curvature_tensors!, einstein_expr, ricci_from_riemann
export cotton_expr, tensor_norm
export bianchi_rules
export gauss_bonnet_rule, weyl_vanishing_rule, ricci_trace_rule
export riemann_vanishing_rule, syzygy_rules
export define_killing!, check_killing
export CovDProperties, define_covd!, get_covd
export covd_to_christoffel, change_covd
export christoffel_to_grad_metric, grad_metric_to_christoffel
export commute_covds
export sort_covds_to_box, sort_covds_to_div, symmetrize_covds
export box, grad_squared, covd_chain, covd_product
export lie_derivative, lie_bracket, lie_to_covd
export SubmanifoldProperties, HypersurfaceProperties
export define_submanifold!, define_hypersurface!
export extrinsic_curvature_expr, induced_metric_expr, projector_expr
export ghy_boundary_term, ibp_with_boundary
export gauss_equation, codazzi_equation, gauss_codazzi_rules
export JunctionData, define_junction!, israel_equation, junction_stress_energy
export MappingProperties, define_mapping!, get_mapping, has_mapping
export pullback, pushforward, pullback_metric
export ProductManifoldProperties, define_product_manifold!
export has_product_manifold, get_product_manifold
export product_metric, product_scalar_curvature
export product_ricci, product_riemann, product_einstein
export product_einstein_equations

# Exports: Symmetry ansatz
export SymmetryAnsatz, SphericalSymmetry, AxialSymmetry, StaticSymmetry, HomogeneousIsotropy
export metric_ansatz

# Exports: Matter
export PerfectFluidProperties, define_perfect_fluid!, perfect_fluid_expr, get_perfect_fluid
export EquationOfState, BarotropicEOS, PolytropicEOS, TabularEOS
export PerfectFluid, pressure, sound_speed

# Exports: Curvature invariants
export InvariantEntry, INVARIANT_CATALOG, curvature_invariant, list_invariants

# Exports: Metric engine
export MetricSignature, lorentzian, euclidean, sign_det
export define_metric!, metric_signature
export set_flat!, is_flat
export freeze_metric!, unfreeze_metric!, is_frozen
export separate_metric
export metric_det_expr, sqrt_det_expr
export gdelta, expand_gdelta
export set_conformal_to!

# Exports: Perturbation
export linearize, δRiemann, δRicci, δRicciScalar
export sorted_partitions, all_compositions, multinomial
export MetricPerturbation, define_metric_perturbation!, perturb, δinverse_metric
export δchristoffel, δriemann, δricci, δricci_scalar, δricci_flat, δricci_scalar_flat, expand_perturbation
export define_tensor_perturbation!, perturbation_order, background_solution!
export gauge_transformation
export maximally_symmetric_background!, cosmological_background!, vacuum_background!
export isaacson_average
export variational_derivative, euler_lagrange
export metric_variation, var_lagrangian

# Exports: SVT / Fourier
export to_fourier, FourierConvention
export transverse_projector, tt_projector
export theta_projector, omega_projector
export spin2_projector, spin1_projector, spin0s_projector, spin0w_projector
export transfer_sw, transfer_ws
export SVTFields, svt_substitute

# Exports: 3+1 foliation
export FoliationProperties, define_foliation!, get_foliation, has_foliation
export classify_component, all_components
export split_spacetime, split_all_spacetime
export svt_rules_bardeen, svt_rules_full, apply_svt, SVTSubstitution
export svt_constraint_rules, lorentzian_contract
export collect_sectors
export foliate_and_decompose
export GaugeChoice, gauge_choice, apply_gauge!
export synchronous_gauge, newtonian_gauge, flat_slicing_gauge, comoving_gauge, uniform_density_gauge
export BianchiIBackground, define_bianchi_I!, is_isotropic, mean_hubble, shear_tensor_name
export BianchiStructureConstants, bianchi_type, structure_constant, verify_jacobi, is_class_A
export bianchi_I, bianchi_II, bianchi_VI0, bianchi_VII0, bianchi_VIII, bianchi_IX

# Exports: Quadratic action
export QuadraticForm, quadratic_form, propagator, determinant, extract_quadratic_form
export sym_det, sym_inv, sym_eval
export SpinSectorDecomposition, moore_penrose_propagator, no_ghost, no_tachyon
export SpinSectorResult, UnitarityAnalysis, unitarity_conditions
export SourceConstraint, source_constraints
export KineticKernel, extract_kernel, extract_kernel_direct, spin_project, contract_momenta
export MultiFieldKernels, extract_kernel_multi
export build_FP_momentum_kernel, build_R2_momentum_kernel, build_Ric2_momentum_kernel
export scale_kernel, combine_kernels, build_6deriv_flat_kernel, flat_6deriv_spin_projections
export _eval_spin_scalar
export BuenoCanoParams, dS_spectrum_6deriv, bc_to_form_factors
export bc_EH, bc_R2, bc_RicSq, bc_R3, bc_RRicSq, bc_Ric3
export bc_RRiem2, bc_RicRiem2, bc_Riem3
export svt_quadratic_forms_6deriv
export vector_spin1_projector, vector_spin0_projector, vector_spin_project
export antisym2_spin1_projector, antisym2_spin0_projector, antisym2_identity
export antisym3_spin1_projector, antisym3_spin0_projector, antisym3_identity

# Exports: PN matching
export FourierEntry, FOURIER_TABLE, fourier_transform_potential
export PNPotentialTerm, classify_pn_order, newton_potential_coeff

# Exports: CAS integration
export simplify_scalar, simplify_quadratic_form
export symbolic_quadratic_form, to_fourier_symbolic

# Exports: Curvature conversions
export riemann_to_weyl, weyl_to_riemann
export ricci_to_einstein, einstein_to_ricci
export contract_curvature
export schouten_to_ricci, ricci_to_schouten
export tfricci_expr, ricci_to_tfricci
export to_riemann, to_ricci
export riemann_to_christoffel, kretschmann_expr
export pontryagin_density, euler_density, chern_simons_action

# Exports: Exterior calculus
export define_form!, form_degree
export wedge, exterior_d, interior_product, hodge_dual
export codifferential, cartan_lie_d
export wedge_power
export connection_form, curvature_form
export cartan_first_structure, cartan_second_structure
export AlgValuedForm, alg_exterior_d, alg_wedge
export connection_1form, curvature_2form, gauge_covd
export field_strength, yang_mills_eom, bianchi_identity
export instanton_density, chern_simons_form, chern_simons_invariant

# Exports: Components
export ChartProperties, define_chart!, get_chart
export CTensor, ctensor_contract, ctensor_trace, ctensor_inverse, ctensor_det
export metric_christoffel, metric_riemann, metric_ricci, metric_ricci_scalar
export metric_einstein, metric_weyl, metric_kretschmann
export ComponentStore, set_component!, get_component, independent_components
export BasisProperties, define_basis!, get_basis, basis_change
export to_basis, component_array, to_ctensor

# Exports: Symbolic components
export SymbolicMetric, symbolic_diagonal_metric, symbolic_metric, sym_deriv
export symbolic_christoffel, symbolic_riemann, symbolic_ricci, symbolic_ricci_scalar
export symbolic_einstein, symbolic_kretschmann, symbolic_curvature_from_metric

# Exports: Macros
export @tensor
export @manifold, @define_tensor, @covd

# Exports: Worldline / PN
export Worldline, define_worldline!, pn_order, truncate_pn

# Exports: Geodesics
export GeodesicEquation, GeodesicSolution, setup_geodesic, geodesic_rhs!, integrate_geodesic

# Exports: TOV solver
export TOVSystem, TOVSolution, setup_tov, tov_rhs!, solve_tov, mass_radius_curve

# Exports: LaTeX parser
export parse_tex, @tex_str

# Exports: DDI
export generalized_delta, is_zero_by_dimension
export generate_ddi_rules, gauss_bonnet_ddi, register_ddi_rules!, has_ddi_rules
export generate_riemann_ddi, riemann_ddi_expr, simplify_with_ddis

# Exports: Invar (RInv/DualRInv)
export RInv, canonicalize_rinv, are_equivalent, rinv_symmetry_group
export DualRInv, left_dual, right_dual, double_dual, pontryagin_rinv
export to_tensor_expr, from_tensor_expr
export simplify_level1, simplify_level2, simplify_level3
export is_riemann_monomial, count_riemann_degree
export apply_bianchi_cyclic, bianchi_relation
export differential_bianchi, contracted_bianchi
export define_lambda_spinor!, lambda_spinor_expr
export define_weyl_spinor!, define_ricci_spinor!, define_curvature_spinors!
export irreducible_decompose
export spin_covd, spin_covd_expr
export define_null_tetrad!, np_completeness
export weyl_scalar, weyl_scalars, ricci_scalar_np, np_lambda
export spin_coefficient, all_spin_coefficients
export weyl_spinor_expr, weyl_spinor_bar_expr, ricci_spinor_expr, riemann_spinor_parts
export spinor_ricci_identity
export GHPWeight, ghp_weight, spin_weight, boost_weight, is_proper_ghp
export WEYL_SCALAR_WEIGHTS, RICCI_SCALAR_WEIGHTS, SPIN_COEFF_WEIGHTS
export GHPDerivative, GHP_DERIVATIVES, ghp_derivative, ghp_weight_shift
export np_directional_derivative, NPCommutatorRelation, np_commutator_table
export GHPCommutatorRelation, ghp_commutator_table, ghp_commutator_weight_consistency

# Exports: Feynman rules
export TensorVertex, TensorPropagator, FeynmanDiagram
export n_point, n_indices, n_loops
export graviton_propagator, propagator_numerator
export matter_graviton_vertex, scalar_matter_vertex, graviton_vertex_n
export gauge_fixing_condition, gauge_fixing_action
export fp_operator, ghost_propagator, ghost_graviton_vertex
export gauge_fixed_kinetic_operator
export build_diagram, tree_exchange_diagram, contract_diagram, DiagramAmplitude
export vertex_from_perturbation
export contract_line, find_loop_momenta, symmetry_factor
export momentum_constraints, impose_momentum_conservation, MomentumConstraint

# Exports: PPN
export PPNParameters, ppn_gr, is_gr, is_fully_conservative, is_preferred_frame_free
export is_semi_conservative, is_preferred_location_free
export define_ppn_potentials!, ppn_metric_ansatz
export ppn_foliation!, PPNMetricComponents, ppn_decompose, ppn_compose
export PPNChristoffelComponents, ppn_christoffel_1pn, ppn_christoffel
export ppn_order, ppn_max_order, truncate_ppn, ppn_truncate_metric
export PPN_ORDER_TABLE, PPN_METRIC_ORDERS
export PPNFieldEquationResult, ppn_solve_gr, ppn_solve_scalar_tensor, ppn_solve
export extract_ppn_parameters, ppn_parameter_table
export ppn_perihelion_factor, ppn_deflection_factor, ppn_shapiro_factor
export ppn_nordtvedt_eta, ppn_geodetic_factor, ppn_observational_bounds

# Exports: Gamma matrices / Clifford algebra
export GammaMatrix, gamma5, clifford_relation, gamma_trace
export Gamma5, gamma5_trace, gamma5_anticommutator, gamma5_squared
export gamma_chain_trace, trace_identity_2, trace_identity_4, slash

# Exports: Metric-affine gravity
export AffineConnection, define_affine_connection!
export is_metric_compatible, is_torsion_free
export set_metric_compatible!, set_torsion_free!
export TorsionDecomposition, decompose_torsion!, torsion_vector_expr, contortion_expr
export NonmetricityDecomposition, decompose_nonmetricity!
export weyl_vector_expr, second_trace_expr
export DistortionDecomposition, decompose_distortion!
export contortion_from_torsion, disformation_from_nonmetricity
export CliffordBasis, CB_S, CB_V, CB_T, CB_A, CB_P
export CLIFFORD_NAMES, CLIFFORD_DIM
export fierz_matrix, fierz_coefficient, fierz_identity_check

# Exports: ADM decomposition
export ADMDecomposition, define_adm!
export hamiltonian_constraint, momentum_constraint
export CanonicalPair, adm_canonical_pair, fundamental_bracket, PoissonBracketResult

# Exports: Bimetric gravity
export BimetricSetup, define_bimetric!, bimetric_field_equations
export HassanRosenParams, elementary_symmetric, hassan_rosen_potential

# Exports: Loop integrals
export PropagatorDenom, MomentumIntegral, ScalarIntegral
export integral_topology, pv_topology, to_momentum_integral
export massless_bubble, massless_triangle
export dimreg_trace, total_propagator_power, superficial_divergence

# Exports: Covariant phase space
export is_divergence, extract_divergence, split_divergence
export LagrangianDensity, EOMResult, eom_extract, extract_eom_and_theta
export SymplecticPotential, symplectic_potential, theta_eh, add_boundary_ambiguity
export NoetherCurrent, noether_current, noether_current_eh, verify_noether_conservation
export NoetherCharge, noether_charge, noether_charge_eh
export SymplecticCurrent, symplectic_current, symplectic_current_eh
export HamiltonianVariation, hamiltonian_variation, hamiltonian_variation_eh
export WaldEntropyIntegrand, wald_entropy_integrand, wald_entropy_integrand_eh

# Exports: Spherical harmonics
export wigner3j, clebsch_gordan
export ScalarHarmonic, harmonic_product, conjugate, inner_product
export EvenVectorHarmonic, OddVectorHarmonic, vector_inner_product
export divergence_eigenvalue, curl_eigenvalue, norm_squared
export EvenTensorHarmonicY, EvenTensorHarmonicZ, OddTensorHarmonic, tensor_inner_product
export angular_integral, gaunt_integral, angular_selection_rule
export vector_gaunt, tensor_gaunt
export angular_laplacian, simplify_laplacian
export ScalarMode, HarmonicDecomposition, decompose_scalar, get_mode
export VectorMode, VectorHarmonicDecomposition, decompose_vector
export TensorMode, TensorHarmonicDecomposition, decompose_symmetric_tensor
export Parity, EVEN, ODD

# Exports: Scalar-tensor gravity
export ScalarTensorFunction, g_tensor_name, differentiate_G
export HorndeskiTheory, define_horndeski!, kinetic_X
export horndeski_L2, horndeski_L3, horndeski_L4, horndeski_L5, horndeski_lagrangian
export horndeski_metric_eom, horndeski_scalar_eom, horndeski_eom
export BeyondHorndeskiTheory, define_beyond_horndeski!
export beyond_horndeski_L4, beyond_horndeski_L5, beyond_horndeski_lagrangian, alpha_H
export DHOSTTheory, define_dhost!
export dhost_L1, dhost_L2, dhost_L3, dhost_L4, dhost_L5, dhost_lagrangian
export degeneracy_conditions, is_degenerate, dhost_class
export horndeski_as_dhost, reduce_to_horndeski, dhost_dof_count
export FRWBackground, define_frw_background, BelliniSawickiAlphas
export compute_alphas, compute_alphas_numerical
export EFTDarkEnergy, eft_from_horndeski, eft_from_beyond_horndeski, eft_from_numerical
export eft_stability, eft_observables, gw170817_constraint
export eft_gr, eft_quintessence, eft_fR, apply_gw170817
export MultiScalarTensorFunction, multi_g_tensor_name, differentiate_MG
export MultiHorndeskiTheory, define_multi_horndeski!
export multi_horndeski_L2, multi_horndeski_L3, multi_horndeski_L4, multi_horndeski_L5
export multi_horndeski_lagrangian, kinetic_matrix, kinetic_matrix_full, to_single_field

# Exports: xIdeal — Petrov/Segre classification, energy conditions
export weyl_scalars, validate_null_tetrad, null_tetrad_from_metric
export petrov_invariants, weyl_contraction_invariants, is_algebraically_special
export PetrovType, TypeI, TypeII, TypeIII, TypeD, TypeN, TypeO
export petrov_classify
export SegreType, segre_classify
export EnergyConditionResult, check_energy_conditions

# Extension stubs (implemented in ext/)
function to_symbolics end
function from_symbolics end
function to_symengine end
function from_symengine end

export to_symbolics, from_symbolics, to_symengine, from_symengine

end # module TensorGR
