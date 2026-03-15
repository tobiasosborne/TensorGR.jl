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

# Layer 1.5: Spherical harmonics
include("harmonics/scalar_harmonics.jl")

# Layer 2: Algebra + Symmetries (Clebsch-Gordan included after arithmetic)
include("gr/symmetries.jl")
include("algebra/arithmetic.jl")
include("harmonics/clebsch_gordan.jl")
include("harmonics/vector_harmonics.jl")
include("harmonics/orthogonality.jl")
include("harmonics/tensor_harmonics.jl")
include("harmonics/laplacian.jl")
include("harmonics/decompose_scalar.jl")
include("harmonics/decompose_tensor.jl")
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

# Layer 2.5: Scalar algebra
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

# Layer 4: 3+1 foliation
include("foliation/foliation.jl")
include("foliation/decompose.jl")
include("foliation/svt_rules.jl")
include("foliation/constraints.jl")
include("foliation/sectors.jl")

# Layer 4: Quadratic action
include("action/quadratic_action.jl")
include("action/extract_quadratic.jl")
include("action/spin_projectors.jl")
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
export ScalarHarmonic, conjugate, angular_laplacian, inner_product
export EvenVectorHarmonic, OddVectorHarmonic, divergence_eigenvalue, curl_eigenvalue, norm_squared
export vector_inner_product, tensor_inner_product
export EvenTensorHarmonicY, EvenTensorHarmonicZ, OddTensorHarmonic
export LaplacianS2, laplacian_S2, simplify_laplacian
export wigner3j, clebsch_gordan, harmonic_product
export ScalarMode, HarmonicDecomposition, decompose_scalar, mode_count, get_mode
export Parity, EVEN, ODD, TensorMode, TensorHarmonicDecomposition, decompose_symmetric_tensor

# Exports: Registry
export TensorRegistry, TensorProperties, ManifoldProperties, VBundleProperties
export has_manifold, has_tensor, has_vbundle, get_manifold, get_tensor, get_vbundle
export register_manifold!, register_tensor!, register_rule!, get_rules
export unregister_tensor!, unregister_manifold!, unregister_covd!
export tex_alias!
export define_vbundle!
export set_vanishing!, set_tracefree!, set_divfree!
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
export tproduct, tsum, contract_metrics, enforce_tracefree, enforce_divfree, canonicalize, expand_derivatives
export expand_products, collect_terms, ibp, ibp_product, fix_dummy_positions, normalize_field_positions
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
export δchristoffel, δriemann, δricci, δricci_scalar, expand_perturbation
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

# Exports: Quadratic action
export QuadraticForm, quadratic_form, propagator, determinant, extract_quadratic_form
export sym_det, sym_inv, sym_eval
export KineticKernel, extract_kernel, extract_kernel_direct, spin_project, contract_momenta
export build_FP_momentum_kernel, build_R2_momentum_kernel, build_Ric2_momentum_kernel
export scale_kernel, combine_kernels, build_6deriv_flat_kernel, flat_6deriv_spin_projections
export _eval_spin_scalar
export BuenoCanoParams, dS_spectrum_6deriv, bc_to_form_factors
export bc_EH, bc_R2, bc_RicSq, bc_R3, bc_RRicSq, bc_Ric3
export bc_RRiem2, bc_RicRiem2, bc_Riem3
export svt_quadratic_forms_6deriv

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
export AlgValuedForm, alg_wedge, alg_exterior_d, connection_1form, curvature_2form
export gauge_covd, bianchi_identity
export field_strength, yang_mills_eom, instanton_density
export chern_simons_form, chern_simons_invariant

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

# Extension stubs (implemented in ext/)
function to_symbolics end
function from_symbolics end
function to_symengine end
function from_symengine end

export to_symbolics, from_symbolics, to_symengine, from_symengine

end # module TensorGR
