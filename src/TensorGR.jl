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
include("algebra/trace.jl")
include("algebra/symmetrize.jl")
include("algebra/young.jl")

# Layer 2.5: Scalar algebra
include("scalar/algebra.jl")
include("scalar/functions.jl")

# Layer 3: GR objects
include("gr/curvature.jl")
include("gr/bianchi.jl")
include("gr/metric.jl")
include("gr/covd.jl")
include("gr/sort_covds.jl")
include("gr/lie.jl")
include("gr/killing.jl")

# Layer 4: Perturbation theory
include("perturbation/partitions.jl")
include("perturbation/linearize.jl")
include("perturbation/metric_perturbation.jl")
include("perturbation/expand.jl")
include("perturbation/gauge.jl")
include("perturbation/variation.jl")

# Layer 4: SVT decomposition
include("svt/fourier.jl")
include("svt/decompose.jl")
include("svt/projectors.jl")

# Layer 4: Quadratic action
include("action/quadratic_action.jl")

# Layer 4.5: Curvature conversions + Exterior calculus
include("gr/conversions.jl")
include("exterior/forms.jl")
include("exterior/operations.jl")
include("exterior/connection_forms.jl")

# Layer 5: Component calculations
include("components/basis.jl")
include("components/ctensor.jl")
include("components/metric_compute.jl")
include("components/values.jl")
include("components/to_basis.jl")

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
export define_vbundle!
export set_vanishing!
export current_registry, with_registry

# Exports: AST primitives
export indices, free_indices, dummy_pairs
export rename_dummy, fresh_index, ensure_no_dummy_clash
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
export make_rule, folded_rule

# Exports: Symmetries
export Symmetric, AntiSymmetric, PairSymmetric, RiemannSymmetry
export FullySymmetric, FullyAntiSymmetric
export symmetry_generators

# Exports: Algebra
export tproduct, tsum, contract_metrics, canonicalize, expand_derivatives
export expand_products, collect_terms, ibp, ibp_product
export simplify
export collect_tensors, remove_constants, remove_tensors, index_collect
export make_ansatz
export abstract_trace, make_traceless
export symmetrize, antisymmetrize, impose_symmetry
export YoungTableau, young_shape, young_symmetrize, young_project
export all_contractions, contraction_ansatz

# Exports: Scalar algebra
export scalar_expand, scalar_collect, scalar_subst, scalar_cancel
export define_scalar_function!, scalar_function_derivative

# Exports: GR
export define_curvature_tensors!, einstein_expr, ricci_from_riemann
export bianchi_rules
export define_killing!
export CovDProperties, define_covd!, get_covd
export covd_to_christoffel, change_covd
export christoffel_to_grad_metric, grad_metric_to_christoffel
export commute_covds
export sort_covds_to_box, sort_covds_to_div, symmetrize_covds
export lie_derivative, lie_bracket, lie_to_covd

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
export variational_derivative, euler_lagrange
export metric_variation, var_lagrangian

# Exports: SVT / Fourier
export to_fourier, FourierConvention
export transverse_projector, tt_projector
export SVTFields, svt_substitute

# Exports: Quadratic action
export QuadraticForm, quadratic_form, propagator, determinant
export sym_det, sym_inv, sym_eval

# Exports: Curvature conversions
export riemann_to_weyl, weyl_to_riemann
export ricci_to_einstein, einstein_to_ricci
export contract_curvature
export schouten_to_ricci, ricci_to_schouten
export tfricci_expr, ricci_to_tfricci
export to_riemann, to_ricci
export riemann_to_christoffel, kretschmann_expr

# Exports: Exterior calculus
export define_form!, form_degree
export wedge, exterior_d, interior_product, hodge_dual
export codifferential, cartan_lie_d
export connection_form, curvature_form
export cartan_first_structure, cartan_second_structure

# Exports: Components
export ChartProperties, define_chart!, get_chart
export CTensor, ctensor_contract, ctensor_trace, ctensor_inverse, ctensor_det
export metric_christoffel, metric_riemann, metric_ricci, metric_ricci_scalar
export metric_einstein, metric_weyl, metric_kretschmann
export ComponentStore, set_component!, get_component, independent_components
export BasisProperties, define_basis!, get_basis, basis_change
export to_basis, component_array, to_ctensor

# Exports: Macros
export @tensor
export @manifold, @define_tensor, @covd

# Exports: LaTeX parser
export parse_tex, @tex_str

# Extension stubs (implemented in ext/)
function to_symbolics end
function from_symbolics end
function to_symengine end
function from_symengine end

export to_symbolics, from_symbolics, to_symengine, from_symengine

end # module TensorGR
