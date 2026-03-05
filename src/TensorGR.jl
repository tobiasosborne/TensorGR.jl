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

# Layer 2.5: Scalar algebra
include("scalar/algebra.jl")

# Layer 3: GR objects
include("gr/curvature.jl")
include("gr/bianchi.jl")
include("gr/covd.jl")
include("gr/sort_covds.jl")
include("gr/lie.jl")
include("gr/killing.jl")

# Layer 4: Perturbation theory
include("perturbation/partitions.jl")
include("perturbation/linearize.jl")
include("perturbation/metric_perturbation.jl")
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

# Layer 5: Component calculations
include("components/basis.jl")
include("components/ctensor.jl")
include("components/metric_compute.jl")

# Macros
include("macros/tensor_macro.jl")
include("macros/definitions.jl")

# Exports: Types
export TensorExpr, Tensor, TProduct, TSum, TDeriv, TScalar
export TIndex, IndexPosition, Up, Down, up, down

# Exports: Registry
export TensorRegistry, TensorProperties, ManifoldProperties
export has_manifold, has_tensor, get_manifold, get_tensor
export register_manifold!, register_tensor!, register_rule!, get_rules
export current_registry, with_registry

# Exports: AST primitives
export indices, free_indices, dummy_pairs
export rename_dummy, fresh_index, ensure_no_dummy_clash
export walk, substitute, children

# Exports: Escape hatch
export to_expr, from_expr, is_well_formed

# Exports: Rules
export RewriteRule, apply_rules, apply_rules_fixpoint, @rule

# Exports: Symmetries
export Symmetric, AntiSymmetric, PairSymmetric, RiemannSymmetry
export symmetry_generators

# Exports: Algebra
export tproduct, tsum, contract_metrics, canonicalize, expand_derivatives
export expand_products, collect_terms, ibp, ibp_product
export simplify
export collect_tensors
export make_ansatz
export abstract_trace, make_traceless

# Exports: Scalar algebra
export scalar_expand, scalar_collect, scalar_subst, scalar_cancel

# Exports: GR
export define_curvature_tensors!, einstein_expr, ricci_from_riemann
export bianchi_rules
export define_killing!
export CovDProperties, define_covd!, get_covd
export covd_to_christoffel, change_covd
export commute_covds
export lie_derivative

# Exports: Perturbation
export linearize, δRiemann, δRicci, δRicciScalar
export sorted_partitions, all_compositions, multinomial
export MetricPerturbation, define_metric_perturbation!, perturb, δinverse_metric
export gauge_transformation
export variational_derivative, euler_lagrange

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

# Exports: Exterior calculus
export define_form!, form_degree
export wedge, exterior_d, interior_product, hodge_dual

# Exports: Components
export ChartProperties, define_chart!, get_chart
export CTensor, ctensor_contract, ctensor_trace
export metric_christoffel, metric_riemann, metric_ricci, metric_ricci_scalar

# Exports: Macros
export @tensor
export @manifold, @define_tensor, @covd

# Extension stubs (implemented in ext/)
function to_symbolics end
function from_symbolics end
function to_symengine end
function from_symengine end

export to_symbolics, from_symbolics, to_symengine, from_symengine

end # module TensorGR
