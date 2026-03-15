#= Algebra-valued differential forms.

A Lie-algebra-valued p-form is omega = omega^I_{a1...ap} T_I dx^{a1} ^ ... ^ dx^{ap}
where T_I are generators of a Lie algebra (or sections of a vector bundle).

This module wraps a TensorExpr carrying both spacetime form indices and an
internal (algebra/bundle) index, tracking the form degree and algebra name.
=#

"""
    AlgValuedForm <: TensorExpr

A differential form valued in a Lie algebra (or more generally, a vector bundle).
Wraps a tensor expression that carries both spacetime form indices and an internal
(algebra/bundle) index.

# Fields
- `degree::Int`       -- form degree (0, 1, 2, ...)
- `algebra::Symbol`   -- name of the Lie algebra / vector bundle (e.g., :su2, :Adj)
- `expr::TensorExpr`  -- the underlying tensor expression with indices
"""
struct AlgValuedForm <: TensorExpr
    degree::Int
    algebra::Symbol
    expr::TensorExpr
    function AlgValuedForm(degree::Int, algebra::Symbol, expr::TensorExpr)
        degree >= 0 || throw(ArgumentError("Form degree must be non-negative, got $degree"))
        new(degree, algebra, expr)
    end
end

Base.:(==)(a::AlgValuedForm, b::AlgValuedForm) =
    a.degree == b.degree && a.algebra == b.algebra && a.expr == b.expr
Base.hash(a::AlgValuedForm, h::UInt) =
    hash(a.expr, hash(a.algebra, hash(a.degree, hash(:AlgValuedForm, h))))

# ── AST integration ─────────────────────────────────────────────────

indices(a::AlgValuedForm) = indices(a.expr)
free_indices(a::AlgValuedForm) = free_indices(a.expr)
children(a::AlgValuedForm) = TensorExpr[a.expr]

function walk(f, a::AlgValuedForm)
    new_expr = walk(f, a.expr)
    f(AlgValuedForm(a.degree, a.algebra, new_expr))
end

derivative_order(a::AlgValuedForm) = derivative_order(a.expr)
is_constant(a::AlgValuedForm) = is_constant(a.expr)

# ── Display ──────────────────────────────────────────────────────────

function Base.show(io::IO, a::AlgValuedForm)
    print(io, "AlgForm(", a.algebra, ", deg=", a.degree, ", ")
    show(io, a.expr)
    print(io, ")")
end

function to_latex(a::AlgValuedForm)
    "\\mathfrak{" * string(a.algebra) * "}" * "\\text{-form}_{" * string(a.degree) * "}\\left(" * to_latex(a.expr) * "\\right)"
end

function to_unicode(a::AlgValuedForm)
    string(a.algebra) * "-form[" * string(a.degree) * "](" * to_unicode(a.expr) * ")"
end

# ── Operations ───────────────────────────────────────────────────────

"""
    alg_exterior_d(omega::AlgValuedForm, deriv_idx::TIndex) -> AlgValuedForm

Exterior derivative acting on form indices only (not the algebra index).
Returns a form of degree+1.
"""
function alg_exterior_d(omega::AlgValuedForm, deriv_idx::TIndex)
    AlgValuedForm(omega.degree + 1, omega.algebra, TDeriv(deriv_idx, omega.expr))
end

"""
    alg_wedge(omega::AlgValuedForm, eta::AlgValuedForm,
              structure_constants::Symbol; registry=current_registry()) -> AlgValuedForm

Wedge product with Lie bracket: [omega ^ eta]^I = f^I_{JK} omega^J ^ eta^K
where f^I_{JK} are the structure constants tensor.

Both forms must be valued in the same algebra.
"""
function alg_wedge(omega::AlgValuedForm, eta::AlgValuedForm,
                   structure_constants::Symbol; registry=current_registry())
    omega.algebra == eta.algebra ||
        throw(ArgumentError("Cannot wedge forms valued in different algebras: $(omega.algebra) vs $(eta.algebra)"))

    # Collect used index names to generate fresh dummies
    used = Set{Symbol}()
    for idx in indices(omega.expr)
        push!(used, idx.name)
    end
    for idx in indices(eta.expr)
        push!(used, idx.name)
    end

    # Fresh dummy indices J, K for algebra contraction: f^I_{JK}
    j_name = fresh_index(used)
    push!(used, j_name)
    k_name = fresh_index(used)
    push!(used, k_name)

    # Find the algebra index on each form (the Up index in the algebra's vbundle)
    omega_alg_idx = _find_algebra_index(omega)
    eta_alg_idx = _find_algebra_index(eta)

    vb = omega_alg_idx.vbundle

    # Build f^I_{JK} with I inheriting the free algebra position
    f = Tensor(structure_constants, [up(omega_alg_idx.name, vb), down(j_name, vb), down(k_name, vb)])

    # Rename algebra indices on omega/eta to J, K for contraction
    omega_renamed = rename_dummy(omega.expr, omega_alg_idx.name, j_name)
    eta_renamed = rename_dummy(eta.expr, eta_alg_idx.name, k_name)

    # Ensure no dummy clashes between renamed expressions
    eta_renamed = ensure_no_dummy_clash(omega_renamed, eta_renamed)

    # Wedge coefficient for form indices
    p, q = omega.degree, eta.degree
    coeff = factorial(p + q) // (factorial(p) * factorial(q))

    inner = tproduct(coeff, TensorExpr[f, omega_renamed, eta_renamed])
    AlgValuedForm(p + q, omega.algebra, inner)
end

"""
    connection_1form(name::Symbol, algebra::Symbol, alg_idx::TIndex,
                     form_idx::TIndex; registry=current_registry()) -> AlgValuedForm

Construct a connection 1-form A = A^I_a T_I dx^a.

# Arguments
- `name`: tensor name for the connection (e.g. `:A`)
- `algebra`: Lie algebra name (e.g. `:su2`)
- `alg_idx`: the algebra (Up) index
- `form_idx`: the spacetime 1-form (Down) index
"""
function connection_1form(name::Symbol, algebra::Symbol, alg_idx::TIndex,
                          form_idx::TIndex; registry=current_registry())
    AlgValuedForm(1, algebra, Tensor(name, [alg_idx, form_idx]))
end

"""
    curvature_2form(A::AlgValuedForm, structure_constants::Symbol;
                    registry=current_registry()) -> AlgValuedForm

Compute the curvature 2-form: F = dA + (1/2)[A ^ A].
This is the standard gauge field strength.

The result is a degree-2 algebra-valued form.
"""
function curvature_2form(A::AlgValuedForm, structure_constants::Symbol;
                         registry=current_registry())
    A.degree == 1 ||
        throw(ArgumentError("curvature_2form requires a 1-form, got degree $(A.degree)"))

    # Collect used names for fresh index generation
    used = Set{Symbol}()
    for idx in indices(A.expr)
        push!(used, idx.name)
    end

    deriv_name = fresh_index(used)
    push!(used, deriv_name)

    # dA
    dA = alg_exterior_d(A, down(deriv_name))

    # Build a second copy of A with fresh indices for the wedge
    alg_idx = _find_algebra_index(A)
    vb = alg_idx.vbundle
    new_alg = fresh_index(used)
    push!(used, new_alg)
    new_form = fresh_index(used)
    push!(used, new_form)

    A2 = AlgValuedForm(1, A.algebra,
        Tensor(A.expr isa Tensor ? A.expr.name : :A,
               [up(new_alg, vb), down(new_form)]))

    # (1/2)[A ^ A]
    half_bracket = alg_wedge(A, A2, structure_constants; registry=registry)
    half_expr = tproduct(1 // 2, TensorExpr[half_bracket.expr])

    AlgValuedForm(2, A.algebra, dA.expr + half_expr)
end

# ── Gauge-covariant derivative ────────────────────────────────────────

"""
    gauge_covd(A::AlgValuedForm, omega::AlgValuedForm, structure_constants::Symbol;
               registry=current_registry()) -> AlgValuedForm

Gauge-covariant exterior derivative: D_A ω = dω + [A ∧ ω]
where [A ∧ ω]^I = f^I_{JK} A^J ∧ ω^K.

`A` must be a 1-form (connection). Result is a (p+1)-form where p = degree(ω).

For a 0-form (no spacetime form indices), the bracket term vanishes and
D_A ω = dω (the ordinary exterior derivative).

Ground truth: Nakahara (2003) Eq 11.5; Eguchi, Gilkey & Hanson (1980) Eq 2.22.
"""
function gauge_covd(A::AlgValuedForm, omega::AlgValuedForm,
                    structure_constants::Symbol;
                    registry=current_registry())
    A.degree == 1 ||
        throw(ArgumentError("gauge_covd requires A to be a 1-form, got degree $(A.degree)"))
    A.algebra == omega.algebra ||
        throw(ArgumentError("A and ω must be in the same algebra: $(A.algebra) vs $(omega.algebra)"))

    # Collect used index names for fresh dummy generation
    used = Set{Symbol}()
    for idx in indices(A.expr)
        push!(used, idx.name)
    end
    for idx in indices(omega.expr)
        push!(used, idx.name)
    end

    deriv_name = fresh_index(used)
    push!(used, deriv_name)

    # dω: exterior derivative on form indices
    d_omega = alg_exterior_d(omega, down(deriv_name))

    # For 0-forms the bracket term vanishes: D_A φ = dφ (Nakahara Eq 11.3)
    if omega.degree == 0
        return d_omega
    end

    # [A ∧ ω]: algebra-valued wedge product with structure constants
    bracket = alg_wedge(A, omega, structure_constants; registry=registry)

    AlgValuedForm(omega.degree + 1, omega.algebra, d_omega.expr + bracket.expr)
end

"""
    bianchi_identity(A::AlgValuedForm, structure_constants::Symbol;
                     registry=current_registry()) -> AlgValuedForm

Compute the Bianchi identity expression: D_A F = dF + [A ∧ F].
For the curvature F = dA + ½[A∧A], this should simplify to zero.

Returns the degree-3 algebra-valued form D_A F.

Ground truth: Nakahara (2003) Eq 11.12; standard Yang-Mills Bianchi identity.
"""
function bianchi_identity(A::AlgValuedForm, structure_constants::Symbol;
                          registry=current_registry())
    A.degree == 1 ||
        throw(ArgumentError("bianchi_identity requires A to be a 1-form, got degree $(A.degree)"))

    F = curvature_2form(A, structure_constants; registry=registry)
    gauge_covd(A, F, structure_constants; registry=registry)
end

# ── Physics aliases and Yang-Mills structures ─────────────────────────

"""
    field_strength(A::AlgValuedForm, structure_constants::Symbol;
                   registry=current_registry()) -> AlgValuedForm

Alias for `curvature_2form`. Computes the gauge field strength:

    F = dA + (1/2)[A ∧ A]

Standard physics nomenclature for the curvature 2-form of a gauge connection.

Ground truth: Nakahara (2003) Eq 11.7; Peskin & Schroeder (1995) Eq 15.47.
"""
field_strength(A::AlgValuedForm, structure_constants::Symbol;
               registry=current_registry()) =
    curvature_2form(A, structure_constants; registry=registry)

"""
    yang_mills_eom(A::AlgValuedForm, structure_constants::Symbol,
                   epsilon::Symbol, metric::Symbol, dim::Int;
                   registry=current_registry()) -> AlgValuedForm

Compute the Yang-Mills equation of motion expression: D_A(★F).

For a connection 1-form A on a d-dimensional manifold, computes the
gauge-covariant exterior derivative of the Hodge dual of the field strength:

    D_A(★F) = d(★F) + [A ∧ ★F]

For a solution, D_A(★F) = ★J where J is the current.
In 4D, ★F is a 2-form, so D_A(★F) is a 3-form (dual to a 1-form current).

Requires the Levi-Civita tensor (epsilon) and metric for the Hodge dual.

Ground truth: Nakahara (2003) Eq 11.28; Peskin & Schroeder (1995) Eq 15.47.
"""
function yang_mills_eom(A::AlgValuedForm, structure_constants::Symbol,
                        epsilon::Symbol, metric::Symbol, dim::Int;
                        registry=current_registry())
    A.degree == 1 ||
        throw(ArgumentError("yang_mills_eom requires A to be a 1-form, got degree $(A.degree)"))

    # Compute field strength F = dA + (1/2)[A^A]
    F = curvature_2form(A, structure_constants; registry=registry)

    # Hodge dual of F: ★F is a (dim-2)-form
    # We need to extract the underlying tensor from F to apply hodge_dual,
    # then wrap back in AlgValuedForm
    star_F_degree = dim - F.degree

    # Collect used index names for fresh index generation
    used = Set{Symbol}()
    for idx in indices(A.expr)
        push!(used, idx.name)
    end
    for idx in indices(F.expr)
        push!(used, idx.name)
    end

    # Build Hodge dual contraction manually for algebra-valued forms:
    # (★F)^I_{b1...b(d-2)} = (1/2!) eps^{a1 a2}_{b1...b(d-2)} F^I_{a1 a2}
    # We contract epsilon with the spacetime form indices of F

    # Create dummy indices for contraction with F's form indices
    up_indices = TIndex[]
    for _ in 1:F.degree
        d = fresh_index(used)
        push!(used, d)
        push!(up_indices, up(d))
    end

    # Create result indices for the dual form
    result_indices = TIndex[]
    for _ in 1:star_F_degree
        d = fresh_index(used)
        push!(used, d)
        push!(result_indices, down(d))
    end

    # Build epsilon tensor
    eps_tensor = Tensor(epsilon, vcat(up_indices, result_indices))

    # Build F with renamed spacetime indices for contraction
    # F.expr carries algebra + form indices; we need to rename form indices to match dummies
    alg_idx = _find_algebra_index(F)

    # Fresh algebra index for F copy
    new_alg = fresh_index(used)
    push!(used, new_alg)

    # Build F tensor with matching dummy indices
    F_renamed = Tensor(:F, vcat([TIndex(new_alg, Up, alg_idx.vbundle)],
                                [down(idx.name) for idx in up_indices]))

    coeff = 1 // factorial(F.degree)
    star_F_expr = tproduct(coeff, TensorExpr[eps_tensor, F_renamed])

    # Rename algebra index back to match A's algebra index
    star_F_expr = rename_dummy(star_F_expr, new_alg, alg_idx.name)

    star_F = AlgValuedForm(star_F_degree, A.algebra, star_F_expr)

    # D_A(★F) = d(★F) + [A ∧ ★F]
    gauge_covd(A, star_F, structure_constants; registry=registry)
end

"""
    instanton_density(F::AlgValuedForm; registry=current_registry()) -> AlgValuedForm

Compute the topological (instanton) density: Tr(F ∧ F).

For a 2-form field strength F^I_{ab}, this computes
    F^I_{ab} F^I_{cd}
which is a scalar-valued 4-form proportional to the second Chern class / Pontryagin density.

The trace over the algebra index contracts F^I with F^I (using the Killing form,
here taken as delta^{IJ}).

Ground truth: Nakahara (2003) Eq 11.76; Eguchi, Gilkey & Hanson (1980) Eq 6.2.
"""
function instanton_density(F::AlgValuedForm; registry=current_registry())
    F.degree == 2 ||
        throw(ArgumentError("instanton_density requires a 2-form (field strength), got degree $(F.degree)"))

    # Collect used index names
    used = Set{Symbol}()
    for idx in indices(F.expr)
        push!(used, idx.name)
    end

    # Create a copy of F with fresh indices for the wedge product
    alg_idx = _find_algebra_index(F)
    vb = alg_idx.vbundle

    # Fresh indices for the second copy of F
    new_alg = fresh_index(used)
    push!(used, new_alg)
    new_form1 = fresh_index(used)
    push!(used, new_form1)
    new_form2 = fresh_index(used)
    push!(used, new_form2)

    F2_tensor = Tensor(:F, [up(new_alg, vb), down(new_form1), down(new_form2)])
    F2 = AlgValuedForm(2, F.algebra, F2_tensor)

    # Wedge product F ∧ F (degree 2 + 2 = 4)
    # The wedge coefficient for two 2-forms: (4!)/(2!2!) = 6
    p, q = F.degree, F2.degree
    coeff = factorial(p + q) // (factorial(p) * factorial(q))

    # Trace over algebra index: contract F^I with F^I
    # Rename the algebra index of F2 to match F for contraction
    F2_traced = rename_dummy(F2.expr, new_alg, alg_idx.name)

    # Ensure no dummy clashes between F and F2 (spacetime indices)
    F2_traced = ensure_no_dummy_clash(F.expr, F2_traced)

    inner = tproduct(coeff, TensorExpr[F.expr, F2_traced])

    # Result is a 4-form. The algebra indices are fully contracted (trace),
    # so this is effectively a scalar-valued form, but we keep it as
    # AlgValuedForm with degree 4 for type consistency.
    AlgValuedForm(p + q, F.algebra, inner)
end

# ── Helpers ──────────────────────────────────────────────────────────

"""Find the first Up-position index that lives in a non-Tangent vbundle,
   or the first Up index if all are Tangent."""
function _find_algebra_index(a::AlgValuedForm)
    all_idx = indices(a.expr)
    # Prefer non-Tangent Up index (the algebra index)
    for idx in all_idx
        if idx.position == Up && idx.vbundle != :Tangent
            return idx
        end
    end
    # Fallback: first Up index
    for idx in all_idx
        if idx.position == Up
            return idx
        end
    end
    error("AlgValuedForm has no Up (algebra) index")
end
