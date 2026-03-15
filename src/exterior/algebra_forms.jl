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
