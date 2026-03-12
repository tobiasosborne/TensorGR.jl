module TensorGRSymbolicsExt

using TensorGR
using Symbolics

# ─── to_symbolics / from_symbolics ───────────────────────────────

"""
    to_symbolics(s::TScalar) -> Symbolics.Num

Convert a TScalar to a Symbolics.jl symbolic number.
"""
function TensorGR.to_symbolics(s::TScalar)
    val = s.val
    if val isa Number
        return Symbolics.Num(val)
    elseif val isa Symbol
        return first(Symbolics.@variables $val)
    elseif val isa Expr
        return _expr_to_symbolics(val)
    end
    error("Cannot convert TScalar($val) to Symbolics")
end

function _expr_to_symbolics(ex::Expr)
    if ex.head == :call
        op = ex.args[1]
        # Normalize: Symbolics.toexpr can produce (*)(a,b) where op is the function itself
        op_sym = op isa Symbol ? op : op isa Function ? Symbol(op) : nameof(op)
        args = [_expr_to_symbolics(a) for a in ex.args[2:end]]
        if op_sym == :+ || op == +
            return sum(args)
        elseif op_sym == :- || op == -
            return length(args) == 1 ? -args[1] : args[1] - args[2]
        elseif op_sym == :* || op == *
            return prod(args)
        elseif op_sym == :/ || op == /
            return args[1] / args[2]
        elseif op_sym == :^ || op == ^
            return args[1] ^ args[2]
        elseif op_sym == :// || op == //
            return Rational{Int}(args[1], args[2])
        end
        # Try calling the function directly on Symbolics args
        if op isa Function
            return op(args...)
        end
    elseif ex.head == :(//)
        return _expr_to_symbolics(ex.args[1]) // _expr_to_symbolics(ex.args[2])
    end
    error("Cannot convert expression: $ex")
end

_expr_to_symbolics(x::Number) = Symbolics.Num(x)
function _expr_to_symbolics(x::Symbol)
    first(Symbolics.@variables $x)
end

"""
    from_symbolics(num::Symbolics.Num) -> TScalar

Convert a Symbolics.jl number to a TScalar.
"""
function TensorGR.from_symbolics(num)
    TScalar(Symbolics.toexpr(num))
end

# ─── CAS-1: simplify hooks ───────────────────────────────────────

function TensorGR._simplify_scalar_val(ex::Expr)
    try
        sym = _expr_to_symbolics(ex)
        simplified = Symbolics.simplify(sym)
        return Symbolics.toexpr(simplified)
    catch
        return ex
    end
end

function TensorGR._simplify_scalar_val(num::Symbolics.Num)
    Symbolics.simplify(num)
end

function TensorGR._try_simplify_entry(ex::Expr)
    try
        sym = _expr_to_symbolics(ex)
        simplified = Symbolics.simplify(sym)
        return Symbolics.toexpr(simplified)
    catch
        return ex
    end
end

# ─── CAS-2: Symbolics.Num dispatch for sym arithmetic ────────────

TensorGR._sym_mul(a::Symbolics.Num, b::Symbolics.Num) = a * b
TensorGR._sym_mul(a::Symbolics.Num, b::Number) = a * b
TensorGR._sym_mul(a::Number, b::Symbolics.Num) = a * b
TensorGR._sym_mul(a::Symbolics.Num, b) = a * _to_num(b)
TensorGR._sym_mul(a, b::Symbolics.Num) = _to_num(a) * b

TensorGR._sym_add(a::Symbolics.Num, b::Symbolics.Num) = a + b
TensorGR._sym_add(a::Symbolics.Num, b::Number) = a + b
TensorGR._sym_add(a::Number, b::Symbolics.Num) = a + b
TensorGR._sym_add(a::Symbolics.Num, b) = a + _to_num(b)
TensorGR._sym_add(a, b::Symbolics.Num) = _to_num(a) + b

TensorGR._sym_sub(a::Symbolics.Num, b::Symbolics.Num) = a - b
TensorGR._sym_sub(a::Symbolics.Num, b::Number) = a - b
TensorGR._sym_sub(a::Number, b::Symbolics.Num) = a - b
TensorGR._sym_sub(a::Symbolics.Num, b) = a - _to_num(b)
TensorGR._sym_sub(a, b::Symbolics.Num) = _to_num(a) - b

TensorGR._sym_neg(a::Symbolics.Num) = -a

TensorGR._sym_div(a::Symbolics.Num, b::Symbolics.Num) = a / b
TensorGR._sym_div(a::Symbolics.Num, b::Number) = a / b
TensorGR._sym_div(a::Number, b::Symbolics.Num) = a / b
TensorGR._sym_div(a::Symbolics.Num, b) = a / _to_num(b)
TensorGR._sym_div(a, b::Symbolics.Num) = _to_num(a) / b

_to_num(x::Symbolics.Num) = x
_to_num(x::Number) = Symbolics.Num(x)
function _to_num(x::Expr)
    _expr_to_symbolics(x)
end
_to_num(x::Symbol) = first(Symbolics.@variables $x)

"""
    sym_eval(expr::Symbolics.Num, vars::Dict) -> Number

Evaluate a Symbolics expression by substituting variable values.
"""
function TensorGR.sym_eval(expr::Symbolics.Num, vars::Dict)
    sym_vars = Dict{Symbolics.Num, Any}()
    for (k, v) in vars
        sym_k = first(Symbolics.@variables $k)
        sym_vars[sym_k] = v
    end
    result = Symbolics.substitute(expr, sym_vars)
    # Extract the underlying numeric value
    Float64(Symbolics.value(result))
end

# ─── CAS-2: symbolic_quadratic_form ──────────────────────────────

function TensorGR.symbolic_quadratic_form(entries::AbstractDict, fields::Vector{Symbol};
                                           variables::Vector{Symbol}=Symbol[])
    # Create Symbolics variables
    sym_vars = Dict{Symbol, Symbolics.Num}()
    for v in variables
        sym_vars[v] = first(Symbolics.@variables $v)
    end

    n = length(fields)
    fidx = Dict(f => i for (i, f) in enumerate(fields))
    M = Matrix{Any}(undef, n, n)
    fill!(M, Symbolics.Num(0))

    for ((f1, f2), val) in entries
        i, j = fidx[f1], fidx[f2]
        sym_val = val isa Symbolics.Num ? val : _to_num(val)
        M[i, j] = sym_val
        M[j, i] = sym_val
    end
    QuadraticForm(fields, M)
end

# ─── CAS-3: Symbolic Fourier transform ───────────────────────────

function TensorGR.to_fourier_symbolic(expr::TensorExpr;
                                       omega::Symbolics.Num,
                                       k_vars::Vector{Symbolics.Num}=Symbolics.Num[])
    _fourier_symbolic(expr, omega, k_vars)
end

function _fourier_symbolic(t::Tensor, ::Symbolics.Num, ::Vector{Symbolics.Num})
    t
end

function _fourier_symbolic(s::TScalar, ::Symbolics.Num, ::Vector{Symbolics.Num})
    s
end

function _fourier_symbolic(s::TSum, omega::Symbolics.Num, k_vars::Vector{Symbolics.Num})
    TSum(TensorExpr[_fourier_symbolic(t, omega, k_vars) for t in s.terms])
end

function _fourier_symbolic(p::TProduct, omega::Symbolics.Num, k_vars::Vector{Symbolics.Num})
    TProduct(p.scalar, TensorExpr[_fourier_symbolic(f, omega, k_vars) for f in p.factors])
end

function _fourier_symbolic(d::TDeriv, omega::Symbolics.Num, k_vars::Vector{Symbolics.Num})
    inner = _fourier_symbolic(d.arg, omega, k_vars)

    # Check if temporal derivative (component 0)
    s = string(d.index.name)
    if startswith(s, "_") && length(s) > 1
        comp = tryparse(Int, s[2:end])
        if comp !== nothing && comp == 0
            # Temporal: ∂_0 → ω factor
            return TProduct(1 // 1, TensorExpr[TScalar(omega), inner])
        elseif comp !== nothing && comp >= 1 && comp <= length(k_vars)
            # Spatial: ∂_i → k_i factor
            return TProduct(1 // 1, TensorExpr[TScalar(k_vars[comp]), inner])
        end
    end

    # If not a component index, use standard momentum tensor
    k = Tensor(:k, [d.index])
    TProduct(1 // 1, TensorExpr[k, inner])
end

# ─── Symbolic Components: metric → curvature pipeline ─────────────

function TensorGR.sym_deriv(expr, coord)
    D = Symbolics.Differential(coord)
    Symbolics.expand_derivatives(D(expr))
end

function TensorGR.symbolic_diagonal_metric(coords::Vector, diag::Vector)
    dim = length(coords)
    @assert length(diag) == dim "diagonal entries must match coordinate count"
    coord_names = [Symbol(c) for c in coords]
    g = Matrix{Any}(undef, dim, dim)
    ginv = Matrix{Any}(undef, dim, dim)
    for i in 1:dim, j in 1:dim
        if i == j
            g[i, i] = diag[i]
            ginv[i, i] = Symbolics.simplify(1 / diag[i])
        else
            g[i, j] = 0
            ginv[i, j] = 0
        end
    end
    SymbolicMetric(coords, coord_names, g, ginv, dim)
end

function TensorGR.symbolic_metric(coords::Vector, g::Matrix)
    dim = length(coords)
    @assert size(g) == (dim, dim) "metric matrix must be dim×dim"
    coord_names = [Symbol(c) for c in coords]
    ginv = Matrix{Any}(undef, dim, dim)
    ginv_raw = Symbolics.simplify.(inv(Symbolics.Num.(g)))
    for i in 1:dim, j in 1:dim
        ginv[i, j] = ginv_raw[i, j]
    end
    g_any = Matrix{Any}(undef, dim, dim)
    for i in 1:dim, j in 1:dim
        g_any[i, j] = g[i, j]
    end
    SymbolicMetric(coords, coord_names, g_any, ginv, dim)
end

function TensorGR.symbolic_christoffel(sm::SymbolicMetric)
    dim = sm.dim
    # Pre-compute metric derivatives: dg[i,j,k] = ∂_k g_{ij}
    dg = Array{Any}(undef, dim, dim, dim)
    for i in 1:dim, j in 1:dim, k in 1:dim
        dg[i, j, k] = Symbolics.simplify(TensorGR.sym_deriv(sm.g[i, j], sm.coords[k]))
    end
    # Γ^a_{bc} = (1/2) g^{ad} (∂_b g_{cd} + ∂_c g_{bd} - ∂_d g_{bc})
    Gamma = Array{Any}(undef, dim, dim, dim)
    for a in 1:dim, b in 1:dim, c in 1:dim
        s = 0
        for d in 1:dim
            s += sm.ginv[a, d] * (dg[c, d, b] + dg[b, d, c] - dg[b, c, d])
        end
        Gamma[a, b, c] = Symbolics.simplify(s / 2)
    end
    Gamma
end

function TensorGR.symbolic_riemann(sm::SymbolicMetric, Gamma)
    dim = sm.dim
    # Pre-compute Christoffel derivatives: dGamma[a,b,c,k] = ∂_k Γ^a_{bc}
    dGamma = Array{Any}(undef, dim, dim, dim, dim)
    for a in 1:dim, b in 1:dim, c in 1:dim, k in 1:dim
        dGamma[a, b, c, k] = TensorGR.sym_deriv(Gamma[a, b, c], sm.coords[k])
    end
    # R^a_{bcd} = ∂_c Γ^a_{db} - ∂_d Γ^a_{cb} + Γ^a_{ce} Γ^e_{db} - Γ^a_{de} Γ^e_{cb}
    Riem = Array{Any}(undef, dim, dim, dim, dim)
    for a in 1:dim, b in 1:dim, c in 1:dim, d in 1:dim
        t1 = dGamma[a, d, b, c]
        t2 = dGamma[a, c, b, d]
        t3 = sum(Gamma[a, c, e] * Gamma[e, d, b] for e in 1:dim)
        t4 = sum(Gamma[a, d, e] * Gamma[e, c, b] for e in 1:dim)
        Riem[a, b, c, d] = Symbolics.simplify(t1 - t2 + t3 - t4)
    end
    Riem
end

function TensorGR.symbolic_ricci(Riem, dim::Int)
    Ric = Matrix{Any}(undef, dim, dim)
    for b in 1:dim, d in 1:dim
        Ric[b, d] = Symbolics.simplify(sum(Riem[a, b, a, d] for a in 1:dim))
    end
    Ric
end

function TensorGR.symbolic_ricci_scalar(Ric, ginv, dim::Int)
    Symbolics.simplify(sum(ginv[a, b] * Ric[a, b] for a in 1:dim, b in 1:dim))
end

function TensorGR.symbolic_einstein(Ric, R, g, dim::Int)
    G = Matrix{Any}(undef, dim, dim)
    for a in 1:dim, b in 1:dim
        G[a, b] = Symbolics.simplify(Ric[a, b] - g[a, b] * R / 2)
    end
    G
end

function TensorGR.symbolic_kretschmann(Riem, g, ginv, dim::Int)
    # K = R_{abcd} R^{abcd}
    # First lower first index: R_{abcd} = g_{ae} R^e_{bcd}
    Riem_down = Array{Any}(undef, dim, dim, dim, dim)
    for a in 1:dim, b in 1:dim, c in 1:dim, d in 1:dim
        Riem_down[a, b, c, d] = Symbolics.simplify(
            sum(g[a, e] * Riem[e, b, c, d] for e in 1:dim))
    end
    # Raise all indices via ginv and contract with R_{abcd}
    K = 0
    for a in 1:dim, b in 1:dim, c in 1:dim, d in 1:dim
        R_up = 0
        for e in 1:dim, f in 1:dim, g_idx in 1:dim, h in 1:dim
            R_up += ginv[a, e] * ginv[b, f] * ginv[c, g_idx] * ginv[d, h] *
                    Riem_down[e, f, g_idx, h]
        end
        K += R_up * Riem_down[a, b, c, d]
    end
    Symbolics.simplify(K)
end

function TensorGR.symbolic_curvature_from_metric(sm::SymbolicMetric)
    Gamma = TensorGR.symbolic_christoffel(sm)
    Riem = TensorGR.symbolic_riemann(sm, Gamma)
    Ric = TensorGR.symbolic_ricci(Riem, sm.dim)
    R = TensorGR.symbolic_ricci_scalar(Ric, sm.ginv, sm.dim)
    G = TensorGR.symbolic_einstein(Ric, R, sm.g, sm.dim)
    K = TensorGR.symbolic_kretschmann(Riem, sm.g, sm.ginv, sm.dim)
    (; Gamma=Gamma, Riem=Riem, Ric=Ric, R=R, G=G, K=K)
end

# ─── Metric ansatz generators ────────────────────────────────────

"""
    metric_ansatz(reg, manifold, ::HomogeneousIsotropy; coords, k) -> NamedTuple

Generate the FLRW metric:
  ds^2 = -dtau^2 + a(tau)^2 [dchi^2/(1-k*chi^2) + chi^2 dOmega^2]

Returns `(metric, free_functions, time_coord)`.
"""
function TensorGR.metric_ansatz(reg::TensorGR.TensorRegistry, manifold::Symbol,
                                 ans::TensorGR.HomogeneousIsotropy;
                                 coords::Vector{Symbol}=[:tau, :chi, :theta, :phi],
                                 k::Int=0)
    length(coords) == 4 || error("FLRW ansatz requires exactly 4 coordinates")
    k in (-1, 0, 1) || error("Spatial curvature k must be -1, 0, or +1, got $k")

    # Create symbolic coordinate variables
    _c1 = coords[1]; _c2 = coords[2]; _c3 = coords[3]; _c4 = coords[4]
    tau_sym = first(Symbolics.@variables $_c1)
    chi_sym = first(Symbolics.@variables $_c2)
    theta_sym = first(Symbolics.@variables $_c3)
    phi_sym = first(Symbolics.@variables $_c4)

    # Scale factor a(tau) as a function of the time coordinate
    a_func = first(Symbolics.@variables a($(tau_sym)))

    # Build diagonal metric entries
    g_tt = Symbolics.Num(-1)
    g_chichi = a_func^2 / (1 - k * chi_sym^2)
    g_thth = a_func^2 * chi_sym^2
    g_phiphi = a_func^2 * chi_sym^2 * sin(theta_sym)^2

    sm = TensorGR.symbolic_diagonal_metric(
        [tau_sym, chi_sym, theta_sym, phi_sym],
        [g_tt, g_chichi, g_thth, g_phiphi]
    )

    return (metric=sm, free_functions=[a_func], time_coord=tau_sym)
end

"""
    metric_ansatz(reg, manifold, ::SphericalSymmetry; coords) -> NamedTuple

Generate a static spherically symmetric metric:
  ds^2 = -A(r) dt^2 + B(r) dr^2 + r^2 dOmega^2

Returns `(metric, free_functions, radial_coord)`.
"""
function TensorGR.metric_ansatz(reg::TensorGR.TensorRegistry, manifold::Symbol,
                                 ans::TensorGR.SphericalSymmetry;
                                 coords::Vector{Symbol}=[:t, :r, :theta, :phi])
    length(coords) == 4 || error("Spherical ansatz requires exactly 4 coordinates")

    # Create symbolic coordinate variables
    _c1 = coords[1]; _c2 = coords[2]; _c3 = coords[3]; _c4 = coords[4]
    t_sym = first(Symbolics.@variables $_c1)
    r_sym = first(Symbolics.@variables $_c2)
    theta_sym = first(Symbolics.@variables $_c3)
    phi_sym = first(Symbolics.@variables $_c4)

    # Free functions A(r) and B(r)
    A_func = first(Symbolics.@variables A($(r_sym)))
    B_func = first(Symbolics.@variables B($(r_sym)))

    # Build diagonal metric entries
    g_tt = -A_func
    g_rr = B_func
    g_thth = r_sym^2
    g_phiphi = r_sym^2 * sin(theta_sym)^2

    sm = TensorGR.symbolic_diagonal_metric(
        [t_sym, r_sym, theta_sym, phi_sym],
        [g_tt, g_rr, g_thth, g_phiphi]
    )

    return (metric=sm, free_functions=[A_func, B_func], radial_coord=r_sym)
end

end # module
