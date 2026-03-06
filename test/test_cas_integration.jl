using TensorGR
using Symbolics

# Helper to evaluate Symbolics expressions at numeric points
function _sym_num_eval(expr::Symbolics.Num, vars::Dict)
    sym_vars = Dict{Symbolics.Num, Any}()
    for (k, v) in vars
        sym_k = first(Symbolics.@variables $k)
        sym_vars[sym_k] = v
    end
    Float64(Symbolics.value(Symbolics.substitute(expr, sym_vars)))
end

# ═══════════════════════════════════════════════════════════════════
# CAS-1: simplify_scalar hook
# ═══════════════════════════════════════════════════════════════════

@testset "simplify_scalar: number unchanged" begin
    s = TScalar(3 // 4)
    @test simplify_scalar(s) == s
end

@testset "simplify_scalar: Expr simplification" begin
    # x + x should simplify to 2x
    ex = TScalar(:(x + x))
    result = simplify_scalar(ex)
    # The result val might be Expr or Number, verify by converting to Symbolics
    result_sym = to_symbolics(result)
    @test _sym_num_eval(result_sym, Dict(:x => 3.0)) ≈ 6.0
end

@testset "simplify_quadratic_form: simplify entries" begin
    # Build a QF with redundant Expr entries
    entries = Dict((:a, :a) => :(k^2 + k^2), (:a, :b) => 0, (:b, :b) => :(p^2))
    qf = quadratic_form(entries, [:a, :b])

    sqf = simplify_quadratic_form(qf)
    # M[1,1] should be simplified form of 2k^2
    # Convert to Symbolics and evaluate
    val = sqf.matrix[1, 1]
    if val isa Expr
        sym = to_symbolics(TScalar(val))
        @test _sym_num_eval(sym, Dict(:k => 2.0)) ≈ 8.0
    else
        @test val == 8  # might simplify to a number at k=2
    end
end

# ═══════════════════════════════════════════════════════════════════
# CAS-2: Symbolics.Num QuadraticForm backend
# ═══════════════════════════════════════════════════════════════════

@testset "symbolic_quadratic_form: construction" begin
    @variables k p β

    entries = Dict(
        (:Φ, :Φ) => (2 - 4β) * k^4,
        (:Φ, :ψ) => -2k^2 * p^2,
        (:ψ, :ψ) => 3p^4
    )

    qf = symbolic_quadratic_form(entries, [:Φ, :ψ]; variables=[:k, :p, :β])
    @test qf isa QuadraticForm
    @test length(qf.fields) == 2
    @test isequal(qf.matrix[1, 2], qf.matrix[2, 1])  # symmetric
end

@testset "sym_det with Symbolics.Num" begin
    @variables a b c d

    M = Matrix{Any}(undef, 2, 2)
    M[1, 1] = a
    M[1, 2] = b
    M[2, 1] = c
    M[2, 2] = d

    det = sym_det(M)
    # Should be a*d - b*c
    @test det isa Symbolics.Num
    det_val = _sym_num_eval(det, Dict(:a => 3.0, :b => 1.0, :c => 2.0, :d => 4.0))
    @test det_val ≈ 10.0  # 3*4 - 1*2
end

@testset "sym_inv with Symbolics.Num" begin
    @variables a b d

    M = Matrix{Any}(undef, 2, 2)
    M[1, 1] = a
    M[1, 2] = b
    M[2, 1] = b  # symmetric
    M[2, 2] = d

    inv_M = sym_inv(M)
    @test inv_M isa Matrix

    # Check (M * M^{-1}) = I at a specific point
    vals = Dict(:a => 3.0, :b => 1.0, :d => 4.0)
    m11 = _sym_num_eval(M[1, 1], vals)
    m12 = _sym_num_eval(M[1, 2], vals)
    m22 = _sym_num_eval(M[2, 2], vals)
    i11 = _sym_num_eval(inv_M[1, 1], vals)
    i12 = _sym_num_eval(inv_M[1, 2], vals)
    i22 = _sym_num_eval(inv_M[2, 2], vals)

    # M * M^{-1} should be identity
    @test m11 * i11 + m12 * i12 ≈ 1.0 atol = 1e-10
    @test m11 * i12 + m12 * i22 ≈ 0.0 atol = 1e-10
end

@testset "sym_eval with Symbolics.Num" begin
    @variables x y
    expr = x^2 + 2x*y + y^2
    val = sym_eval(expr, Dict(:x => 3.0, :y => 2.0))
    @test val ≈ 25.0  # (3+2)^2
end

@testset "propagator with Symbolics.Num" begin
    @variables k p β

    entries = Dict(
        (:Φ, :Φ) => (2 - 4β) * k^4,
        (:Φ, :ψ) => -2k^2 * p^2,
        (:ψ, :ψ) => 3p^4
    )

    qf = symbolic_quadratic_form(entries, [:Φ, :ψ]; variables=[:k, :p, :β])
    det_expr = determinant(qf)
    @test det_expr isa Symbolics.Num

    prop = propagator(qf)
    @test prop isa QuadraticForm

    # Numerical check: at β=0, k=1, p=1
    # M = [2  -2; -2  3], det = 6-4 = 2
    vals = Dict(:β => 0.0, :k => 1.0, :p => 1.0)
    det_num = sym_eval(det_expr, vals)
    @test det_num ≈ 2.0

    # Propagator G_{ΦΦ} = M_{ψψ}/det = 3/2
    g11 = sym_eval(prop.matrix[1, 1], vals)
    @test g11 ≈ 1.5
end

# ═══════════════════════════════════════════════════════════════════
# CAS-3: Symbolic Fourier transform
# ═══════════════════════════════════════════════════════════════════

@testset "to_fourier_symbolic: temporal derivative" begin
    @variables ω k1 k2 k3

    T = Tensor(:T, TIndex[])
    d = TDeriv(TIndex(Symbol("_", 0), Down), T)

    result = to_fourier_symbolic(d; omega=ω, k_vars=Symbolics.Num[k1, k2, k3])
    @test result isa TProduct
    @test any(f -> f isa TScalar, result.factors)
end

@testset "to_fourier_symbolic: spatial derivative" begin
    @variables ω k1 k2 k3

    T = Tensor(:T, TIndex[])
    d = TDeriv(TIndex(Symbol("_", 1), Down), T)

    result = to_fourier_symbolic(d; omega=ω, k_vars=Symbolics.Num[k1, k2, k3])
    @test result isa TProduct
    scalar_factors = filter(f -> f isa TScalar, result.factors)
    @test !isempty(scalar_factors)
end

@testset "to_fourier_symbolic: abstract index falls back to momentum tensor" begin
    @variables ω k1

    T = Tensor(:T, TIndex[])
    d = TDeriv(down(:a), T)

    result = to_fourier_symbolic(d; omega=ω, k_vars=Symbolics.Num[k1])
    @test result isa TProduct
    k_factors = filter(f -> f isa Tensor && f.name == :k, result.factors)
    @test length(k_factors) == 1
end

@testset "to_fourier_symbolic: double derivative" begin
    @variables ω k1 k2 k3

    T = Tensor(:T, TIndex[])
    d1 = TDeriv(TIndex(Symbol("_", 1), Down), T)
    d2 = TDeriv(TIndex(Symbol("_", 2), Down), d1)

    result = to_fourier_symbolic(d2; omega=ω, k_vars=Symbolics.Num[k1, k2, k3])
    @test result isa TProduct
end

# ═══════════════════════════════════════════════════════════════════
# CAS-4: Postquantum gravity reference values
# ═══════════════════════════════════════════════════════════════════

@testset "Postquantum gravity: det(M) reference check" begin
    @variables k ω β

    M_PP = (2 - 4β) * k^4
    M_Pp = (-2ω^2 + 4β * (3ω^2 - 4k^2)) * k^2
    M_pp = 2k^4 - 4k^2 * ω^2 + 12ω^4 - 4β * (3ω^2 - 4k^2)^2

    entries = Dict((:Φ, :Φ) => M_PP, (:Φ, :ψ) => M_Pp, (:ψ, :ψ) => M_pp)
    qf = symbolic_quadratic_form(entries, [:Φ, :ψ]; variables=[:k, :ω, :β])

    det_expr = determinant(qf)

    # Numerical check at β=1, ω=2, k=1
    vals = Dict(:β => 1.0, :ω => 2.0, :k => 1.0)
    det_num = sym_eval(det_expr, vals)

    # M_PP = (2-4)*1 = -2, M_Pp = (-8+4*(12-4))*1 = 24
    # M_pp = 2-16+192-4*(12-4)^2 = 2-16+192-256 = -78
    # det = (-2)*(-78) - 24^2 = 156 - 576 = -420
    @test det_num ≈ -420.0

    # At conformal point β=1/3
    vals_conf = Dict(:β => 1.0/3.0, :ω => 2.0, :k => 1.0)
    det_conf = sym_eval(det_expr, vals_conf)
    @test det_conf ≈ 492.0 / 9.0 atol = 1e-10
end
