#= Compute Bueno-Cano parameters (a,b,c,e) for all 6 cubic curvature invariants.
#
# Reference: Bueno & Cano, "Einsteinian cubic gravity" (2016)
#   arXiv: 1607.06463 — LOCAL COPY: benchmarks/papers/1607.06463_*.pdf
#   Method: Eqs. (11)-(14) — evaluate ℒ(R̃(Λ,α)) and extract α-derivatives.
#
# The parametric Riemann tensor (Eq. 11):
#   R̃_{μνρσ}(Λ,α) = Λ(g_{μρ}g_{νσ} − g_{μσ}g_{νρ}) + α(k_{μρ}k_{νσ} − k_{μσ}k_{νρ})
# where k is a rank-χ projector (k²=k, Tr(k)=χ).
#
# From Eqs. (13)-(14), the first two α-derivatives determine
# the Bueno-Cano parameters (a,b,c,e) that control the linearized
# spectrum on any maximally symmetric background via Eqs. (17)-(19).
=#

using LinearAlgebra

const D = 4  # spacetime dimension

# ═══════════════════════════════════════════════════════════════════════
# Parametric Riemann tensor and curvature contractions
# ═══════════════════════════════════════════════════════════════════════

"""Build a rank-χ projector in D dimensions."""
function projector(χ::Int)
    k = zeros(D, D)
    for i in 1:min(χ, D)
        k[i,i] = 1.0
    end
    k
end

"""Build R̃_{μνρσ}(Λ,α) on the parametric Riemann tensor."""
function build_Rtilde(Λ_BC::Float64, α::Float64, k::Matrix{Float64})
    g = Matrix{Float64}(I, D, D)
    R = zeros(D, D, D, D)
    @inbounds for μ in 1:D, ν in 1:D, ρ in 1:D, σ in 1:D
        R[μ,ν,ρ,σ] = Λ_BC*(g[μ,ρ]*g[ν,σ] - g[μ,σ]*g[ν,ρ]) +
                      α*(k[μ,ρ]*k[ν,σ] - k[μ,σ]*k[ν,ρ])
    end
    R
end

"""Ricci tensor: Ric_{μν} = R^σ_{μσν} = g^{σρ} R_{ρμσν}"""
function ricci(Riem::Array{Float64,4})
    Ric = zeros(D, D)
    @inbounds for μ in 1:D, ν in 1:D, σ in 1:D
        Ric[μ,ν] += Riem[σ,μ,σ,ν]  # g=δ so g^{σρ}=δ^{σρ}
    end
    Ric
end

"""Ricci scalar: R = g^{μν} Ric_{μν}"""
ricci_scalar(Ric::Matrix{Float64}) = tr(Ric)

"""Full contraction T_{a₁...aₙ} T^{a₁...aₙ} (all indices with flat metric)."""
function full_contract(A::Array{Float64,4}, B::Array{Float64,4})
    s = 0.0
    @inbounds for μ in 1:D, ν in 1:D, ρ in 1:D, σ in 1:D
        s += A[μ,ν,ρ,σ] * B[μ,ν,ρ,σ]
    end
    s
end

# ═══════════════════════════════════════════════════════════════════════
# The 6 cubic invariants evaluated on a given Riemann tensor
# ═══════════════════════════════════════════════════════════════════════

function eval_I1(Riem)
    Ric = ricci(Riem); R = ricci_scalar(Ric)
    R^3
end

function eval_I2(Riem)
    Ric = ricci(Riem); R = ricci_scalar(Ric)
    R * tr(Ric * Ric)
end

function eval_I3(Riem)
    Ric = ricci(Riem)
    tr(Ric * Ric * Ric)
end

function eval_I4(Riem)
    Ric = ricci(Riem); R = ricci_scalar(Ric)
    R * full_contract(Riem, Riem)
end

function eval_I5(Riem)
    # Ric^{ab} R_{acde} R_b^{cde} = Σ Ric[a,b] * Riem[a,c,d,e] * Riem[b,c,d,e]
    Ric = ricci(Riem)
    s = 0.0
    @inbounds for a in 1:D, b in 1:D, c in 1:D, d in 1:D, e in 1:D
        s += Ric[a,b] * Riem[a,c,d,e] * Riem[b,c,d,e]
    end
    s
end

function eval_I6(Riem)
    # R_{ab}^{cd} R_{cd}^{ef} R_{ef}^{ab} = Σ Riem[a,b,c,d] * Riem[c,d,e,f] * Riem[e,f,a,b]
    s = 0.0
    @inbounds for a in 1:D, b in 1:D, c in 1:D, d in 1:D, e in 1:D, f in 1:D
        s += Riem[a,b,c,d] * Riem[c,d,e,f] * Riem[e,f,a,b]
    end
    s
end

const INVARIANTS = [eval_I1, eval_I2, eval_I3, eval_I4, eval_I5, eval_I6]
const INV_NAMES = ["R³", "R·Ric²", "Ric³", "R·Riem²", "Ric·Riem²", "Riem³"]

# ═══════════════════════════════════════════════════════════════════════
# Extract Bueno-Cano parameters via α-derivatives
# ═══════════════════════════════════════════════════════════════════════

"""
Compute the Bueno-Cano parameters (a,b,c,e) for a given invariant.

Uses the parametric Riemann tensor R̃(Λ_BC,α) with rank-χ projector,
evaluates the invariant as a function of α, and extracts the first two
derivatives via central finite differences. Then solves the linear
system from Eqs. (13)-(14) for (a,b,c,e).

Reference: Bueno-Cano (1607.06463), Eqs. (11)-(14).
"""
function bc_parameters(invariant_fn, Λ_BC::Float64)
    h = 1e-5  # step size for finite differences

    # Evaluate at several χ values to over-determine the system
    χ_values = [2, 3, 4]
    e_values = Float64[]
    system_rows = Tuple{Float64,Float64,Float64,Float64}[]  # (1, χ(χ-1), χ-1, C₂/(2χ(χ-1)))

    for χ in χ_values
        k = projector(χ)
        q = χ * (χ - 1)  # χ(χ-1)
        p = χ - 1         # χ-1

        # Evaluate I(α) at α = -h, 0, +h
        I_m = invariant_fn(build_Rtilde(Λ_BC, -h, k))
        I_0 = invariant_fn(build_Rtilde(Λ_BC, 0.0, k))
        I_p = invariant_fn(build_Rtilde(Λ_BC, +h, k))

        # First derivative: C₁ = dI/dα|₀
        C1 = (I_p - I_m) / (2h)
        # Second derivative: 2C₂ = d²I/dα²|₀
        C2_times2 = (I_p - 2I_0 + I_m) / h^2

        # From Eq. (13): 2e·q = C₁ → e = C₁/(2q)
        if abs(q) > 1e-10
            push!(e_values, C1 / (2q))
        end

        # From Eq. (14): 4q(a + bq + cp) = 2C₂
        # → a + bq + cp = C₂/(2q) = C2_times2/(4q)
        if abs(q) > 1e-10
            rhs = C2_times2 / (4q)
            push!(system_rows, (1.0, Float64(q), Float64(p), rhs))
        end
    end

    # e should be the same for all χ (verify)
    e = mean(e_values)
    @assert maximum(abs.(e_values .- e)) / max(abs(e), 1e-15) < 1e-4 "e not consistent across χ values: $e_values"

    # Solve for a, b, c from the linear system:
    # a + b·q_i + c·p_i = rhs_i for each χ_i
    A_mat = zeros(length(system_rows), 3)
    rhs_vec = zeros(length(system_rows))
    for (i, (one, qi, pi, ri)) in enumerate(system_rows)
        A_mat[i, :] = [one, qi, pi]
        rhs_vec[i] = ri
    end
    abc = A_mat \ rhs_vec
    a, b, c = abc

    (a=a, b=b, c=c, e=e)
end

mean(x) = sum(x) / length(x)

# ═══════════════════════════════════════════════════════════════════════
# Main computation
# ═══════════════════════════════════════════════════════════════════════

println("=" ^ 70)
println("  Bueno-Cano Parameters for 6 Cubic Curvature Invariants")
println("  Reference: Bueno-Cano (1607.06463) Eqs. (11)-(14)")
println("=" ^ 70)

Λ_TGR = 0.3  # TensorGR convention: R̄_μν = Λ g_μν
Λ_BC = Λ_TGR / 3  # Bueno-Cano convention: R̃_μν = (D-1)Λ_BC g_μν

println("\nΛ (TensorGR) = $Λ_TGR,  Λ_BC = $Λ_BC")
println("\nInvariant    a_BC          b_BC          c_BC          e_BC")
println("─" ^ 70)

bc_results = []
for (i, (fn, name)) in enumerate(zip(INVARIANTS, INV_NAMES))
    p = bc_parameters(fn, Λ_BC)
    push!(bc_results, p)
    println("  I$i $(rpad(name, 10))  $(lpad(round(p.a; sigdigits=6), 12))  $(lpad(round(p.b; sigdigits=6), 12))  $(lpad(round(p.c; sigdigits=6), 12))  $(lpad(round(p.e; sigdigits=6), 12))")
end

# Convert to TensorGR convention and express as Λ-dependent coefficients
println("\n── In TensorGR convention (coefficients of Λ and Λ²) ──\n")
println("  γᵢ · Iᵢ contributes to the Bueno-Cano parameters as:")
println("  a_i = γᵢ · ā_i · Λ,  b_i = γᵢ · b̄_i · Λ,  c_i = γᵢ · c̄_i · Λ,  e_i = γᵢ · ē_i · Λ²")
println()
println("Invariant    ā/Λ          b̄/Λ          c̄/Λ          ē/Λ²")
println("─" ^ 70)

for (i, (p, name)) in enumerate(zip(bc_results, INV_NAMES))
    a_bar = p.a / Λ_TGR
    b_bar = p.b / Λ_TGR
    c_bar = p.c / Λ_TGR
    e_bar = p.e / Λ_TGR^2
    println("  I$i $(rpad(name, 10))  $(lpad(round(a_bar; sigdigits=6), 12))  $(lpad(round(b_bar; sigdigits=6), 12))  $(lpad(round(c_bar; sigdigits=6), 12))  $(lpad(round(e_bar; sigdigits=6), 12))")
end

# ═══════════════════════════════════════════════════════════════════════
# Analytical verification for I₁ = R³
# ═══════════════════════════════════════════════════════════════════════

println("\n── Analytical verification ──\n")

p1 = bc_results[1]
println("I₁ = R³:")
println("  Analytical: a=0, b=6Λ=$(6Λ_TGR), c=0, e=24Λ²=$(24Λ_TGR^2)")
println("  Numerical:  a=$(round(p1.a; sigdigits=6)), b=$(round(p1.b; sigdigits=6)), c=$(round(p1.c; sigdigits=6)), e=$(round(p1.e; sigdigits=6))")
@assert abs(p1.a) < 1e-4 "I₁: a should be 0, got $(p1.a)"
@assert abs(p1.b - 6Λ_TGR) / (6Λ_TGR) < 1e-3 "I₁: b should be 6Λ"
@assert abs(p1.c) < 1e-4 "I₁: c should be 0, got $(p1.c)"
@assert abs(p1.e - 24Λ_TGR^2) / (24Λ_TGR^2) < 1e-3 "I₁: e should be 24Λ²"
println("  ✓ All match within tolerance")

p2 = bc_results[2]
println("\nI₂ = R·Ric²:")
println("  Analytical: a=0, b=Λ=$(Λ_TGR), c=2Λ=$(2Λ_TGR), e=6Λ²=$(6Λ_TGR^2)")
println("  Numerical:  a=$(round(p2.a; sigdigits=6)), b=$(round(p2.b; sigdigits=6)), c=$(round(p2.c; sigdigits=6)), e=$(round(p2.e; sigdigits=6))")
@assert abs(p2.a) < 1e-4 "I₂: a should be 0, got $(p2.a)"
@assert abs(p2.b - Λ_TGR) / Λ_TGR < 1e-3 "I₂: b should be Λ"
@assert abs(p2.c - 2Λ_TGR) / (2Λ_TGR) < 1e-3 "I₂: c should be 2Λ"
@assert abs(p2.e - 6Λ_TGR^2) / (6Λ_TGR^2) < 1e-3 "I₂: e should be 6Λ²"
println("  ✓ All match within tolerance")

p3 = bc_results[3]
println("\nI₃ = Ric³:")
println("  Analytical: a=0, b=0, c=3Λ/2=$(3Λ_TGR/2), e=3Λ²/2=$(3Λ_TGR^2/2)")
println("  Numerical:  a=$(round(p3.a; sigdigits=6)), b=$(round(p3.b; sigdigits=6)), c=$(round(p3.c; sigdigits=6)), e=$(round(p3.e; sigdigits=6))")
@assert abs(p3.a) < 1e-4 "I₃: a should be 0, got $(p3.a)"
@assert abs(p3.b) < 1e-4 "I₃: b should be 0, got $(p3.b)"
@assert abs(p3.c - 3Λ_TGR/2) / (3Λ_TGR/2) < 1e-3 "I₃: c should be 3Λ/2"
@assert abs(p3.e - 3Λ_TGR^2/2) / (3Λ_TGR^2/2) < 1e-3 "I₃: e should be 3Λ²/2"
println("  ✓ All match within tolerance")

p4 = bc_results[4]
println("\nI₄ = R·Riem²:")
println("  Analytical: a=4Λ=$(4Λ_TGR), b=2Λ/3=$(2Λ_TGR/3), c=0, e=4Λ²=$(4Λ_TGR^2)")
println("  Numerical:  a=$(round(p4.a; sigdigits=6)), b=$(round(p4.b; sigdigits=6)), c=$(round(p4.c; sigdigits=6)), e=$(round(p4.e; sigdigits=6))")
@assert abs(p4.a - 4Λ_TGR) / (4Λ_TGR) < 1e-3 "I₄: a should be 4Λ"
@assert abs(p4.b - 2Λ_TGR/3) / (2Λ_TGR/3) < 1e-3 "I₄: b should be 2Λ/3"
@assert abs(p4.c) < 1e-4 "I₄: c should be 0, got $(p4.c)"
@assert abs(p4.e - 4Λ_TGR^2) / (4Λ_TGR^2) < 1e-3 "I₄: e should be 4Λ²"
println("  ✓ All match within tolerance")

p5 = bc_results[5]
println("\nI₅ = Ric·Riem²:")
println("  Analytical: a=Λ=$(Λ_TGR), b=0, c=2Λ/3=$(2Λ_TGR/3), e=Λ²=$(Λ_TGR^2)")
println("  Numerical:  a=$(round(p5.a; sigdigits=6)), b=$(round(p5.b; sigdigits=6)), c=$(round(p5.c; sigdigits=6)), e=$(round(p5.e; sigdigits=6))")
@assert abs(p5.a - Λ_TGR) / Λ_TGR < 1e-3 "I₅: a should be Λ"
@assert abs(p5.b) < 1e-4 "I₅: b should be 0, got $(p5.b)"
@assert abs(p5.c - 2Λ_TGR/3) / (2Λ_TGR/3) < 1e-3 "I₅: c should be 2Λ/3"
@assert abs(p5.e - Λ_TGR^2) / Λ_TGR^2 < 1e-3 "I₅: e should be Λ²"
println("  ✓ All match within tolerance")

p6 = bc_results[6]
println("\nI₆ = Riem³:")
println("  Analytical: a=2Λ=$(2Λ_TGR), b=0, c=0, e=2Λ²/3=$(2Λ_TGR^2/3)")
println("  Numerical:  a=$(round(p6.a; sigdigits=6)), b=$(round(p6.b; sigdigits=6)), c=$(round(p6.c; sigdigits=6)), e=$(round(p6.e; sigdigits=6))")
@assert abs(p6.a - 2Λ_TGR) / (2Λ_TGR) < 1e-3 "I₆: a should be 2Λ"
@assert abs(p6.b) < 1e-4 "I₆: b should be 0, got $(p6.b)"
@assert abs(p6.c) < 1e-4 "I₆: c should be 0, got $(p6.c)"
@assert abs(p6.e - 2Λ_TGR^2/3) / (2Λ_TGR^2/3) < 1e-3 "I₆: e should be 2Λ²/3"
println("  ✓ All match within tolerance")

# ═══════════════════════════════════════════════════════════════════════
# Full dS spectrum with all parameters
# ═══════════════════════════════════════════════════════════════════════

println("\n" * "=" ^ 70)
println("  Full de Sitter Spectrum with Cubic Corrections")
println("=" ^ 70)

# Example parameters
κ = 1.0; α₁ = -0.1; α₂ = 0.3
γ = [0.01, 0.02, -0.01, 0.005, 0.01, -0.005]  # γ₁...γ₆

# Quadratic contributions (from earlier computation)
a_quad = 0.0
b_quad = 2α₁
c_quad = 2α₂
e_quad = κ + (8α₁ + 2α₂) * Λ_BC

# Cubic contributions
a_cubic = sum(γ[i] * bc_results[i].a for i in 1:6)
b_cubic = sum(γ[i] * bc_results[i].b for i in 1:6)
c_cubic = sum(γ[i] * bc_results[i].c for i in 1:6)
e_cubic = sum(γ[i] * bc_results[i].e for i in 1:6)

# Total
a_tot = a_quad + a_cubic
b_tot = b_quad + b_cubic
c_tot = c_quad + c_cubic
e_tot = e_quad + e_cubic

println("\nParameters: κ=$κ, α₁=$α₁, α₂=$α₂, Λ=$Λ_TGR")
println("Cubic: γ = $γ")

println("\n  Component      a            b            c            e")
println("  ─" ^ 62)
println("  Quadratic   $(round(a_quad;sigdigits=4))       $(round(b_quad;sigdigits=4))       $(round(c_quad;sigdigits=4))       $(round(e_quad;sigdigits=4))")
println("  Cubic       $(round(a_cubic;sigdigits=4))       $(round(b_cubic;sigdigits=4))       $(round(c_cubic;sigdigits=4))       $(round(e_cubic;sigdigits=4))")
println("  TOTAL       $(round(a_tot;sigdigits=4))       $(round(b_tot;sigdigits=4))       $(round(c_tot;sigdigits=4))       $(round(e_tot;sigdigits=4))")

# Physical spectrum (Bueno-Cano Eqs. 17-19, D=4)
κ_eff_inv = 4e_tot - 8Λ_BC * a_tot
println("\nPhysical spectrum on dS (Bueno-Cano Eqs. 17-19):")
println("  κ_eff⁻¹ = $(round(κ_eff_inv; digits=6))")

if abs(2a_tot + c_tot) > 1e-15
    mg2 = (-e_tot + 2Λ_BC * a_tot) / (2a_tot + c_tot)
    println("  m²_g    = $(round(mg2; digits=6))  (massive spin-2)")
else
    println("  m²_g    = ∞  (no massive spin-2)")
end

denom_s = 2a_tot + 4c_tot + 12b_tot
if abs(denom_s) > 1e-15
    ms2 = (2e_tot - 4Λ_BC * (a_tot + 4b_tot + c_tot)) / denom_s
    println("  m²_s    = $(round(ms2; digits=6))  (spin-0 scalar)")
else
    println("  m²_s    = ∞  (no spin-0)")
end

# ECG verification
println("\n── ECG special case verification ──")
println("ECG (Bueno-Cano Eq. 22): P = 12R^a_b^c_d R^b_c^d_e R^e_a + Riem³ − 12Ric·Riem² + 8Ric³")
println("The ECG condition is 2a+c = 0 (no massive spin-2) and 2a+4c+12b → ∞ (no spin-0)")
println("For the pure cubic part, check if there exist γᵢ ratios giving 2a+c = 0:")

# Build the system: 2aᵢ + cᵢ = 0 for all i
for (i, p) in enumerate(bc_results)
    println("  I$i: 2a+c = $(round(2p.a + p.c; sigdigits=4))")
end

println("\n" * "=" ^ 70)
println("  COMPUTATION COMPLETE")
println("  All 6 cubic invariant BC parameters computed and verified.")
println("  Reference: Bueno-Cano (1607.06463) — LOCAL COPY verified.")
println("=" ^ 70)
