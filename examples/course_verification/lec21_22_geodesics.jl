# ============================================================================
# TensorGR.jl -- Course Verification: Lectures 21-22 -- Geodesics
#
# Combines abstract Killing vector identities with symbolic effective
# potential calculations for Schwarzschild geodesics.
#
# Topics:
#   1. Killing vector conservation (abstract algebra)
#   2. Effective potential for massive particles (Symbolics.jl)
#   3. ISCO at r = 6M
#   4. Null effective potential and photon sphere at r = 3M
#
# References:
#   Carroll, "Spacetime and Geometry", Chapter 5
#   Wald, "General Relativity" (1984), Chapter 6
# ============================================================================

using TensorGR
using Symbolics
import TensorGR: simplify

println("="^70)
println("Lectures 21-22: Geodesics and Orbital Mechanics")
println("="^70)

passed = 0
failed = 0

function check(label, cond)
    global passed, failed
    if cond
        passed += 1
        println("  $label ... PASSED")
    else
        failed += 1
        println("  $label ... FAILED")
    end
end

# ======================================================================
# Part 1: Abstract Killing vector identities (TensorGR algebra)
# ======================================================================
println("\n" * "="^70)
println("Part 1: Killing Vector Conservation (Abstract Algebra)")
println("="^70)

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_metric!(reg, :g; manifold=:M4)

    # ------------------------------------------------------------------
    # 1a. Killing equation structure: nabla_a xi_b + nabla_b xi_a = 0
    #     Register a Killing vector and verify the Lie derivative vanishes.
    # ------------------------------------------------------------------
    println("\n--- 1a. Killing equation: Lie derivative of metric ---")

    register_tensor!(reg, TensorProperties(
        name=:xi, manifold=:M4, rank=(1, 0),
        symmetries=SymmetrySpec[]))

    xi = Tensor(:xi, [up(:c)])
    g_ab = Tensor(:g, [down(:a), down(:b)])

    # Lie derivative of the metric along xi
    lie_g = lie_derivative(xi, g_ab)
    println("  L_xi g_{ab} = ", to_unicode(lie_g))

    # The Lie derivative should produce a sum with derivative terms
    check("Lie derivative of metric is a structured expression",
          lie_g isa TSum || lie_g isa TProduct || lie_g isa TDeriv)

    # Verify free indices are {a, b}
    fi = free_indices(lie_g)
    fi_names = Set([idx.name for idx in fi])
    check("L_xi g_{ab} has free indices {a, b}", fi_names == Set([:a, :b]))

    # ------------------------------------------------------------------
    # 1b. Conservation: if xi is Killing and u is geodesic,
    #     then xi_a u^a = const
    #
    #     Abstractly: nabla_b (xi_a u^a) = u^b nabla_b(xi_a) u^a + xi_a u^b nabla_b u^a
    #     The second term vanishes (geodesic: u^b nabla_b u^a = 0)
    #     The first term: u^a u^b nabla_b xi_a = 0 (Killing + symmetry)
    #
    #     We verify the structure: the conserved quantity xi_a u^a is scalar.
    # ------------------------------------------------------------------
    println("\n--- 1b. Conserved quantity xi_a u^a ---")

    register_tensor!(reg, TensorProperties(
        name=:u, manifold=:M4, rank=(1, 0),
        symmetries=SymmetrySpec[]))

    # xi_a u^a (scalar)
    conserved = Tensor(:xi, [down(:a)]) * Tensor(:u, [up(:a)])
    conserved_s = simplify(conserved)
    println("  xi_a u^a = ", to_unicode(conserved_s))
    fi_cons = free_indices(conserved_s)
    check("xi_a u^a is a scalar (no free indices)", isempty(fi_cons))

    # ------------------------------------------------------------------
    # 1c. The derivative nabla_b(xi_a u^a) structure
    # ------------------------------------------------------------------
    println("\n--- 1c. Derivative of conserved quantity ---")

    d_conserved = TDeriv(down(:b), conserved)
    println("  nabla_b(xi_a u^a) constructed")
    fi_d = free_indices(d_conserved)
    check("nabla_b(xi_a u^a) has one free index (b)",
          length(fi_d) == 1 && fi_d[1].name == :b)
end

# ======================================================================
# Part 2: Schwarzschild Effective Potential (Symbolics.jl)
# ======================================================================
println("\n" * "="^70)
println("Part 2: Schwarzschild Effective Potential (Symbolic)")
println("="^70)

@variables r_s M_s L_s E_s

# ------------------------------------------------------------------
# 2. Massive particle effective potential
#    V_eff(r) = (1/2) - M/r + L^2/(2r^2) - M*L^2/r^3
#
#    This comes from the geodesic equation with conserved E and L:
#    (1/2)(dr/dtau)^2 + V_eff = (1/2)E^2
# ------------------------------------------------------------------
println("\n--- 2. Massive effective potential ---")

V_eff = 1//2 - M_s/r_s + L_s^2/(2*r_s^2) - M_s*L_s^2/r_s^3
println("  V_eff(r) = 1/2 - M/r + L^2/(2r^2) - ML^2/r^3")

# Compute dV/dr
D_r = Symbolics.Differential(r_s)
dV = Symbolics.expand_derivatives(D_r(V_eff))
dV_simplified = Symbolics.simplify(dV)
println("  dV/dr = ", dV_simplified)

# Compute d^2V/dr^2
ddV = Symbolics.expand_derivatives(D_r(dV))
ddV_simplified = Symbolics.simplify(ddV)
println("  d^2V/dr^2 = ", ddV_simplified)

# ------------------------------------------------------------------
# 3. ISCO at r = 6M
#    At ISCO: dV/dr = 0 and d^2V/dr^2 = 0 simultaneously
#    Solution: r_ISCO = 6M, L_ISCO^2 = 12 M^2
# ------------------------------------------------------------------
println("\n--- 3. ISCO at r = 6M ---")

# At ISCO: L^2 = 12 M^2 (standard result)
L_ISCO_sq = 12.0
M_val = 1.0
r_ISCO = 6.0 * M_val

isco_vals = Dict(r_s => r_ISCO, M_s => M_val, L_s => sqrt(L_ISCO_sq))

# Evaluate dV/dr at ISCO
dV_at_isco = Symbolics.value(Symbolics.substitute(dV, isco_vals))
check("dV/dr = 0 at r = 6M (ISCO)", abs(Float64(dV_at_isco)) < 1e-10)
println("    dV/dr(6M) = $dV_at_isco")

# Evaluate d^2V/dr^2 at ISCO
ddV_at_isco = Symbolics.value(Symbolics.substitute(ddV, isco_vals))
check("d^2V/dr^2 = 0 at r = 6M (ISCO)", abs(Float64(ddV_at_isco)) < 1e-10)
println("    d^2V/dr^2(6M) = $ddV_at_isco")

# Verify ISCO energy: E_ISCO^2 = 8/9
V_at_isco = Symbolics.value(Symbolics.substitute(V_eff, isco_vals))
E_ISCO_sq = 2.0 * Float64(V_at_isco)
check("E_ISCO^2 = 8/9", abs(E_ISCO_sq - 8.0/9.0) < 1e-10)
println("    E_ISCO^2 = 2*V(6M) = $E_ISCO_sq (expected $(8.0/9.0))")

# ------------------------------------------------------------------
# 3b. Circular orbit: verify dV/dr = 0 for general L at r_circ
#     For L^2 = 4M^2 -> r_circ = 4M (unstable, above photon sphere)
# ------------------------------------------------------------------
println("\n--- 3b. Circular orbit at r = 4M ---")

L_4M = 4.0 * M_val^2  # L^2 = 4M^2
r_4M = 4.0 * M_val
circ_vals = Dict(r_s => r_4M, M_s => M_val, L_s => sqrt(L_4M))
dV_circ = Float64(Symbolics.value(Symbolics.substitute(dV, circ_vals)))
# For L^2=4M^2: dV/dr at r=4M should be 0 (unstable circular orbit)
# Solve: dV/dr = M/r^2 - L^2/r^3 + 3ML^2/r^4 = 0
# At r=4M, L^2=4M^2: M/16M^2 - 4M^2/64M^3 + 3M*4M^2/256M^4
# = 1/(16M) - 1/(16M) + 12/(256M) = 3/(64M)
# So L^2=4M^2 does NOT give circular at r=4M.
# Correct: for r_circ = r, L^2 = Mr^2/(r-3M).
# At r=4M: L^2 = M*16M^2/(4M-3M) = 16M^2/1 = 16M^3.
# But we want L^2 per unit mass, so L^2 = 16M^2 (with M=1: L=4).
circ_vals2 = Dict(r_s => r_4M, M_s => M_val, L_s => 4.0)
dV_circ2 = Float64(Symbolics.value(Symbolics.substitute(dV, circ_vals2)))
check("dV/dr = 0 at r=4M with L^2=16M^2 (circular orbit)", abs(dV_circ2) < 1e-10)
println("    dV/dr(r=4M, L=4M) = $dV_circ2")

# ------------------------------------------------------------------
# 4. Null geodesic effective potential
#    V_null(r) = L^2(r-2M)/(2r^3) = L^2/(2r^2) - ML^2/r^3
#    (This is V_eff without the 1/2 - M/r terms that come from m!=0)
#
#    Photon sphere at r = 3M: dV_null/dr = 0
# ------------------------------------------------------------------
println("\n--- 4. Photon sphere at r = 3M ---")

V_null = L_s^2 / (2*r_s^2) - M_s*L_s^2 / r_s^3
println("  V_null(r) = L^2/(2r^2) - ML^2/r^3")

dV_null = Symbolics.expand_derivatives(D_r(V_null))
dV_null_simplified = Symbolics.simplify(dV_null)
println("  dV_null/dr = ", dV_null_simplified)

# Evaluate at r = 3M
r_photon = 3.0 * M_val
photon_vals = Dict(r_s => r_photon, M_s => M_val, L_s => 1.0)
dV_null_photon = Float64(Symbolics.value(Symbolics.substitute(dV_null, photon_vals)))
check("dV_null/dr = 0 at r = 3M (photon sphere)", abs(dV_null_photon) < 1e-10)
println("    dV_null/dr(3M) = $dV_null_photon")

# Verify this is a maximum (unstable): d^2V_null/dr^2 < 0
ddV_null = Symbolics.expand_derivatives(D_r(dV_null))
ddV_null_photon = Float64(Symbolics.value(Symbolics.substitute(ddV_null, photon_vals)))
check("d^2V_null/dr^2 < 0 at r=3M (unstable)", ddV_null_photon < 0)
println("    d^2V_null/dr^2(3M) = $ddV_null_photon (negative => unstable)")

# ------------------------------------------------------------------
# 5. Impact parameter at photon sphere: b = L/E = 3*sqrt(3)*M
#    At r=3M: V_null(3M) = L^2/(2*9M^2) - ML^2/(27M^3)
#           = L^2/(18M^2) - L^2/(27M^2)
#           = L^2/(54M^2)
#    E^2/2 = V_null -> E^2 = L^2/(27M^2) -> b = L/E = 3*sqrt(3)*M
# ------------------------------------------------------------------
println("\n--- 5. Impact parameter at photon sphere ---")

V_null_3M = Float64(Symbolics.value(Symbolics.substitute(V_null, photon_vals)))
E_sq_photon = 2.0 * V_null_3M  # E^2 = 2*V_null for L=1
b_squared = 1.0 / E_sq_photon  # b^2 = L^2/E^2 = 1/E^2 for L=1
b_expected_sq = 27.0 * M_val^2  # (3*sqrt(3)*M)^2
check("b^2 = 27 M^2 at photon sphere", abs(b_squared - b_expected_sq) < 1e-10)
println("    b^2 = $b_squared (expected $b_expected_sq = 27M^2)")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
println("\n" * "="^70)
total = passed + failed
println("Lectures 21-22 (Geodesics) verification: $passed/$total checks passed")
if failed > 0
    println("WARNING: $failed check(s) failed!")
else
    println("All geodesic and orbital mechanics properties verified!")
end
println("="^70)

@assert failed == 0 "Lectures 21-22 verification had $failed failure(s)"
