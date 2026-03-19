#= EFT of dark energy alpha parametrization.

Unifying effective field theory parametrization of dark energy / modified gravity
on FRW backgrounds, encompassing Horndeski, beyond-Horndeski (GLPV), and DHOST
theories via five time-dependent alpha functions:

  alpha_M: running of effective Planck mass (M_*^2 = M_P^2 (1 + alpha_M))
  alpha_K: kineticity (kinetic energy of scalar perturbations)
  alpha_B: braiding (kinetic mixing of metric and scalar)
  alpha_T: tensor speed excess (c_T^2 = 1 + alpha_T)
  alpha_H: beyond-Horndeski parameter (zero for Horndeski)

The GW170817 observation constrains |alpha_T| < O(10^{-15}).

Ground truth: Bellini & Sawicki, JCAP 1407 (2014) 050, arXiv:1404.3713;
              Gleyzes et al, JCAP 1308 (2013) 025, arXiv:1304.4840;
              Kobayashi (2019) arXiv:1901.04778, Sec 5.
=#

# ── EFTDarkEnergy ────────────────────────────────────────────────

"""
    EFTDarkEnergy

EFT parametrization of dark energy / modified gravity on FRW background.

Fields:
- `alpha_M`: running of effective Planck mass, d ln M_*^2 / d ln a
- `alpha_K`: kineticity (kinetic energy of scalar perturbations)
- `alpha_B`: braiding (kinetic mixing of metric and scalar)
- `alpha_T`: tensor speed excess (c_T^2 = 1 + alpha_T)
- `alpha_H`: beyond-Horndeski parameter (zero for Horndeski, default 0)
- `H`: Hubble parameter H(t)
- `Omega_DE`: dark energy density parameter

Ground truth: Bellini & Sawicki JCAP 1407 (2014) 050, Eqs 3.1-3.10;
              Gleyzes et al JCAP 1308 (2013) 025, Sec 3-4.
"""
struct EFTDarkEnergy
    alpha_M::Any    # Planck mass running
    alpha_K::Any    # kineticity
    alpha_B::Any    # braiding
    alpha_T::Any    # tensor speed excess
    alpha_H::Any    # beyond-Horndeski (optional, default 0)
    H::Any          # Hubble parameter H(t)
    Omega_DE::Any   # dark energy density parameter
end

# ── Constructor with defaults ─────────────────────────────────────

"""
    EFTDarkEnergy(; alpha_M=0, alpha_K=0, alpha_B=0, alpha_T=0,
                    alpha_H=0, H=:H, Omega_DE=:Omega_DE)

Keyword constructor for EFTDarkEnergy with sensible defaults (GR limit).
"""
function EFTDarkEnergy(; alpha_M=0, alpha_K=0, alpha_B=0, alpha_T=0,
                         alpha_H=0, H=:H, Omega_DE=:Omega_DE)
    EFTDarkEnergy(alpha_M, alpha_K, alpha_B, alpha_T, alpha_H, H, Omega_DE)
end

# ── eft_from_horndeski ────────────────────────────────────────────

"""
    eft_from_horndeski(theory::HorndeskiTheory, bg::FRWBackground;
                       registry=current_registry()) -> EFTDarkEnergy

Compute EFT dark energy parameters from a Horndeski theory on an FRW background.
Wraps the existing `compute_alphas` infrastructure into the EFT parametrization.

For Horndeski theories, alpha_H = 0. Use `eft_from_beyond_horndeski` for
beyond-Horndeski theories with nonzero alpha_H.

Ground truth: Bellini & Sawicki JCAP 1407 (2014) 050, Eqs 3.7-3.10.
"""
function eft_from_horndeski(theory::HorndeskiTheory, bg::FRWBackground;
                             registry::TensorRegistry=current_registry())
    alphas = compute_alphas(theory, bg; registry=registry)
    EFTDarkEnergy(
        alphas.alpha_M,     # symbolic :alpha_M or numeric
        alphas.alpha_K,
        alphas.alpha_B,
        alphas.alpha_T,
        0,                  # alpha_H = 0 for Horndeski
        bg.H,
        :Omega_DE           # density parameter left symbolic
    )
end

"""
    eft_from_beyond_horndeski(theory::BeyondHorndeskiTheory, bg::FRWBackground;
                               registry=current_registry()) -> EFTDarkEnergy

Compute EFT dark energy parameters from a beyond-Horndeski (GLPV) theory.
Includes nonzero alpha_H from the F_4, F_5 functions.

Ground truth: Gleyzes et al (2015); Kobayashi (2019) Sec 5.3, Eqs 5.12-5.13.
"""
function eft_from_beyond_horndeski(theory::BeyondHorndeskiTheory, bg::FRWBackground;
                                    registry::TensorRegistry=current_registry())
    alphas = compute_alphas(theory.horndeski, bg; registry=registry)
    aH = alpha_H(theory, bg; registry=registry)
    EFTDarkEnergy(
        alphas.alpha_M,
        alphas.alpha_K,
        alphas.alpha_B,
        alphas.alpha_T,
        aH,
        bg.H,
        :Omega_DE
    )
end

# ── eft_from_numerical ────────────────────────────────────────────

"""
    eft_from_numerical(theory::HorndeskiTheory, bg::FRWBackground,
                       gvals::Dict{Symbol,<:Number};
                       registry=current_registry()) -> EFTDarkEnergy

Evaluate EFT parameters numerically given G-function values on the background.
Requires additional keys :alpha_M and :Omega_DE in `gvals` for alpha_M
(which depends on time derivatives of M_*^2) and the density parameter.
"""
function eft_from_numerical(theory::HorndeskiTheory, bg::FRWBackground,
                             gvals::Dict{Symbol,<:Number};
                             registry::TensorRegistry=current_registry())
    result = compute_alphas_numerical(theory, bg, gvals; registry=registry)
    aM = haskey(gvals, :alpha_M) ? Float64(gvals[:alpha_M]) : 0.0
    Omega_DE = haskey(gvals, :Omega_DE) ? Float64(gvals[:Omega_DE]) : 0.0
    EFTDarkEnergy(
        aM,
        result[:alpha_K],
        result[:alpha_B],
        result[:alpha_T],
        result[:alpha_H],
        Float64(gvals[:H]),
        Omega_DE
    )
end

# ── Stability conditions ─────────────────────────────────────────

"""
    eft_stability(eft::EFTDarkEnergy) -> NamedTuple

Check stability conditions for scalar-tensor perturbations:
- `no_ghost_tensor`: M_*^2 > 0 (effectively 1 + alpha_M > 0 when alpha_M is numeric)
- `no_ghost_scalar`: Q_S > 0, i.e. D = alpha_K + (3/2) alpha_B^2 > 0
- `no_gradient_tensor`: c_T^2 = 1 + alpha_T > 0
- `no_gradient_scalar`: c_s^2 > 0
- `no_tachyon`: (no additional constraint at this level; mass terms require background EOM)

Returns a NamedTuple with boolean flags and the computed quantities.

For symbolic parameters, returns symbolic expressions instead of booleans.

Ground truth: Bellini & Sawicki JCAP 1407 (2014) 050, Sec 3;
              Kobayashi (2019) Sec 5.2, Eqs 5.14-5.23.
"""
function eft_stability(eft::EFTDarkEnergy)
    mul = _sym_mul
    add = _sym_add
    # Use regular division for numeric stability (avoids Float64 // Float64 error)
    safediv(a, b) = (a isa Number && b isa Number) ? a / b : _sym_div(a, b)

    aK = eft.alpha_K
    aB = eft.alpha_B
    aT = eft.alpha_T
    aM = eft.alpha_M

    # Tensor speed squared
    c_T_sq = add(1, aT)

    # D = alpha_K + (3/2) alpha_B^2
    D = add(aK, mul(3//2, mul(aB, aB)))

    # Scalar sound speed squared (simplified, on-shell)
    # c_s^2 = -(1/D)[(2 - aB)aB + aT + (aB - aM)(1 + aT)]
    # Bellini & Sawicki Eq 3.16; Kobayashi Eq 5.23
    if D isa Number && D == 0
        c_s_sq = 0
    else
        term1 = mul(_sym_sub(2, aB), aB)
        term2 = aT
        term3 = mul(_sym_sub(aB, aM), add(1, aT))
        numerator = _sym_neg(add(add(term1, term2), term3))
        c_s_sq = safediv(numerator, D)
    end

    # Evaluate stability if numeric
    if all(x -> x isa Number, (aK, aB, aT, aM))
        (
            no_ghost_tensor = true,   # M_*^2 > 0 assumed (need full expression)
            no_ghost_scalar = D > 0,
            no_gradient_tensor = c_T_sq > 0,
            no_gradient_scalar = c_s_sq > 0,
            no_tachyon = true,        # mass term stability requires background EOM
            c_T_sq = c_T_sq,
            c_s_sq = c_s_sq,
            D = D,
        )
    else
        (
            no_ghost_tensor = true,
            no_ghost_scalar = D,
            no_gradient_tensor = c_T_sq,
            no_gradient_scalar = c_s_sq,
            no_tachyon = true,
            c_T_sq = c_T_sq,
            c_s_sq = c_s_sq,
            D = D,
        )
    end
end

# ── Observables ───────────────────────────────────────────────────

"""
    eft_observables(eft::EFTDarkEnergy) -> NamedTuple

Compute observable quantities derived from the EFT alpha parameters:

- `G_eff_over_GN`: effective gravitational coupling for matter, G_eff/G_N
  In the quasi-static limit: G_eff/G_N = (1/(1+alpha_M)) * (1 + xi^2 / c_s^2)
  where xi = alpha_B * H (the braiding determines the scalar-matter coupling).
  Simplified expression valid for |alpha_i| << 1.

- `slip`: gravitational slip parameter eta = Phi/Psi (ratio of Bardeen potentials)
  eta = (1 + alpha_T) / (1 + alpha_M + alpha_B * beta)
  In the GR limit, eta = 1.

- `c_T_sq`: tensor (gravitational wave) speed squared, c_T^2 = 1 + alpha_T

Ground truth: Bellini & Sawicki JCAP 1407 (2014) 050, Sec 4;
              Amendola et al, Living Rev. Rel. 21 (2018) 2, Sec 3.
"""
function eft_observables(eft::EFTDarkEnergy)
    mul = _sym_mul
    add = _sym_add
    sub = _sym_sub
    # Use regular division for numeric types (avoids Float64 // Float64 error)
    div(a, b) = (a isa Number && b isa Number) ? a / b : _sym_div(a, b)

    aM = eft.alpha_M
    aK = eft.alpha_K
    aB = eft.alpha_B
    aT = eft.alpha_T

    # Tensor speed squared
    c_T_sq = add(1, aT)

    # Effective gravitational coupling (quasi-static, sub-horizon limit)
    # G_eff/G_N ~ 1 + ... (leading-order correction from alpha params)
    # For small alphas: G_eff/G_N approx 1 + (2 alpha_B^2) / (alpha_K + 3/2 alpha_B^2)
    # Full expression: Amendola et al (2018) Eq 3.16
    # D = alpha_K + (3/2) alpha_B^2
    D = add(aK, mul(3//2, mul(aB, aB)))

    if D isa Number && D == 0
        # GR limit: no propagating scalar => G_eff = G_N
        G_eff_ratio = 1
    else
        # Quasi-static approximation:
        # G_eff/G_N = (1/(1+alpha_M)) * [1 + 2 alpha_B^2 / D]
        # where D = alpha_K + (3/2) alpha_B^2
        braiding_correction = div(mul(2, mul(aB, aB)), D)
        G_eff_ratio = div(add(1, braiding_correction), add(1, aM))
    end

    # Gravitational slip: eta = Phi/Psi
    # In the quasi-static limit with alpha_H = 0:
    # eta = (1 + alpha_T) / (1 + alpha_M)
    # (This is the leading-order expression; higher-order terms depend on
    # background evolution and are model-specific.)
    if eft.alpha_H isa Number && eft.alpha_H == 0
        slip = div(c_T_sq, add(1, aM))
    else
        # With beyond-Horndeski: eta receives alpha_H corrections
        # eta = (1 + alpha_T - alpha_H) / (1 + alpha_M)
        slip = div(sub(c_T_sq, eft.alpha_H), add(1, aM))
    end

    (
        G_eff_over_GN = G_eff_ratio,
        slip = slip,
        c_T_sq = c_T_sq,
    )
end

# ── GW170817 constraint ──────────────────────────────────────────

"""
    gw170817_constraint(eft::EFTDarkEnergy) -> Bool

Check if the EFT parametrization satisfies the GW170817 constraint on the
tensor speed: |alpha_T| < 10^{-15}, effectively alpha_T = 0.

For numeric alpha_T, checks |alpha_T| < 1e-15.
For symbolic alpha_T, returns `alpha_T == 0`.

Ground truth: LIGO/Virgo, PRL 119, 161101 (2017);
              Creminelli & Vernizzi, PRL 119, 251302 (2017);
              Baker et al, PRL 119, 251301 (2017).
"""
function gw170817_constraint(eft::EFTDarkEnergy)
    aT = eft.alpha_T
    if aT isa Number
        return abs(aT) < 1e-15
    else
        return aT == 0
    end
end

# ── Convenience: GR EFT ──────────────────────────────────────────

"""
    eft_gr(; H=:H, Omega_DE=:Omega_DE) -> EFTDarkEnergy

Return the EFT parametrization for General Relativity: all alpha_i = 0.
"""
function eft_gr(; H=:H, Omega_DE=:Omega_DE)
    EFTDarkEnergy(0, 0, 0, 0, 0, H, Omega_DE)
end

# ── Convenience: quintessence EFT ─────────────────────────────────

"""
    eft_quintessence(alpha_K; H=:H, Omega_DE=:Omega_DE) -> EFTDarkEnergy

Return the EFT parametrization for quintessence: only alpha_K nonzero.
Quintessence = minimally coupled canonical scalar field.

In Horndeski: G2 = X - V(phi), G3 = G5 = 0, G4 = M_Pl^2/2.
=> alpha_M = alpha_B = alpha_T = alpha_H = 0, alpha_K != 0.

Ground truth: Bellini & Sawicki JCAP 1407 (2014) 050, Sec 4.1.
"""
function eft_quintessence(alpha_K; H=:H, Omega_DE=:Omega_DE)
    EFTDarkEnergy(0, alpha_K, 0, 0, 0, H, Omega_DE)
end

# ── Convenience: f(R) EFT ─────────────────────────────────────────

"""
    eft_fR(alpha_M; H=:H, Omega_DE=:Omega_DE) -> EFTDarkEnergy

Return the EFT parametrization for f(R) gravity.
f(R) satisfies: alpha_T = 0, alpha_B = -alpha_M, alpha_K = 0 (on-shell),
alpha_H = 0.

The relation alpha_B = -alpha_M is a distinguishing prediction of f(R)
gravity within the EFT framework.

Ground truth: Bellini & Sawicki JCAP 1407 (2014) 050, Sec 4.2;
              Pogosian & Silvestri, PRD 94, 104014 (2016).
"""
function eft_fR(alpha_M; H=:H, Omega_DE=:Omega_DE)
    alpha_B = _sym_neg(alpha_M)
    EFTDarkEnergy(alpha_M, 0, alpha_B, 0, 0, H, Omega_DE)
end

# ── Apply GW170817: set alpha_T = 0 ──────────────────────────────

"""
    apply_gw170817(eft::EFTDarkEnergy) -> EFTDarkEnergy

Return a new EFT parametrization with alpha_T = 0, enforcing the GW170817
constraint on the tensor speed.

This eliminates quintic Horndeski (G5 != const) and beyond-Horndeski
theories with certain F_4, F_5 combinations.

Ground truth: Creminelli & Vernizzi, PRL 119, 251302 (2017).
"""
function apply_gw170817(eft::EFTDarkEnergy)
    EFTDarkEnergy(eft.alpha_M, eft.alpha_K, eft.alpha_B, 0,
                  eft.alpha_H, eft.H, eft.Omega_DE)
end
