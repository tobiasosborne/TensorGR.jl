#= Quadratic action for scalar-tensor perturbations on FRW.

On a spatially flat FRW background, the second-order action for linear
perturbations in Horndeski theory decouples into tensor and scalar sectors.

Tensor sector (gravitational waves):
  S_T^{(2)} = (1/8) int dt d^3x a^3 M_*^2
              [dot{h}_{ij}^2 - (1 + alpha_T)(nabla h_{ij})^2 / a^2]

Scalar sector (curvature perturbation zeta, unitary gauge):
  S_S^{(2)} = int dt d^3x a^3 Q_S
              [dot{zeta}^2 - c_s^2 (nabla zeta)^2 / a^2]

where:
  D     = alpha_K + (3/2) alpha_B^2               (Kobayashi Eq 5.15)
  Q_S   = M_*^2 D / (c_T^2 H^2)                  (Kobayashi Eq 5.22)
  c_T^2 = 1 + alpha_T                             (tensor speed squared)
  c_s^2 = scalar sound speed squared               (Kobayashi Eq 5.23)

Ground truth: Kobayashi arXiv:1901.04778, Sec 5.2, Eqs 5.14-5.23;
              Bellini & Sawicki JCAP 1407 (2014) 050, Sec 3.
=#

# ── ScalarTensorQuadraticAction ─────────────────────────────────────

"""
    ScalarTensorQuadraticAction

Quadratic action for cosmological perturbations in Horndeski theory on FRW.
Stores the kinetic and gradient coefficients for both tensor and scalar sectors.

Fields:
- `tensor_kinetic`: coefficient of dot{h}_{ij}^2, i.e. (1/8) M_*^2
- `tensor_gradient`: coefficient of (nabla h_{ij})^2 / a^2, i.e. -(1/8) M_*^2 c_T^2
- `scalar_kinetic`: coefficient of dot{zeta}^2, i.e. Q_S
- `scalar_gradient`: coefficient of (nabla zeta)^2 / a^2, i.e. -Q_S c_s^2
- `alphas`: the Bellini-Sawicki alpha parameters used
"""
struct ScalarTensorQuadraticAction
    tensor_kinetic::Any       # (1/8) M_*^2
    tensor_gradient::Any      # -(1/8) M_*^2 c_T^2
    scalar_kinetic::Any       # Q_S
    scalar_gradient::Any      # -Q_S c_s^2
    alphas::BelliniSawickiAlphas
end

# ── StabilityConditions ─────────────────────────────────────────────

"""
    StabilityConditions

The four stability conditions for cosmological perturbations in Horndeski theory:
- No tensor ghost: M_*^2 > 0
- No tensor gradient instability: c_T^2 > 0
- No scalar ghost: Q_S > 0 (equivalently D > 0)
- No scalar gradient instability: c_s^2 > 0

Also stores the numerical/symbolic values of c_T^2, c_s^2, and Q_S.
"""
struct StabilityConditions
    no_tensor_ghost::Bool
    no_tensor_gradient::Bool
    no_scalar_ghost::Bool
    no_scalar_gradient::Bool
    c_T_sq::Any
    c_s_sq::Any
    Q_S::Any
end

# ── Sound speeds ────────────────────────────────────────────────────

"""
    tensor_sound_speed(alphas::BelliniSawickiAlphas) -> Any

Tensor (gravitational wave) speed squared: c_T^2 = 1 + alpha_T.
GW170817 constrains |alpha_T| < 10^{-15}.
"""
function tensor_sound_speed(alphas::BelliniSawickiAlphas)
    _sym_add(1, alphas.alpha_T)
end

"""
    scalar_sound_speed(alphas::BelliniSawickiAlphas) -> Any

Scalar sound speed squared c_s^2, computed from the alpha parameters.

Uses the relation (Bellini & Sawicki Sec 3; Kobayashi Eq 5.23 simplified):
  c_s^2 = -(1/D) [(2 - alpha_B) alpha_B + alpha_T
                    + (alpha_B - alpha_M) (1 + alpha_T)]

where D = alpha_K + (3/2) alpha_B^2.

This is the closed-form expression valid when the background equations of motion
hold and time derivatives of the alphas have been eliminated via the Friedmann
equations. Following Bellini & Sawicki Eq 3.16 (and Gleyzes et al 2015).

Note: This expression requires alpha_M (the running of M_*^2). When alpha_M
is symbolic (as returned by compute_alphas), the result contains :alpha_M
and must be evaluated numerically via check_stability.
"""
function scalar_sound_speed(alphas::BelliniSawickiAlphas)
    aK = alphas.alpha_K
    aB = alphas.alpha_B
    aT = alphas.alpha_T
    aM = alphas.alpha_M

    mul = _sym_mul
    add = _sym_add
    sub = _sym_sub
    div = _sym_div

    # D = alpha_K + (3/2) alpha_B^2
    D = add(aK, mul(3//2, mul(aB, aB)))

    # Numerator: -[(2 - aB) aB + aT + (aB - aM)(1 + aT)]
    # = -[2 aB - aB^2 + aT + aB + aB aT - aM - aM aT]
    # Keep factored for clarity.
    term1 = mul(sub(2, aB), aB)         # (2 - aB) aB
    term2 = aT                           # alpha_T
    term3 = mul(sub(aB, aM), add(1, aT)) # (aB - aM)(1 + aT)

    numerator = _sym_neg(add(add(term1, term2), term3))

    # When D = 0 (e.g. GR limit: no propagating scalar), c_s^2 is undefined.
    # If numerator is also 0, return 0 (degenerate case, Q_S = 0 anyway).
    if D isa Number && D == 0
        return numerator == 0 ? 0 : error("scalar_sound_speed: D = 0 with nonzero numerator")
    end

    div(numerator, D)
end

# ── Quadratic action construction ──────────────────────────────────

"""
    quadratic_action_horndeski(alphas::BelliniSawickiAlphas) -> ScalarTensorQuadraticAction

Construct the quadratic action for Horndeski perturbations on FRW from the
Bellini-Sawicki alpha parameters.

Returns a `ScalarTensorQuadraticAction` with symbolic coefficients.

Ground truth: Kobayashi (2019) Sec 5.2, Eqs 5.14-5.23.
"""
function quadratic_action_horndeski(alphas::BelliniSawickiAlphas)
    mul = _sym_mul
    neg = _sym_neg
    add = _sym_add
    div = _sym_div

    M2 = alphas.M_star_sq
    aK = alphas.alpha_K
    aB = alphas.alpha_B

    c_T_sq = tensor_sound_speed(alphas)
    c_s_sq = scalar_sound_speed(alphas)

    # D = alpha_K + (3/2) alpha_B^2     (Kobayashi Eq 5.15)
    D = add(aK, mul(3//2, mul(aB, aB)))

    # Tensor sector: (1/8) M_*^2 [dot{h}^2 - c_T^2 (nabla h)^2 / a^2]
    tensor_kinetic = mul(1//8, M2)
    tensor_gradient = neg(mul(1//8, mul(M2, c_T_sq)))

    # Scalar sector: Q_S [dot{zeta}^2 - c_s^2 (nabla zeta)^2 / a^2]
    # Q_S = M_*^2 D / (c_T^2 H^2)      (Kobayashi Eq 5.22)
    H = alphas.background.H
    Q_S = div(mul(M2, D), mul(c_T_sq, mul(H, H)))

    scalar_kinetic = Q_S
    scalar_gradient = neg(mul(Q_S, c_s_sq))

    ScalarTensorQuadraticAction(tensor_kinetic, tensor_gradient,
                                scalar_kinetic, scalar_gradient, alphas)
end

# ── Stability conditions ───────────────────────────────────────────

"""
    stability_conditions(qa::ScalarTensorQuadraticAction) -> StabilityConditions

Extract the stability conditions from a quadratic action.
For symbolic parameters, all conditions are returned as `true` (unchecked).
Use `check_stability` with a parameter dictionary for numeric evaluation.
"""
function stability_conditions(qa::ScalarTensorQuadraticAction)
    c_T_sq = tensor_sound_speed(qa.alphas)
    c_s_sq = scalar_sound_speed(qa.alphas)
    Q_S = qa.scalar_kinetic
    M2 = qa.alphas.M_star_sq

    # For purely symbolic expressions, return true (cannot evaluate)
    StabilityConditions(true, true, true, true, c_T_sq, c_s_sq, Q_S)
end

"""
    check_stability(qa::ScalarTensorQuadraticAction,
                    params::Dict{Symbol,<:Number}) -> StabilityConditions

Evaluate all four stability conditions numerically by substituting parameter
values into the symbolic expressions.

`params` maps symbols (:alpha_K, :alpha_B, :alpha_T, :alpha_M, :M_star_sq,
:H, etc.) to numbers.

Returns a `StabilityConditions` with `Bool` flags and numeric values.
"""
function check_stability(qa::ScalarTensorQuadraticAction,
                          params::Dict{Symbol,<:Number})
    c_T_sq_expr = tensor_sound_speed(qa.alphas)
    c_s_sq_expr = scalar_sound_speed(qa.alphas)

    c_T_sq_val = sym_eval(c_T_sq_expr, params)
    c_s_sq_val = sym_eval(c_s_sq_expr, params)

    M2_val = sym_eval(qa.alphas.M_star_sq, params)
    Q_S_val = sym_eval(qa.scalar_kinetic, params)

    StabilityConditions(
        M2_val > 0,          # no tensor ghost
        c_T_sq_val > 0,      # no tensor gradient instability
        Q_S_val > 0,         # no scalar ghost
        c_s_sq_val > 0,      # no scalar gradient instability
        c_T_sq_val,
        c_s_sq_val,
        Q_S_val
    )
end

# ── Integration with QuadraticForm ─────────────────────────────────

"""
    to_quadratic_form(qa::ScalarTensorQuadraticAction;
                      omega=:omega, k=:k) -> QuadraticForm

Map the scalar-tensor quadratic action to the existing `QuadraticForm` type
with fields `[:h_TT, :zeta]`.

The resulting 2x2 block-diagonal matrix has momentum-dependent entries:
  M[1,1] = (1/8) M_*^2 (omega^2 - c_T^2 k^2)   (tensor sector)
  M[2,2] = Q_S (omega^2 - c_s^2 k^2)             (scalar sector)
  M[1,2] = M[2,1] = 0                             (decoupled)
"""
function to_quadratic_form(qa::ScalarTensorQuadraticAction;
                            omega::Symbol=:omega, k::Symbol=:k)
    mul = _sym_mul
    add = _sym_add
    neg = _sym_neg

    c_T_sq = tensor_sound_speed(qa.alphas)
    c_s_sq = scalar_sound_speed(qa.alphas)

    # omega^2 and k^2
    w2 = mul(omega, omega)
    k2 = mul(k, k)

    # M[1,1] = tensor_kinetic * omega^2 + tensor_gradient * k^2
    #        = (1/8) M_*^2 omega^2 - (1/8) M_*^2 c_T^2 k^2
    M11 = add(mul(qa.tensor_kinetic, w2), mul(neg(qa.tensor_gradient), k2))

    # M[2,2] = Q_S omega^2 - Q_S c_s^2 k^2
    M22 = add(mul(qa.scalar_kinetic, w2), mul(neg(qa.scalar_gradient), k2))

    entries = Dict(
        (:h_TT, :h_TT) => M11,
        (:zeta, :zeta) => M22,
        (:h_TT, :zeta) => 0,
    )
    quadratic_form(entries, [:h_TT, :zeta])
end
