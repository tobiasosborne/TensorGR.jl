#= Hassan-Rosen bimetric interaction potential.
#
# V(g,f) = m² Σ_{n=0}^{4} β_n e_n(S)
#
# where S^a_b = (√(g⁻¹f))^a_b and e_n are elementary symmetric polynomials:
#   e_0(S) = 1
#   e_1(S) = Tr(S)
#   e_2(S) = (1/2)((TrS)² - Tr(S²))
#   e_3(S) = (1/6)((TrS)³ - 3·TrS·Tr(S²) + 2·Tr(S³))
#   e_4(S) = det(S)
#
# Ground truth: Hassan & Rosen, JHEP 02 (2012) 126;
#              de Rham, Living Rev. Rel. 17, 7 (2014) Sec 8.
=#

"""
    HassanRosenParams

Parameters for the Hassan-Rosen bimetric interaction potential.

# Fields
- `m_sq::Any`      -- mass parameter m²
- `beta::NTuple{5,Any}` -- β₀, β₁, β₂, β₃, β₄ coefficients
"""
struct HassanRosenParams
    m_sq::Any
    beta::NTuple{5,Any}
end

function HassanRosenParams(; m_sq=:m2, beta0=0, beta1=0, beta2=0, beta3=0, beta4=0)
    HassanRosenParams(m_sq, (beta0, beta1, beta2, beta3, beta4))
end

function Base.show(io::IO, p::HassanRosenParams)
    print(io, "HR(m²=$(p.m_sq), β=", p.beta, ")")
end

"""
    elementary_symmetric(n::Int, S_name::Symbol;
                          registry::TensorRegistry=current_registry()) -> TensorExpr

Construct the n-th elementary symmetric polynomial e_n(S) of the
matrix square root S^a_b.

    e_0 = 1
    e_1 = S^a_a = Tr(S)
    e_2 = (1/2)((Tr S)² - Tr(S²))
    e_3 = (1/6)((Tr S)³ - 3·Tr S·Tr(S²) + 2·Tr(S³))
    e_4 = det(S)

Ground truth: Hassan & Rosen (2012) Eq 2.4.
"""
function elementary_symmetric(n::Int, S_name::Symbol;
                                registry::TensorRegistry=current_registry())
    n in 0:4 || error("Elementary symmetric polynomial order must be 0-4, got $n")

    if n == 0
        return TScalar(1 // 1)
    end

    used = Set{Symbol}()

    if n == 1
        # e_1 = Tr(S) = S^a_a
        a = fresh_index(used)
        return Tensor(S_name, [up(a), down(a)])
    end

    if n == 2
        # e_2 = (1/2)((Tr S)² - Tr(S²))
        a = fresh_index(used); push!(used, a)
        b = fresh_index(used); push!(used, b)
        c = fresh_index(used)

        trS = Tensor(S_name, [up(a), down(a)])
        trS_sq = trS * trS
        # Tr(S²) = S^a_b S^b_a
        S_ab = Tensor(S_name, [up(b), down(c)])
        S_ba = Tensor(S_name, [up(c), down(b)])
        trS2 = S_ab * S_ba

        return tproduct(1 // 2, TensorExpr[trS_sq - trS2])
    end

    if n == 3
        # e_3 = (1/6)((Tr S)³ - 3·Tr S·Tr(S²) + 2·Tr(S³))
        a = fresh_index(used); push!(used, a)
        b = fresh_index(used); push!(used, b)
        c = fresh_index(used); push!(used, c)
        d = fresh_index(used); push!(used, d)
        e = fresh_index(used)

        trS = Tensor(S_name, [up(a), down(a)])
        trS_cubed = trS * trS * trS

        S1 = Tensor(S_name, [up(b), down(c)])
        S2 = Tensor(S_name, [up(c), down(b)])
        trS2 = S1 * S2
        trS_trS2 = tproduct(-3 // 1, TensorExpr[trS, trS2])

        S3a = Tensor(S_name, [up(d), down(e)])
        S3b = Tensor(S_name, [up(e), down(b)])
        S3c = Tensor(S_name, [up(b), down(d)])
        trS3 = tproduct(2 // 1, TensorExpr[S3a, S3b, S3c])

        return tproduct(1 // 6, TensorExpr[trS_cubed + trS_trS2 + trS3])
    end

    # n == 4: det(S) — represented symbolically
    TScalar(Symbol(:det_, S_name))
end

"""
    hassan_rosen_potential(bs::BimetricSetup, params::HassanRosenParams;
                           registry::TensorRegistry=current_registry()) -> TensorExpr

Construct the Hassan-Rosen interaction potential:

    V = m² √(-g) Σ_{n=0}^{4} β_n e_n(S)

Returns the potential as a TensorExpr (without the √(-g) factor).

Ground truth: Hassan & Rosen (2012) Eq 2.4.
"""
function hassan_rosen_potential(bs::BimetricSetup, params::HassanRosenParams;
                                 registry::TensorRegistry=current_registry())
    S_name = Symbol(:S_, bs.metric_g, :_, bs.metric_f)

    terms = TensorExpr[]
    for n in 0:4
        coeff = params.beta[n + 1]
        (coeff isa Number && coeff == 0) && continue

        en = elementary_symmetric(n, S_name; registry=registry)

        if coeff isa Number && coeff == 1
            push!(terms, en)
        else
            push!(terms, tproduct(1 // 1, TensorExpr[TScalar(coeff), en]))
        end
    end

    isempty(terms) && return TScalar(0 // 1)

    potential = length(terms) == 1 ? terms[1] : tsum(terms)

    # Multiply by m²
    if params.m_sq isa Number && params.m_sq == 1
        return potential
    end
    tproduct(1 // 1, TensorExpr[TScalar(params.m_sq), potential])
end

# ── Matrix square root algebra ─────────────────────────────────────

"""
    _S_power_chain(S_name, n, idx_top, idx_bot, used) -> TensorExpr

Build S^n as a chain of n factors of S contracted along internal indices:

    (S^n)^{idx_top}_{idx_bot} = S^{idx_top}_{c₁} S^{c₁}_{c₂} ⋯ S^{c_{n-1}}_{idx_bot}

For n=0, returns the Kronecker delta δ^{idx_top}_{idx_bot}.
For n=1, returns S^{idx_top}_{idx_bot}.

The `used` set is updated with any fresh indices consumed.
"""
function _S_power_chain(S_name::Symbol, n::Int, idx_top::Symbol, idx_bot::Symbol,
                        used::Set{Symbol})
    n >= 0 || error("Power must be non-negative, got $n")

    if n == 0
        return Tensor(:δ, [up(idx_top), down(idx_bot)])
    end

    if n == 1
        return Tensor(S_name, [up(idx_top), down(idx_bot)])
    end

    # n >= 2: chain S^{idx_top}_{c₁} S^{c₁}_{c₂} ⋯ S^{c_{n-1}}_{idx_bot}
    internal = Symbol[]
    for _ in 1:(n - 1)
        c = fresh_index(used)
        push!(used, c)
        push!(internal, c)
    end

    factors = TensorExpr[]
    # First factor: S^{idx_top}_{c₁}
    push!(factors, Tensor(S_name, [up(idx_top), down(internal[1])]))
    # Middle factors: S^{c_i}_{c_{i+1}}
    for i in 1:(n - 2)
        push!(factors, Tensor(S_name, [up(internal[i]), down(internal[i + 1])]))
    end
    # Last factor: S^{c_{n-1}}_{idx_bot}
    push!(factors, Tensor(S_name, [up(internal[end]), down(idx_bot)]))

    tproduct(1 // 1, factors)
end

"""
    sqrt_matrix_identity(bs::BimetricSetup;
                          registry::TensorRegistry=current_registry()) -> TensorExpr

Return the defining identity for the matrix square root S = √(g⁻¹f):

    S^a_c S^c_b - g^{ac} f_{cb} = 0

Returns the LHS as a TensorExpr with free indices (a, b). This is the
fundamental algebraic identity: S² = g⁻¹ f.

Ground truth: Hassan & Rosen, JHEP 02 (2012) 126, Sec 2.
"""
function sqrt_matrix_identity(bs::BimetricSetup;
                               registry::TensorRegistry=current_registry())
    S_name = Symbol(:S_, bs.metric_g, :_, bs.metric_f)

    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used)

    # S^a_c S^c_b  (= S² with free indices a, b)
    S_ac = Tensor(S_name, [up(a), down(c)])
    S_cb = Tensor(S_name, [up(c), down(b)])
    S_squared = S_ac * S_cb

    # g^{ac} f_{cb}  (= g⁻¹ f with free indices a, b)
    g_up = Tensor(bs.metric_g, [up(a), up(c)])
    f_dn = Tensor(bs.metric_f, [down(c), down(b)])
    ginv_f = g_up * f_dn

    # LHS = S² - g⁻¹f
    S_squared - ginv_f
end

"""
    cayley_hamilton_S(bs::BimetricSetup, params::HassanRosenParams;
                       registry::TensorRegistry=current_registry()) -> TensorExpr

Return the Cayley-Hamilton identity for the 4×4 matrix S = √(g⁻¹f):

    S⁴ - e₁(S) S³ + e₂(S) S² - e₃(S) S + e₄(S) I = 0

Returns the LHS as a TensorExpr with free indices (a, b). The e_n are the
elementary symmetric polynomials of S, computed via `elementary_symmetric`.

By the Cayley-Hamilton theorem, every matrix satisfies its own characteristic
equation. For a 4×4 matrix, the characteristic polynomial is degree 4 with
coefficients given by the elementary symmetric polynomials of the eigenvalues.

Ground truth: Cayley-Hamilton theorem; Hassan & Rosen (2012) Sec 2.
"""
function cayley_hamilton_S(bs::BimetricSetup, params::HassanRosenParams;
                            registry::TensorRegistry=current_registry())
    S_name = Symbol(:S_, bs.metric_g, :_, bs.metric_f)

    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)

    # Build S^n for n = 0, 1, 2, 3, 4
    # Each call to _S_power_chain may consume indices from `used`
    S0 = _S_power_chain(S_name, 0, a, b, copy(used))  # δ^a_b
    S1 = _S_power_chain(S_name, 1, a, b, copy(used))  # S^a_b
    S2 = _S_power_chain(S_name, 2, a, b, copy(used))  # S^a_c S^c_b
    S3 = _S_power_chain(S_name, 3, a, b, copy(used))  # S^a_c S^c_d S^d_b
    S4 = _S_power_chain(S_name, 4, a, b, copy(used))  # S^a_c S^c_d S^d_e S^e_b

    # Compute e_n(S) — these are scalars (fully contracted)
    e1 = elementary_symmetric(1, S_name; registry=registry)
    e2 = elementary_symmetric(2, S_name; registry=registry)
    e3 = elementary_symmetric(3, S_name; registry=registry)
    e4 = elementary_symmetric(4, S_name; registry=registry)

    # Cayley-Hamilton: S⁴ - e₁ S³ + e₂ S² - e₃ S + e₄ I = 0
    terms = TensorExpr[
        S4,                                                    # + S⁴
        tproduct(-1 // 1, TensorExpr[e1, S3]),                # - e₁ S³
        tproduct(1 // 1, TensorExpr[e2, S2]),                 # + e₂ S²
        tproduct(-1 // 1, TensorExpr[e3, S1]),                # - e₃ S
        tproduct(1 // 1, TensorExpr[e4, S0]),                 # + e₄ I
    ]

    tsum(terms)
end

"""
    register_sqrt_rules!(reg::TensorRegistry, bs::BimetricSetup;
                          max_power::Int=4) -> Vector{RewriteRule}

Register rewrite rules that reduce powers of S using the Cayley-Hamilton theorem.

In d=4, the Cayley-Hamilton identity gives:

    S⁴ = e₁ S³ - e₂ S² + e₃ S - e₄ I

This function creates a functional rewrite rule that detects a chain of
`max_power` or more S factors contracted in sequence and replaces them using
the identity above.

Returns the vector of rules created. The rules are also added to the registry.

Ground truth: Cayley-Hamilton theorem for 4×4 matrices.
"""
function register_sqrt_rules!(reg::TensorRegistry, bs::BimetricSetup;
                                max_power::Int=4)
    S_name = Symbol(:S_, bs.metric_g, :_, bs.metric_f)

    # Functional rule: detect S^{max_power} chains in TProduct factors
    # and replace using Cayley-Hamilton reduction.
    rule = RewriteRule(
        function(expr::TensorExpr)
            # Match a TProduct containing a chain of max_power S factors
            expr isa TProduct || return false
            chain = _find_S_chain(expr, S_name, max_power)
            return chain !== nothing
        end,
        function(expr::TensorExpr)
            _reduce_S_chain(expr, S_name, max_power, bs, reg)
        end
    )

    push!(reg.rules, rule)
    RewriteRule[rule]
end

"""Find a chain of `n` S-factors contracted in sequence within a TProduct."""
function _find_S_chain(expr::TProduct, S_name::Symbol, n::Int)
    # Collect S-factor indices
    s_indices = Int[]
    for (i, f) in enumerate(expr.factors)
        if f isa Tensor && f.name == S_name
            push!(s_indices, i)
        end
    end

    length(s_indices) < n && return nothing

    # Check all length-n subsequences for chain contraction:
    # S^{a₁}_{b₁}, S^{a₂}_{b₂}, ... where b_i == a_{i+1}
    for start_idx in 1:(length(s_indices) - n + 1)
        candidate = s_indices[start_idx:start_idx + n - 1]
        is_chain = true
        for k in 1:(n - 1)
            f_k = expr.factors[candidate[k]]
            f_next = expr.factors[candidate[k + 1]]
            # Check: down index of f_k matches up index of f_next
            if !(f_k.indices[2].name == f_next.indices[1].name &&
                 f_k.indices[2].position == Down &&
                 f_next.indices[1].position == Up)
                is_chain = false
                break
            end
        end
        if is_chain
            return candidate
        end
    end
    nothing
end

"""Replace the first S^n chain in a TProduct using Cayley-Hamilton."""
function _reduce_S_chain(expr::TProduct, S_name::Symbol, n::Int,
                          bs::BimetricSetup, reg::TensorRegistry)
    chain = _find_S_chain(expr, S_name, n)
    chain === nothing && return expr

    # Extract the free indices of the chain: up index of first, down index of last
    first_S = expr.factors[chain[1]]
    last_S = expr.factors[chain[end]]
    idx_top = first_S.indices[1].name   # the upper free index
    idx_bot = last_S.indices[2].name    # the lower free index

    # Collect all index names in the expression to avoid clashes
    all_idxs = indices(expr)
    used = Set{Symbol}(idx.name for idx in all_idxs)

    # Build the Cayley-Hamilton replacement: S⁴ → e₁S³ - e₂S² + e₃S - e₄I
    e1 = elementary_symmetric(1, S_name; registry=reg)
    e2 = elementary_symmetric(2, S_name; registry=reg)
    e3 = elementary_symmetric(3, S_name; registry=reg)
    e4 = elementary_symmetric(4, S_name; registry=reg)

    S0 = _S_power_chain(S_name, 0, idx_top, idx_bot, copy(used))
    S1 = _S_power_chain(S_name, 1, idx_top, idx_bot, copy(used))
    S2 = _S_power_chain(S_name, 2, idx_top, idx_bot, copy(used))
    S3 = _S_power_chain(S_name, 3, idx_top, idx_bot, copy(used))

    replacement = tsum(TensorExpr[
        tproduct(1 // 1, TensorExpr[e1, S3]),        # + e₁ S³
        tproduct(-1 // 1, TensorExpr[e2, S2]),       # - e₂ S²
        tproduct(1 // 1, TensorExpr[e3, S1]),         # + e₃ S
        tproduct(-1 // 1, TensorExpr[e4, S0]),        # - e₄ I
    ])

    # Rebuild the TProduct: replace the chain factors with the replacement
    other_factors = TensorExpr[]
    chain_set = Set(chain)
    for (i, f) in enumerate(expr.factors)
        if i == chain[1]
            push!(other_factors, replacement)
        elseif i in chain_set
            # skip — these are part of the chain we replaced
        else
            push!(other_factors, f)
        end
    end

    tproduct(expr.scalar, other_factors)
end

"""
    sqrt_matrix_variation(bs::BimetricSetup;
                           registry::TensorRegistry=current_registry()) -> TensorExpr

Return the Sylvester equation for the variation of the matrix square root:

    S δS + δS S = δ(g⁻¹f) = g⁻¹(δf - δg · g⁻¹f)

Equivalently, using S² = g⁻¹f:

    S^a_c (δS)^c_b + (δS)^a_c S^c_b - g^{ac}(δf_{cb} - δg_{cd} g^{de} f_{eb}) = 0

Returns the LHS (which = 0) as a TensorExpr with free indices (a, b).
The variation fields δg, δf are represented as tensors named `delta_g`, `delta_f`.

The Sylvester equation SX + XS = C (with C = δ(g⁻¹f)) has a unique solution
when S has no pair of eigenvalues that sum to zero, which is generically true.

Ground truth: Hassan & Rosen (2012) Sec 2;
             de Rham, Living Rev. Rel. 17, 7 (2014) Sec 8.1.
"""
function sqrt_matrix_variation(bs::BimetricSetup;
                                registry::TensorRegistry=current_registry())
    S_name = Symbol(:S_, bs.metric_g, :_, bs.metric_f)
    dS_name = Symbol(:deltaS_, bs.metric_g, :_, bs.metric_f)

    # Register δS if not already present
    if !has_tensor(registry, dS_name)
        register_tensor!(registry, TensorProperties(
            name=dS_name, manifold=bs.manifold, rank=(1, 1),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_variation => true,
                :of_tensor => S_name)))
    end

    # Register the variation tensors δg, δf
    dg_name = Symbol(:delta_, bs.metric_g)
    df_name = Symbol(:delta_, bs.metric_f)
    for (dn, desc) in [(dg_name, "variation of $(bs.metric_g)"),
                         (df_name, "variation of $(bs.metric_f)")]
        if !has_tensor(registry, dn)
            register_tensor!(registry, TensorProperties(
                name=dn, manifold=bs.manifold, rank=(0, 2),
                symmetries=SymmetrySpec[Symmetric(1, 2)],
                options=Dict{Symbol,Any}(
                    :is_variation => true,
                    :description => desc)))
        end
    end

    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used); push!(used, d)
    e_idx = fresh_index(used)

    # LHS term 1: S^a_c (δS)^c_b
    S_ac = Tensor(S_name, [up(a), down(c)])
    dS_cb = Tensor(dS_name, [up(c), down(b)])
    term1 = S_ac * dS_cb

    # LHS term 2: (δS)^a_c S^c_b
    dS_ac = Tensor(dS_name, [up(a), down(c)])
    S_cb = Tensor(S_name, [up(c), down(b)])
    term2 = dS_ac * S_cb

    # RHS: g^{ac}(δf_{cb} - δg_{cd} g^{de} f_{eb})
    # = g^{ac} δf_{cb} - g^{ac} δg_{cd} g^{de} f_{eb}
    g_up_ac = Tensor(bs.metric_g, [up(a), up(c)])
    df_cb = Tensor(df_name, [down(c), down(b)])
    rhs_term1 = g_up_ac * df_cb                          # g^{ac} δf_{cb}

    g_up_ac2 = Tensor(bs.metric_g, [up(a), up(c)])
    dg_cd = Tensor(dg_name, [down(c), down(d)])
    g_up_de = Tensor(bs.metric_g, [up(d), up(e_idx)])
    f_eb = Tensor(bs.metric_f, [down(e_idx), down(b)])
    rhs_term2 = tproduct(1 // 1, TensorExpr[g_up_ac2, dg_cd, g_up_de, f_eb])

    # LHS - RHS = S δS + δS S - g⁻¹ δf + g⁻¹ δg g⁻¹ f = 0
    tsum(TensorExpr[term1, term2, -rhs_term1, rhs_term2])
end

# ── Y-tensors and field equations ─────────────────────────────────

"""
    y_tensor(n::Int, S_name::Symbol;
             registry::TensorRegistry=current_registry()) -> TensorExpr

Construct the n-th Y-tensor (cofactor polynomial) of the matrix square root S.

The Y-tensors are defined as:

    Y_0(S) = I                                    (identity)
    Y_1(S) = S - e₁(S) I
    Y_2(S) = S² - e₁(S) S + e₂(S) I
    Y_3(S) = S³ - e₁(S) S² + e₂(S) S - e₃(S) I

They satisfy the recursion: Y_{n+1}(S) = S · Y_n(S) + (-1)^{n+1} e_{n+1}(S) · I

The result is a mixed-index expression with one free upper index and one free
lower index: Y_n(S)^a_b.

Ground truth: Hassan & Rosen, JHEP 02 (2012) 126, Eq 2.7;
             de Rham, Living Rev. Rel. 17, 7 (2014) Sec 1.1.3 Eq (1.11).
"""
function y_tensor(n::Int, S_name::Symbol;
                  registry::TensorRegistry=current_registry())
    n in 0:3 || error("Y-tensor order must be 0-3, got $n")

    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)

    if n == 0
        # Y_0 = δ^a_b  (identity)
        return Tensor(:δ, [up(a), down(b)])
    end

    # Build explicit Y_n = Σ_{k=0}^{n} (-1)^{n-k} e_{n-k}(S) · S^k
    # i.e. Y_n = S^n - e_1 S^{n-1} + e_2 S^{n-2} - ... + (-1)^n e_n I
    terms = TensorExpr[]
    for k in n:-1:0
        # Coefficient: (-1)^{n-k} × e_{n-k}(S)
        sign = iseven(n - k) ? 1 : -1
        en_k = elementary_symmetric(n - k, S_name; registry=registry)
        Sk = _S_power_chain(S_name, k, a, b, copy(used))

        if n - k == 0
            # e_0 = 1, so coefficient is just the sign
            if sign == 1
                push!(terms, Sk)
            else
                push!(terms, tproduct(-1 // 1, TensorExpr[Sk]))
            end
        else
            push!(terms, tproduct(sign // 1, TensorExpr[en_k, Sk]))
        end
    end

    length(terms) == 1 ? terms[1] : tsum(terms)
end

"""
    interaction_tensor_g(bs::BimetricSetup, params::HassanRosenParams;
                          registry::TensorRegistry=current_registry()) -> TensorExpr

Construct the g-sector interaction tensor V_{ab}^{(g)} from the Hassan-Rosen
bimetric potential.

    V_{ab}^{(g)} = -Σ_{n=0}^{3} (-1)^n β_n g_{ac} Y_n(S)^c_b

The result is a rank-(0,2) expression with both free indices down: V^{(g)}_{ab}.

Ground truth: Hassan & Rosen, JHEP 02 (2012) 126, Eq 2.8;
             de Rham, Living Rev. Rel. 17, 7 (2014) Sec 8.1.
"""
function interaction_tensor_g(bs::BimetricSetup, params::HassanRosenParams;
                               registry::TensorRegistry=current_registry())
    S_name = Symbol(:S_, bs.metric_g, :_, bs.metric_f)

    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)

    # V^{(g)}_{ab} = -Σ_{n=0}^{3} (-1)^n β_n g_{ac} Y_n(S)^c_b
    terms = TensorExpr[]
    for n in 0:3
        coeff = params.beta[n + 1]  # β_n (0-indexed in physics, 1-indexed in tuple)
        (coeff isa Number && coeff == 0) && continue

        # Overall sign: -(-1)^n = (-1)^{n+1}
        overall_sign = iseven(n) ? -1 : 1

        # Build Y_n(S)^c_b with specified contraction index c and free index b
        Yn_cb = _build_y_tensor_with_indices(n, S_name, c, b, copy(used), registry)

        # g_{ac}
        g_ac = Tensor(bs.metric_g, [down(a), down(c)])

        # Combine: overall_sign * β_n * g_{ac} * Y_n^c_b
        if coeff isa Number && coeff == 1
            push!(terms, tproduct(overall_sign // 1, TensorExpr[g_ac, Yn_cb]))
        else
            push!(terms, tproduct(overall_sign // 1, TensorExpr[TScalar(coeff), g_ac, Yn_cb]))
        end
    end

    isempty(terms) && return TScalar(0 // 1)
    length(terms) == 1 ? terms[1] : tsum(terms)
end

"""
    _build_y_tensor_with_indices(n, S_name, idx_top, idx_bot, used, registry) -> TensorExpr

Build Y_n(S)^{idx_top}_{idx_bot} with specified free indices.
Internal helper for interaction_tensor_g and interaction_tensor_f.
"""
function _build_y_tensor_with_indices(n::Int, S_name::Symbol, idx_top::Symbol,
                                       idx_bot::Symbol, used::Set{Symbol},
                                       registry::TensorRegistry)
    if n == 0
        return Tensor(:δ, [up(idx_top), down(idx_bot)])
    end

    # Y_n = S^n - e_1 S^{n-1} + e_2 S^{n-2} - ... + (-1)^n e_n I
    terms = TensorExpr[]
    for k in n:-1:0
        sign = iseven(n - k) ? 1 : -1
        en_k = elementary_symmetric(n - k, S_name; registry=registry)
        Sk = _S_power_chain(S_name, k, idx_top, idx_bot, copy(used))

        if n - k == 0
            # e_0 = 1
            if sign == 1
                push!(terms, Sk)
            else
                push!(terms, tproduct(-1 // 1, TensorExpr[Sk]))
            end
        else
            push!(terms, tproduct(sign // 1, TensorExpr[en_k, Sk]))
        end
    end

    length(terms) == 1 ? terms[1] : tsum(terms)
end

"""
    interaction_tensor_f(bs::BimetricSetup, params::HassanRosenParams;
                          registry::TensorRegistry=current_registry()) -> TensorExpr

Construct the f-sector interaction tensor V_{ab}^{(f)} from the Hassan-Rosen
bimetric potential.

    V_{ab}^{(f)} = -Σ_{n=0}^{3} (-1)^n β_{4-n} f_{ac} [Y_n(γ)]^c_b

where γ = S⁻¹ = √(f⁻¹g) is the inverse matrix square root.

The inverse square root tensor γ^a_b is automatically registered (as
`Sinv_<g>_<f>`) if not already present in the registry.

The Y-tensors Y_n(γ) are built from powers of γ and its elementary symmetric
polynomials, using the same `_S_power_chain` infrastructure as Y_n(S).

The result is a rank-(0,2) expression with both free indices down: V^{(f)}_{ab}.

Ground truth: Hassan & Rosen, JHEP 02 (2012) 126, Eq 2.9;
             de Rham, Living Rev. Rel. 17, 7 (2014) Sec 8.1;
             Baccetta, Martin-Moruno & Visser, JHEP 1208 (2012) 148, Eq 2.16.
"""
function interaction_tensor_f(bs::BimetricSetup, params::HassanRosenParams;
                               registry::TensorRegistry=current_registry())
    # γ = S⁻¹ = √(f⁻¹g), the inverse matrix square root
    Sinv_name = Symbol(:Sinv_, bs.metric_g, :_, bs.metric_f)

    # Register γ if not already present
    if !has_tensor(registry, Sinv_name)
        register_tensor!(registry, TensorProperties(
            name=Sinv_name, manifold=bs.manifold, rank=(1, 1),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_sqrt_matrix_inv => true,
                :bimetric => true,
                :metric_g => bs.metric_g,
                :metric_f => bs.metric_f)))
    end

    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)

    # V^{(f)}_{ab} = -Σ_{n=0}^{3} (-1)^n β_{4-n} f_{ac} Y_n(γ)^c_b
    terms = TensorExpr[]
    for n in 0:3
        coeff = params.beta[5 - n]  # β_{4-n}: n=0 → β_4, n=1 → β_3, n=2 → β_2, n=3 → β_1
        (coeff isa Number && coeff == 0) && continue

        # Overall sign: -(-1)^n = (-1)^{n+1}
        overall_sign = iseven(n) ? -1 : 1

        # Build Y_n(γ)^c_b
        Yn_cb = _build_y_tensor_with_indices(n, Sinv_name, c, b, copy(used), registry)

        # f_{ac}
        f_ac = Tensor(bs.metric_f, [down(a), down(c)])

        # Combine: overall_sign * β_{4-n} * f_{ac} * Y_n(γ)^c_b
        if coeff isa Number && coeff == 1
            push!(terms, tproduct(overall_sign // 1, TensorExpr[f_ac, Yn_cb]))
        else
            push!(terms, tproduct(overall_sign // 1, TensorExpr[TScalar(coeff), f_ac, Yn_cb]))
        end
    end

    isempty(terms) && return TScalar(0 // 1)
    length(terms) == 1 ? terms[1] : tsum(terms)
end

"""
    bimetric_eom_g(bs::BimetricSetup, params::HassanRosenParams;
                    registry::TensorRegistry=current_registry()) -> TensorExpr

Construct the g-sector bimetric field equation (LHS = 0 on-shell):

    G_{ab}[g] + m² V_{ab}^{(g)} = 0

Returns the LHS as a TensorExpr with both free indices down.
The factor m² is taken from params.m_sq.

Ground truth: Hassan & Rosen, JHEP 02 (2012) 126, Eq 2.8.
"""
function bimetric_eom_g(bs::BimetricSetup, params::HassanRosenParams;
                         registry::TensorRegistry=current_registry())
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used)

    # G_{ab}[g]
    ein_g_name = bs.curvature_g[:einstein]
    G_g = Tensor(ein_g_name, [down(a), down(b)])

    # V^{(g)}_{ab}
    V_g = interaction_tensor_g(bs, params; registry=registry)

    # m² * V^{(g)}
    m2 = params.m_sq
    if m2 isa Number && m2 == 1
        m2_V_g = V_g
    elseif m2 isa Number && m2 == 0
        return G_g
    else
        m2_V_g = tproduct(1 // 1, TensorExpr[TScalar(m2), V_g])
    end

    # G_{ab}[g] + m² V^{(g)}_{ab}
    G_g + m2_V_g
end

"""
    bimetric_eom_f(bs::BimetricSetup, params::HassanRosenParams;
                    registry::TensorRegistry=current_registry()) -> TensorExpr

Construct the f-sector bimetric field equation (LHS = 0 on-shell):

    G_{ab}[f] + m² V_{ab}^{(f)} = 0

Returns the LHS as a TensorExpr with both free indices down.
The factor m² is taken from params.m_sq.

Ground truth: Hassan & Rosen, JHEP 02 (2012) 126, Eq 2.9.
"""
function bimetric_eom_f(bs::BimetricSetup, params::HassanRosenParams;
                         registry::TensorRegistry=current_registry())
    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used)

    # G_{ab}[f]
    ein_f_name = bs.curvature_f[:einstein]
    G_f = Tensor(ein_f_name, [down(a), down(b)])

    # V^{(f)}_{ab}
    V_f = interaction_tensor_f(bs, params; registry=registry)

    # m² * V^{(f)}
    m2 = params.m_sq
    if m2 isa Number && m2 == 1
        m2_V_f = V_f
    elseif m2 isa Number && m2 == 0
        return G_f
    else
        m2_V_f = tproduct(1 // 1, TensorExpr[TScalar(m2), V_f])
    end

    # G_{ab}[f] + m² V^{(f)}_{ab}
    G_f + m2_V_f
end
