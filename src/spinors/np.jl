# Newman-Penrose null tetrad: l^a, n^a, m^a, mbar^a.
#
# The four null vectors satisfy:
#   l_a n^a = -1,  m_a mbar^a = 1  (normalization)
#   l_a l^a = n_a n^a = m_a m^a = 0  (null)
#   all other inner products zero
#
# Completeness: g^{ab} = -l^a n^b - n^a l^b + m^a mbar^b + mbar^a m^b
#
# Reference: Newman & Penrose, J. Math. Phys. 3, 566 (1962), Eq 2.1.
#            Penrose & Rindler Vol 1 (1984), Eq 4.5.11.

"""
    define_null_tetrad!(reg::TensorRegistry; manifold::Symbol=:M4, metric::Symbol=:g)

Register the Newman-Penrose null tetrad vectors `l^a`, `n^a`, `m^a`, `mbar^a`
and their fundamental contraction/normalization rules.

The tetrad vectors are registered as rank-(1,0) tensors on the Tangent bundle.
Function-based rules handle null conditions, orthogonality, and normalization.

# Registered tensors
- `:np_l` — real null vector l^a
- `:np_n` — real null vector n^a
- `:np_m` — complex null vector m^a
- `:np_mbar` — complex null vector m̄^a (conjugate of m)

# Contraction rules (via function-based RewriteRule)
- Null: `l_a l^a = 0`, `n_a n^a = 0`, `m_a m^a = 0`, `mbar_a mbar^a = 0`
- Normalization: `l_a n^a = -1`, `m_a mbar^a = 1`
- Orthogonality: all other pairs = 0

# Reference
Newman & Penrose, J. Math. Phys. 3, 566 (1962).
"""
function define_null_tetrad!(reg::TensorRegistry; manifold::Symbol=:M4, metric::Symbol=:g)
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")

    # Register the four tetrad vectors
    for (name, desc) in [(:np_l, "real null l"),
                          (:np_n, "real null n"),
                          (:np_m, "complex null m"),
                          (:np_mbar, "complex null mbar")]
        if !has_tensor(reg, name)
            register_tensor!(reg, TensorProperties(
                name=name, manifold=manifold, rank=(1, 0),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}(
                    :is_null_tetrad => true,
                    :description => desc)))
        end
    end

    # Inner product table: (v1, v2) -> scalar value
    # Only pairs where contraction is nonzero are listed; all others are zero.
    np_names = Set([:np_l, :np_n, :np_m, :np_mbar])
    nonzero_pairs = Dict{Tuple{Symbol,Symbol}, Rational{Int}}(
        (:np_l, :np_n) => -1 // 1,
        (:np_n, :np_l) => -1 // 1,
        (:np_m, :np_mbar) => 1 // 1,
        (:np_mbar, :np_m) => 1 // 1,
    )

    # Register a single function-based rule that detects any contracted
    # pair of NP tetrad vectors in a product and replaces with the scalar.
    rule = RewriteRule(
        function(expr)
            expr isa TProduct || return false
            _find_np_contraction(expr, np_names) !== nothing
        end,
        function(expr)
            info = _find_np_contraction(expr, np_names)
            info === nothing && return expr
            (i1, i2, name1, name2) = info

            val = get(nonzero_pairs, (name1, name2), 0 // 1)

            # Remove the two NP factors from the product
            remaining = TensorExpr[]
            for (k, fk) in enumerate(expr.factors)
                k == i1 && continue
                k == i2 && continue
                push!(remaining, fk)
            end

            if val == 0 // 1
                return TScalar(0)
            elseif isempty(remaining)
                return TScalar(expr.scalar * val)
            else
                return tproduct(expr.scalar * val, remaining)
            end
        end
    )
    register_rule!(reg, rule)

    nothing
end

"""
Find a pair of NP tetrad vectors in a product that share a contracted Tangent
index (one Up, one Down). Returns `(idx1, idx2, name1, name2)` or `nothing`.
name1 is the vector with the Down index, name2 has the Up index.
"""
function _find_np_contraction(p::TProduct, np_names::Set{Symbol})
    factors = p.factors
    np_positions = Int[]
    for (i, f) in enumerate(factors)
        f isa Tensor && f.name in np_names && push!(np_positions, i)
    end
    length(np_positions) < 2 && return nothing

    for a in 1:length(np_positions)
        for b in (a+1):length(np_positions)
            i1 = np_positions[a]
            i2 = np_positions[b]
            t1 = factors[i1]::Tensor
            t2 = factors[i2]::Tensor

            # Each NP vector has exactly one Tangent index
            length(t1.indices) == 1 && length(t2.indices) == 1 || continue
            idx1 = t1.indices[1]
            idx2 = t2.indices[1]

            # Check if contracted (same name, opposite positions, Tangent vbundle)
            if idx1.vbundle == :Tangent && idx2.vbundle == :Tangent &&
               idx1.name == idx2.name && idx1.position != idx2.position
                # Return with Down-index vector first (convention for inner product table)
                if idx1.position == Down
                    return (i1, i2, t1.name, t2.name)
                else
                    return (i1, i2, t2.name, t1.name)
                end
            end
        end
    end
    nothing
end

# ── Weyl scalars Psi_0 ... Psi_4 ──────────────────────────────────────────

"""
    weyl_scalar(n::Int; registry::TensorRegistry=current_registry()) -> TensorExpr

Return the NP Weyl scalar Psi_n (n=0..4) as a contraction of the Weyl
tensor with null tetrad vectors.

Definitions (NP 1962, Eq 4.3):
- Psi_0 = C_{abcd} l^a m^b l^c m^d
- Psi_1 = C_{abcd} l^a n^b l^c m^d
- Psi_2 = C_{abcd} l^a m^b mbar^c n^d
- Psi_3 = C_{abcd} l^a n^b mbar^c n^d
- Psi_4 = C_{abcd} n^a mbar^b n^c mbar^d
"""
function weyl_scalar(n::Int; registry::TensorRegistry=current_registry())
    0 <= n <= 4 || error("Weyl scalar index must be 0-4, got $n")

    C = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])

    # Tetrad vectors for each scalar (contracted with the 4 Weyl indices)
    tetrad_defs = [
        # Psi_0: l^a m^b l^c m^d
        (:np_l, :np_m, :np_l, :np_m),
        # Psi_1: l^a n^b l^c m^d
        (:np_l, :np_n, :np_l, :np_m),
        # Psi_2: l^a m^b mbar^c n^d
        (:np_l, :np_m, :np_mbar, :np_n),
        # Psi_3: l^a n^b mbar^c n^d
        (:np_l, :np_n, :np_mbar, :np_n),
        # Psi_4: n^a mbar^b n^c mbar^d
        (:np_n, :np_mbar, :np_n, :np_mbar),
    ]

    (v1, v2, v3, v4) = tetrad_defs[n + 1]
    idx_names = [:a, :b, :c, :d]
    factors = TensorExpr[
        C,
        Tensor(v1, [up(idx_names[1])]),
        Tensor(v2, [up(idx_names[2])]),
        Tensor(v3, [up(idx_names[3])]),
        Tensor(v4, [up(idx_names[4])]),
    ]
    tproduct(1 // 1, factors)
end

"""
    weyl_scalars(; registry::TensorRegistry=current_registry()) -> Vector{TensorExpr}

Return all 5 NP Weyl scalars [Psi_0, ..., Psi_4] as TensorExprs.
"""
function weyl_scalars(; registry::TensorRegistry=current_registry())
    [weyl_scalar(n; registry=registry) for n in 0:4]
end

# ── Ricci scalars Phi_{ij} and Lambda ─────────────────────────────────────

"""
    ricci_scalar_np(i::Int, j::Int; registry::TensorRegistry=current_registry()) -> TensorExpr

Return the NP Ricci scalar Phi_{ij} (i,j in 0..2) as a contraction of the
Ricci tensor with null tetrad vectors.

Definitions (NP 1962, Eq 4.4):
- Phi_{00} = -(1/2) R_{ab} l^a l^b
- Phi_{01} = -(1/2) R_{ab} l^a m^b
- Phi_{02} = -(1/2) R_{ab} m^a m^b
- Phi_{10} = -(1/2) R_{ab} l^a mbar^b  (= conj Phi_{01})
- Phi_{11} = -(1/4) R_{ab} (l^a n^b + m^a mbar^b)
- Phi_{12} = -(1/2) R_{ab} n^a m^b
- Phi_{20} = -(1/2) R_{ab} mbar^a mbar^b  (= conj Phi_{02})
- Phi_{21} = -(1/2) R_{ab} n^a mbar^b  (= conj Phi_{12})
- Phi_{22} = -(1/2) R_{ab} n^a n^b
"""
function ricci_scalar_np(i::Int, j::Int; registry::TensorRegistry=current_registry())
    0 <= i <= 2 && 0 <= j <= 2 || error("Ricci scalar indices must be 0-2, got ($i,$j)")

    R = Tensor(:Ric, [down(:a), down(:b)])

    # Phi_{11} is special: -(1/4)(R_{ab} l^a n^b + R_{ab} m^a mbar^b)
    if i == 1 && j == 1
        term1 = tproduct(-1 // 4, TensorExpr[
            R,
            Tensor(:np_l, [up(:a)]),
            Tensor(:np_n, [up(:b)])
        ])
        term2 = tproduct(-1 // 4, TensorExpr[
            Tensor(:Ric, [down(:c), down(:d)]),
            Tensor(:np_m, [up(:c)]),
            Tensor(:np_mbar, [up(:d)])
        ])
        return tsum(TensorExpr[term1, term2])
    end

    # Map (i,j) to tetrad vector pair for R_{ab} v1^a v2^b
    tetrad_map = Dict(
        (0, 0) => (:np_l, :np_l),
        (0, 1) => (:np_l, :np_m),
        (0, 2) => (:np_m, :np_m),
        (1, 0) => (:np_l, :np_mbar),
        (1, 2) => (:np_n, :np_m),
        (2, 0) => (:np_mbar, :np_mbar),
        (2, 1) => (:np_n, :np_mbar),
        (2, 2) => (:np_n, :np_n),
    )

    (v1, v2) = tetrad_map[(i, j)]
    tproduct(-1 // 2, TensorExpr[
        R,
        Tensor(v1, [up(:a)]),
        Tensor(v2, [up(:b)])
    ])
end

"""
    np_lambda(; registry::TensorRegistry=current_registry()) -> TensorExpr

Return the NP scalar Lambda = R/24 (equivalent to the scalar curvature spinor).
"""
function np_lambda(; registry::TensorRegistry=current_registry())
    tproduct(1 // 24, TensorExpr[Tensor(:RicScalar, TIndex[])])
end

# ── 12 NP spin coefficients ───────────────────────────────────────────────

# Each spin coefficient is a scalar formed from tetrad projections of the
# covariant derivative of a tetrad vector.
# General form: v1^a v2^b nabla_b v3_a  (with appropriate lowering)
#
# Reference: Newman & Penrose, J. Math. Phys. 3, 566 (1962), Eq 4.2.

"""
    spin_coefficient(name::Symbol; covd_name::Symbol=:D,
                     registry::TensorRegistry=current_registry()) -> TensorExpr

Return the NP spin coefficient `name` as a TensorExpr.

Available names: `:kappa`, `:sigma_np`, `:lambda_np`, `:nu_np`, `:rho_np`,
`:mu_np`, `:tau_np`, `:pi_np`, `:epsilon_np`, `:gamma_np`, `:alpha_np`, `:beta_np`.

(Names suffixed with `_np` to avoid conflicts with common Julia symbols.)

Definitions (NP 1962, Eq 4.2):
- kappa    = m^a l^b nabla_b l_a
- sigma_np = m^a m^b nabla_b l_a
- rho_np   = m^a mbar^b nabla_b l_a
- tau_np   = m^a n^b nabla_b l_a
- nu_np    = mbar^a n^b nabla_b n_a
- lambda_np = mbar^a mbar^b nabla_b n_a
- mu_np    = mbar^a m^b nabla_b n_a
- pi_np    = mbar^a l^b nabla_b n_a
- epsilon_np = (1/2)(n^a l^b nabla_b l_a - mbar^a l^b nabla_b m_a)
- gamma_np   = (1/2)(n^a n^b nabla_b l_a - mbar^a n^b nabla_b m_a)
- alpha_np   = (1/2)(n^a mbar^b nabla_b l_a - mbar^a mbar^b nabla_b m_a)
- beta_np    = (1/2)(n^a m^b nabla_b l_a - mbar^a m^b nabla_b m_a)
"""
function spin_coefficient(name::Symbol; covd_name::Symbol=:D,
                          registry::TensorRegistry=current_registry())
    # Helper: build v1^a v2^b nabla_b v3_a
    function _sc_term(v1::Symbol, v2::Symbol, v3::Symbol)
        tproduct(1 // 1, TensorExpr[
            Tensor(v1, [up(:a)]),
            Tensor(v2, [up(:b)]),
            TDeriv(down(:b), Tensor(v3, [down(:a)]), covd_name)
        ])
    end

    # Simple spin coefficients: v1^a v2^b nabla_b v3_a
    simple = Dict{Symbol, Tuple{Symbol, Symbol, Symbol}}(
        :kappa     => (:np_m, :np_l, :np_l),
        :sigma_np  => (:np_m, :np_m, :np_l),
        :rho_np    => (:np_m, :np_mbar, :np_l),
        :tau_np    => (:np_m, :np_n, :np_l),
        :nu_np     => (:np_mbar, :np_n, :np_n),
        :lambda_np => (:np_mbar, :np_mbar, :np_n),
        :mu_np     => (:np_mbar, :np_m, :np_n),
        :pi_np     => (:np_mbar, :np_l, :np_n),
    )

    if haskey(simple, name)
        (v1, v2, v3) = simple[name]
        return _sc_term(v1, v2, v3)
    end

    # Compound spin coefficients: (1/2)(term1 - term2)
    compound = Dict{Symbol, Tuple{Tuple{Symbol,Symbol,Symbol}, Tuple{Symbol,Symbol,Symbol}}}(
        :epsilon_np => ((:np_n, :np_l, :np_l),     (:np_mbar, :np_l, :np_m)),
        :gamma_np   => ((:np_n, :np_n, :np_l),     (:np_mbar, :np_n, :np_m)),
        :alpha_np   => ((:np_n, :np_mbar, :np_l),  (:np_mbar, :np_mbar, :np_m)),
        :beta_np    => ((:np_n, :np_m, :np_l),     (:np_mbar, :np_m, :np_m)),
    )

    if haskey(compound, name)
        ((a1, a2, a3), (b1, b2, b3)) = compound[name]
        t1 = _sc_term(a1, a2, a3)
        t2 = _sc_term(b1, b2, b3)
        return tsum(TensorExpr[
            tproduct(1 // 2, TensorExpr[t1]),
            tproduct(-1 // 2, TensorExpr[t2])
        ])
    end

    error("Unknown spin coefficient: $name. Valid names: kappa, sigma_np, lambda_np, " *
          "nu_np, rho_np, mu_np, tau_np, pi_np, epsilon_np, gamma_np, alpha_np, beta_np")
end

"""
    all_spin_coefficients(; covd_name::Symbol=:D,
                           registry::TensorRegistry=current_registry()) -> Dict{Symbol, TensorExpr}

Return all 12 NP spin coefficients as a Dict.
"""
function all_spin_coefficients(; covd_name::Symbol=:D,
                                registry::TensorRegistry=current_registry())
    names = [:kappa, :sigma_np, :lambda_np, :nu_np, :rho_np, :mu_np,
             :tau_np, :pi_np, :epsilon_np, :gamma_np, :alpha_np, :beta_np]
    Dict(n => spin_coefficient(n; covd_name=covd_name, registry=registry) for n in names)
end

"""
    np_completeness(; registry::TensorRegistry=current_registry()) -> TensorExpr

Return the NP completeness relation as a TensorExpr:
  g^{ab} = -l^a n^b - n^a l^b + m^a mbar^b + mbar^a m^b

This is an identity, not a rule; use it for verification or substitution.
"""
function np_completeness(; registry::TensorRegistry=current_registry())
    la = Tensor(:np_l, [up(:a)])
    lb = Tensor(:np_l, [up(:b)])
    na = Tensor(:np_n, [up(:a)])
    nb = Tensor(:np_n, [up(:b)])
    ma = Tensor(:np_m, [up(:a)])
    mb = Tensor(:np_m, [up(:b)])
    mba = Tensor(:np_mbar, [up(:a)])
    mbb = Tensor(:np_mbar, [up(:b)])

    tsum(TensorExpr[
        tproduct(-1 // 1, TensorExpr[la, nb]),
        tproduct(-1 // 1, TensorExpr[na, lb]),
        tproduct(1 // 1, TensorExpr[ma, mbb]),
        tproduct(1 // 1, TensorExpr[mba, mb])
    ])
end
