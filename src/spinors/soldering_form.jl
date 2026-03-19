# Soldering form (Infeld-van der Waerden symbol) sigma^a_{AA'}.
#
# The soldering form provides the isomorphism between the tangent bundle
# and the tensor product of the two spin bundles:
#   V_a <-> V_{AA'} = sigma^a_{AA'} V_a
#
# Key identities:
#   sigma^a_{AA'} sigma_a^{BB'} = delta^B_A delta^{B'}_{A'}   (completeness)
#   sigma^a_{AA'} sigma^{b AA'}  = g^{ab}                     (metric reconstruction)
#   g_{ab} = sigma_a^{AA'} sigma_{b AA'}                       (metric from soldering)
#
# Reference: Penrose & Rindler, Spinors and Space-Time Vol 1 (1984),
#            Eqs 3.1.20, 3.1.23, 3.1.39.
#            Wald, General Relativity (1984), Ch 13, Eqs 13.1.17-13.1.22.

"""
    define_soldering_form!(reg::TensorRegistry; manifold::Symbol=:M4, name::Symbol=:sigma)

Register the soldering form (Infeld-van der Waerden symbol) `sigma^a_{AA'}`
and its fundamental contraction rules.

The soldering form has three index slots:
- Slot 1: spacetime (`:Tangent`), contravariant
- Slot 2: undotted spinor (`:SL2C`), covariant
- Slot 3: dotted spinor (`:SL2C_dot`), covariant

No slot symmetries (the two down indices live on *different* vbundles and are
thus not interchangeable).

Requires that spinor bundles and spin metrics are already registered via
[`define_spinor_bundles!`](@ref) and [`define_spin_metric!`](@ref).

# Contraction rules registered

1. **Completeness**: `sigma^a_{AA'} sigma_a^{BB'} = delta^B_A delta^{B'}_{A'}`
2. **Metric reconstruction**: `sigma^a_{AA'} sigma^{b AA'} = g^{ab}`

# Reference
Penrose & Rindler, *Spinors and Space-Time* Vol 1 (1984), Eqs 3.1.20, 3.1.23.
"""
function define_soldering_form!(reg::TensorRegistry;
                                manifold::Symbol=:M4,
                                name::Symbol=:sigma)
    has_vbundle(reg, :SL2C) || error("SL2C bundle not registered; call define_spinor_bundles! first")
    has_vbundle(reg, :SL2C_dot) || error("SL2C_dot bundle not registered; call define_spinor_bundles! first")
    has_tensor(reg, :eps_spin) || error("Spin metric not registered; call define_spin_metric! first")

    if !has_tensor(reg, name)
        register_tensor!(reg, TensorProperties(
            name=name, manifold=manifold, rank=(1, 2),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_soldering => true,
                :index_vbundles => [:Tangent, :SL2C, :SL2C_dot])))
    end

    # Store the soldering form name in the registry options for downstream use
    # (e.g., to_spinor_indices needs to know which tensor is the soldering form).
    # We store it keyed by manifold so multiple manifolds could each have one.
    if !haskey(reg.tensors[name].options, :soldering_manifold)
        reg.tensors[name].options[:soldering_manifold] = manifold
    end

    # ── Rule 1: Completeness ──────────────────────────────────────────────
    # sigma^a_{AA'} sigma_a^{BB'} -> delta^B_A delta^{B'}_{A'}
    #
    # Implemented as a structural rewrite rule: when a product contains two
    # sigma tensors contracted on their Tangent index, replace with deltas.
    _register_soldering_completeness_rule!(reg, name)

    # ── Rule 2: Metric reconstruction ────────────────────────────────────
    # sigma^a_{AA'} sigma^{b AA'} -> g^{ab}
    # (contracted on both spinor indices)
    _register_soldering_metric_rule!(reg, name, manifold)

    nothing
end

# ── Internal: register completeness contraction rule ──────────────────────

"""
Register the completeness rule:
  sigma^a_{AA'} sigma_a^{BB'} -> delta^B_A delta^{B'}_{A'}

Detects two sigma tensors in a product sharing a contracted Tangent index,
then replaces them with the product of undotted and dotted Kronecker deltas.
"""
function _register_soldering_completeness_rule!(reg::TensorRegistry, sigma_name::Symbol)
    rule = RewriteRule(
        function(expr)
            expr isa TProduct || return false
            _find_soldering_tangent_contraction(expr, sigma_name) !== nothing
        end,
        function(expr)
            info = _find_soldering_tangent_contraction(expr, sigma_name)
            info === nothing && return expr
            (i1, i2, sig1, sig2) = info
            # sig1 and sig2 are sigma tensors contracted on their Tangent index.
            # sig1 has indices [tangent, SL2C, SL2C_dot]
            # sig2 has indices [tangent, SL2C, SL2C_dot]
            # The Tangent indices are contracted (one Up, one Down).
            # Result: delta^B_A on SL2C * delta^{B'}_{A'} on SL2C_dot
            # where A,A' come from the sigma with tangent-down (or whichever)
            # and B,B' from the other.

            # Identify which sigma has tangent Up vs Down
            s1_tangent_pos = sig1.indices[1].position
            s2_tangent_pos = sig2.indices[1].position
            # The one with tangent Up is "sigma^a_{CD}" (canonical form)
            # The one with tangent Down is "sigma_a^{EF}" (raised spinor indices)
            if s1_tangent_pos == Up
                sigma_up = sig1  # sigma^a_{AA'}
                sigma_dn = sig2  # sigma_a^{BB'}
            else
                sigma_up = sig2
                sigma_dn = sig1
            end

            # sigma^a_{AA'}: spinor indices are Down
            idx_A  = sigma_up.indices[2]   # SL2C Down
            idx_Ap = sigma_up.indices[3]   # SL2C_dot Down
            # sigma_a^{BB'}: spinor indices are Up
            idx_B  = sigma_dn.indices[2]   # SL2C Up
            idx_Bp = sigma_dn.indices[3]   # SL2C_dot Up

            # Build delta^B_A (undotted) and delta^{B'}_{A'} (dotted)
            delta_undotted_name = get(reg.delta_cache, :SL2C, :delta_spin)
            delta_dotted_name = get(reg.delta_cache, :SL2C_dot, :delta_spin_dot)

            d1 = Tensor(delta_undotted_name, [
                TIndex(idx_B.name, Up, :SL2C),
                TIndex(idx_A.name, Down, :SL2C)
            ])
            d2 = Tensor(delta_dotted_name, [
                TIndex(idx_Bp.name, Up, :SL2C_dot),
                TIndex(idx_Ap.name, Down, :SL2C_dot)
            ])

            # Remove the two sigma factors, add the two deltas
            remaining = TensorExpr[]
            for (k, fk) in enumerate(expr.factors)
                k == i1 && continue
                k == i2 && continue
                push!(remaining, fk)
            end
            push!(remaining, d1)
            push!(remaining, d2)
            tproduct(expr.scalar, remaining)
        end
    )
    register_rule!(reg, rule)
end

# ── Internal: register metric reconstruction rule ────────────────────────

"""
Register the metric reconstruction rule:
  sigma^a_{AA'} sigma^{b AA'} -> g^{ab}

Detects two sigma tensors in a product with both spinor indices contracted
(one pair on SL2C, one pair on SL2C_dot), then replaces with the metric.
"""
function _register_soldering_metric_rule!(reg::TensorRegistry, sigma_name::Symbol, manifold::Symbol)
    rule = RewriteRule(
        function(expr)
            expr isa TProduct || return false
            _find_soldering_spinor_contraction(expr, sigma_name) !== nothing
        end,
        function(expr)
            info = _find_soldering_spinor_contraction(expr, sigma_name)
            info === nothing && return expr
            (i1, i2, sig1, sig2) = info

            # Both spinor index pairs are contracted.
            # The surviving indices are the two Tangent indices.
            tangent1 = sig1.indices[1]
            tangent2 = sig2.indices[1]

            metric_name = get(reg.metric_cache, manifold, :g)
            g = Tensor(metric_name, [
                TIndex(tangent1.name, tangent1.position, tangent1.vbundle),
                TIndex(tangent2.name, tangent2.position, tangent2.vbundle)
            ])

            remaining = TensorExpr[]
            for (k, fk) in enumerate(expr.factors)
                k == i1 && continue
                k == i2 && continue
                push!(remaining, fk)
            end
            push!(remaining, g)
            tproduct(expr.scalar, remaining)
        end
    )
    register_rule!(reg, rule)
end

# ── Internal: find two sigmas contracted on Tangent index ─────────────────

"""
Find a pair of sigma tensors in a product that share a contracted Tangent
index (one Up, one Down). Returns `(idx1, idx2, sigma1, sigma2)` or `nothing`.
"""
function _find_soldering_tangent_contraction(p::TProduct, sigma_name::Symbol)
    factors = p.factors
    sigma_positions = Int[]
    for (i, f) in enumerate(factors)
        f isa Tensor && f.name == sigma_name && push!(sigma_positions, i)
    end
    length(sigma_positions) < 2 && return nothing

    for a in 1:length(sigma_positions)
        for b in (a+1):length(sigma_positions)
            i1 = sigma_positions[a]
            i2 = sigma_positions[b]
            sig1 = factors[i1]::Tensor
            sig2 = factors[i2]::Tensor
            # Check if Tangent indices (slot 1) are contracted
            t1 = sig1.indices[1]
            t2 = sig2.indices[1]
            if t1.vbundle == :Tangent && t2.vbundle == :Tangent &&
               t1.name == t2.name && t1.position != t2.position
                return (i1, i2, sig1, sig2)
            end
        end
    end
    nothing
end

# ── Internal: find two sigmas contracted on both spinor indices ───────────

"""
Find a pair of sigma tensors in a product where both spinor index pairs
(SL2C and SL2C_dot) are contracted. Returns `(idx1, idx2, sigma1, sigma2)`
or `nothing`.
"""
function _find_soldering_spinor_contraction(p::TProduct, sigma_name::Symbol)
    factors = p.factors
    sigma_positions = Int[]
    for (i, f) in enumerate(factors)
        f isa Tensor && f.name == sigma_name && push!(sigma_positions, i)
    end
    length(sigma_positions) < 2 && return nothing

    for a in 1:length(sigma_positions)
        for b in (a+1):length(sigma_positions)
            i1 = sigma_positions[a]
            i2 = sigma_positions[b]
            sig1 = factors[i1]::Tensor
            sig2 = factors[i2]::Tensor
            # Check SL2C contraction (slot 2)
            s1 = sig1.indices[2]
            s2 = sig2.indices[2]
            sl2c_contracted = (s1.vbundle == :SL2C && s2.vbundle == :SL2C &&
                               s1.name == s2.name && s1.position != s2.position)
            # Check SL2C_dot contraction (slot 3)
            d1 = sig1.indices[3]
            d2 = sig2.indices[3]
            sl2c_dot_contracted = (d1.vbundle == :SL2C_dot && d2.vbundle == :SL2C_dot &&
                                   d1.name == d2.name && d1.position != d2.position)
            if sl2c_contracted && sl2c_dot_contracted
                return (i1, i2, sig1, sig2)
            end
        end
    end
    nothing
end

# ── Tensor-to-spinor conversion ──────────────────────────────────────────

"""
    to_spinor_indices(expr::TensorExpr, reg::TensorRegistry) -> TensorExpr

Convert tensor (Tangent) indices to spinor index pairs using the soldering
form. Each Tangent index `a` is replaced by a pair `(A, A')` via insertion
of `sigma^a_{AA'}`.

For a vector `V_a`:
  `to_spinor_indices(V_a)` -> `sigma^a_{AA'} V_a`  (which contracts to `V_{AA'}`)

For a rank-2 tensor `T_{ab}`:
  -> `sigma^a_{AA'} sigma^b_{BB'} T_{ab}`  (contracts to `T_{AA'BB'}`)

The sigma tensors are inserted with fresh dummy indices on the Tangent bundle
so that the original indices become contracted dummies.

The result is NOT automatically simplified; call `simplify` or `contract_metrics`
to perform the actual contractions.

# Reference
Penrose & Rindler, *Spinors and Space-Time* Vol 1 (1984), Eq 3.1.20.
"""
function to_spinor_indices(expr::TensorExpr, reg::TensorRegistry)
    free = free_indices(expr)
    tangent_free = filter(idx -> idx.vbundle == :Tangent, free)
    isempty(tangent_free) && return expr

    # Find the soldering form name
    sigma_name = _find_soldering_form(reg)
    sigma_name === nothing && error("No soldering form registered; call define_soldering_form! first")

    # Collect all used index names to avoid collisions
    all_idxs = indices(expr)
    used = Set{Symbol}(idx.name for idx in all_idxs)

    # For each tangent free index, insert a sigma
    sigma_factors = TensorExpr[]
    for tidx in tangent_free
        # Generate fresh spinor index names
        sl2c_name = fresh_index(used; vbundle=:SL2C)
        push!(used, sl2c_name)
        sl2c_dot_name = fresh_index(used; vbundle=:SL2C_dot)
        push!(used, sl2c_dot_name)

        # sigma^a_{AA'}: tangent index matches the free index position (flipped for contraction)
        # If the free index is Down (V_a), sigma needs Up tangent index to contract.
        # If the free index is Up (V^a), sigma needs Down tangent index to contract.
        tangent_pos = (tidx.position == Up) ? Down : Up
        sig = Tensor(sigma_name, [
            TIndex(tidx.name, tangent_pos, :Tangent),
            TIndex(sl2c_name, Down, :SL2C),
            TIndex(sl2c_dot_name, Down, :SL2C_dot)
        ])
        push!(sigma_factors, sig)
    end

    # Build the product: sigma_1 * sigma_2 * ... * expr
    all_factors = copy(sigma_factors)
    push!(all_factors, expr)
    tproduct(1 // 1, all_factors)
end

"""
    to_tensor_indices(expr::TensorExpr, reg::TensorRegistry) -> TensorExpr

Convert spinor index pairs back to tensor indices using the soldering form.
For each pair of free spinor indices `(A, A')` (one SL2C, one SL2C_dot),
insert `sigma_{AA'}^a` to contract them into a single Tangent index.

The result is NOT automatically simplified; call `simplify` or `contract_metrics`
to perform the actual contractions.

# Reference
Penrose & Rindler, *Spinors and Space-Time* Vol 1 (1984), Eq 3.1.20.
"""
function to_tensor_indices(expr::TensorExpr, reg::TensorRegistry)
    free = free_indices(expr)
    sl2c_free = filter(idx -> idx.vbundle == :SL2C, free)
    sl2c_dot_free = filter(idx -> idx.vbundle == :SL2C_dot, free)

    # We need at least one of each to form a pair
    n_pairs = min(length(sl2c_free), length(sl2c_dot_free))
    n_pairs == 0 && return expr

    sigma_name = _find_soldering_form(reg)
    sigma_name === nothing && error("No soldering form registered; call define_soldering_form! first")

    all_idxs = indices(expr)
    used = Set{Symbol}(idx.name for idx in all_idxs)

    sigma_factors = TensorExpr[]
    for i in 1:n_pairs
        undotted = sl2c_free[i]
        dotted = sl2c_dot_free[i]

        # Generate fresh tangent index
        tangent_name = fresh_index(used; vbundle=:Tangent)
        push!(used, tangent_name)

        # sigma with tangent index and spinor indices positioned to contract
        # with the free spinor indices in expr.
        # If undotted is Up (A^), sigma needs SL2C Down to contract.
        # If undotted is Down (A_), sigma needs SL2C Up to contract.
        sl2c_pos = (undotted.position == Up) ? Down : Up
        sl2c_dot_pos = (dotted.position == Up) ? Down : Up

        sig = Tensor(sigma_name, [
            TIndex(tangent_name, Up, :Tangent),
            TIndex(undotted.name, sl2c_pos, :SL2C),
            TIndex(dotted.name, sl2c_dot_pos, :SL2C_dot)
        ])
        push!(sigma_factors, sig)
    end

    all_factors = copy(sigma_factors)
    push!(all_factors, expr)
    tproduct(1 // 1, all_factors)
end

# ── Internal: find the registered soldering form ─────────────────────────

"""
Find the name of the registered soldering form tensor. Returns the first
tensor with `:is_soldering => true` in options, or `nothing`.
"""
function _find_soldering_form(reg::TensorRegistry)
    for (name, props) in reg.tensors
        get(props.options, :is_soldering, false) && return name
    end
    nothing
end
