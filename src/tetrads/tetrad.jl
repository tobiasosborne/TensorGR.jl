# Tetrad (vierbein/vielbein) field definition and registration.
#
# A tetrad e^I_a relates the spacetime metric g_{ab} to the frame metric
# eta_{IJ} via the metricity condition:
#
#     e^I_a e^J_b eta_{IJ} = g_{ab}
#
# The inverse tetrad E^a_I satisfies completeness:
#
#     e^I_a E^a_J = delta^I_J     (frame completeness)
#     E^a_I e^I_b = delta^a_b     (spacetime completeness)
#
# References:
#   Wald, *General Relativity* (1984), Sec 3.4b.
#   Carroll, *Spacetime and Geometry* (2004), Sec 3.10.
#   Nakahara, *Geometry, Topology and Physics* (2003), Sec 7.8.

"""
    TetradProperties

Properties of a tetrad field e^I_a (vierbein/vielbein).

Fields:
- `name`: symbol for the tetrad (e.g. :e)
- `manifold`: base manifold (e.g. :M4)
- `metric`: spacetime metric (e.g. :g)
- `frame_metric`: frame (Lorentz) metric (e.g. :eta)
- `inverse_name`: symbol for the inverse tetrad (e.g. :E)
"""
struct TetradProperties
    name::Symbol           # e.g. :e
    manifold::Symbol       # e.g. :M4
    metric::Symbol         # spacetime metric g_{ab}
    frame_metric::Symbol   # frame metric eta_{IJ}
    inverse_name::Symbol   # e.g. :E (inverse tetrad E^a_I)
end

"""
    has_tetrad(reg::TensorRegistry, name::Symbol) -> Bool

Check if a tetrad with the given name is registered.
"""
has_tetrad(reg::TensorRegistry, name::Symbol) = haskey(reg.tetrads, name)

"""
    get_tetrad(reg::TensorRegistry, name::Symbol) -> TetradProperties

Retrieve the properties of a registered tetrad.
"""
function get_tetrad(reg::TensorRegistry, name::Symbol)
    reg.tetrads[name]::TetradProperties
end

"""
    define_tetrad!(reg, name; manifold, metric, frame_metric=:eta, inverse_name=nothing)

Register a tetrad e^I_a and its inverse E^a_I on `manifold`.

The tetrad has mixed index structure: one frame (Lorentz) index and one
spacetime (Tangent) index. Specifically:
- `name` (e.g. :e) is registered with indices [frame_up, down] (e^I_a)
- `inverse_name` (e.g. :E) is registered with indices [up, frame_down] (E^a_I)

Also registers rewrite rules implementing:
1. **Completeness**: e^I_a E^a_J -> delta^I_J  and  E^a_I e^I_b -> delta^a_b
2. **Metricity**: e^I_a e^J_b eta_{IJ} -> g_{ab}  (via contracted-frame rule)

The metricity rule fires after `contract_metrics` reduces eta_{IJ} contractions,
detecting two tetrads sharing a frame (Lorentz) dummy index.

# Prerequisites
- The manifold must be registered with `@manifold`.
- The frame bundle must be set up with `define_frame_bundle!`.
- The spacetime metric must be registered.

# Example
```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_frame_bundle!(reg; manifold=:M4, dim=4)
    define_tetrad!(reg, :e; manifold=:M4, metric=:g)
end
```
"""
function define_tetrad!(reg::TensorRegistry, name::Symbol;
                        manifold::Symbol,
                        metric::Symbol,
                        frame_metric::Symbol=:eta,
                        inverse_name::Union{Symbol,Nothing}=nothing)
    # Validate prerequisites
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")
    has_tensor(reg, metric) || error("Metric $metric not registered")
    has_tensor(reg, frame_metric) || error("Frame metric $frame_metric not registered")
    has_vbundle(reg, :Lorentz) || error("Lorentz VBundle not registered; call define_frame_bundle! first")
    has_tetrad(reg, name) && error("Tetrad $name already registered")

    # Default inverse name: uppercase of tetrad name
    inv_name = inverse_name !== nothing ? inverse_name : Symbol(uppercase(string(name)))

    # Look up the metric's delta and the frame delta
    delta_name = get(reg.delta_cache, manifold, :delta)
    has_tensor(reg, delta_name) || (delta_name = :delta)
    frame_delta_name = get(reg.delta_cache, :Lorentz, :delta_frame)
    has_tensor(reg, frame_delta_name) || (frame_delta_name = :delta_frame)

    # ---- Register the tetrad tensor e^I_a ----
    # Convention: e^I_a has rank (1,1), upper Lorentz index, lower Tangent index
    if !has_tensor(reg, name)
        register_tensor!(reg, TensorProperties(
            name=name, manifold=manifold, rank=(1, 1),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_tetrad => true,
                :up_vbundle => :Lorentz,
                :down_vbundle => :Tangent,
                :metric => metric,
                :frame_metric => frame_metric,
                :inverse => inv_name)))
    end

    # ---- Register the inverse tetrad E^a_I ----
    # Convention: E^a_I has rank (1,1), upper Tangent index, lower Lorentz index
    if !has_tensor(reg, inv_name)
        register_tensor!(reg, TensorProperties(
            name=inv_name, manifold=manifold, rank=(1, 1),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_tetrad => true,
                :is_inverse_tetrad => true,
                :up_vbundle => :Tangent,
                :down_vbundle => :Lorentz,
                :metric => metric,
                :frame_metric => frame_metric,
                :inverse => name)))
    end

    # ---- Completeness rules ----
    # e^I_a E^a_J -> delta^I_J  (contract on Tangent index a -> frame delta)
    _register_completeness_rule!(reg, name, inv_name,
                                 frame_delta_name, :Lorentz, :Tangent)
    # E^a_I e^I_b -> delta^a_b  (contract on Lorentz index I -> spacetime delta)
    _register_completeness_rule!(reg, inv_name, name,
                                 delta_name, :Tangent, :Lorentz)

    # ---- Metricity rule ----
    # After contract_metrics eliminates eta, detects e_{I,a} e^{I,b} -> g_{ab}
    # (two tetrads sharing a Lorentz dummy index)
    _register_metricity_contracted_rule!(reg, name, metric)

    # Also register the explicit 3-factor form for direct use without contract_metrics
    _register_metricity_rule!(reg, name, frame_metric, metric)

    # ---- Store tetrad properties ----
    tp = TetradProperties(name, manifold, metric, frame_metric, inv_name)
    reg.tetrads[name] = tp
    tp
end

# ---- Internal helpers ----

"""
Register a completeness rule: T1 * T2 -> delta
where the contracted index must be in `contract_vbundle`, and the
remaining free indices are in `result_vbundle`.
"""
function _register_completeness_rule!(reg::TensorRegistry,
                                       t1_name::Symbol, t2_name::Symbol,
                                       delta_name::Symbol,
                                       result_vbundle::Symbol,
                                       contract_vbundle::Symbol)
    rule = RewriteRule(
        function(expr)
            expr isa TProduct || return false
            length(expr.factors) >= 2 || return false
            _has_tetrad_pair(expr, t1_name, t2_name, contract_vbundle)
        end,
        function(expr)
            _apply_completeness(expr, t1_name, t2_name, delta_name,
                                result_vbundle, contract_vbundle)
        end
    )
    register_rule!(reg, rule)
end

"""
Check if a product contains a contractible pair of t1 and t2 where
the contracted index lives in `contract_vbundle`.
"""
function _has_tetrad_pair(p::TProduct, t1_name::Symbol, t2_name::Symbol,
                          contract_vbundle::Symbol)
    tensors1 = Int[]
    tensors2 = Int[]
    for (i, f) in enumerate(p.factors)
        f isa Tensor || continue
        if f.name == t1_name
            push!(tensors1, i)
        elseif f.name == t2_name
            push!(tensors2, i)
        end
    end
    isempty(tensors1) && return false
    isempty(tensors2) && return false

    for i1 in tensors1, i2 in tensors2
        f1 = p.factors[i1]::Tensor
        f2 = p.factors[i2]::Tensor
        for idx1 in f1.indices, idx2 in f2.indices
            if idx1.name == idx2.name &&
               idx1.position != idx2.position &&
               idx1.vbundle == idx2.vbundle &&
               idx1.vbundle === contract_vbundle
                return true
            end
        end
    end
    false
end

"""
Apply completeness: replace the contracted tetrad pair with a delta.
Only contracts on indices in `contract_vbundle`.
"""
function _apply_completeness(p::TProduct, t1_name::Symbol, t2_name::Symbol,
                              delta_name::Symbol, result_vbundle::Symbol,
                              contract_vbundle::Symbol)
    factors = collect(p.factors)

    for i1 in eachindex(factors), i2 in eachindex(factors)
        i1 == i2 && continue
        f1 = factors[i1]
        f2 = factors[i2]
        (f1 isa Tensor && f1.name == t1_name) || continue
        (f2 isa Tensor && f2.name == t2_name) || continue

        for (si1, idx1) in enumerate(f1.indices), (si2, idx2) in enumerate(f2.indices)
            if idx1.name == idx2.name &&
               idx1.position != idx2.position &&
               idx1.vbundle == idx2.vbundle &&
               idx1.vbundle === contract_vbundle
                # The remaining (non-contracted) indices form the delta
                other1 = f1.indices[3 - si1]
                other2 = f2.indices[3 - si2]
                # Build delta: the Up index first, Down second
                up_idx = other1.position == Up ? other1 : other2
                dn_idx = other1.position == Down ? other1 : other2
                delta = Tensor(delta_name, [
                    TIndex(up_idx.name, Up, result_vbundle),
                    TIndex(dn_idx.name, Down, result_vbundle)
                ])
                # Rebuild product without the two tetrad factors
                new_factors = TensorExpr[]
                for (k, f) in enumerate(factors)
                    k == i1 && continue
                    k == i2 && continue
                    push!(new_factors, f)
                end
                push!(new_factors, delta)
                return tproduct(p.scalar, new_factors)
            end
        end
    end
    p
end

"""
Register metricity rule for the post-contraction form:
two tetrads e sharing a Lorentz dummy index -> spacetime metric g.

After contract_metrics reduces `e^I_a e^J_b eta_{IJ}` to `e_{K,a} e^{K,b}`,
this rule detects the pair and produces `g_{ab}`.
"""
function _register_metricity_contracted_rule!(reg::TensorRegistry,
                                               tetrad_name::Symbol,
                                               spacetime_metric::Symbol)
    rule = RewriteRule(
        function(expr)
            expr isa TProduct || return false
            _has_tetrad_frame_contraction(expr, tetrad_name)
        end,
        function(expr)
            _apply_metricity_contracted(expr, tetrad_name, spacetime_metric)
        end
    )
    register_rule!(reg, rule)
end

"""
Check if a product has two tetrads sharing a Lorentz dummy index.
"""
function _has_tetrad_frame_contraction(p::TProduct, tetrad_name::Symbol)
    tetrad_idxs = [i for (i,f) in enumerate(p.factors)
                   if f isa Tensor && f.name == tetrad_name]
    length(tetrad_idxs) >= 2 || return false

    # Check if any pair of tetrads shares a Lorentz dummy
    for a in 1:length(tetrad_idxs), b in (a+1):length(tetrad_idxs)
        f1 = p.factors[tetrad_idxs[a]]::Tensor
        f2 = p.factors[tetrad_idxs[b]]::Tensor
        for idx1 in f1.indices, idx2 in f2.indices
            if idx1.name == idx2.name &&
               idx1.position != idx2.position &&
               idx1.vbundle === :Lorentz
                return true
            end
        end
    end
    false
end

"""
Apply metricity for post-contraction form: replace two tetrads
sharing a Lorentz dummy with the spacetime metric.
"""
function _apply_metricity_contracted(p::TProduct, tetrad_name::Symbol,
                                      spacetime_metric::Symbol)
    factors = collect(p.factors)
    tetrad_idxs = [i for (i,f) in enumerate(factors)
                   if f isa Tensor && f.name == tetrad_name]

    for a in 1:length(tetrad_idxs), b in (a+1):length(tetrad_idxs)
        ia = tetrad_idxs[a]
        ib = tetrad_idxs[b]
        f1 = factors[ia]::Tensor
        f2 = factors[ib]::Tensor

        for (si1, idx1) in enumerate(f1.indices), (si2, idx2) in enumerate(f2.indices)
            if idx1.name == idx2.name &&
               idx1.position != idx2.position &&
               idx1.vbundle === :Lorentz
                # Extract spacetime (Tangent) indices
                sp1 = f1.indices[3 - si1]
                sp2 = f2.indices[3 - si2]
                # Build g_{ab}
                g = Tensor(spacetime_metric, [
                    TIndex(sp1.name, Down, :Tangent),
                    TIndex(sp2.name, Down, :Tangent)
                ])
                # Rebuild product
                remove = Set([ia, ib])
                new_factors = TensorExpr[]
                for (k, f) in enumerate(factors)
                    k in remove && continue
                    push!(new_factors, f)
                end
                push!(new_factors, g)
                return tproduct(p.scalar, new_factors)
            end
        end
    end
    p
end

"""
Register explicit metricity rule: e^I_a e^J_b eta_{IJ} -> g_{ab}
(for use when contract_metrics hasn't run yet)
"""
function _register_metricity_rule!(reg::TensorRegistry,
                                    tetrad_name::Symbol,
                                    frame_metric::Symbol,
                                    spacetime_metric::Symbol)
    rule = RewriteRule(
        function(expr)
            expr isa TProduct || return false
            length(expr.factors) >= 3 || return false
            _has_metricity_pattern(expr, tetrad_name, frame_metric)
        end,
        function(expr)
            _apply_metricity(expr, tetrad_name, frame_metric, spacetime_metric)
        end
    )
    register_rule!(reg, rule)
end

"""
Check if a product has the pattern e^I_a e^J_b eta_{IJ}.
"""
function _has_metricity_pattern(p::TProduct, tetrad_name::Symbol, frame_metric::Symbol)
    tetrad_count = count(f -> f isa Tensor && f.name == tetrad_name, p.factors)
    tetrad_count >= 2 || return false
    eta_count = count(f -> f isa Tensor && f.name == frame_metric, p.factors)
    eta_count >= 1 || return false

    tetrad_idxs = [i for (i,f) in enumerate(p.factors) if f isa Tensor && f.name == tetrad_name]
    eta_idxs = [i for (i,f) in enumerate(p.factors) if f isa Tensor && f.name == frame_metric]

    for ie in eta_idxs
        eta = p.factors[ie]::Tensor
        matched = Int[]
        for it in tetrad_idxs
            tet = p.factors[it]::Tensor
            for eidx in eta.indices, tidx in tet.indices
                if eidx.name == tidx.name &&
                   eidx.position != tidx.position &&
                   eidx.vbundle == tidx.vbundle &&
                   tidx.vbundle === :Lorentz
                    push!(matched, it)
                    break
                end
            end
        end
        length(unique(matched)) >= 2 && return true
    end
    false
end

"""
Apply metricity: replace e^I_a e^J_b eta_{IJ} with g_{ab}.
"""
function _apply_metricity(p::TProduct, tetrad_name::Symbol,
                           frame_metric::Symbol, spacetime_metric::Symbol)
    factors = collect(p.factors)

    tetrad_idxs = [i for (i,f) in enumerate(factors) if f isa Tensor && f.name == tetrad_name]
    eta_idxs = [i for (i,f) in enumerate(factors) if f isa Tensor && f.name == frame_metric]

    for ie in eta_idxs
        eta = factors[ie]::Tensor
        matched_tetrads = Tuple{Int, Int, TIndex}[]
        for it in tetrad_idxs
            tet = factors[it]::Tensor
            for (si, tidx) in enumerate(tet.indices)
                tidx.vbundle === :Lorentz || continue
                for eidx in eta.indices
                    if eidx.name == tidx.name &&
                       eidx.position != tidx.position
                        spacetime_idx = tet.indices[3 - si]
                        push!(matched_tetrads, (it, si, spacetime_idx))
                        break
                    end
                end
            end
        end

        if length(matched_tetrads) >= 2
            (it1, _, sp_idx1) = matched_tetrads[1]
            (it2, _, sp_idx2) = matched_tetrads[2]
            it1 == it2 && continue

            g = Tensor(spacetime_metric, [
                TIndex(sp_idx1.name, Down, :Tangent),
                TIndex(sp_idx2.name, Down, :Tangent)
            ])

            remove = Set([ie, it1, it2])
            new_factors = TensorExpr[]
            for (k, f) in enumerate(factors)
                k in remove && continue
                push!(new_factors, f)
            end
            push!(new_factors, g)
            return tproduct(p.scalar, new_factors)
        end
    end
    p
end
