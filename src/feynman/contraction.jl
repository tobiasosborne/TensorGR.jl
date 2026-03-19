#= Tensor contraction engine for Feynman diagrams.
#
# Extends the basic contract_diagram in types.jl with:
#   - contract_line:  contract a single propagator with two vertex expressions
#   - find_loop_momenta:  identify independent loop momenta from topology
#   - impose_momentum_conservation!:  apply sum-of-momenta = 0 at each vertex
#   - symmetry_factor:  compute from graph automorphisms
#
# Ground truth: Goldberger & Rothstein hep-th/0409156; Porto 1104.2712.
=#

# ────────────────────────────────────────────────────────────────────
# Single-line contraction
# ────────────────────────────────────────────────────────────────────

"""
    contract_line(prop::TensorPropagator, v1_expr::TensorExpr, v2_expr::TensorExpr,
                  idx_group_1::Vector{TIndex}, idx_group_2::Vector{TIndex};
                  registry::TensorRegistry=current_registry()) -> TensorExpr

Contract a single propagator with two vertex expressions.

The propagator's left indices are renamed to match `idx_group_1` (from vertex 1),
and right indices to match `idx_group_2` (from vertex 2). The result is the
product `v1 * prop_renamed * v2` with all dummy indices properly managed.

Rank compatibility is checked: both index groups must match the propagator's
left/right ranks.
"""
function contract_line(prop::TensorPropagator, v1_expr::TensorExpr, v2_expr::TensorExpr,
                       idx_group_1::Vector{TIndex}, idx_group_2::Vector{TIndex};
                       registry::TensorRegistry=current_registry())
    length(idx_group_1) == length(prop.indices_left) ||
        error("Left index group rank ($(length(idx_group_1))) does not match " *
              "propagator left rank ($(length(prop.indices_left)))")
    length(idx_group_2) == length(prop.indices_right) ||
        error("Right index group rank ($(length(idx_group_2))) does not match " *
              "propagator right rank ($(length(prop.indices_right)))")

    # Rename propagator indices to contract with vertex index groups
    prop_renamed = prop.expr
    for (old_idx, new_idx) in zip(prop.indices_left, idx_group_1)
        if old_idx.name != new_idx.name
            prop_renamed = rename_dummy(prop_renamed, old_idx.name, new_idx.name)
        end
    end
    for (old_idx, new_idx) in zip(prop.indices_right, idx_group_2)
        if old_idx.name != new_idx.name
            prop_renamed = rename_dummy(prop_renamed, old_idx.name, new_idx.name)
        end
    end

    # Ensure no dummy clashes between the three expressions
    v2_safe = ensure_no_dummy_clash(v1_expr, v2_expr)
    prop_safe = ensure_no_dummy_clash(v1_expr, prop_renamed)
    prop_safe = ensure_no_dummy_clash(v2_safe, prop_safe)

    tproduct(1 // 1, TensorExpr[v1_expr, prop_safe, v2_safe])
end

# ────────────────────────────────────────────────────────────────────
# Loop momentum identification
# ────────────────────────────────────────────────────────────────────

"""
    find_loop_momenta(diagram::FeynmanDiagram) -> Vector{Symbol}

Identify independent loop momenta from the diagram topology.

Uses a spanning tree algorithm: edges NOT in the spanning tree each
carry an independent loop momentum. For a connected diagram with
V vertices and E internal edges, there are L = E - V + 1 loop momenta.

Tree-level diagrams (L=0) return an empty vector.
"""
function find_loop_momenta(diagram::FeynmanDiagram)
    V = length(diagram.vertices)
    E = length(diagram.propagators)
    V == 0 && return Symbol[]

    L = E - V + 1
    L <= 0 && return Symbol[]

    # Build spanning tree via BFS
    adj = Dict{Int, Vector{Tuple{Int,Int}}}()  # vertex => [(neighbor, edge_idx)]
    for v in 1:V
        adj[v] = Tuple{Int,Int}[]
    end
    for (i, (va, _, vb, _)) in enumerate(diagram.topology)
        push!(adj[va], (vb, i))
        if va != vb  # skip self-loops in reverse direction
            push!(adj[vb], (va, i))
        end
    end

    # BFS from vertex 1
    visited = Set{Int}(1)
    tree_edges = Set{Int}()
    queue = Int[1]

    while !isempty(queue)
        v = popfirst!(queue)
        for (w, edge_idx) in adj[v]
            if w ∉ visited
                push!(visited, w)
                push!(tree_edges, edge_idx)
                push!(queue, w)
            end
        end
    end

    # Edges not in spanning tree carry loop momenta
    loop_moms = Symbol[]
    for i in 1:E
        if i ∉ tree_edges
            push!(loop_moms, Symbol(:q, length(loop_moms) + 1))
        end
    end

    loop_moms
end

# ────────────────────────────────────────────────────────────────────
# Momentum conservation
# ────────────────────────────────────────────────────────────────────

"""
    MomentumConstraint

At each vertex: sum of incoming momenta = 0.
Stores which momenta participate and their signs.
"""
struct MomentumConstraint
    vertex::Int
    momenta::Vector{Symbol}
    signs::Vector{Int}       # +1 incoming, -1 outgoing
end

"""
    momentum_constraints(diagram::FeynmanDiagram) -> Vector{MomentumConstraint}

Compute the momentum conservation constraint at each vertex.

For each vertex, all legs (internal and external) contribute a momentum.
Internal propagators connect two vertices, so their momentum enters one
vertex with +1 and the other with -1. External momenta are all incoming (+1).

Only V-1 constraints are independent for a connected diagram.
"""
function momentum_constraints(diagram::FeynmanDiagram)
    V = length(diagram.vertices)
    constraints = MomentumConstraint[]

    for vi in 1:V
        moms = Symbol[]
        signs = Int[]

        # Internal lines: each propagator connects two vertices
        for (i, (va, _, vb, _)) in enumerate(diagram.topology)
            p = diagram.propagators[i]
            if va == vi
                push!(moms, p.momentum)
                push!(signs, +1)   # momentum flows out of vertex a
            end
            if vb == vi
                push!(moms, p.momentum)
                push!(signs, -1)   # momentum flows into vertex b
            end
        end

        # External legs at this vertex
        for el in diagram.external_legs
            if el.vertex == vi
                push!(moms, el.momentum)
                push!(signs, +1)   # external momenta are incoming
            end
        end

        push!(constraints, MomentumConstraint(vi, moms, signs))
    end

    constraints
end

"""
    impose_momentum_conservation(diagram::FeynmanDiagram) -> Dict{Symbol, Vector{Pair{Int,Symbol}}}

Determine momentum routing: solve momentum conservation at each vertex
to express internal momenta in terms of external and loop momenta.

Returns a Dict mapping each internal propagator momentum to a linear
combination of external momenta and loop momenta:
  `q_internal => [(+1, :p1), (-1, :p2), (+1, :q1)]`
meaning `q_internal = p1 - p2 + q1`.

For tree diagrams (L=0), all internal momenta are fixed by external momenta.
"""
function impose_momentum_conservation(diagram::FeynmanDiagram)
    V = length(diagram.vertices)
    V == 0 && return Dict{Symbol, Vector{Pair{Int,Symbol}}}()

    # Identify external and loop momenta (these are independent)
    ext_moms = Set{Symbol}(el.momentum for el in diagram.external_legs)
    loop_moms = isempty(diagram.loop_momenta) ?
        Set{Symbol}(find_loop_momenta(diagram)) :
        Set{Symbol}(diagram.loop_momenta)
    independent = union(ext_moms, loop_moms)

    # Internal momenta that need solving
    internal_moms = Symbol[]
    for p in diagram.propagators
        if p.momentum ∉ independent
            push!(internal_moms, p.momentum)
        end
    end

    # For tree-level with default momentum labels, assign via BFS
    routing = Dict{Symbol, Vector{Pair{Int,Symbol}}}()
    constraints = momentum_constraints(diagram)

    # Simple greedy solver: at each constraint, if exactly one unknown
    # remains, solve for it
    solved = copy(independent)
    remaining = Set{Symbol}(internal_moms)
    changed = true

    while changed && !isempty(remaining)
        changed = false
        for c in constraints
            unknowns = Symbol[]
            for m in c.momenta
                if m ∉ solved
                    push!(unknowns, m)
                end
            end

            if length(unknowns) == 1
                target = unknowns[1]
                # Solve: target = -sum(sign_i * m_i) for all other momenta
                target_sign = 0
                combo = Pair{Int,Symbol}[]
                for (m, s) in zip(c.momenta, c.signs)
                    if m == target
                        target_sign = s
                    elseif m ∈ solved
                        # This known momentum contributes with opposite sign
                        push!(combo, Pair(-s, m))
                    end
                end
                # If target has sign +1, then target = -(sum of rest)
                # If target has sign -1, then -target = -(sum of rest) => target = sum of rest
                if target_sign == -1
                    # Flip all signs
                    combo = [Pair(-p.first, p.second) for p in combo]
                end

                routing[target] = combo
                push!(solved, target)
                delete!(remaining, target)
                changed = true
            end
        end
    end

    routing
end

# ────────────────────────────────────────────────────────────────────
# Symmetry factor computation
# ────────────────────────────────────────────────────────────────────

"""
    symmetry_factor(diagram::FeynmanDiagram) -> Rational{Int}

Compute the symmetry factor of the diagram.

The symmetry factor accounts for identical propagators between the same
vertex pair. For each group of `n` identical propagators connecting the
same pair of vertices, contribute a factor `1/n!`.

Self-energy (bubble) with 2 identical propagators: 1/2.
Sunset with 3 identical propagators: 1/6.
Tadpole (vertex connected to itself): no additional factor from topology
(the 1/n! from the action expansion already handles it).

Ground truth: Goldberger & Rothstein hep-th/0409156, Sec 4.
"""
function symmetry_factor(diagram::FeynmanDiagram)
    # Count identical propagators between each (ordered) vertex pair
    edge_counts = Dict{Tuple{Int,Int}, Int}()
    for (va, _, vb, _) in diagram.topology
        key = va <= vb ? (va, vb) : (vb, va)
        edge_counts[key] = get(edge_counts, key, 0) + 1
    end

    factor = 1 // 1
    for (_, n) in edge_counts
        if n > 1
            factor *= 1 // factorial(n)
        end
    end

    factor
end
