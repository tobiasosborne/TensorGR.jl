#= Feynman diagram type hierarchy and builder API.
=#

# ────────────────────────────────────────────────────────────────────
# Type Hierarchy
# ────────────────────────────────────────────────────────────────────

"""
    TensorVertex

An n-point tensor vertex in momentum space.
Represents V^{(n)}_{a1b1, a2b2, ..., anbn}(k1, k2, ..., kn)
with momentum conservation at each vertex.
"""
struct TensorVertex
    name::Symbol
    index_groups::Vector{Vector{TIndex}}
    momenta::Vector{Symbol}
    expr::TensorExpr
    coupling_order::Int
    symmetry_group::Vector{Vector{Int}}
end

function TensorVertex(name::Symbol, index_groups::Vector{Vector{TIndex}},
                      momenta::Vector{Symbol}, expr::TensorExpr;
                      coupling_order::Int=length(index_groups) - 2,
                      symmetry_group::Vector{Vector{Int}}=Vector{Int}[])
    length(index_groups) == length(momenta) ||
        error("Number of index groups ($(length(index_groups))) must match momenta ($(length(momenta)))")
    TensorVertex(name, index_groups, momenta, expr, coupling_order, symmetry_group)
end

"""Number of legs (n-point) on the vertex."""
n_point(v::TensorVertex) = length(v.index_groups)

"""Total number of free indices across all legs."""
n_indices(v::TensorVertex) = sum(length(g) for g in v.index_groups; init=0)

function Base.show(io::IO, v::TensorVertex)
    np = n_point(v)
    print(io, "TensorVertex(:$(v.name), $(np)-point, ",
          "kappa^$(v.coupling_order), $(n_indices(v)) indices)")
end

"""
    TensorPropagator

A propagator connecting two index groups.
Carries a single momentum and optional gauge parameter.
"""
struct TensorPropagator
    name::Symbol
    indices_left::Vector{TIndex}
    indices_right::Vector{TIndex}
    momentum::Symbol
    expr::TensorExpr
    gauge_param::Any
end

function TensorPropagator(name::Symbol, indices_left::Vector{TIndex},
                          indices_right::Vector{TIndex}, momentum::Symbol,
                          expr::TensorExpr;
                          gauge_param=nothing)
    TensorPropagator(name, indices_left, indices_right, momentum, expr, gauge_param)
end

function Base.show(io::IO, p::TensorPropagator)
    nl = length(p.indices_left)
    nr = length(p.indices_right)
    gauge_str = p.gauge_param === nothing ? "" : ", gauge=$(p.gauge_param)"
    print(io, "TensorPropagator(:$(p.name), rank ($nl,$nr), k=$(p.momentum)$(gauge_str))")
end

"""
    FeynmanDiagram

A complete Feynman diagram: graph of vertices connected by propagators.

`topology` entries `(va, la, vb, lb)` mean: vertex `va` leg `la` is
connected to vertex `vb` leg `lb` via the corresponding propagator.
"""
struct FeynmanDiagram
    vertices::Vector{TensorVertex}
    propagators::Vector{TensorPropagator}
    topology::Vector{Tuple{Int,Int,Int,Int}}
    external_legs::Vector{@NamedTuple{vertex::Int, leg::Int, momentum::Symbol}}
    loop_momenta::Vector{Symbol}
    symmetry_factor::Rational{Int}
end

function FeynmanDiagram(vertices, propagators, topology, external_legs;
                        loop_momenta::Vector{Symbol}=Symbol[],
                        symmetry_factor::Rational{Int}=1//1)
    FeynmanDiagram(vertices, propagators, topology, external_legs,
                   loop_momenta, symmetry_factor)
end

function Base.show(io::IO, d::FeynmanDiagram)
    nv = length(d.vertices)
    ne = length(d.propagators)
    nl = n_loops(d)
    nx = length(d.external_legs)
    print(io, "FeynmanDiagram($(nv) vertices, $(ne) propagators, ",
          "$(nl) loops, $(nx) external legs)")
end

"""
    DiagramAmplitude

Result of contracting a Feynman diagram: external indices, momenta,
and the contracted tensor expression.
"""
struct DiagramAmplitude
    external_indices::Vector{Vector{TIndex}}
    external_momenta::Vector{Symbol}
    expr::TensorExpr
end

function Base.show(io::IO, da::DiagramAmplitude)
    ni = sum(length(g) for g in da.external_indices; init=0)
    print(io, "DiagramAmplitude($(ni) external indices, ",
          "$(length(da.external_momenta)) momenta)")
end

# ────────────────────────────────────────────────────────────────────
# Loop Counting
# ────────────────────────────────────────────────────────────────────

"""
    n_loops(diagram::FeynmanDiagram) -> Int

Compute the loop count via the Euler relation L = E - V + 1
for a connected diagram (E = internal edges, V = vertices).
"""
function n_loops(diagram::FeynmanDiagram)
    E = length(diagram.propagators)
    V = length(diagram.vertices)
    V == 0 && return 0
    E - V + 1
end

# ────────────────────────────────────────────────────────────────────
# Builder API
# ────────────────────────────────────────────────────────────────────

"""
    build_diagram(vertices, propagators, connections;
                  external_momenta=Symbol[]) -> FeynmanDiagram

Assemble a Feynman diagram from vertices, propagators, and a
connection topology.

`connections` is a vector of `(vertex_a, leg_a, vertex_b, leg_b)` tuples,
one per propagator. The `i`-th connection uses `propagators[i]`.

Index-group rank compatibility is checked: both ends of each internal
line must have the same number of indices.

External legs are all vertex legs that do not appear in any connection.
"""
function build_diagram(vertices::Vector{TensorVertex},
                       propagators::Vector{TensorPropagator},
                       connections::Vector{Tuple{Int,Int,Int,Int}};
                       external_momenta::Vector{Symbol}=Symbol[])
    length(propagators) == length(connections) ||
        error("Number of propagators ($(length(propagators))) must match " *
              "connections ($(length(connections)))")

    # Validate topology references and rank compatibility
    for (i, (va, la, vb, lb)) in enumerate(connections)
        (1 <= va <= length(vertices)) ||
            error("Connection $i: vertex_a=$va out of range [1, $(length(vertices))]")
        (1 <= vb <= length(vertices)) ||
            error("Connection $i: vertex_b=$vb out of range [1, $(length(vertices))]")
        (1 <= la <= n_point(vertices[va])) ||
            error("Connection $i: leg_a=$la out of range for vertex $va " *
                  "($(n_point(vertices[va])) legs)")
        (1 <= lb <= n_point(vertices[vb])) ||
            error("Connection $i: leg_b=$lb out of range for vertex $vb " *
                  "($(n_point(vertices[vb])) legs)")

        rank_a = length(vertices[va].index_groups[la])
        rank_b = length(vertices[vb].index_groups[lb])
        rank_a == rank_b ||
            error("Connection $i: incompatible ranks ($rank_a vs $rank_b) " *
                  "at vertex $va leg $la and vertex $vb leg $lb")
    end

    # Identify external legs: those not appearing in any connection
    connected = Set{Tuple{Int,Int}}()
    for (va, la, vb, lb) in connections
        push!(connected, (va, la))
        push!(connected, (vb, lb))
    end

    ext_legs = @NamedTuple{vertex::Int, leg::Int, momentum::Symbol}[]
    mom_idx = 1
    for (vi, v) in enumerate(vertices)
        for li in 1:n_point(v)
            if (vi, li) in connected
                continue
            end
            mom = if mom_idx <= length(external_momenta)
                external_momenta[mom_idx]
            else
                Symbol(:p, mom_idx)
            end
            push!(ext_legs, (vertex=vi, leg=li, momentum=mom))
            mom_idx += 1
        end
    end

    # Determine loop momenta from explicit loop_momenta in the diagram
    # (for tree-level diagrams this is empty)
    loop_mom = Symbol[]

    FeynmanDiagram(vertices, propagators, connections, ext_legs;
                   loop_momenta=loop_mom)
end

"""
    tree_exchange_diagram(v1::TensorVertex, v2::TensorVertex,
                          prop::TensorPropagator;
                          leg1::Int=1, leg2::Int=1,
                          external_momenta::Vector{Symbol}=Symbol[]) -> FeynmanDiagram

Convenience constructor for a tree-level single-exchange diagram:
two vertices connected by one propagator.

`leg1` and `leg2` specify which leg of each vertex is the internal line
(default: leg 1 of each vertex).
"""
function tree_exchange_diagram(v1::TensorVertex, v2::TensorVertex,
                               prop::TensorPropagator;
                               leg1::Int=1, leg2::Int=1,
                               external_momenta::Vector{Symbol}=Symbol[])
    connections = [(1, leg1, 2, leg2)]
    build_diagram([v1, v2], [prop], connections;
                  external_momenta=external_momenta)
end

"""
    vertex_from_perturbation(expr::TensorExpr, order::Int, field::Symbol;
                              name::Symbol=Symbol(:V, order, :_, field),
                              momenta::Vector{Symbol}=Symbol[]) -> TensorVertex

Create a TensorVertex from a perturbation expansion expression.

The expression `expr` should be the order-`order` perturbation of a
Lagrangian density (e.g., output of `expand_perturbation`). The `field`
symbol identifies which tensor field's indices form the leg groups.

Each occurrence of `field` in the expression becomes one leg of the vertex.
The resulting vertex is an `order`-point vertex (order = number of field legs).
"""
function vertex_from_perturbation(expr::TensorExpr, order::Int, field::Symbol;
                                   name::Symbol=Symbol(:V, order, :_, field),
                                   momenta::Vector{Symbol}=Symbol[])
    # Collect index groups from field occurrences
    index_groups = Vector{TIndex}[]
    _collect_field_index_groups!(index_groups, expr, field)

    # Assign default momenta if not provided
    if isempty(momenta)
        momenta = [Symbol(:k, i) for i in 1:length(index_groups)]
    end

    length(index_groups) == length(momenta) ||
        error("Found $(length(index_groups)) occurrences of field :$field " *
              "but $(length(momenta)) momenta provided")

    TensorVertex(name, index_groups, momenta, expr;
                 coupling_order=order - 2)
end

"""Walk expression tree collecting index groups from field tensor occurrences."""
function _collect_field_index_groups!(groups::Vector{Vector{TIndex}},
                                      expr::Tensor, field::Symbol)
    if expr.name == field
        push!(groups, copy(expr.indices))
    end
end

function _collect_field_index_groups!(groups::Vector{Vector{TIndex}},
                                      expr::TProduct, field::Symbol)
    for f in expr.factors
        _collect_field_index_groups!(groups, f, field)
    end
end

function _collect_field_index_groups!(groups::Vector{Vector{TIndex}},
                                      expr::TSum, field::Symbol)
    # For a sum, collect from the first term (all terms should have the
    # same field structure in a well-formed perturbation expansion)
    if !isempty(expr.terms)
        _collect_field_index_groups!(groups, expr.terms[1], field)
    end
end

function _collect_field_index_groups!(groups::Vector{Vector{TIndex}},
                                      expr::TDeriv, field::Symbol)
    _collect_field_index_groups!(groups, expr.arg, field)
end

function _collect_field_index_groups!(::Vector{Vector{TIndex}},
                                      ::TScalar, ::Symbol)
    # scalars have no field occurrences
end

"""
    contract_diagram(diag::FeynmanDiagram; registry::TensorRegistry=current_registry()) -> DiagramAmplitude

Contract a Feynman diagram by substituting propagator expressions into
vertex expressions along each internal line.

Returns a DiagramAmplitude with the contracted tensor expression and
the external index/momentum data.

For tree-level diagrams, the result is a single TensorExpr. For loop
diagrams, the expression contains uncontracted loop momenta.
"""
function contract_diagram(diag::FeynmanDiagram;
                          registry::TensorRegistry=current_registry())
    # Start with vertex expressions
    vertex_exprs = TensorExpr[v.expr for v in diag.vertices]

    # Collect propagator expressions
    prop_exprs = TensorExpr[p.expr for p in diag.propagators]

    # Build the full product: all vertices times all propagators
    all_factors = TensorExpr[]
    for ve in vertex_exprs
        push!(all_factors, ve)
    end
    for pe in prop_exprs
        push!(all_factors, pe)
    end

    # Form the product (ensure no dummy clashes between factors)
    if isempty(all_factors)
        amplitude_expr = TScalar(1 // 1)
    else
        result = all_factors[1]
        for i in 2:length(all_factors)
            next = ensure_no_dummy_clash(result, all_factors[i])
            result = tproduct(1 // 1, TensorExpr[result, next])
        end
        amplitude_expr = tproduct(diag.symmetry_factor, TensorExpr[result])
    end

    # Gather external index groups and momenta
    ext_indices = Vector{TIndex}[]
    ext_momenta = Symbol[]
    for el in diag.external_legs
        push!(ext_indices, diag.vertices[el.vertex].index_groups[el.leg])
        push!(ext_momenta, el.momentum)
    end

    DiagramAmplitude(ext_indices, ext_momenta, amplitude_expr)
end
