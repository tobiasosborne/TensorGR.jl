#= Convenience macros for defining manifolds, tensors, and covariant derivatives.

@manifold M4 dim=4 metric=g indices=[a,b,c,d,e,f]
@define_tensor T on=M4 rank=(0,2) symmetry=Symmetric(1,2)
@covd ∇ on=M4 metric=g
=#

"""
    @manifold name dim=d metric=g indices=[a,b,c,d,e,f]

Define a manifold with dimension, metric, and index alphabet.
Performs full setup: registers manifold, metric (with signature), delta, epsilon,
curvature tensors (Riem, Ric, RicScalar, Ein, Weyl, Sch), covariant derivative,
and Bianchi rules. Equivalent to `register_manifold!` + `define_metric!`.
"""
macro manifold(name, kwargs...)
    _parse_manifold(name, kwargs)
end

function _parse_manifold(name, kwargs)
    dim = 4
    metric = :g
    derivative = :∂
    idx_list = [:a, :b, :c, :d, :e, :f, :m, :n, :p, :q]

    for kw in kwargs
        if kw isa Expr && kw.head == :(=)
            key = kw.args[1]
            val = kw.args[2]
            if key == :dim
                dim = val
            elseif key == :metric
                metric = val
            elseif key == :derivative
                derivative = val
            elseif key == :indices
                if val isa Expr && val.head == :vect
                    idx_list = [x isa QuoteNode ? x.value : x for x in val.args]
                end
            end
        end
    end

    quote
        let reg = current_registry()
            register_manifold!(reg, ManifoldProperties(
                $(QuoteNode(name)), $dim, $(QuoteNode(metric)),
                $(QuoteNode(derivative)), Symbol[$(QuoteNode.(idx_list)...)]))
            define_metric!(reg, $(QuoteNode(metric)); manifold=$(QuoteNode(name)))
        end
    end |> esc
end

"""
    @define_tensor name on=manifold rank=(p,q) [symmetry=Symmetric(1,2)]

Register a tensor with the given properties.
"""
macro define_tensor(name, kwargs...)
    _parse_define_tensor(name, kwargs)
end

function _parse_define_tensor(name, kwargs)
    manifold = :M4
    rank = (0, 0)
    syms = SymmetrySpec[]

    for kw in kwargs
        if kw isa Expr && kw.head == :(=)
            key = kw.args[1]
            val = kw.args[2]
            if key == :on
                manifold = val
            elseif key == :rank
                rank = val
            elseif key == :symmetry
                syms = [val]
            elseif key == :symmetries
                if val isa Expr && val.head == :vect
                    syms = val.args
                end
            end
        end
    end

    quote
        register_tensor!(current_registry(), TensorProperties(
            name=$(QuoteNode(name)), manifold=$(QuoteNode(manifold)),
            rank=$rank,
            symmetries=SymmetrySpec[$(syms...)]))
    end |> esc
end

"""
    @covd name on=manifold metric=g

Define a covariant derivative with associated Christoffel symbols.
"""
macro covd(name, kwargs...)
    _parse_covd(name, kwargs)
end

function _parse_covd(name, kwargs)
    manifold = :M4
    metric = :g

    for kw in kwargs
        if kw isa Expr && kw.head == :(=)
            key = kw.args[1]
            val = kw.args[2]
            if key == :on
                manifold = val
            elseif key == :metric
                metric = val
            end
        end
    end

    quote
        define_covd!(current_registry(), $(QuoteNode(name));
                     manifold=$(QuoteNode(manifold)), metric=$(QuoteNode(metric)))
    end |> esc
end
