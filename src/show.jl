function Base.show(io::IO, idx::TIndex)
    if idx.position == Down
        print(io, "-", idx.name)
    else
        print(io, idx.name)
    end
end

function Base.show(io::IO, t::Tensor)
    print(io, t.name)
    if !isempty(t.indices)
        print(io, "[")
        join(io, t.indices, ", ")
        print(io, "]")
    end
end

function Base.show(io::IO, s::TScalar)
    print(io, s.val)
end

function Base.show(io::IO, p::TProduct)
    s = p.scalar
    if isempty(p.factors)
        print(io, s)
        return
    end
    if s == 1//1
        # no prefix
    elseif s == -1//1
        print(io, "-")
    else
        print(io, "(", s, ") * ")
    end
    for (i, f) in enumerate(p.factors)
        i > 1 && print(io, " * ")
        show(io, f)
    end
end

function Base.show(io::IO, s::TSum)
    for (i, t) in enumerate(s.terms)
        i > 1 && print(io, " + ")
        show(io, t)
    end
end

function Base.show(io::IO, d::TDeriv)
    print(io, "∂[")
    show(io, d.index)
    print(io, "](")
    show(io, d.arg)
    print(io, ")")
end
