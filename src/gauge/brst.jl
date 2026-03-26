#= BRST differential for Yang-Mills gauge theories.
#
# The BRST transformation s acts on gauge and ghost fields:
#   s(A^I_a)    = D_a c^I = ∂_a c^I + f^I_{JK} A^J_a c^K
#   s(c^I)      = -(1/2) f^I_{JK} c^J c^K
#   s(c̄^I)      = B^I                         (anti-ghost → NL field)
#   s(B^I)      = 0                            (NL field is BRST-closed)
#
# Ghost numbers: A=0, c=+1, c̄=-1, B=0.
# BRST increases ghost number by 1: gh(s(X)) = gh(X) + 1.
# Nilpotency: s² = 0 (requires Jacobi identity for f^I_{JK}).
#
# The gauge group has structure constants f^I_{JK} satisfying:
#   f^I_{JK} = -f^I_{KJ}                     (antisymmetric)
#   f^I_{[JK} f^J_{LM]} = 0                  (Jacobi identity)
#
# References:
#   Henneaux & Teitelboim, *Quantization of Gauge Systems* (1992), Ch 8.
#   Fröb (2020), arXiv:2008.12422, Sec 4.
#   Weinberg, *The Quantum Theory of Fields* Vol II (1996), Ch 15.
=#

"""
    GaugeGroupProperties

Stores the definition of a gauge group for BRST.
"""
struct GaugeGroupProperties
    name::Symbol
    manifold::Symbol
    vbundle::Symbol
    structure_constants::Symbol
    gauge_field::Symbol
    ghost::Symbol
    anti_ghost::Symbol
    nl_field::Symbol
end

"""
    define_gauge_group!(reg, name; manifold=:M4, dim=3,
                         vbundle=:Gauge, struct_const=:f_struct,
                         gauge_field=:A, ghost=:c_ghost,
                         anti_ghost=:c_bar, nl_field=:B_NL)

Register a gauge group and all associated BRST fields.

Registers:
- VBundle for the gauge group (if not already present)
- Structure constants f^I_{JK} (antisymmetric in J,K)
- Gauge field A^I_a (one Up gauge index + one Down Tangent index)
- Ghost c^I (Grassmann-odd, ghost number +1)
- Anti-ghost c̄^I (Grassmann-odd, ghost number -1)
- Nakanishi-Lautrup field B^I (ghost number 0)

Returns a `GaugeGroupProperties` storing the definitions.
"""
function define_gauge_group!(reg::TensorRegistry, name::Symbol;
                              manifold::Symbol=:M4,
                              dim::Int=3,
                              vbundle::Symbol=:Gauge,
                              struct_const::Symbol=:f_struct,
                              gauge_field::Symbol=:A,
                              ghost::Symbol=:c_ghost,
                              anti_ghost::Symbol=:c_bar,
                              nl_field::Symbol=:B_NL)
    @lock reg.lock begin
    has_manifold(reg, manifold) ||
        error("define_gauge_group!: manifold '$manifold' not registered")

    # Register gauge VBundle
    if !has_vbundle(reg, vbundle)
        define_vbundle!(reg, vbundle;
            manifold=manifold, dim=dim,
            indices=[:I, :J, :K, :L, :M, :N])
    end

    # Structure constants f^I_{JK} (antisymmetric in last two indices)
    if !has_tensor(reg, struct_const)
        register_tensor!(reg, TensorProperties(
            name=struct_const, manifold=manifold, rank=(1, 2),
            symmetries=SymmetrySpec[AntiSymmetric(2, 3)],
            options=Dict{Symbol,Any}(
                :is_structure_constant => true,
                :gauge_group => name,
                :vbundle => vbundle)))
    end

    # Gauge field A^I_a (gauge up + tangent down)
    if !has_tensor(reg, gauge_field)
        register_tensor!(reg, TensorProperties(
            name=gauge_field, manifold=manifold, rank=(1, 1),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_gauge_field => true,
                :gauge_group => name,
                :ghost_number => 0,
                :vbundle => vbundle)))
    end

    # Ghost c^I (Grassmann-odd, ghost number +1)
    if !has_tensor(reg, ghost)
        register_grassmann_field!(reg, ghost;
            manifold=manifold, rank=(1, 0),
            options=Dict{Symbol,Any}(
                :is_ghost => true,
                :gauge_group => name,
                :ghost_number => 1,
                :vbundle => vbundle))
    end

    # Anti-ghost c̄^I (Grassmann-odd, ghost number -1)
    if !has_tensor(reg, anti_ghost)
        register_grassmann_field!(reg, anti_ghost;
            manifold=manifold, rank=(1, 0),
            options=Dict{Symbol,Any}(
                :is_anti_ghost => true,
                :gauge_group => name,
                :ghost_number => -1,
                :vbundle => vbundle))
    end

    # Nakanishi-Lautrup field B^I (ghost number 0)
    if !has_tensor(reg, nl_field)
        register_tensor!(reg, TensorProperties(
            name=nl_field, manifold=manifold, rank=(1, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_nl_field => true,
                :gauge_group => name,
                :ghost_number => 0,
                :vbundle => vbundle)))
    end

    ggp = GaugeGroupProperties(name, manifold, vbundle,
        struct_const, gauge_field, ghost, anti_ghost, nl_field)
    reg.foliations[Symbol(:gauge_group_, name)] = ggp
    ggp
    end
end

"""
    get_gauge_group(reg, name) -> GaugeGroupProperties

Retrieve a previously defined gauge group.
"""
function get_gauge_group(reg::TensorRegistry, name::Symbol)
    key = Symbol(:gauge_group_, name)
    haskey(reg.foliations, key) ||
        error("No gauge group :$name registered")
    reg.foliations[key]::GaugeGroupProperties
end

# ── BRST transformation ──────────────────────────────────────────────

"""
    brst_gauge_field(ggp, I, a; registry) -> TSum

BRST transformation of the gauge field:

    s(A^I_a) = ∂_a c^I + f^I_{JK} A^J_a c^K

# Arguments
- `ggp::GaugeGroupProperties` -- gauge group definition
- `I::TIndex` -- Up gauge index
- `a::TIndex` -- Down Tangent index
"""
function brst_gauge_field(ggp::GaugeGroupProperties,
                           I::TIndex, a::TIndex;
                           registry::TensorRegistry=current_registry())
    I.position === Up || error("brst_gauge_field: I must be Up")
    a.position === Down || error("brst_gauge_field: a must be Down")

    vb = ggp.vbundle
    used = Set{Symbol}([I.name, a.name])
    J_sym = fresh_index(used; vbundle=vb)
    push!(used, J_sym)
    K_sym = fresh_index(used; vbundle=vb)

    J_up   = TIndex(J_sym, Up, vb)
    J_down = TIndex(J_sym, Down, vb)
    K_up   = TIndex(K_sym, Up, vb)
    K_down = TIndex(K_sym, Down, vb)

    # Term 1: ∂_a c^I
    term1 = TDeriv(a, Tensor(ggp.ghost, [I]), :partial)

    # Term 2: f^I_{JK} A^J_a c^K
    term2 = TProduct(1 // 1, TensorExpr[
        Tensor(ggp.structure_constants, [I, J_down, K_down]),
        Tensor(ggp.gauge_field, [J_up, a]),
        Tensor(ggp.ghost, [K_up])
    ])

    TSum(TensorExpr[term1, term2])
end

"""
    brst_ghost(ggp, I; registry) -> TProduct

BRST transformation of the ghost field:

    s(c^I) = -(1/2) f^I_{JK} c^J c^K
"""
function brst_ghost(ggp::GaugeGroupProperties,
                     I::TIndex;
                     registry::TensorRegistry=current_registry())
    I.position === Up || error("brst_ghost: I must be Up")

    vb = ggp.vbundle
    used = Set{Symbol}([I.name])
    J_sym = fresh_index(used; vbundle=vb)
    push!(used, J_sym)
    K_sym = fresh_index(used; vbundle=vb)

    J_up   = TIndex(J_sym, Up, vb)
    J_down = TIndex(J_sym, Down, vb)
    K_up   = TIndex(K_sym, Up, vb)
    K_down = TIndex(K_sym, Down, vb)

    TProduct(-1 // 2, TensorExpr[
        Tensor(ggp.structure_constants, [I, J_down, K_down]),
        Tensor(ggp.ghost, [J_up]),
        Tensor(ggp.ghost, [K_up])
    ])
end

"""
    brst_anti_ghost(ggp, I; registry) -> Tensor

BRST transformation of the anti-ghost:

    s(c̄^I) = B^I
"""
function brst_anti_ghost(ggp::GaugeGroupProperties,
                          I::TIndex;
                          registry::TensorRegistry=current_registry())
    I.position === Up || error("brst_anti_ghost: I must be Up")
    Tensor(ggp.nl_field, [I])
end

"""
    brst_nl_field(ggp, I; registry) -> TScalar

BRST transformation of the Nakanishi-Lautrup field:

    s(B^I) = 0
"""
function brst_nl_field(ggp::GaugeGroupProperties,
                        I::TIndex;
                        registry::TensorRegistry=current_registry())
    TScalar(0 // 1)
end

# ── Ghost number ─────────────────────────────────────────────────────

"""
    ghost_number(expr; registry=current_registry()) -> Int

Compute the total ghost number of an expression.

Ghost numbers: gauge field A=0, ghost c=+1, anti-ghost c̄=-1, B=0.
Ghost number is additive in products.
"""
function ghost_number(expr::TensorExpr;
                       registry::TensorRegistry=current_registry())
    _ghost_number(expr, registry)
end

function _ghost_number(t::Tensor, reg::TensorRegistry)
    has_tensor(reg, t.name) || return 0
    props = get_tensor(reg, t.name)
    get(props.options, :ghost_number, 0)::Int
end

function _ghost_number(p::TProduct, reg::TensorRegistry)
    s = 0
    for f in p.factors
        s += _ghost_number(f, reg)
    end
    s
end

function _ghost_number(s::TSum, reg::TensorRegistry)
    isempty(s.terms) && return 0
    _ghost_number(s.terms[1], reg)
end

function _ghost_number(::TScalar, ::TensorRegistry)
    0
end

function _ghost_number(d::TDeriv, reg::TensorRegistry)
    _ghost_number(d.arg, reg)
end

function _ghost_number(::GammaMatrix, ::TensorRegistry)
    0
end

function _ghost_number(::TensorExpr, ::TensorRegistry)
    0
end

"""
    filter_by_ghost_number(expr::TSum, n::Int;
                            registry=current_registry()) -> TSum

Extract terms from a TSum that have ghost number `n`.
"""
function filter_by_ghost_number(expr::TSum, n::Int;
                                 registry::TensorRegistry=current_registry())
    matching = TensorExpr[]
    for term in expr.terms
        if _ghost_number(term, registry) == n
            push!(matching, term)
        end
    end
    isempty(matching) ? TScalar(0 // 1) : TSum(matching)
end
