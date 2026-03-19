#= DHOST degeneracy conditions ensuring 3 propagating DOF.

Algebraic constraints on the DHOST coefficient functions a_i(phi,X) that
eliminate the Ostrogradsky ghost, ensuring the theory propagates exactly
3 degrees of freedom (1 scalar + 2 tensor polarizations).

The conditions are organized into three classes following:

  Langlois & Noui, JCAP 1602 (2016) 034, arXiv:1510.06930, Sec 3;
  Langlois, "Dark energy and modified gravity in DHOST theories: A review",
    IJMPD 28 (2019) 1942006, arXiv:1811.06271, Sec 3.2, Eqs 3.8-3.10;
  Ben Achour et al, PRD 93 (2016) 124005, arXiv:1602.08398, Table I.

CONVENTION NOTE:
  Our code:   L_1 = (Box phi)^2,      L_2 = phi_{ab} phi^{ab}
  Langlois:   L_1 = phi_{ab} phi^{ab}, L_2 = (Box phi)^2
  Mapping:    our a_1 = paper a_2,     our a_2 = paper a_1
  a_3, a_4, a_5 are the same in both conventions.

The three degeneracy conditions for Class I (f=0, pure quadratic sector)
in our code convention are (derived from Langlois 2019, Eqs 3.8-3.10 with
f_X=0 and the a_1<->a_2 swap):

  C1: 2 a_2 (a_1 + a_2) + 3 a_3^2 = 0
  C2: a_2 a_4 + a_3 (a_2 + a_3) = 0
  C3: 2 a_2^2 a_5 + a_3^2 (a_2 + a_3) = 0

These reduce correctly for:
  - Horndeski (a_1 = -a_2, a_3 = a_4 = a_5 = 0): all three = 0 identically.
  - Generic a_i: non-zero, indicating Ostrogradsky ghost is present.
=#

# -- Degeneracy condition expressions -----------------------------------------

"""
    degeneracy_conditions(theory::DHOSTTheory;
                          registry::TensorRegistry=current_registry()) -> Vector{Any}

Return the degeneracy conditions as symbolic scalar expressions that must all
vanish for the DHOST theory to be degenerate (3 propagating DOF).

Returns three conditions C1, C2, C3 as symbolic expressions built with
the `_sym_*` arithmetic. Each is a polynomial in the a_i coefficient names.

Ground truth: Langlois (2019) arXiv:1811.06271, Eqs 3.8-3.10 (with f_X=0).
"""
function degeneracy_conditions(theory::DHOSTTheory;
                               registry::TensorRegistry=current_registry())
    mul = _sym_mul
    add = _sym_add

    # Return 0 for vanishing coefficients so conditions simplify algebraically
    function _coeff(stf)
        name = g_tensor_name(stf)
        (has_tensor(registry, name) && get_tensor(registry, name).vanishing) ? 0 : name
    end

    a1 = _coeff(theory.a[1])
    a2 = _coeff(theory.a[2])
    a3 = _coeff(theory.a[3])
    a4 = _coeff(theory.a[4])
    a5 = _coeff(theory.a[5])

    # C1: 2 a_2(a_1 + a_2) + 3 a_3^2 = 0
    C1 = add(mul(2, mul(a2, add(a1, a2))),
             mul(3, mul(a3, a3)))

    # C2: a_2 a_4 + a_3(a_2 + a_3) = 0
    C2 = add(mul(a2, a4),
             mul(a3, add(a2, a3)))

    # C3: 2 a_2^2 a_5 + a_3^2(a_2 + a_3) = 0
    C3 = add(mul(2, mul(a2, mul(a2, a5))),
             mul(mul(a3, a3), add(a2, a3)))

    Any[C1, C2, C3]
end

# -- Check degeneracy ---------------------------------------------------------

"""
    is_degenerate(theory::DHOSTTheory;
                  registry::TensorRegistry=current_registry(),
                  values::Dict{Symbol,<:Number}=Dict{Symbol,Int}()) -> Bool

Check if the DHOST theory satisfies all degeneracy conditions.

If `values` is provided, evaluates the conditions numerically. Otherwise,
checks structurally: returns `true` only if all conditions evaluate to
exactly zero (works for Horndeski-like configurations where a_3=a_4=a_5=0).

Ground truth: Langlois & Noui (2016) arXiv:1510.06930, Sec 3.
"""
function is_degenerate(theory::DHOSTTheory;
                       registry::TensorRegistry=current_registry(),
                       values::Dict{Symbol,<:Number}=Dict{Symbol,Int}())
    conds = degeneracy_conditions(theory; registry=registry)

    if isempty(values)
        return all(c -> c == 0, conds)
    else
        return all(c -> abs(sym_eval(c, values)) < 1e-12, conds)
    end
end

# -- DHOST class identification ------------------------------------------------

"""
    dhost_class(theory::DHOSTTheory;
                registry::TensorRegistry=current_registry(),
                values::Dict{Symbol,<:Number}=Dict{Symbol,Int}()) -> Symbol

Classify the DHOST theory:
  - `:class_Ia`  -- Horndeski-like (a_1+a_2=0, a_3=a_4=a_5=0 in our convention)
  - `:class_Ib`  -- General Class I degenerate (a_2 != 0, conditions satisfied)
  - `:class_II`  -- a_2 = 0, a_1 != 0 (our convention)
  - `:class_III` -- a_1 = a_2 = 0 (our convention)
  - `:not_degenerate` -- degeneracy conditions not satisfied

Ground truth: Ben Achour et al (2016) arXiv:1602.08398, Table I.
"""
function dhost_class(theory::DHOSTTheory;
                     registry::TensorRegistry=current_registry(),
                     values::Dict{Symbol,<:Number}=Dict{Symbol,Int}())
    a1_name = g_tensor_name(theory.a[1])
    a2_name = g_tensor_name(theory.a[2])
    a3_name = g_tensor_name(theory.a[3])
    a4_name = g_tensor_name(theory.a[4])
    a5_name = g_tensor_name(theory.a[5])

    if !isempty(values)
        return _dhost_class_numerical(theory, values; registry=registry)
    end

    # Structural classification via vanishing rules
    a1_zero = _is_vanishing(registry, a1_name)
    a2_zero = _is_vanishing(registry, a2_name)
    a3_zero = _is_vanishing(registry, a3_name)
    a4_zero = _is_vanishing(registry, a4_name)
    a5_zero = _is_vanishing(registry, a5_name)

    if a1_zero && a2_zero
        return :class_III
    elseif a2_zero
        return :class_II
    else
        # Class I: check degeneracy
        if is_degenerate(theory; registry=registry)
            return (a3_zero && a4_zero && a5_zero) ? :class_Ia : :class_Ib
        else
            return :not_degenerate
        end
    end
end

function _dhost_class_numerical(theory::DHOSTTheory,
                                 values::Dict{Symbol,<:Number};
                                 registry::TensorRegistry=current_registry())
    tol = 1e-12
    a1v = get(values, g_tensor_name(theory.a[1]), 0.0)
    a2v = get(values, g_tensor_name(theory.a[2]), 0.0)
    a3v = get(values, g_tensor_name(theory.a[3]), 0.0)
    a4v = get(values, g_tensor_name(theory.a[4]), 0.0)
    a5v = get(values, g_tensor_name(theory.a[5]), 0.0)

    if abs(a1v) < tol && abs(a2v) < tol
        return :class_III
    elseif abs(a2v) < tol
        return :class_II
    else
        if is_degenerate(theory; registry=registry, values=values)
            if abs(a1v + a2v) < tol && abs(a3v) < tol &&
               abs(a4v) < tol && abs(a5v) < tol
                return :class_Ia
            else
                return :class_Ib
            end
        else
            return :not_degenerate
        end
    end
end

# -- Helper: check if a tensor is set to vanish --------------------------------

"""
    _is_vanishing(reg::TensorRegistry, name::Symbol) -> Bool

Check if a tensor has been set to vanish (via `set_vanishing!`).
"""
function _is_vanishing(reg::TensorRegistry, name::Symbol)
    for rule in get_rules(reg)
        if rule isa RewriteRule &&
           rule.pattern isa Tensor && rule.pattern.name == name &&
           rule.replacement == ZERO
            return true
        end
    end
    false
end

# -- Horndeski embedding -------------------------------------------------------

"""
    horndeski_as_dhost(ht::HorndeskiTheory;
                       registry::TensorRegistry=current_registry()) -> DHOSTTheory

Embed Horndeski as DHOST. The L_4 kinetic part
G_{4,X}[(Box phi)^2 - phi_{ab}^2] maps to:

  a_1 = G_{4,X}  (our L_1 = (Box phi)^2)
  a_2 = -G_{4,X} (our L_2 = phi_{ab}^2)
  a_3 = a_4 = a_5 = 0

Sets a_3, a_4, a_5 to vanish in the registry.

Ground truth: Langlois & Noui (2016) arXiv:1510.06930, Sec 4.1.
"""
function horndeski_as_dhost(ht::HorndeskiTheory;
                            registry::TensorRegistry=current_registry())
    dht = define_dhost!(registry; manifold=ht.manifold, metric=ht.metric,
                        scalar_field=ht.scalar_field, covd=ht.covd)

    set_vanishing!(registry, g_tensor_name(dht.a[3]))
    set_vanishing!(registry, g_tensor_name(dht.a[4]))
    set_vanishing!(registry, g_tensor_name(dht.a[5]))

    dht
end

# -- Reduce to Horndeski -------------------------------------------------------

"""
    reduce_to_horndeski(theory::DHOSTTheory;
                        registry::TensorRegistry=current_registry()) -> Union{HorndeskiTheory, Nothing}

If the DHOST theory is Class Ia (reduces to Horndeski), return the equivalent
HorndeskiTheory. Otherwise return `nothing`.

Ground truth: Langlois & Noui (2016) arXiv:1510.06930, Sec 4.1.
"""
function reduce_to_horndeski(theory::DHOSTTheory;
                             registry::TensorRegistry=current_registry())
    cls = dhost_class(theory; registry=registry)
    cls == :class_Ia || return nothing

    define_horndeski!(registry; manifold=theory.manifold,
                      metric=theory.metric,
                      scalar_field=theory.scalar_field,
                      covd=theory.covd)
end

# -- DOF count -----------------------------------------------------------------

"""
    dhost_dof_count(theory::DHOSTTheory;
                    registry::TensorRegistry=current_registry(),
                    values::Dict{Symbol,<:Number}=Dict{Symbol,Int}()) -> Int

Number of propagating degrees of freedom:
  3 for degenerate theories (1 scalar + 2 tensor)
  4 for non-degenerate (extra Ostrogradsky ghost)

Ground truth: Langlois & Noui (2016) arXiv:1510.06930, Sec 1.
"""
function dhost_dof_count(theory::DHOSTTheory;
                         registry::TensorRegistry=current_registry(),
                         values::Dict{Symbol,<:Number}=Dict{Symbol,Int}())
    is_degenerate(theory; registry=registry, values=values) ? 3 : 4
end
