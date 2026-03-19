#= DHOST (Degenerate Higher-Order Scalar-Tensor) class I Lagrangian.

Implements the DHOST class I Lagrangian quadratic in second covariant
derivatives of the scalar field, following Langlois & Noui (2016),
JCAP 1602, 034, arXiv:1510.06930, Eqs 2.1-2.5:

  L_DHOST = f_0(phi,X) + f_1(phi,X) Box(phi) + sum_{i=1}^{5} a_i(phi,X) L_i

The five quadratic building blocks are:

  L_1 = (Box phi)^2
  L_2 = (nabla_a nabla_b phi)(nabla^a nabla^b phi)
  L_3 = (Box phi)(nabla_a phi)(nabla_b phi)(nabla^a nabla^b phi) / X
  L_4 = (nabla_a phi)(nabla^a nabla^b phi)(nabla_b nabla_c phi)(nabla^c phi) / X
  L_5 = [(nabla_a phi)(nabla_b phi)(nabla^a nabla^b phi)]^2 / X^2

where X = -(1/2) g^{ab} nabla_a phi nabla_b phi is the kinetic term.

The coefficient functions f_0, f_1, a_1, ..., a_5 are arbitrary functions of
(phi, X). Degenerate subclasses impose algebraic constraints on these
coefficients to ensure only 3 propagating degrees of freedom.
=#

# -- DHOSTTheory ---------------------------------------------------------------

"""
    DHOSTTheory

Container for a DHOST class I theory, linking manifold, metric, scalar field,
the background functions f_0, f_1, and the five quadratic coefficients a_1..a_5.
"""
struct DHOSTTheory
    manifold::Symbol
    metric::Symbol
    scalar_field::Symbol
    covd::Symbol
    f0::ScalarTensorFunction
    f1::ScalarTensorFunction
    a::NTuple{5, ScalarTensorFunction}   # a_1 ... a_5
end

# -- Registration --------------------------------------------------------------

"""
    define_dhost!(reg; manifold, metric, scalar_field=:phi, covd=:nabla)

Register the scalar field and all DHOST coefficient functions (f0, f1, a1..a5)
as rank-0 tensors.  Returns a `DHOSTTheory` struct.
"""
function define_dhost!(reg::TensorRegistry;
                       manifold::Symbol, metric::Symbol,
                       scalar_field::Symbol=:phi,
                       covd::Symbol=:nabla)
    has_manifold(reg, manifold) || error("Manifold $manifold not registered")

    # Register scalar field (rank-0) if not present
    if !has_tensor(reg, scalar_field)
        register_tensor!(reg, TensorProperties(
            name=scalar_field, manifold=manifold, rank=(0, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}()))
    end

    # Coefficient functions
    f0 = ScalarTensorFunction(:f0, 0, 0)
    f1 = ScalarTensorFunction(:f1, 0, 0)
    a1 = ScalarTensorFunction(:a1, 0, 0)
    a2 = ScalarTensorFunction(:a2, 0, 0)
    a3 = ScalarTensorFunction(:a3, 0, 0)
    a4 = ScalarTensorFunction(:a4, 0, 0)
    a5 = ScalarTensorFunction(:a5, 0, 0)

    needed = [f0, f1, a1, a2, a3, a4, a5]

    for stf in needed
        tname = g_tensor_name(stf)
        if !has_tensor(reg, tname)
            register_tensor!(reg, TensorProperties(
                name=tname, manifold=manifold, rank=(0, 0),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}(:is_scalar_tensor_function => true,
                                         :stf_base => stf.name,
                                         :stf_phi_derivs => stf.phi_derivs,
                                         :stf_X_derivs => stf.X_derivs)))
        end
    end

    # Define covariant derivative if not already present
    if !has_tensor(reg, covd)
        define_covd!(reg, covd; manifold=manifold, metric=metric)
    end

    DHOSTTheory(manifold, metric, scalar_field, covd,
                f0, f1, (a1, a2, a3, a4, a5))
end

# -- Individual DHOST Lagrangian pieces ----------------------------------------

"""
    dhost_L1(dht; registry) -> TensorExpr

L_1 = (Box phi)^2.  Langlois & Noui (2016) Eq 2.1.
"""
function dhost_L1(dht::DHOSTTheory;
                  registry::TensorRegistry=current_registry())
    phi = Tensor(dht.scalar_field, TIndex[])
    box1 = box(phi, dht.metric; registry=registry)
    box2 = box(phi, dht.metric; registry=registry)
    box1 * box2
end

"""
    dhost_L2(dht; registry) -> TensorExpr

L_2 = (nabla_a nabla_b phi)(nabla^a nabla^b phi).  Langlois & Noui (2016) Eq 2.2.
"""
function dhost_L2(dht::DHOSTTheory;
                  registry::TensorRegistry=current_registry())
    with_registry(registry) do
        phi = Tensor(dht.scalar_field, TIndex[])
        used = Set{Symbol}()
        a = fresh_index(used); push!(used, a)
        b = fresh_index(used); push!(used, b)
        c = fresh_index(used); push!(used, c)
        d = fresh_index(used)

        dd1 = covd_chain(phi, [down(a), down(b)])
        dd2 = covd_chain(phi, [down(c), down(d)])
        g_ac = Tensor(dht.metric, [up(a), up(c)])
        g_bd = Tensor(dht.metric, [up(b), up(d)])
        g_ac * g_bd * dd1 * dd2
    end
end

"""
    dhost_L3(dht; registry) -> TensorExpr

L_3 = (Box phi)(nabla^a phi)(nabla^b phi)(nabla_a nabla_b phi) / X.
Langlois & Noui (2016) Eq 2.3.

Since X is a scalar function of the field, we represent 1/X as an explicit
TScalar(:Xinv) factor.  The caller must provide a rule or evaluation for Xinv.
Alternatively, the full Lagrangian multiplies a_3 * L_3 where a_3 absorbs X.
"""
function dhost_L3(dht::DHOSTTheory;
                  registry::TensorRegistry=current_registry())
    with_registry(registry) do
        phi = Tensor(dht.scalar_field, TIndex[])
        used = Set{Symbol}()
        a = fresh_index(used); push!(used, a)
        b = fresh_index(used); push!(used, b)
        c = fresh_index(used); push!(used, c)
        d = fresh_index(used)

        box_phi = box(phi, dht.metric; registry=registry)

        # nabla^a phi = g^{ac} d_c phi
        dphi_up_a = Tensor(dht.metric, [up(a), up(c)]) * TDeriv(down(c), phi)
        # nabla^b phi = g^{bd} d_d phi
        dphi_up_b = Tensor(dht.metric, [up(b), up(d)]) * TDeriv(down(d), phi)

        dd_ab = covd_chain(phi, [down(a), down(b)])

        box_phi * dphi_up_a * dphi_up_b * dd_ab
    end
end

"""
    dhost_L4(dht; registry) -> TensorExpr

L_4 = (nabla^a phi)(nabla_a nabla_b phi)(nabla^b nabla^c phi)(nabla_c phi) / X.
Langlois & Noui (2016) Eq 2.4.  (Division by X absorbed into coefficient a_4.)
"""
function dhost_L4(dht::DHOSTTheory;
                  registry::TensorRegistry=current_registry())
    with_registry(registry) do
        phi = Tensor(dht.scalar_field, TIndex[])
        used = Set{Symbol}()
        a = fresh_index(used); push!(used, a)
        b = fresh_index(used); push!(used, b)
        c = fresh_index(used); push!(used, c)
        d = fresh_index(used); push!(used, d)
        e = fresh_index(used); push!(used, e)
        f = fresh_index(used)

        # nabla^a phi = g^{ad} d_d phi
        dphi_up_a = Tensor(dht.metric, [up(a), up(d)]) * TDeriv(down(d), phi)
        # nabla_c phi = d_c phi
        dphi_down_c = TDeriv(down(c), phi)

        # nabla_a nabla_b phi
        dd_ab = covd_chain(phi, [down(a), down(b)])
        # nabla_e nabla_f phi  (will become nabla^b nabla^c via metric contraction)
        dd_ef = covd_chain(phi, [down(e), down(f)])
        g_be = Tensor(dht.metric, [up(b), up(e)])
        g_cf = Tensor(dht.metric, [up(c), up(f)])

        dphi_up_a * dd_ab * g_be * g_cf * dd_ef * dphi_down_c
    end
end

"""
    dhost_L5(dht; registry) -> TensorExpr

L_5 = [(nabla^a phi)(nabla^b phi)(nabla_a nabla_b phi)]^2 / X^2.
Langlois & Noui (2016) Eq 2.5.  (Division by X^2 absorbed into coefficient a_5.)
"""
function dhost_L5(dht::DHOSTTheory;
                  registry::TensorRegistry=current_registry())
    with_registry(registry) do
        phi = Tensor(dht.scalar_field, TIndex[])
        used = Set{Symbol}()

        # First copy: (nabla^a phi)(nabla^b phi)(nabla_a nabla_b phi)
        a = fresh_index(used); push!(used, a)
        b = fresh_index(used); push!(used, b)
        c = fresh_index(used); push!(used, c)
        d = fresh_index(used); push!(used, d)
        dphi_up_a = Tensor(dht.metric, [up(a), up(c)]) * TDeriv(down(c), phi)
        dphi_up_b = Tensor(dht.metric, [up(b), up(d)]) * TDeriv(down(d), phi)
        dd_ab = covd_chain(phi, [down(a), down(b)])

        # Second copy with fresh indices
        e = fresh_index(used); push!(used, e)
        f = fresh_index(used); push!(used, f)
        p = fresh_index(used); push!(used, p)
        q = fresh_index(used)
        dphi_up_e = Tensor(dht.metric, [up(e), up(p)]) * TDeriv(down(p), phi)
        dphi_up_f = Tensor(dht.metric, [up(f), up(q)]) * TDeriv(down(q), phi)
        dd_ef = covd_chain(phi, [down(e), down(f)])

        dphi_up_a * dphi_up_b * dd_ab * dphi_up_e * dphi_up_f * dd_ef
    end
end

# -- Full DHOST Lagrangian -----------------------------------------------------

"""
    dhost_lagrangian(dht::DHOSTTheory; registry) -> TensorExpr

Full DHOST class I Lagrangian:

  L = f_0(phi,X) + f_1(phi,X) Box(phi) + sum_{i=1}^{5} a_i(phi,X) L_i

The 1/X and 1/X^2 factors in L_3, L_4, L_5 are absorbed into the coefficient
functions a_3, a_4, a_5 respectively, following the convention of Langlois &
Noui (2016) arXiv:1510.06930.

Ground truth: Langlois & Noui, JCAP 1602 (2016) 034, arXiv:1510.06930, Eq 2.1.
"""
function dhost_lagrangian(dht::DHOSTTheory;
                          registry::TensorRegistry=current_registry())
    with_registry(registry) do
        phi = Tensor(dht.scalar_field, TIndex[])

        # f_0 + f_1 Box(phi)
        f0t = Tensor(g_tensor_name(dht.f0), TIndex[])
        f1t = Tensor(g_tensor_name(dht.f1), TIndex[])
        box_phi = box(phi, dht.metric; registry=registry)
        base = f0t + f1t * box_phi

        # sum a_i L_i
        L_funcs = [dhost_L1, dhost_L2, dhost_L3, dhost_L4, dhost_L5]
        sum_term = ZERO
        for (i, Lfn) in enumerate(L_funcs)
            ai = Tensor(g_tensor_name(dht.a[i]), TIndex[])
            Li = Lfn(dht; registry=registry)
            sum_term = sum_term + ai * Li
        end

        base + sum_term
    end
end
