# Metric-affine Riemann tensor for general affine connection.
#
# R^a_{bcd}(Γ) = ∂_c Γ^a_{bd} - ∂_d Γ^a_{bc} + Γ^a_{ce}Γ^e_{bd} - Γ^a_{de}Γ^e_{bc}
#
# Decomposition into Riemannian + distortion parts:
#   R^a_{bcd}(Γ) = R^a_{bcd}(LC) + ∇_c N^a_{bd} - ∇_d N^a_{bc}
#                  + N^a_{ce}N^e_{bd} - N^a_{de}N^e_{bc}
#
# Key differences from Riemannian case:
#   - R_{abcd} ≠ R_{cdab} in general (pair symmetry only with metric compatibility)
#   - R_{[abc]d} ≠ 0 in general (first Bianchi with torsion)
#   - R_{abcd} = -R_{abdc} still holds (antisymmetry in last two indices)
#
# Ground truth: Hehl et al, Phys. Rep. 258 (1995), Sec 3.

"""
    MAFieldStrength

Container for the metric-affine curvature tensors.

# Fields
- `riemann_name::Symbol`      -- R^a_{bcd}(Γ) full curvature
- `ricci_name::Symbol`        -- R_{bd} = R^a_{bad}
- `scalar_name::Symbol`       -- R = g^{bd} R_{bd}
- `connection::Symbol`        -- parent connection
"""
struct MAFieldStrength
    riemann_name::Symbol
    ricci_name::Symbol
    scalar_name::Symbol
    connection::Symbol
end

function Base.show(io::IO, fs::MAFieldStrength)
    print(io, "MAFieldStrength(Riem=:$(fs.riemann_name), Ric=:$(fs.ricci_name))")
end

"""
    define_ma_curvature!(reg::TensorRegistry, ac::AffineConnection;
                          manifold::Symbol=:M4) -> MAFieldStrength

Register the curvature tensors for a general affine connection.

Creates:
- Full Riemann R^a_{bcd}(Γ) — antisymmetric in c,d only (NOT pair-symmetric)
- Ricci tensor R_{bd} = R^a_{bad} — NOT necessarily symmetric (asymmetric Ricci)
- Ricci scalar R = g^{bd} R_{bd}

Note: for a general affine connection, the Ricci tensor has an antisymmetric
part: R_{[ab]} ∝ (curvature of the trace of the connection), which
relates to the electromagnetic-like U(1) piece (Weyl vector).

Ground truth: Hehl et al (1995) Sec 3.
"""
function define_ma_curvature!(reg::TensorRegistry, ac::AffineConnection;
                               manifold::Symbol=ac.manifold)
    riem_name = Symbol(:Riem_, ac.name)
    ric_name = Symbol(:Ric_, ac.name)
    scalar_name = Symbol(:RicScalar_, ac.name)

    # Full Riemann: antisymmetric in last two indices ONLY
    if !has_tensor(reg, riem_name)
        register_tensor!(reg, TensorProperties(
            name=riem_name, manifold=manifold, rank=(1, 3),
            symmetries=SymmetrySpec[AntiSymmetric(3, 4)],
            options=Dict{Symbol,Any}(
                :is_curvature => true,
                :is_metric_affine => true,
                :connection => ac.name,
                :no_pair_symmetry => true)))
    end

    # Ricci: NO symmetry (asymmetric in general)
    if !has_tensor(reg, ric_name)
        register_tensor!(reg, TensorProperties(
            name=ric_name, manifold=manifold, rank=(0, 2),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_ricci => true,
                :is_metric_affine => true,
                :connection => ac.name,
                :asymmetric => true)))
    end

    # Ricci scalar
    if !has_tensor(reg, scalar_name)
        register_tensor!(reg, TensorProperties(
            name=scalar_name, manifold=manifold, rank=(0, 0),
            symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}(
                :is_ricci_scalar => true,
                :is_metric_affine => true,
                :connection => ac.name)))
    end

    MAFieldStrength(riem_name, ric_name, scalar_name, ac.name)
end

"""
    ma_riemann_decomposition(ac::AffineConnection;
                              registry::TensorRegistry=current_registry()) -> TensorExpr

Return the decomposition of the metric-affine Riemann tensor into
Riemannian part + distortion contributions:

    R^a_{bcd}(Γ) = R^a_{bcd}(LC) + ∇_c N^a_{bd} - ∇_d N^a_{bc}
                   + N^a_{ce} N^e_{bd} - N^a_{de} N^e_{bc}

Returns a TSum with the Riemannian term + 4 distortion terms.
"""
function ma_riemann_decomposition(ac::AffineConnection;
                                   registry::TensorRegistry=current_registry())
    N = ac.distortion_name

    # R^a_{bcd}(LC) — standard Riemannian curvature
    R_LC = Tensor(:Riem, [up(:a), down(:b), down(:c), down(:d)])

    # ∇_c N^a_{bd}
    covd_N1 = TDeriv(down(:c), Tensor(N, [up(:a), down(:b), down(:d)]), :D)

    # -∇_d N^a_{bc}
    covd_N2 = TDeriv(down(:d), Tensor(N, [up(:a), down(:b), down(:c)]), :D)

    # N^a_{ce} N^e_{bd}
    NN1 = tproduct(1 // 1, TensorExpr[
        Tensor(N, [up(:a), down(:c), down(:e)]),
        Tensor(N, [up(:e), down(:b), down(:d)])
    ])

    # -N^a_{de} N^e_{bc}
    NN2 = tproduct(1 // 1, TensorExpr[
        Tensor(N, [up(:a), down(:d), down(:e)]),
        Tensor(N, [up(:e), down(:b), down(:c)])
    ])

    tsum(TensorExpr[
        R_LC,
        covd_N1,
        tproduct(-1 // 1, TensorExpr[covd_N2]),
        NN1,
        tproduct(-1 // 1, TensorExpr[NN2])
    ])
end

# ──────────────────────────────────────────────────────────────────────
#  Poincare Gauge Theory action
#
#  The most general parity-even quadratic action in torsion and curvature
#  for a metric-compatible connection with torsion:
#
#    S_PGT = ∫ d⁴x √g [ a₀ R + T²-sector + R²-sector ]
#
#  Torsion sector (3 independent quadratic invariants):
#    I₁ = T_{abc} T^{abc}
#    I₂ = T_{abc} T^{bac}
#    I₃ = T^a T_a       (torsion trace squared)
#
#  Curvature sector (6 independent quadratic invariants):
#    J₁ = R_{abcd} R^{abcd}
#    J₂ = R_{abcd} R^{cdab}
#    J₃ = R_{ab} R^{ab}
#    J₄ = R_{ab} R^{ba}
#    J₅ = R²
#    J₆ = R_{[ab]} R^{[ab]}   (antisymmetric Ricci squared)
#
#  Note: In PGT, the Riemann tensor has only antisymmetry in the last two
#  indices (NOT pair symmetry). The Ricci tensor R_{ab} is NOT symmetric.
#  Hence J₃ ≠ J₄ and J₆ is independent.
#
#  Ground truth: Blagojevic & Hehl, "Gauge Theories of Gravitation" (2013), Ch 5;
#                Hehl, McCrea, Mielke & Ne'eman, Phys. Rep. 258 (1995), Sec 5.
# ──────────────────────────────────────────────────────────────────────

"""
    PGTParams

Parameters for the Poincare gauge theory action.

# Fields
- `a0`                 -- Einstein-Hilbert coefficient (multiplies R)
- `t::NTuple{3,Any}`   -- torsion sector couplings (t₁, t₂, t₃)
- `r::NTuple{6,Any}`   -- curvature sector couplings (r₁, ..., r₆)

The full Lagrangian is:
    L = a₀ R + t₁ I₁ + t₂ I₂ + t₃ I₃ + r₁ J₁ + ... + r₆ J₆

Ground truth: Hehl et al, Phys. Rep. 258 (1995), Sec 5.
"""
struct PGTParams
    a0::Any
    t::NTuple{3,Any}
    r::NTuple{6,Any}
end

function PGTParams(; a0=1, t1=0, t2=0, t3=0,
                     r1=0, r2=0, r3=0, r4=0, r5=0, r6=0)
    PGTParams(a0, (t1, t2, t3), (r1, r2, r3, r4, r5, r6))
end

function Base.show(io::IO, p::PGTParams)
    print(io, "PGTParams(a₀=$(p.a0), t=$(p.t), r=$(p.r))")
end

"""
    torsion_quadratic(ac::AffineConnection;
                       registry::TensorRegistry=current_registry())
        -> NamedTuple{(:I1,:I2,:I3), NTuple{3,TensorExpr}}

Compute the 3 independent torsion quadratic invariants:
- I₁ = T_{abc} T^{abc}      (full contraction, same index order)
- I₂ = T_{abc} T^{bac}      (full contraction, permuted indices)
- I₃ = T^a T_a              (torsion trace squared)

These correspond to the three independent quadratic contractions
of the torsion tensor T^a_{bc} (antisymmetric in bc).

All returned expressions are scalars (no free indices).

Ground truth: Hehl et al, Phys. Rep. 258 (1995), Sec 5.
"""
function torsion_quadratic(ac::AffineConnection;
                            registry::TensorRegistry=current_registry())
    T = ac.torsion_name

    # I₁ = T_{abc} T^{abc} = g_{ae} g^{bf} g^{cg} T^a_{bc} T^e_{fg}
    # Contracts all three index pairs between lowered T₁ and raised T₂.

    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    e = fresh_index(used); push!(used, e)
    f = fresh_index(used); push!(used, f)
    g_idx = fresh_index(used); push!(used, g_idx)

    I1 = tproduct(1 // 1, TensorExpr[
        Tensor(:g, [down(a), down(e)]),
        Tensor(:g, [up(b), up(f)]),
        Tensor(:g, [up(c), up(g_idx)]),
        Tensor(T, [up(a), down(b), down(c)]),
        Tensor(T, [up(e), down(f), down(g_idx)])
    ])

    # I₂ = T_{abc} T^{bac} = g_{ae} g^{bf} g^{cg} T^a_{bc} T^e_{gf}
    # Same metrics as I₁ but second torsion has swapped lower indices (g<->f).

    I2 = tproduct(1 // 1, TensorExpr[
        Tensor(:g, [down(a), down(e)]),
        Tensor(:g, [up(b), up(f)]),
        Tensor(:g, [up(c), up(g_idx)]),
        Tensor(T, [up(a), down(b), down(c)]),
        Tensor(T, [up(e), down(g_idx), down(f)])
    ])

    # I₃ = T^a T_a (torsion trace squared)
    # T_b = T^a_{ba} (self-contraction), then T^a T_a = g^{bf} T_b T_f

    I3 = tproduct(1 // 1, TensorExpr[
        Tensor(T, [up(a), down(b), down(a)]),
        Tensor(T, [up(e), down(f), down(e)]),
        Tensor(:g, [up(b), up(f)])
    ])

    (I1=I1, I2=I2, I3=I3)
end

"""
    curvature_quadratic_ma(ac::AffineConnection, fs::MAFieldStrength;
                            registry::TensorRegistry=current_registry())
        -> NamedTuple{(:J1,:J2,:J3,:J4,:J5,:J6), NTuple{6,TensorExpr}}

Compute the 6 independent curvature quadratic invariants for a
metric-affine connection:

- J₁ = R_{abcd} R^{abcd}         (Kretschmann-like, same index order)
- J₂ = R_{abcd} R^{cdab}         (pair-exchanged contraction)
- J₃ = R_{ab} R^{ab}             (Ricci squared, same order)
- J₄ = R_{ab} R^{ba}             (Ricci squared, transposed)
- J₅ = R²                        (scalar squared)
- J₆ = R_{[ab]} R^{[ab]}         (antisymmetric Ricci squared)

In PGT (metric-affine with metric compatibility), the Riemann tensor
has only antisymmetry in the last two indices, and the Ricci tensor
is NOT symmetric. Hence J₃ ≠ J₄ and J₆ is independent.

All returned expressions are scalars (no free indices).

Ground truth: Hehl et al, Phys. Rep. 258 (1995), Sec 5.
"""
function curvature_quadratic_ma(ac::AffineConnection, fs::MAFieldStrength;
                                 registry::TensorRegistry=current_registry())
    Riem = fs.riemann_name
    Ric = fs.ricci_name
    R = fs.scalar_name

    used = Set{Symbol}()
    a = fresh_index(used); push!(used, a)
    b = fresh_index(used); push!(used, b)
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used); push!(used, d)
    e = fresh_index(used); push!(used, e)
    f = fresh_index(used); push!(used, f)
    g_idx = fresh_index(used); push!(used, g_idx)
    h = fresh_index(used); push!(used, h)

    # J₁ = R_{abcd} R^{abcd} = g_{ae} g^{bf} g^{cg} g^{dh} R^a_{bcd} R^e_{fgh}
    # Contracts all four index pairs between lowered R₁ and raised R₂.

    J1 = tproduct(1 // 1, TensorExpr[
        Tensor(:g, [down(a), down(e)]),
        Tensor(:g, [up(b), up(f)]),
        Tensor(:g, [up(c), up(g_idx)]),
        Tensor(:g, [up(d), up(h)]),
        Tensor(Riem, [up(a), down(b), down(c), down(d)]),
        Tensor(Riem, [up(e), down(f), down(g_idx), down(h)])
    ])

    # J₂ = R_{abcd} R^{cdab}
    # R_{a,b,c,d} = g_{a,m} R^m_{b,c,d}  [lower slot 1]
    # R^{c,d,a,b} = g^{d,p} g^{a,q} g^{b,r} R^c_{p,q,r}  [raise slots 2,3,4]
    #
    # Product: g_{a,m} R^m_{b,c,d} · g^{d,p} g^{a,q} g^{b,r} R^c_{p,q,r}
    # Index check: a(2), b(2), c(2), d(2), m(2), p(2), q(2), r(2) — all pairs ✓

    used3 = Set{Symbol}()
    aa = fresh_index(used3); push!(used3, aa)
    bb = fresh_index(used3); push!(used3, bb)
    cc = fresh_index(used3); push!(used3, cc)
    dd = fresh_index(used3); push!(used3, dd)
    mm = fresh_index(used3); push!(used3, mm)
    pp = fresh_index(used3); push!(used3, pp)
    qq = fresh_index(used3); push!(used3, qq)
    rr = fresh_index(used3); push!(used3, rr)

    J2 = tproduct(1 // 1, TensorExpr[
        Tensor(:g, [down(aa), down(mm)]),
        Tensor(Riem, [up(mm), down(bb), down(cc), down(dd)]),
        Tensor(:g, [up(dd), up(pp)]),
        Tensor(:g, [up(aa), up(qq)]),
        Tensor(:g, [up(bb), up(rr)]),
        Tensor(Riem, [up(cc), down(pp), down(qq), down(rr)])
    ])

    # J₃ = R_{ab} R^{ab}
    # Ric has rank (0,2), so R_{ab} = Ric_{ab} directly (both down).
    # R^{ab} = g^{ac} g^{bd} R_{cd}
    # J₃ = Ric_{a,b} · g^{a,c} g^{b,d} · Ric_{c,d}

    used4 = Set{Symbol}()
    a4 = fresh_index(used4); push!(used4, a4)
    b4 = fresh_index(used4); push!(used4, b4)
    c4 = fresh_index(used4); push!(used4, c4)
    d4 = fresh_index(used4); push!(used4, d4)

    J3 = tproduct(1 // 1, TensorExpr[
        Tensor(Ric, [down(a4), down(b4)]),
        Tensor(:g, [up(a4), up(c4)]),
        Tensor(:g, [up(b4), up(d4)]),
        Tensor(Ric, [down(c4), down(d4)])
    ])

    # J₄ = R_{ab} R^{ba}
    # = Ric_{a,b} · g^{b,c} g^{a,d} · Ric_{c,d}
    # Same as J₃ but with transposed contraction on second Ricci.
    # = Ric_{a,b} · g^{a,c} g^{b,d} · Ric_{d,c}

    used5 = Set{Symbol}()
    a5 = fresh_index(used5); push!(used5, a5)
    b5 = fresh_index(used5); push!(used5, b5)
    c5 = fresh_index(used5); push!(used5, c5)
    d5 = fresh_index(used5); push!(used5, d5)

    J4 = tproduct(1 // 1, TensorExpr[
        Tensor(Ric, [down(a5), down(b5)]),
        Tensor(:g, [up(a5), up(c5)]),
        Tensor(:g, [up(b5), up(d5)]),
        Tensor(Ric, [down(d5), down(c5)])
    ])

    # J₅ = R²
    # = RicScalar · RicScalar

    J5 = tproduct(1 // 1, TensorExpr[
        Tensor(R, TIndex[]),
        Tensor(R, TIndex[])
    ])

    # J₆ = R_{[ab]} R^{[ab]}
    # The antisymmetric part of Ricci: R_{[ab]} = (1/2)(R_{ab} - R_{ba})
    # R_{[ab]} R^{[ab]} = (1/4)(R_{ab} - R_{ba})(R^{ab} - R^{ba})
    # = (1/4)(R_{ab}R^{ab} - R_{ab}R^{ba} - R_{ba}R^{ab} + R_{ba}R^{ba})
    # = (1/4)(J₃ - J₄ - J₄ + J₃) = (1/2)(J₃ - J₄)
    #
    # But we want to express J₆ as an independent expression, not in terms
    # of J₃ and J₄ (even though algebraically J₆ = (J₃-J₄)/2).
    # For clarity and testing, build it explicitly as (1/2)(J₃ - J₄).

    used6 = Set{Symbol}()
    a6 = fresh_index(used6); push!(used6, a6)
    b6 = fresh_index(used6); push!(used6, b6)
    c6 = fresh_index(used6); push!(used6, c6)
    d6 = fresh_index(used6); push!(used6, d6)

    # J₆ = (1/2)(R_{ab}R^{ab} - R_{ab}R^{ba})
    J6_plus = tproduct(1 // 2, TensorExpr[
        Tensor(Ric, [down(a6), down(b6)]),
        Tensor(:g, [up(a6), up(c6)]),
        Tensor(:g, [up(b6), up(d6)]),
        Tensor(Ric, [down(c6), down(d6)])
    ])

    # Need fresh indices for the second term
    used7 = Set{Symbol}()
    a7 = fresh_index(used7); push!(used7, a7)
    b7 = fresh_index(used7); push!(used7, b7)
    c7 = fresh_index(used7); push!(used7, c7)
    d7 = fresh_index(used7); push!(used7, d7)

    J6_minus = tproduct(-1 // 2, TensorExpr[
        Tensor(Ric, [down(a7), down(b7)]),
        Tensor(:g, [up(a7), up(c7)]),
        Tensor(:g, [up(b7), up(d7)]),
        Tensor(Ric, [down(d7), down(c7)])
    ])

    J6 = tsum(TensorExpr[J6_plus, J6_minus])

    (J1=J1, J2=J2, J3=J3, J4=J4, J5=J5, J6=J6)
end

"""
    pgt_action(ac::AffineConnection, fs::MAFieldStrength, params::PGTParams;
               registry::TensorRegistry=current_registry()) -> TensorExpr

Build the full Poincare gauge theory Lagrangian density:

    L = a₀ R + t₁ I₁ + t₂ I₂ + t₃ I₃ + r₁ J₁ + ... + r₆ J₆

where Iₖ are the torsion quadratic invariants and Jₖ are the curvature
quadratic invariants.

Returns a scalar TensorExpr (the Lagrangian density, not including √g d⁴x).

Ground truth: Blagojevic & Hehl (2013) Ch 5; Hehl et al (1995) Sec 5.
"""
function pgt_action(ac::AffineConnection, fs::MAFieldStrength, params::PGTParams;
                    registry::TensorRegistry=current_registry())
    R = fs.scalar_name

    # Safe zero check that handles symbolic (Symbol) couplings
    _is_zero(x) = x isa Number && iszero(x)

    terms = TensorExpr[]

    # Einstein-Hilbert term: a₀ R
    a0 = params.a0
    if !_is_zero(a0)
        if a0 isa Rational{Int}
            push!(terms, tproduct(a0, TensorExpr[Tensor(R, TIndex[])]))
        elseif a0 isa Integer
            push!(terms, tproduct(a0 // 1, TensorExpr[Tensor(R, TIndex[])]))
        else
            push!(terms, tproduct(1 // 1, TensorExpr[TScalar(a0), Tensor(R, TIndex[])]))
        end
    end

    # Torsion sector: t₁ I₁ + t₂ I₂ + t₃ I₃
    tq = torsion_quadratic(ac; registry=registry)
    tor_invariants = (tq.I1, tq.I2, tq.I3)
    for (k, tk) in enumerate(params.t)
        _is_zero(tk) && continue
        inv_k = tor_invariants[k]
        if tk isa Rational{Int}
            push!(terms, tproduct(tk, TensorExpr[inv_k]))
        elseif tk isa Integer
            push!(terms, tproduct(tk // 1, TensorExpr[inv_k]))
        else
            push!(terms, tproduct(1 // 1, TensorExpr[TScalar(tk), inv_k]))
        end
    end

    # Curvature sector: r₁ J₁ + ... + r₆ J₆
    cq = curvature_quadratic_ma(ac, fs; registry=registry)
    curv_invariants = (cq.J1, cq.J2, cq.J3, cq.J4, cq.J5, cq.J6)
    for (k, rk) in enumerate(params.r)
        _is_zero(rk) && continue
        inv_k = curv_invariants[k]
        if rk isa Rational{Int}
            push!(terms, tproduct(rk, TensorExpr[inv_k]))
        elseif rk isa Integer
            push!(terms, tproduct(rk // 1, TensorExpr[inv_k]))
        else
            push!(terms, tproduct(1 // 1, TensorExpr[TScalar(rk), inv_k]))
        end
    end

    isempty(terms) && return TScalar(0)
    length(terms) == 1 && return terms[1]
    tsum(terms)
end

"""
    einstein_cartan_action(ac::AffineConnection, fs::MAFieldStrength;
                           registry::TensorRegistry=current_registry()) -> TensorExpr

The Einstein-Cartan action: the simplest Poincare gauge theory with only
the Ricci scalar (no quadratic torsion or curvature terms).

    L_EC = R

This is the metric-compatible connection with torsion, but the Lagrangian
is just the Ricci scalar of the full (torsioned) connection. The field
equations yield Einstein's equations plus an algebraic equation relating
torsion to spin density.

Equivalent to `pgt_action` with a₀=1 and all other couplings zero.

Ground truth: Cartan (1922); Kibble (1961); Sciama (1964).
"""
function einstein_cartan_action(ac::AffineConnection, fs::MAFieldStrength;
                                registry::TensorRegistry=current_registry())
    Tensor(fs.scalar_name, TIndex[])
end
