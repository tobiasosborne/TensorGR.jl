# -- Regge-Wheeler-Zerilli decomposition on Schwarzschild background -----------
#
# Martel & Poisson, Phys. Rev. D 71, 104003 (2005), Secs IV-V.
#
# Decomposes the metric perturbation h_{ab} on a Schwarzschild background
# into even-parity (Zerilli) and odd-parity (Regge-Wheeler) sectors using
# tensor spherical harmonics.
#
# Even parity (MP Eqs 4.1-4.3):
#   p_{ab} = h^{lm}_{ab} Y_{lm}                (M^2 block: 3 coeffs)
#   p_{aB} = j^{lm}_a    Y^{lm}_B              (mixed:     2 coeffs, l >= 1)
#   p_{AB} = r^2 (K^{lm} Omega_{AB} Y + G^{lm} Z^{lm}_{AB})  (S^2 block, G l >= 2)
#
# Odd parity (MP Eqs 5.1-5.3):
#   p_{ab} = 0                                  (no odd M^2 block)
#   p_{aB} = h^{lm}_a X^{lm}_B                 (mixed:    2 coeffs, l >= 1)
#   p_{AB} = h^{lm}_2   X^{lm}_{AB}            (S^2:      1 coeff,  l >= 2)
#
# Gauge-invariant variables (MP Eqs 4.10-4.12, 5.7):
#   Even: h_tilde_{ab} = h_{ab} - nabla_a eps_b - nabla_b eps_a
#         K_tilde      = K + (1/2)l(l+1)G - (2/r) r^a eps_a
#         eps_a        = j_a - (1/2) r^2 nabla_a G
#   Odd:  h_tilde_a    = h_a - (1/2) nabla_a h_2 + (1/r) r_a h_2   (MP Eq 5.7)

"""
    SchwarzschildPerturbation

Schwarzschild metric perturbation decomposed into RW/Zerilli sectors.

Fields:
- `mass::Symbol`    -- symbolic mass parameter M
- `l::Int`          -- angular momentum quantum number
- `m::Int`          -- magnetic quantum number
- `parity::Parity`  -- EVEN (Zerilli) or ODD (Regge-Wheeler)
- `coeffs::Dict{Symbol,Symbol}` -- sector => coefficient name
- `gauge_invariant::Dict{Symbol,Symbol}` -- gauge-invariant variable names
"""
struct SchwarzschildPerturbation
    mass::Symbol
    l::Int
    m::Int
    parity::Parity
    coeffs::Dict{Symbol,Symbol}
    gauge_invariant::Dict{Symbol,Symbol}
end

Base.:(==)(a::SchwarzschildPerturbation, b::SchwarzschildPerturbation) =
    a.mass == b.mass && a.l == b.l && a.m == b.m && a.parity == b.parity &&
    a.coeffs == b.coeffs && a.gauge_invariant == b.gauge_invariant
Base.hash(a::SchwarzschildPerturbation, h::UInt) =
    hash(a.gauge_invariant, hash(a.coeffs, hash(a.parity,
    hash(a.m, hash(a.l, hash(a.mass, hash(:SchwarzschildPerturbation, h)))))))

function Base.show(io::IO, sp::SchwarzschildPerturbation)
    p = sp.parity == EVEN ? "even/Zerilli" : "odd/RW"
    print(io, "SchwarzschildPert(M=", sp.mass, ", l=", sp.l, ", m=", sp.m,
          ", ", p, ", ", length(sp.coeffs), " coeffs)")
end

# ── Even-parity coefficients (Zerilli sector) ──────────────────────────────

function _schw_even_coeffs(prefix::Symbol, l::Int, m::Int)
    ms = _m_str(m)
    coeffs = Dict{Symbol,Symbol}()
    # M^2 block: h_{tt}, h_{tr}, h_{rr} -- always present (l >= 0)
    coeffs[:H0] = Symbol(prefix, "_H0_", l, "_", ms)   # MP: h_{tt} = f H_0
    coeffs[:H1] = Symbol(prefix, "_H1_", l, "_", ms)   # MP: h_{tr} = H_1
    coeffs[:H2] = Symbol(prefix, "_H2_", l, "_", ms)   # MP: h_{rr} = H_2/f
    # Mixed block: j_t, j_r (l >= 1)
    if l >= 1
        coeffs[:jt] = Symbol(prefix, "_jt_", l, "_", ms)
        coeffs[:jr] = Symbol(prefix, "_jr_", l, "_", ms)
    end
    # S^2 block: K (l >= 0)
    coeffs[:K] = Symbol(prefix, "_K_", l, "_", ms)
    # S^2 block: G (l >= 2)
    if l >= 2
        coeffs[:G] = Symbol(prefix, "_G_", l, "_", ms)
    end
    coeffs
end

# ── Even-parity gauge-invariant variables (MP Eqs 4.10-4.12) ───────────────

function _schw_even_gi(prefix::Symbol, l::Int, m::Int)
    ms = _m_str(m)
    gi = Dict{Symbol,Symbol}()
    # h_tilde_{ab}: gauge-invariant metric perturbation on M^2
    gi[:htilde_tt] = Symbol(prefix, "_htilde_tt_", l, "_", ms)
    gi[:htilde_tr] = Symbol(prefix, "_htilde_tr_", l, "_", ms)
    gi[:htilde_rr] = Symbol(prefix, "_htilde_rr_", l, "_", ms)
    # K_tilde: gauge-invariant K (MP Eq 4.11)
    gi[:Ktilde] = Symbol(prefix, "_Ktilde_", l, "_", ms)
    gi
end

# ── Odd-parity coefficients (Regge-Wheeler sector) ─────────────────────────

function _schw_odd_coeffs(prefix::Symbol, l::Int, m::Int)
    ms = _m_str(m)
    coeffs = Dict{Symbol,Symbol}()
    # Mixed block: h_t, h_r (l >= 1)
    if l >= 1
        coeffs[:ht] = Symbol(prefix, "_ht_", l, "_", ms)
        coeffs[:hr] = Symbol(prefix, "_hr_", l, "_", ms)
    end
    # S^2 block: h_2 (l >= 2)
    if l >= 2
        coeffs[:h2] = Symbol(prefix, "_h2_", l, "_", ms)
    end
    coeffs
end

# ── Odd-parity gauge-invariant variable (MP Eq 5.7) ────────────────────────

function _schw_odd_gi(prefix::Symbol, l::Int, m::Int)
    ms = _m_str(m)
    gi = Dict{Symbol,Symbol}()
    if l >= 1
        gi[:htilde_t] = Symbol(prefix, "_htilde_t_", l, "_", ms)
        gi[:htilde_r] = Symbol(prefix, "_htilde_r_", l, "_", ms)
    end
    gi
end

# ── Main decomposition function ────────────────────────────────────────────

"""
    decompose_schwarzschild(h_name::Symbol, mass::Symbol, lmax::Int;
                            registry=nothing) -> Vector{SchwarzschildPerturbation}

Decompose metric perturbation `h_name` on a Schwarzschild background with
mass `mass` into Regge-Wheeler (odd) and Zerilli (even) sectors for all
modes (l,m) with 0 <= l <= lmax, |m| <= l.

Returns a vector of `SchwarzschildPerturbation` structs, one per (l,m,parity).

If `registry` is provided, registers the radial coefficient tensors as
scalar fields on the Schwarzschild M^2 submanifold.

Ground truth: Martel & Poisson (2005) Secs IV-V.
"""
function decompose_schwarzschild(h_name::Symbol, mass::Symbol, lmax::Int;
                                  registry::Union{TensorRegistry,Nothing}=nothing)
    lmax >= 0 || throw(ArgumentError("lmax must be non-negative, got lmax=$lmax"))
    results = SchwarzschildPerturbation[]

    for l in 0:lmax
        for m in -l:l
            # Even parity (Zerilli sector)
            ec = _schw_even_coeffs(h_name, l, m)
            egi = _schw_even_gi(h_name, l, m)
            push!(results, SchwarzschildPerturbation(mass, l, m, EVEN, ec, egi))

            # Odd parity (RW sector) -- l=0 has no odd degrees of freedom
            oc = _schw_odd_coeffs(h_name, l, m)
            if !isempty(oc)
                ogi = _schw_odd_gi(h_name, l, m)
                push!(results, SchwarzschildPerturbation(mass, l, m, ODD, oc, ogi))
            end
        end
    end

    # Register coefficient tensors in the registry if provided
    if registry !== nothing
        _register_schw_coeffs!(registry, results)
    end

    results
end

function _register_schw_coeffs!(reg::TensorRegistry, decomps::Vector{SchwarzschildPerturbation})
    for sp in decomps
        for (_, cname) in sp.coeffs
            if !has_tensor(reg, cname)
                register_tensor!(reg, TensorProperties(
                    name=cname, manifold=:Schw, rank=(0, 0),
                    symmetries=SymmetrySpec[],
                    options=Dict{Symbol,Any}(:is_radial_coeff => true)))
            end
        end
        for (_, gname) in sp.gauge_invariant
            if !has_tensor(reg, gname)
                register_tensor!(reg, TensorProperties(
                    name=gname, manifold=:Schw, rank=(0, 0),
                    symmetries=SymmetrySpec[],
                    options=Dict{Symbol,Any}(:is_gauge_invariant => true)))
            end
        end
    end
end

# ── Gauge fixing ───────────────────────────────────────────────────────────

"""
    regge_wheeler_gauge(sp::SchwarzschildPerturbation) -> SchwarzschildPerturbation

Apply Regge-Wheeler gauge conditions to a Schwarzschild perturbation.

Even parity (MP below Eq 4.9): set j_a = 0, G = 0.
  Then h_tilde_{ab} = h_{ab} and K_tilde = K.

Odd parity (MP Eq 5.6): set h_2 = 0.
  Then h_tilde_a = h_a.

Returns a new `SchwarzschildPerturbation` with the gauge-fixed coefficients
removed and gauge-invariant variables identified with the remaining coefficients.
"""
function regge_wheeler_gauge(sp::SchwarzschildPerturbation)
    new_coeffs = copy(sp.coeffs)
    new_gi = copy(sp.gauge_invariant)

    if sp.parity == EVEN
        # RW gauge: j_a = 0, G = 0
        delete!(new_coeffs, :jt)
        delete!(new_coeffs, :jr)
        delete!(new_coeffs, :G)
        # In RW gauge: h_tilde_{ab} = h_{ab}, K_tilde = K (MP Eqs 4.10-4.12)
        if haskey(new_coeffs, :H0)
            new_gi[:htilde_tt] = new_coeffs[:H0]
        end
        if haskey(new_coeffs, :H1)
            new_gi[:htilde_tr] = new_coeffs[:H1]
        end
        if haskey(new_coeffs, :H2)
            new_gi[:htilde_rr] = new_coeffs[:H2]
        end
        if haskey(new_coeffs, :K)
            new_gi[:Ktilde] = new_coeffs[:K]
        end
    else  # ODD
        # RW gauge: h_2 = 0 (MP Eq 5.6)
        delete!(new_coeffs, :h2)
        # In RW gauge: h_tilde_a = h_a (MP Eq 5.7)
        if haskey(new_coeffs, :ht) && haskey(new_gi, :htilde_t)
            new_gi[:htilde_t] = new_coeffs[:ht]
        end
        if haskey(new_coeffs, :hr) && haskey(new_gi, :htilde_r)
            new_gi[:htilde_r] = new_coeffs[:hr]
        end
    end

    SchwarzschildPerturbation(sp.mass, sp.l, sp.m, sp.parity, new_coeffs, new_gi)
end

"""
    zerilli_gauge(sp::SchwarzschildPerturbation) -> SchwarzschildPerturbation

Apply Zerilli gauge conditions to a Schwarzschild perturbation.

Even parity: set j_a = 0, G = 0, h_{rr} = 0 (three gauge conditions fix
three gauge functions). This is an overcomplete gauge -- only works when
the field equations are satisfied.

Odd parity: same as Regge-Wheeler gauge (h_2 = 0).

Returns a new `SchwarzschildPerturbation` with the gauge-fixed coefficients removed.
"""
function zerilli_gauge(sp::SchwarzschildPerturbation)
    if sp.parity == ODD
        return regge_wheeler_gauge(sp)
    end

    # Even parity: Zerilli gauge = RW gauge + h_{rr} = 0
    rw = regge_wheeler_gauge(sp)
    new_coeffs = copy(rw.coeffs)
    delete!(new_coeffs, :H2)
    SchwarzschildPerturbation(rw.mass, rw.l, rw.m, rw.parity, new_coeffs, rw.gauge_invariant)
end
