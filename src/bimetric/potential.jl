#= Hassan-Rosen bimetric interaction potential.
#
# V(g,f) = m² Σ_{n=0}^{4} β_n e_n(S)
#
# where S^a_b = (√(g⁻¹f))^a_b and e_n are elementary symmetric polynomials:
#   e_0(S) = 1
#   e_1(S) = Tr(S)
#   e_2(S) = (1/2)((TrS)² - Tr(S²))
#   e_3(S) = (1/6)((TrS)³ - 3·TrS·Tr(S²) + 2·Tr(S³))
#   e_4(S) = det(S)
#
# Ground truth: Hassan & Rosen, JHEP 02 (2012) 126;
#              de Rham, Living Rev. Rel. 17, 7 (2014) Sec 8.
=#

"""
    HassanRosenParams

Parameters for the Hassan-Rosen bimetric interaction potential.

# Fields
- `m_sq::Any`      -- mass parameter m²
- `beta::NTuple{5,Any}` -- β₀, β₁, β₂, β₃, β₄ coefficients
"""
struct HassanRosenParams
    m_sq::Any
    beta::NTuple{5,Any}
end

function HassanRosenParams(; m_sq=:m2, beta0=0, beta1=0, beta2=0, beta3=0, beta4=0)
    HassanRosenParams(m_sq, (beta0, beta1, beta2, beta3, beta4))
end

function Base.show(io::IO, p::HassanRosenParams)
    print(io, "HR(m²=$(p.m_sq), β=", p.beta, ")")
end

"""
    elementary_symmetric(n::Int, S_name::Symbol;
                          registry::TensorRegistry=current_registry()) -> TensorExpr

Construct the n-th elementary symmetric polynomial e_n(S) of the
matrix square root S^a_b.

    e_0 = 1
    e_1 = S^a_a = Tr(S)
    e_2 = (1/2)((Tr S)² - Tr(S²))
    e_3 = (1/6)((Tr S)³ - 3·Tr S·Tr(S²) + 2·Tr(S³))
    e_4 = det(S)

Ground truth: Hassan & Rosen (2012) Eq 2.4.
"""
function elementary_symmetric(n::Int, S_name::Symbol;
                                registry::TensorRegistry=current_registry())
    n in 0:4 || error("Elementary symmetric polynomial order must be 0-4, got $n")

    if n == 0
        return TScalar(1 // 1)
    end

    used = Set{Symbol}()

    if n == 1
        # e_1 = Tr(S) = S^a_a
        a = fresh_index(used)
        return Tensor(S_name, [up(a), down(a)])
    end

    if n == 2
        # e_2 = (1/2)((Tr S)² - Tr(S²))
        a = fresh_index(used); push!(used, a)
        b = fresh_index(used); push!(used, b)
        c = fresh_index(used)

        trS = Tensor(S_name, [up(a), down(a)])
        trS_sq = trS * trS
        # Tr(S²) = S^a_b S^b_a
        S_ab = Tensor(S_name, [up(b), down(c)])
        S_ba = Tensor(S_name, [up(c), down(b)])
        trS2 = S_ab * S_ba

        return tproduct(1 // 2, TensorExpr[trS_sq - trS2])
    end

    if n == 3
        # e_3 = (1/6)((Tr S)³ - 3·Tr S·Tr(S²) + 2·Tr(S³))
        a = fresh_index(used); push!(used, a)
        b = fresh_index(used); push!(used, b)
        c = fresh_index(used); push!(used, c)
        d = fresh_index(used); push!(used, d)
        e = fresh_index(used)

        trS = Tensor(S_name, [up(a), down(a)])
        trS_cubed = trS * trS * trS

        S1 = Tensor(S_name, [up(b), down(c)])
        S2 = Tensor(S_name, [up(c), down(b)])
        trS2 = S1 * S2
        trS_trS2 = tproduct(-3 // 1, TensorExpr[trS, trS2])

        S3a = Tensor(S_name, [up(d), down(e)])
        S3b = Tensor(S_name, [up(e), down(b)])
        S3c = Tensor(S_name, [up(b), down(d)])
        trS3 = tproduct(2 // 1, TensorExpr[S3a, S3b, S3c])

        return tproduct(1 // 6, TensorExpr[trS_cubed + trS_trS2 + trS3])
    end

    # n == 4: det(S) — represented symbolically
    TScalar(Symbol(:det_, S_name))
end

"""
    hassan_rosen_potential(bs::BimetricSetup, params::HassanRosenParams;
                           registry::TensorRegistry=current_registry()) -> TensorExpr

Construct the Hassan-Rosen interaction potential:

    V = m² √(-g) Σ_{n=0}^{4} β_n e_n(S)

Returns the potential as a TensorExpr (without the √(-g) factor).

Ground truth: Hassan & Rosen (2012) Eq 2.4.
"""
function hassan_rosen_potential(bs::BimetricSetup, params::HassanRosenParams;
                                 registry::TensorRegistry=current_registry())
    S_name = Symbol(:S_, bs.metric_g, :_, bs.metric_f)

    terms = TensorExpr[]
    for n in 0:4
        coeff = params.beta[n + 1]
        (coeff isa Number && coeff == 0) && continue

        en = elementary_symmetric(n, S_name; registry=registry)

        if coeff isa Number && coeff == 1
            push!(terms, en)
        else
            push!(terms, tproduct(1 // 1, TensorExpr[TScalar(coeff), en]))
        end
    end

    isempty(terms) && return TScalar(0 // 1)

    potential = length(terms) == 1 ? terms[1] : tsum(terms)

    # Multiply by m²
    if params.m_sq isa Number && params.m_sq == 1
        return potential
    end
    tproduct(1 // 1, TensorExpr[TScalar(params.m_sq), potential])
end
