#= Topological density constructors.

Pontryagin density (Chern-Pontryagin): ★(R∧R) = ε^{abcd} R_{abef} R_{cd}^{ef}
Euler density (Gauss-Bonnet): ε^{abcd} ε^{efgh} R_{abef} R_{cdgh} / 4
                              = R^2 - 4 Ric^2 + Riem^2  (in 4D)
=#

"""
    pontryagin_density(metric::Symbol; registry=current_registry()) -> TensorExpr

Construct the Pontryagin (Chern-Pontryagin) density in 4D:
`★(R∧R) = ε^{abcd} R_{ab}^{ef} R_{cdef}`

This is a pseudoscalar, a total derivative in 4D.
"""
function pontryagin_density(metric::Symbol;
                             registry::TensorRegistry=current_registry())
    with_registry(registry) do
        reg = registry
        mprops = get_tensor(reg, metric)
        eps_name = Symbol(:ε, metric)

        used = Set{Symbol}()
        a = fresh_index(used); push!(used, a)
        b = fresh_index(used); push!(used, b)
        c = fresh_index(used); push!(used, c)
        d = fresh_index(used); push!(used, d)
        e = fresh_index(used); push!(used, e)
        f = fresh_index(used); push!(used, f)

        eps = Tensor(eps_name, [up(a), up(b), up(c), up(d)])
        R1 = Tensor(:Riem, [down(a), down(b), up(e), up(f)])
        R2 = Tensor(:Riem, [down(c), down(d), down(e), down(f)])

        eps * R1 * R2
    end
end

"""
    euler_density(metric::Symbol; dim::Int=4, registry=current_registry()) -> TensorExpr

Construct the Euler (Gauss-Bonnet) density in 4D:
`E₄ = R² - 4 R_{ab}R^{ab} + R_{abcd}R^{abcd}`

In the epsilon form: `E₄ = (1/4) ε^{abcd} ε^{efgh} R_{abef} R_{cdgh}`.
"""
function euler_density(metric::Symbol; dim::Int=4,
                        registry::TensorRegistry=current_registry())
    with_registry(registry) do
        used = Set{Symbol}()

        # Build in index form: Riem² - 4 Ric² + R²
        a = fresh_index(used); push!(used, a)
        b = fresh_index(used); push!(used, b)
        c = fresh_index(used); push!(used, c)
        d = fresh_index(used); push!(used, d)

        # Kretschner: R_{abcd} R^{abcd}
        Riem_down = Tensor(:Riem, [down(a), down(b), down(c), down(d)])
        Riem_up = Tensor(:Riem, [up(a), up(b), up(c), up(d)])
        kretschner = Riem_down * Riem_up

        # Ricci squared: R_{ab} R^{ab}
        e = fresh_index(used); push!(used, e)
        f = fresh_index(used); push!(used, f)
        Ric_down = Tensor(:Ric, [down(e), down(f)])
        Ric_up = Tensor(:Ric, [up(e), up(f)])
        ricci_sq = Ric_down * Ric_up

        # Scalar squared: R²
        R = Tensor(:RicScalar, TIndex[])
        scalar_sq = R * R

        # E₄ = Riem² - 4 Ric² + R²
        kretschner + tproduct(-4 // 1, TensorExpr[ricci_sq]) + scalar_sq
    end
end

"""
    chern_simons_action(scalar_field::Tensor, metric::Symbol;
                         registry=current_registry()) -> TensorExpr

Construct the Chern-Simons gravitational coupling:
`S_CS = ϑ · ★(R∧R) = ϑ · ε^{abcd} R_{ab}^{ef} R_{cdef}`

where `ϑ` is the scalar field (axion/dilaton).
"""
function chern_simons_action(scalar_field::Tensor, metric::Symbol;
                              registry::TensorRegistry=current_registry())
    with_registry(registry) do
        scalar_field * pontryagin_density(metric; registry=registry)
    end
end
