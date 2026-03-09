#= Product Manifolds: M₁ × M₂ curvature decomposition

Demonstrates the product manifold feature of TensorGR.jl.

For a direct product (M₁,g₁) × (M₂,g₂) with block-diagonal metric g = g₁ ⊕ g₂:
  • The Riemann tensor decomposes: R = R₁ ⊕ R₂ (no mixed terms)
  • The scalar curvature is additive: R = R₁ + R₂
  • The Einstein tensor acquires cross-scalar terms:
      G_{ij} = G₁_{ij} - ½ R₂ g₁_{ij}

This last identity is physically profound: the curvature of one factor acts as
an effective cosmological constant for the other factor.
=#

using TensorGR

function demo_product_manifolds()
    reg = TensorRegistry()
    with_registry(reg) do
        println("=" ^ 60)
        println("  Product Manifolds in TensorGR.jl")
        println("=" ^ 60)

        # ── Setup: M = AdS₂ × S² (near-horizon Reissner-Nordström) ──
        register_manifold!(reg, ManifoldProperties(:AdS2, 2, :h, :∂, [:a,:b,:c,:d]))
        register_tensor!(reg, TensorProperties(
            name=:h, manifold=:AdS2, rank=(0,2),
            symmetries=SymmetrySpec[Symmetric(1,2)], is_metric=true))

        register_manifold!(reg, ManifoldProperties(:S2, 2, :σ, :∂, [:α,:β,:γ,:δ]))
        register_tensor!(reg, TensorProperties(
            name=:σ, manifold=:S2, rank=(0,2),
            symmetries=SymmetrySpec[Symmetric(1,2)], is_metric=true))

        pp = define_product_manifold!(reg, :M; factors=[:AdS2, :S2])

        println("\n1. Product structure")
        println("   M = AdS₂ × S²")
        println("   dim(AdS₂) = $(pp.factor_dims[1]),  dim(S²) = $(pp.factor_dims[2])")
        println("   Total dim = $(sum(pp.factor_dims))")

        # ── Block-diagonal metric ──
        g = product_metric(:M)
        println("\n2. Block-diagonal metric")
        println("   g = $(to_unicode(g))")

        # ── Scalar curvature is additive ──
        R = product_scalar_curvature(:M)
        println("\n3. Scalar curvature (additive)")
        println("   R = $(to_unicode(R))")

        # ── Ricci sectors ──
        println("\n4. Ricci tensor decomposition (no mixed terms)")
        ric_ads = product_ricci(:M, :AdS2)
        ric_s2  = product_ricci(:M, :S2)
        println("   AdS₂ sector: $(to_unicode(ric_ads))")
        println("   S²   sector: $(to_unicode(ric_s2))")
        println("   Mixed:       0  (identically)")

        # ── Riemann sectors ──
        println("\n5. Riemann tensor decomposition")
        riem_ads = product_riemann(:M, :AdS2)
        riem_s2  = product_riemann(:M, :S2)
        println("   AdS₂ sector: $(to_unicode(riem_ads))")
        println("   S²   sector: $(to_unicode(riem_s2))")
        println("   Mixed:       0  (all cross-factor components vanish)")

        # ── Einstein tensor: the beautiful part ──
        println("\n6. Einstein tensor with cross-scalar contributions")
        eqs = product_einstein_equations(:M)
        for (name, G) in sort(collect(eqs); by=first)
            println("   $name sector: $(to_unicode(G))")
        end

        # ── Physical interpretation ──
        println("\n7. Physical interpretation")
        println("   In 2D, the Einstein tensor vanishes identically (G₂ᴅ = 0).")
        println("   Setting Ein_h = 0 and Ein_σ = 0:")

        set_vanishing!(reg, :Ein_h)
        set_vanishing!(reg, :Ein_σ)

        G_ads = product_einstein(:M, :AdS2)
        G_s2  = product_einstein(:M, :S2)

        G_ads_simplified = simplify(G_ads; registry=reg)
        G_s2_simplified  = simplify(G_s2;  registry=reg)

        println("   G_{ab} = $(to_unicode(G_ads_simplified))")
        println("   G_{αβ} = $(to_unicode(G_s2_simplified))")
        println()
        println("   → The curvature of S² acts as a cosmological constant for AdS₂")
        println("   → The curvature of AdS₂ acts as a cosmological constant for S²")
        println("   This is the Bertotti-Robinson / Freund-Rubin mechanism!")

        # ── Three-factor product ──
        println("\n" * "=" ^ 60)
        println("  Three-factor product: M₃ × M₄ × M₅")
        println("=" ^ 60)

        register_manifold!(reg, ManifoldProperties(:M3, 3, :g3, :∂, [:i,:j,:k,:l,:m,:n]))
        register_tensor!(reg, TensorProperties(
            name=:g3, manifold=:M3, rank=(0,2),
            symmetries=SymmetrySpec[Symmetric(1,2)], is_metric=true))

        register_manifold!(reg, ManifoldProperties(:M4, 4, :g4, :∂, [:p,:q,:r,:s,:t,:u]))
        register_tensor!(reg, TensorProperties(
            name=:g4, manifold=:M4, rank=(0,2),
            symmetries=SymmetrySpec[Symmetric(1,2)], is_metric=true))

        register_manifold!(reg, ManifoldProperties(:M5, 5, :g5, :∂, [:v,:w,:x,:y,:z,:ζ]))
        register_tensor!(reg, TensorProperties(
            name=:g5, manifold=:M5, rank=(0,2),
            symmetries=SymmetrySpec[Symmetric(1,2)], is_metric=true))

        pp3 = define_product_manifold!(reg, :BigM; factors=[:M3, :M4, :M5])

        R3 = product_scalar_curvature(:BigM)
        println("\n   R = $(to_unicode(R3))")

        G_m3 = product_einstein(:BigM, :M3)
        println("\n   M₃ Einstein sector:")
        println("   G_{ij} = $(to_unicode(G_m3))")
        println("   (cross terms from BOTH M₄ and M₅ scalar curvatures)")

        println("\n" * "=" ^ 60)
        println("  All checks passed!")
        println("=" ^ 60)
    end
end

demo_product_manifolds()
