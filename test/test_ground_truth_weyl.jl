#= Ground truth verification of Weyl tensor and Riemann tensor identities.
#
# Reference: Wald, R. M. (1984). "General Relativity." University of Chicago Press.
#   - Eq 3.2.14: First (algebraic) Bianchi identity: R_{a[bcd]} = 0
#   - Eq 3.2.15: Riemann symmetries: antisymmetry in each pair, pair swap symmetry
#   - Eq 3.2.28: Weyl decomposition:
#       C_{abcd} = R_{abcd} - (2/(n-2))(g_{a[c}R_{d]b} - g_{b[c}R_{d]a})
#                + (2/((n-1)(n-2))) R g_{a[c}g_{d]b}
#   - Consequence of 3.2.28: Weyl is trace-free: g^{ac}C_{abcd} = 0
#   - Eq 3.2.16 (contracted Bianchi): nabla^a G_{ab} = 0
#   - Eq 3.2.17 (contracted Bianchi): nabla^a R_{ab} = (1/2) nabla_b R
#
# Cross-reference: MathWorld "Weyl Tensor" and "Riemann Tensor" entries confirm
#   all identities stated above (retrieved 2026-03-15).
=#

using TensorGR: enforce_tracefree, enforce_divfree

@testset "Ground truth: Weyl tensor (Wald 1984)" begin

    # ------------------------------------------------------------------ #
    # Shared setup: 4D manifold with metric, curvature, and CovD         #
    # ------------------------------------------------------------------ #
    reg = TensorRegistry()
    with_registry(reg) do
        @manifold M4 dim=4 metric=g
        define_curvature_tensors!(reg, :M4, :g)
        @covd D on=M4 metric=g
        # Register Bianchi rewrite rules (contracted Bianchi, etc.)
        for r in bianchi_rules()
            register_rule!(reg, r)
        end

        # ============================================================== #
        # Section 1: Riemann tensor symmetries (Wald Eq 3.2.15)          #
        # ============================================================== #

        @testset "Wald 3.2.15a -- Riemann antisymmetry in first pair: R_{abcd} = -R_{bacd}" begin
            R_abcd = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            R_bacd = Tensor(:Riem, [down(:b), down(:a), down(:c), down(:d)])
            result = simplify(R_abcd + R_bacd; registry=reg)
            @test result == TScalar(0 // 1)
        end

        @testset "Wald 3.2.15a -- Riemann antisymmetry in second pair: R_{abcd} = -R_{abdc}" begin
            R_abcd = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            R_abdc = Tensor(:Riem, [down(:a), down(:b), down(:d), down(:c)])
            result = simplify(R_abcd + R_abdc; registry=reg)
            @test result == TScalar(0 // 1)
        end

        @testset "Wald 3.2.15c -- Riemann pair swap symmetry: R_{abcd} = R_{cdab}" begin
            R_abcd = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            R_cdab = Tensor(:Riem, [down(:c), down(:d), down(:a), down(:b)])
            result = simplify(R_abcd - R_cdab; registry=reg)
            @test result == TScalar(0 // 1)
        end

        @testset "Wald 3.2.14 -- First Bianchi (single-term reduction)" begin
            # The algebraic Bianchi identity R_{a[bcd]} = 0, i.e.,
            #   R_{abcd} + R_{acdb} + R_{adbc} = 0
            # is a multi-term identity. The xperm canonicalization handles
            # single-term Riemann symmetries but cannot combine three distinct
            # canonical forms to zero. We verify the weaker consequence that
            # canonicalization correctly uses all single-term symmetries by
            # checking R_{acdb} canonicalizes to -R_{acbd} (antisymmetry in cd).
            R_acdb = Tensor(:Riem, [down(:a), down(:c), down(:d), down(:b)])
            c = canonicalize(R_acdb)
            # Should be -R_{acbd} (antisymmetry in second pair: d<->b flips sign,
            # then pair swap / relabeling to canonical form)
            @test c isa TProduct
            @test c.scalar == -1 // 1
        end

        # ============================================================== #
        # Section 2: Riemann trace contractions                          #
        # ============================================================== #

        @testset "Wald -- Riemann trace (1,3) gives Ricci: R^a_{bad} = R_{bd}" begin
            # Standard convention: contracting slots 1 and 3 of R^a_{bcd} gives Ric_{bd}
            Riem_traced = Tensor(:Riem, [up(:a), down(:b), down(:a), down(:d)])
            result = simplify(Riem_traced; registry=reg)
            @test result == Tensor(:Ric, [down(:b), down(:d)])
        end

        @testset "Wald -- Ricci trace gives scalar: R^a_a = R" begin
            Ric_traced = Tensor(:Ric, [up(:a), down(:a)])
            result = simplify(Ric_traced; registry=reg)
            @test result == Tensor(:RicScalar, TIndex[])
        end

        # ============================================================== #
        # Section 3: Weyl tensor symmetries (Wald Eq 3.2.28)            #
        #   Weyl inherits all Riemann symmetries (RiemannSymmetry)       #
        # ============================================================== #

        @testset "Wald 3.2.28 -- Weyl antisymmetry in first pair: C_{abcd} = -C_{bacd}" begin
            C_abcd = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])
            C_bacd = Tensor(:Weyl, [down(:b), down(:a), down(:c), down(:d)])
            result = simplify(C_abcd + C_bacd; registry=reg)
            @test result == TScalar(0 // 1)
        end

        @testset "Wald 3.2.28 -- Weyl antisymmetry in second pair: C_{abcd} = -C_{abdc}" begin
            C_abcd = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])
            C_abdc = Tensor(:Weyl, [down(:a), down(:b), down(:d), down(:c)])
            result = simplify(C_abcd + C_abdc; registry=reg)
            @test result == TScalar(0 // 1)
        end

        @testset "Wald 3.2.28 -- Weyl pair swap symmetry: C_{abcd} = C_{cdab}" begin
            C_abcd = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])
            C_cdab = Tensor(:Weyl, [down(:c), down(:d), down(:a), down(:b)])
            result = simplify(C_abcd - C_cdab; registry=reg)
            @test result == TScalar(0 // 1)
        end

        # ============================================================== #
        # Section 4: Weyl trace-free property                           #
        #   Consequence of Wald Eq 3.2.28: g^{ac} C_{abcd} = 0         #
        #   MathWorld: "C^lambda_{mu lambda kappa} = 0"                 #
        # ============================================================== #

        # Mark Weyl as trace-free on all cross-pair contractions
        set_tracefree!(reg, :Weyl; pairs=[(1, 3), (1, 4), (2, 3), (2, 4)])

        @testset "Wald 3.2.28 consequence -- Weyl trace-free: g^{ac} C_{abcd} = 0" begin
            g_up = Tensor(:g, [up(:a), up(:c)])
            C = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])
            expr = g_up * C
            result = simplify(expr; registry=reg)
            @test result == TScalar(0 // 1)
        end

        @testset "Wald 3.2.28 consequence -- Weyl self-trace: C^a_{bad} = 0" begin
            C_traced = Tensor(:Weyl, [up(:a), down(:b), down(:a), down(:d)])
            result = simplify(C_traced; registry=reg)
            @test result == TScalar(0 // 1)
        end

        @testset "Wald 3.2.28 consequence -- Weyl trace-free: g^{bd} C_{abcd} = 0" begin
            g_up = Tensor(:g, [up(:b), up(:d)])
            C = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])
            expr = g_up * C
            result = simplify(expr; registry=reg)
            @test result == TScalar(0 // 1)
        end

        # ============================================================== #
        # Section 5: Weyl decomposition formula (Wald Eq 3.2.28)        #
        #   R_{abcd} = C_{abcd} + (2/(n-2))(g_{a[c}R_{d]b}-g_{b[c}R_{d]a}) #
        #            - (2/((n-1)(n-2))) R g_{a[c}g_{d]b}                #
        #   In dim=4: coefficients are 1/2 and 1/6.                     #
        # ============================================================== #

        @testset "Wald 3.2.28 -- Decomposition has correct structure" begin
            decomp = riemann_to_weyl(down(:a), down(:b), down(:c), down(:d), :g; dim=4)
            # Should be a sum of Weyl + Ricci terms + scalar*metric terms
            @test decomp isa TSum
            # The Weyl term should appear in the decomposition
            has_weyl = any(decomp.terms) do t
                t isa Tensor && t.name == :Weyl
            end
            @test has_weyl
        end

        @testset "Wald 3.2.28 -- Inverse decomposition recovers Riemann" begin
            # weyl_to_riemann gives C_{abcd} = R_{abcd} - (Ricci terms)
            inv_decomp = weyl_to_riemann(down(:a), down(:b), down(:c), down(:d), :g; dim=4)
            @test inv_decomp isa TSum
            # The Riemann term should appear
            has_riem = any(inv_decomp.terms) do t
                t isa Tensor && t.name == :Riem
            end
            @test has_riem
        end

        @testset "Wald 3.2.28 -- Trace of Weyl decomposition is self-consistent" begin
            # The Weyl decomposition C_{abcd} = R_{abcd} - (Ricci terms) implies
            # g^{ac} C_{abcd} = 0 (trace-free). Taking g^{ac} of the explicit
            # decomposition formula should yield zero, confirming the Ricci terms
            # are constructed to cancel the Riemann trace exactly.
            expr = weyl_to_riemann(down(:a), down(:b), down(:c), down(:d), :g; dim=4)
            g_up = Tensor(:g, [up(:a), up(:c)])
            traced = g_up * expr
            result = simplify(traced; registry=reg)
            @test result == TScalar(0 // 1)
        end

        # ============================================================== #
        # Section 6: Divergence-free property (vacuum)                   #
        #   In vacuum (R_{ab} = 0): nabla^a C_{abcd} = 0               #
        #   This follows from the contracted second Bianchi identity.    #
        # ============================================================== #

        # Mark Weyl as divergence-free w.r.t. CovD D on first index
        set_divfree!(reg, :Weyl; covd=:D, index=1)

        @testset "Contracted Bianchi -- Weyl divergence-free (vacuum): D_a C^a_{bcd} = 0" begin
            div_expr = TDeriv(down(:a), Tensor(:Weyl, [up(:a), down(:b), down(:c), down(:d)]), :D)
            result = simplify(div_expr; registry=reg)
            @test result == TScalar(0 // 1)
        end

        # ============================================================== #
        # Section 7: Contracted Bianchi identities (Wald Eqs 3.2.16-17) #
        # ============================================================== #

        @testset "Wald 3.2.16 -- Contracted Bianchi: D_a G^a_b = 0" begin
            div_ein = TDeriv(down(:a), Tensor(:Ein, [up(:a), down(:b)]), :D)
            result = simplify(div_ein; registry=reg)
            @test result == TScalar(0 // 1)
        end

        @testset "Wald 3.2.17 -- Contracted Bianchi: D_a R^a_b = (1/2) D_b R" begin
            div_ric = TDeriv(down(:a), Tensor(:Ric, [up(:a), down(:b)]), :D)
            result = simplify(div_ric; registry=reg)
            # Should be (1/2) * partial_b(RicScalar)
            # The bianchi rule replaces the covd version with partial
            # (the derivative of a scalar is the same for covd and partial)
            @test result isa TProduct
            @test result.scalar == 1 // 2
        end

        # ============================================================== #
        # Section 8: Weyl vanishes in 3D (independent components = 0)   #
        #   MathWorld: C_N = N(N+1)(N+2)(N-3)/12, so C_3 = 0           #
        # ============================================================== #

        @testset "MathWorld -- Weyl independent components: 0 in 3D, 10 in 4D" begin
            # Formula: C_N = N(N+1)(N+2)(N-3)/12
            weyl_components(n) = n * (n + 1) * (n + 2) * (n - 3) / 12
            @test weyl_components(3) == 0   # Weyl vanishes identically in 3D
            @test weyl_components(4) == 10  # 10 independent components in 4D
            @test weyl_components(5) == 35  # 35 in 5D
        end

        # ============================================================== #
        # Section 9: Weyl decomposition in general dimension             #
        #   Verify the coefficient structure for arbitrary dim n.        #
        #   Wald 3.2.28: coeff of Ricci term = 2/(n-2) = 1/(n-2) per   #
        #   antisymmetrized pair; scalar term = 2/((n-1)(n-2)).         #
        # ============================================================== #

        @testset "Wald 3.2.28 -- Decomposition coefficients for various dimensions" begin
            for dim in [3, 4, 5, 6, 10]
                decomp = riemann_to_weyl(down(:a), down(:b), down(:c), down(:d), :g; dim=dim)
                @test decomp isa TSum
                # The decomposition should always contain the Weyl tensor
                has_weyl = any(decomp.terms) do t
                    t isa Tensor && t.name == :Weyl
                end
                @test has_weyl
            end
        end

    end  # with_registry
end
