# Tests for Regge-Wheeler-Zerilli decomposition on Schwarzschild background.
# Ground truth: Martel & Poisson (2005), PRD 71, 104003, Secs IV-V.

@testset "Schwarzschild RW/Zerilli decomposition" begin

    @testset "SchwarzschildPerturbation construction" begin
        coeffs = Dict(:H0 => :h_H0_2_0, :H1 => :h_H1_2_0, :H2 => :h_H2_2_0,
                      :jt => :h_jt_2_0, :jr => :h_jr_2_0, :K => :h_K_2_0,
                      :G => :h_G_2_0)
        gi = Dict(:htilde_tt => :h_htilde_tt_2_0, :htilde_tr => :h_htilde_tr_2_0,
                  :htilde_rr => :h_htilde_rr_2_0, :Ktilde => :h_Ktilde_2_0)
        sp = SchwarzschildPerturbation(:M, 2, 0, EVEN, coeffs, gi)
        @test sp.mass == :M
        @test sp.l == 2
        @test sp.m == 0
        @test sp.parity == EVEN
        @test length(sp.coeffs) == 7
        @test length(sp.gauge_invariant) == 4
    end

    @testset "SchwarzschildPerturbation equality and hashing" begin
        c1 = Dict(:H0 => :h_H0_2_0)
        c2 = Dict(:H0 => :h_H0_2_0)
        gi1 = Dict(:htilde_tt => :h_htilde_tt_2_0)
        gi2 = Dict(:htilde_tt => :h_htilde_tt_2_0)
        sp1 = SchwarzschildPerturbation(:M, 2, 0, EVEN, c1, gi1)
        sp2 = SchwarzschildPerturbation(:M, 2, 0, EVEN, c2, gi2)
        @test sp1 == sp2
        @test hash(sp1) == hash(sp2)

        sp3 = SchwarzschildPerturbation(:M, 2, 0, ODD, Dict(:ht => :x), Dict{Symbol,Symbol}())
        @test sp1 != sp3
    end

    @testset "decompose_schwarzschild: basic structure" begin
        decomps = decompose_schwarzschild(:h, :M, 2)
        @test decomps isa Vector{SchwarzschildPerturbation}
        @test all(sp -> sp.mass == :M, decomps)
    end

    @testset "decompose_schwarzschild: mode count" begin
        decomps = decompose_schwarzschild(:h, :M, 2)
        # l=0: 1 even (no odd -- empty coeffs)
        # l=1: 3 even + 3 odd = 6
        # l=2: 5 even + 5 odd = 10
        # Total: 1 + 6 + 10 = 17
        @test length(decomps) == 17
    end

    @testset "decompose_schwarzschild: lmax=0 -> even only" begin
        decomps = decompose_schwarzschild(:h, :M, 0)
        @test length(decomps) == 1
        sp = decomps[1]
        @test sp.parity == EVEN
        @test sp.l == 0
        @test sp.m == 0
        # l=0 even: H0, H1, H2, K (4 coeffs); no jt, jr, G
        @test length(sp.coeffs) == 4
        @test haskey(sp.coeffs, :H0)
        @test haskey(sp.coeffs, :H1)
        @test haskey(sp.coeffs, :H2)
        @test haskey(sp.coeffs, :K)
        @test !haskey(sp.coeffs, :jt)
        @test !haskey(sp.coeffs, :G)
    end

    @testset "Even parity l=0: G and j vanish (MP Sec VIII)" begin
        # At l=0, the even-parity decomposition has only H0, H1, H2, K.
        # Vector harmonics Y^A vanish at l=0 (gradient of Y_{00}=const is zero),
        # and Z^{AB} requires l >= 2, so G is absent.
        decomps = decompose_schwarzschild(:h, :M, 0)
        sp = decomps[1]
        @test !haskey(sp.coeffs, :jt)
        @test !haskey(sp.coeffs, :jr)
        @test !haskey(sp.coeffs, :G)
    end

    @testset "Odd parity l=0 and l=1: tensor sector vanishes (MP Sec VIII)" begin
        decomps = decompose_schwarzschild(:h, :M, 2)

        # l=0: no odd mode at all
        l0_odd = filter(sp -> sp.l == 0 && sp.parity == ODD, decomps)
        @test isempty(l0_odd)

        # l=1 odd: ht, hr present but no h2 (X^{AB} requires l >= 2)
        l1_odd = filter(sp -> sp.l == 1 && sp.m == 0 && sp.parity == ODD, decomps)
        @test length(l1_odd) == 1
        sp = l1_odd[1]
        @test haskey(sp.coeffs, :ht)
        @test haskey(sp.coeffs, :hr)
        @test !haskey(sp.coeffs, :h2)
    end

    @testset "Even parity l=2: full 7 coefficients (MP Eqs 4.1-4.3)" begin
        decomps = decompose_schwarzschild(:h, :M, 2)
        l2_even = filter(sp -> sp.l == 2 && sp.m == 0 && sp.parity == EVEN, decomps)
        @test length(l2_even) == 1
        sp = l2_even[1]
        @test length(sp.coeffs) == 7  # H0, H1, H2, jt, jr, K, G
        @test haskey(sp.coeffs, :H0)
        @test haskey(sp.coeffs, :H1)
        @test haskey(sp.coeffs, :H2)
        @test haskey(sp.coeffs, :jt)
        @test haskey(sp.coeffs, :jr)
        @test haskey(sp.coeffs, :K)
        @test haskey(sp.coeffs, :G)
    end

    @testset "Odd parity l=2: full 3 coefficients (MP Eqs 5.1-5.3)" begin
        decomps = decompose_schwarzschild(:h, :M, 2)
        l2_odd = filter(sp -> sp.l == 2 && sp.m == 0 && sp.parity == ODD, decomps)
        @test length(l2_odd) == 1
        sp = l2_odd[1]
        @test length(sp.coeffs) == 3  # ht, hr, h2
        @test haskey(sp.coeffs, :ht)
        @test haskey(sp.coeffs, :hr)
        @test haskey(sp.coeffs, :h2)
    end

    @testset "Gauge-invariant variables: even parity (MP Eqs 4.10-4.12)" begin
        decomps = decompose_schwarzschild(:h, :M, 2)
        l2_even = filter(sp -> sp.l == 2 && sp.m == 0 && sp.parity == EVEN, decomps)
        sp = l2_even[1]
        @test haskey(sp.gauge_invariant, :htilde_tt)
        @test haskey(sp.gauge_invariant, :htilde_tr)
        @test haskey(sp.gauge_invariant, :htilde_rr)
        @test haskey(sp.gauge_invariant, :Ktilde)
        @test length(sp.gauge_invariant) == 4
    end

    @testset "Gauge-invariant variables: odd parity (MP Eq 5.7)" begin
        decomps = decompose_schwarzschild(:h, :M, 2)
        l2_odd = filter(sp -> sp.l == 2 && sp.m == 0 && sp.parity == ODD, decomps)
        sp = l2_odd[1]
        @test haskey(sp.gauge_invariant, :htilde_t)
        @test haskey(sp.gauge_invariant, :htilde_r)
        @test length(sp.gauge_invariant) == 2
    end

    @testset "Coefficient naming conventions" begin
        decomps = decompose_schwarzschild(:h, :M, 2)
        l2_even = filter(sp -> sp.l == 2 && sp.m == 1 && sp.parity == EVEN, decomps)
        sp = l2_even[1]
        @test sp.coeffs[:H0] == :h_H0_2_1
        @test sp.coeffs[:G] == :h_G_2_1
        @test sp.gauge_invariant[:Ktilde] == :h_Ktilde_2_1

        # Negative m
        l2_even_neg = filter(sp -> sp.l == 2 && sp.m == -1 && sp.parity == EVEN, decomps)
        sp_neg = l2_even_neg[1]
        @test sp_neg.coeffs[:H0] == :h_H0_2_neg1
    end

    @testset "regge_wheeler_gauge: even parity (MP: j_a=0, G=0)" begin
        decomps = decompose_schwarzschild(:h, :M, 2)
        l2_even = filter(sp -> sp.l == 2 && sp.m == 0 && sp.parity == EVEN, decomps)
        sp = l2_even[1]
        rw = regge_wheeler_gauge(sp)

        # RW gauge removes j_t, j_r, G
        @test !haskey(rw.coeffs, :jt)
        @test !haskey(rw.coeffs, :jr)
        @test !haskey(rw.coeffs, :G)

        # Remaining: H0, H1, H2, K
        @test haskey(rw.coeffs, :H0)
        @test haskey(rw.coeffs, :H1)
        @test haskey(rw.coeffs, :H2)
        @test haskey(rw.coeffs, :K)
        @test length(rw.coeffs) == 4

        # In RW gauge, gauge-invariant = original: h_tilde = h, K_tilde = K
        @test rw.gauge_invariant[:htilde_tt] == sp.coeffs[:H0]
        @test rw.gauge_invariant[:htilde_tr] == sp.coeffs[:H1]
        @test rw.gauge_invariant[:htilde_rr] == sp.coeffs[:H2]
        @test rw.gauge_invariant[:Ktilde] == sp.coeffs[:K]
    end

    @testset "regge_wheeler_gauge: odd parity (MP: h_2=0)" begin
        decomps = decompose_schwarzschild(:h, :M, 2)
        l2_odd = filter(sp -> sp.l == 2 && sp.m == 0 && sp.parity == ODD, decomps)
        sp = l2_odd[1]
        rw = regge_wheeler_gauge(sp)

        # RW gauge removes h_2
        @test !haskey(rw.coeffs, :h2)

        # Remaining: ht, hr
        @test haskey(rw.coeffs, :ht)
        @test haskey(rw.coeffs, :hr)
        @test length(rw.coeffs) == 2

        # h_tilde_a = h_a in RW gauge
        @test rw.gauge_invariant[:htilde_t] == sp.coeffs[:ht]
        @test rw.gauge_invariant[:htilde_r] == sp.coeffs[:hr]
    end

    @testset "zerilli_gauge: even parity (RW + h_{rr}=0)" begin
        decomps = decompose_schwarzschild(:h, :M, 2)
        l2_even = filter(sp -> sp.l == 2 && sp.m == 0 && sp.parity == EVEN, decomps)
        sp = l2_even[1]
        zg = zerilli_gauge(sp)

        # Zerilli gauge removes j_t, j_r, G, H2
        @test !haskey(zg.coeffs, :jt)
        @test !haskey(zg.coeffs, :jr)
        @test !haskey(zg.coeffs, :G)
        @test !haskey(zg.coeffs, :H2)

        # Remaining: H0, H1, K
        @test haskey(zg.coeffs, :H0)
        @test haskey(zg.coeffs, :H1)
        @test haskey(zg.coeffs, :K)
        @test length(zg.coeffs) == 3
    end

    @testset "zerilli_gauge: odd parity = RW gauge" begin
        decomps = decompose_schwarzschild(:h, :M, 2)
        l2_odd = filter(sp -> sp.l == 2 && sp.m == 0 && sp.parity == ODD, decomps)
        sp = l2_odd[1]
        zg = zerilli_gauge(sp)
        rw = regge_wheeler_gauge(sp)

        @test zg == rw
    end

    @testset "regge_wheeler_gauge: l=0 even (no j or G to remove)" begin
        decomps = decompose_schwarzschild(:h, :M, 0)
        sp = decomps[1]  # l=0, m=0, even
        rw = regge_wheeler_gauge(sp)
        # l=0 has no j_a, no G, so nothing removed
        @test length(rw.coeffs) == 4  # H0, H1, H2, K
    end

    @testset "regge_wheeler_gauge: l=1 odd (no h2 to remove)" begin
        decomps = decompose_schwarzschild(:h, :M, 1)
        l1_odd = filter(sp -> sp.l == 1 && sp.m == 0 && sp.parity == ODD, decomps)
        sp = l1_odd[1]
        rw = regge_wheeler_gauge(sp)
        # l=1 odd has no h2, so nothing removed beyond what is absent
        @test length(rw.coeffs) == 2  # ht, hr
    end

    @testset "Validation" begin
        @test_throws ArgumentError decompose_schwarzschild(:h, :M, -1)
    end

    @testset "Pretty printing" begin
        decomps = decompose_schwarzschild(:h, :M, 2)
        sp = decomps[1]
        s = sprint(show, sp)
        @test occursin("M", s)
        @test occursin("Zerilli", s) || occursin("RW", s)
    end

    @testset "Registry integration" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:Schw, 2, nothing, nothing, Symbol[]))
        decomps = decompose_schwarzschild(:h, :M, 2; registry=reg)

        # Check that coefficient tensors are registered
        l2_even = filter(sp -> sp.l == 2 && sp.m == 0 && sp.parity == EVEN, decomps)
        sp = l2_even[1]
        @test has_tensor(reg, sp.coeffs[:H0])
        @test has_tensor(reg, sp.coeffs[:K])
        @test has_tensor(reg, sp.coeffs[:G])
        @test has_tensor(reg, sp.gauge_invariant[:Ktilde])
    end

    @testset "Total coefficient count at l >= 2: 10 per (l,m) (MP Sec III.A)" begin
        decomps = decompose_schwarzschild(:h, :M, 3)
        for l in 2:3
            for m in -l:l
                even_modes = filter(sp -> sp.l == l && sp.m == m && sp.parity == EVEN, decomps)
                odd_modes = filter(sp -> sp.l == l && sp.m == m && sp.parity == ODD, decomps)
                @test length(even_modes) == 1
                @test length(odd_modes) == 1
                n_even = length(even_modes[1].coeffs)
                n_odd = length(odd_modes[1].coeffs)
                @test n_even == 7  # H0, H1, H2, jt, jr, K, G
                @test n_odd == 3   # ht, hr, h2
                @test n_even + n_odd == 10
            end
        end
    end

    @testset "RW gauge degree-of-freedom count: even=4, odd=2 (l >= 2)" begin
        decomps = decompose_schwarzschild(:h, :M, 2)
        for sp in decomps
            sp.l < 2 && continue
            rw = regge_wheeler_gauge(sp)
            if sp.parity == EVEN
                @test length(rw.coeffs) == 4  # H0, H1, H2, K
            else
                @test length(rw.coeffs) == 2  # ht, hr
            end
        end
    end

end
