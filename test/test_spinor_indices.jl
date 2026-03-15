@testset "Spinor Dummy Pair Analysis" begin

    @testset "fresh_spinor_index: undotted" begin
        used = Set{Symbol}()
        @test fresh_spinor_index(used; dotted=false) == :A
        used = Set{Symbol}([:A])
        @test fresh_spinor_index(used; dotted=false) == :B
        used = Set{Symbol}([:A, :B, :C, :D, :E, :F])
        # Exhausted base alphabet, should get A1
        @test fresh_spinor_index(used; dotted=false) == :A1
    end

    @testset "fresh_spinor_index: dotted" begin
        used = Set{Symbol}()
        @test fresh_spinor_index(used; dotted=true) == :Ap
        used = Set{Symbol}([:Ap])
        @test fresh_spinor_index(used; dotted=true) == :Bp
        used = Set{Symbol}([:Ap, :Bp, :Cp, :Dp, :Ep, :Fp])
        @test fresh_spinor_index(used; dotted=true) == :Ap1
    end

    @testset "fresh_index with vbundle keyword" begin
        used = Set{Symbol}()
        # Default (Tangent) behavior unchanged
        @test fresh_index(used) == :a
        @test fresh_index(used; vbundle=:Tangent) == :a
        # SL2C dispatch
        @test fresh_index(used; vbundle=:SL2C) == :A
        @test fresh_index(Set{Symbol}([:A]); vbundle=:SL2C) == :B
        # SL2C_dot dispatch
        @test fresh_index(used; vbundle=:SL2C_dot) == :Ap
        @test fresh_index(Set{Symbol}([:Ap]); vbundle=:SL2C_dot) == :Bp
    end

    @testset "spinor_dummy_pairs: single SL2C contraction" begin
        # psi^A chi_A (both SL2C)
        T1 = Tensor(:psi, [spin_up(:A)])
        T2 = Tensor(:chi, [spin_down(:A)])
        prod = T1 * T2

        sp = spinor_dummy_pairs(prod)
        @test length(sp) == 1
        @test sp[1][1].name == :A
        @test sp[1][1].vbundle == :SL2C
    end

    @testset "spinor_dummy_pairs: dotted contraction" begin
        # phi^{Ap} xi_{Ap} (both SL2C_dot)
        T1 = Tensor(:phi, [spin_dot_up(:Ap)])
        T2 = Tensor(:xi, [spin_dot_down(:Ap)])
        prod = T1 * T2

        sp = spinor_dummy_pairs(prod)
        @test length(sp) == 1
        @test sp[1][1].vbundle == :SL2C_dot
    end

    @testset "spinor_dummy_pairs: no cross-bundle pairing" begin
        # spin_up(:A, :SL2C) and spin_down(:A, :SL2C_dot) are both free
        T1 = Tensor(:psi, [spin_up(:A)])
        T2 = Tensor(:chi, [spin_dot_down(:A)])
        prod = T1 * T2

        sp = spinor_dummy_pairs(prod)
        @test isempty(sp)
        # Both should be free
        fi = free_indices(prod)
        @test length(fi) == 2
    end

    @testset "spinor_dummy_pairs: mixed expression with Tangent indices" begin
        # T^a_{A} psi^A  -- Tangent index a is free, SL2C index A is contracted
        T1 = Tensor(:T, [up(:a), spin_down(:A)])
        T2 = Tensor(:psi, [spin_up(:A)])
        prod = T1 * T2

        sp = spinor_dummy_pairs(prod)
        @test length(sp) == 1
        @test sp[1][1].name == :A

        # Tangent contraction should not appear in spinor_dummy_pairs
        all_dp = dummy_pairs(prod)
        @test length(all_dp) == 1  # only A is a dummy
    end

    @testset "spinor_dummy_pairs: tangent-only expression returns empty" begin
        T1 = Tensor(:T, [up(:a), down(:b)])
        T2 = Tensor(:S, [up(:b)])
        prod = T1 * T2

        sp = spinor_dummy_pairs(prod)
        @test isempty(sp)
    end

    @testset "spinor_dummy_pairs: multiple contractions" begin
        # psi^A_{Bp} chi_A^{Bp}  -- one SL2C pair (A), one SL2C_dot pair (Bp)
        T1 = Tensor(:psi, [spin_up(:A), spin_dot_down(:Bp)])
        T2 = Tensor(:chi, [spin_down(:A), spin_dot_up(:Bp)])
        prod = T1 * T2

        sp = spinor_dummy_pairs(prod)
        @test length(sp) == 2
        vbundles = Set(p[1].vbundle for p in sp)
        @test :SL2C in vbundles
        @test :SL2C_dot in vbundles
    end

    @testset "normalize_spinor_dummies: basic SL2C rename" begin
        # psi^A chi_A  -> dummies renamed to P
        T1 = Tensor(:psi, [spin_up(:A)])
        T2 = Tensor(:chi, [spin_down(:A)])
        prod = T1 * T2

        result = normalize_spinor_dummies(prod)
        sp = spinor_dummy_pairs(result)
        @test length(sp) == 1
        @test sp[1][1].name == :P
    end

    @testset "normalize_spinor_dummies: dotted rename" begin
        T1 = Tensor(:phi, [spin_dot_up(:Ap)])
        T2 = Tensor(:xi, [spin_dot_down(:Ap)])
        prod = T1 * T2

        result = normalize_spinor_dummies(prod)
        sp = spinor_dummy_pairs(result)
        @test length(sp) == 1
        @test sp[1][1].name == :Pp
    end

    @testset "normalize_spinor_dummies: mixed undotted+dotted" begin
        # psi^A_{Bp} chi_A^{Bp}  -> A->P, Bp->Pp
        T1 = Tensor(:psi, [spin_up(:A), spin_dot_down(:Bp)])
        T2 = Tensor(:chi, [spin_down(:A), spin_dot_up(:Bp)])
        prod = T1 * T2

        result = normalize_spinor_dummies(prod)
        sp = spinor_dummy_pairs(result)
        @test length(sp) == 2

        sl2c_names = Set(p[1].name for p in sp if p[1].vbundle === :SL2C)
        sl2c_dot_names = Set(p[1].name for p in sp if p[1].vbundle === :SL2C_dot)
        @test :P in sl2c_names
        @test :Pp in sl2c_dot_names
    end

    @testset "normalize_spinor_dummies: tangent dummies untouched" begin
        # T^a_{A} S_a^{A}  -- both a (Tangent) and A (SL2C) are dummies
        T1 = Tensor(:T, [up(:a), spin_down(:A)])
        T2 = Tensor(:S, [down(:a), spin_up(:A)])
        prod = T1 * T2

        result = normalize_spinor_dummies(prod)
        # SL2C dummy A -> P
        sp = spinor_dummy_pairs(result)
        @test length(sp) == 1
        @test sp[1][1].name == :P

        # Tangent dummy 'a' should remain as 'a' (unchanged)
        all_dp = dummy_pairs(result)
        tangent_dp = filter(p -> p[1].vbundle === :Tangent, all_dp)
        @test length(tangent_dp) == 1
        @test tangent_dp[1][1].name == :a
    end

    @testset "normalize_spinor_dummies: no-op on non-spinor expression" begin
        T1 = Tensor(:T, [up(:a), down(:b)])
        T2 = Tensor(:S, [up(:b)])
        prod = T1 * T2

        result = normalize_spinor_dummies(prod)
        # Should be identical (no spinor dummies to rename)
        dp = dummy_pairs(result)
        @test length(dp) == 1
        @test dp[1][1].name == :b
    end

    @testset "normalize_spinor_dummies: ordering by first occurrence" begin
        # Two SL2C contractions: C appears before B in the index list
        # psi^C_{B} chi_C^{B}  -> first occurrence: C=1, B=2  -> C->P, B->Q
        T1 = Tensor(:psi, [spin_up(:C), spin_down(:B)])
        T2 = Tensor(:chi, [spin_down(:C), spin_up(:B)])
        prod = T1 * T2

        result = normalize_spinor_dummies(prod)
        sp = spinor_dummy_pairs(result)
        names = sort([p[1].name for p in sp])
        @test names == [:P, :Q]
    end

    @testset "free_indices with spinor indices" begin
        T = Tensor(:psi, [spin_down(:A)])
        fi = free_indices(T)
        @test length(fi) == 1
        @test fi[1] == spin_down(:A)
    end

end
