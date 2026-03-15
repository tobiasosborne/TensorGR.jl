@testset "SpinIndex convenience layer" begin

    @testset "spinor(:A) creates correct SL2C index" begin
        idx = spinor(:A)
        @test idx == TIndex(:A, Up, :SL2C)
        @test idx.name == :A
        @test idx.position == Up
        @test idx.vbundle == :SL2C
    end

    @testset "spinor_dot(:A) creates correct SL2C_dot index" begin
        idx = spinor_dot(:A)
        @test idx == TIndex(:A, Up, :SL2C_dot)
        @test idx.name == :A
        @test idx.position == Up
        @test idx.vbundle == :SL2C_dot
    end

    @testset "spinor(:A, up=false) creates lower index" begin
        idx = spinor(:A, up=false)
        @test idx == TIndex(:A, Down, :SL2C)
        @test idx.position == Down
        @test idx.vbundle == :SL2C
    end

    @testset "spinor_dot(:B, up=false) creates lower dotted index" begin
        idx = spinor_dot(:B, up=false)
        @test idx == TIndex(:B, Down, :SL2C_dot)
        @test idx.position == Down
        @test idx.vbundle == :SL2C_dot
    end

    @testset "is_undotted / is_dotted_spinor predicates" begin
        # Undotted indices
        @test is_undotted(spinor(:A)) == true
        @test is_undotted(spinor(:B, up=false)) == true
        @test is_undotted(spin_up(:C)) == true
        @test is_undotted(spin_down(:D)) == true

        # Dotted indices
        @test is_dotted_spinor(spinor_dot(:A)) == true
        @test is_dotted_spinor(spinor_dot(:B, up=false)) == true
        @test is_dotted_spinor(spin_dot_up(:Cp)) == true
        @test is_dotted_spinor(spin_dot_down(:Dp)) == true

        # Cross-checks: undotted is not dotted, dotted is not undotted
        @test is_undotted(spinor_dot(:A)) == false
        @test is_dotted_spinor(spinor(:A)) == false

        # Tangent indices are neither
        @test is_undotted(up(:a)) == false
        @test is_dotted_spinor(up(:a)) == false
        @test is_undotted(down(:b)) == false
        @test is_dotted_spinor(down(:b)) == false
    end

    @testset "spinor_dummy creates matched undotted pairs" begin
        A_up, A_dn = spinor_dummy(:A)

        @test A_up == TIndex(:A, Up, :SL2C)
        @test A_dn == TIndex(:A, Down, :SL2C)

        # Same name, opposite positions, same vbundle
        @test A_up.name == A_dn.name
        @test A_up.position == Up
        @test A_dn.position == Down
        @test A_up.vbundle == A_dn.vbundle == :SL2C

        # Should form a valid dummy pair when used in a product
        T1 = Tensor(:psi, [A_up])
        T2 = Tensor(:chi, [A_dn])
        prod = T1 * T2
        dp = dummy_pairs(prod)
        @test length(dp) == 1
        @test dp[1][1].name == :A
    end

    @testset "spinor_dot_dummy creates matched dotted pairs" begin
        Ap_up, Ap_dn = spinor_dot_dummy(:Ap)

        @test Ap_up == TIndex(:Ap, Up, :SL2C_dot)
        @test Ap_dn == TIndex(:Ap, Down, :SL2C_dot)

        # Same name, opposite positions, same vbundle
        @test Ap_up.name == Ap_dn.name
        @test Ap_up.position == Up
        @test Ap_dn.position == Down
        @test Ap_up.vbundle == Ap_dn.vbundle == :SL2C_dot

        # Should form a valid dummy pair when used in a product
        T1 = Tensor(:phi, [Ap_up])
        T2 = Tensor(:xi, [Ap_dn])
        prod = T1 * T2
        dp = dummy_pairs(prod)
        @test length(dp) == 1
        @test dp[1][1].name == :Ap
    end

    @testset "spinor_pair returns correct undotted+dotted pair" begin
        A, Adot = spinor_pair(:A)

        @test A == TIndex(:A, Up, :SL2C)
        @test Adot == TIndex(:A, Up, :SL2C_dot)

        # Both upper, same name, different vbundles
        @test A.name == Adot.name == :A
        @test A.position == Up
        @test Adot.position == Up
        @test A.vbundle == :SL2C
        @test Adot.vbundle == :SL2C_dot

        # The two indices should NOT form a dummy pair (different vbundles)
        T1 = Tensor(:psi, [A])
        T2 = Tensor(:chi, [Adot])
        prod = T1 * T2
        dp = dummy_pairs(prod)
        @test isempty(dp)

        # Both should be free
        fi = free_indices(prod)
        @test length(fi) == 2
    end

    @testset "Interoperability with existing spin_up/spin_down" begin
        # spinor(:A) == spin_up(:A)
        @test spinor(:A) == spin_up(:A)

        # spinor(:B, up=false) == spin_down(:B)
        @test spinor(:B, up=false) == spin_down(:B)

        # spinor_dot(:Ap) == spin_dot_up(:Ap)
        @test spinor_dot(:Ap) == spin_dot_up(:Ap)

        # spinor_dot(:Bp, up=false) == spin_dot_down(:Bp)
        @test spinor_dot(:Bp, up=false) == spin_dot_down(:Bp)

        # Can mix old and new API in the same expression
        sigma = Tensor(:sigma, [up(:a), spinor(:A, up=false), spinor_dot(:Ap, up=false)])
        @test sigma.indices[1].vbundle == :Tangent
        @test sigma.indices[2].vbundle == :SL2C
        @test sigma.indices[3].vbundle == :SL2C_dot
        @test sigma.indices[2] == spin_down(:A)
        @test sigma.indices[3] == spin_dot_down(:Ap)
    end

    @testset "SpinIndex struct wrapping" begin
        # Can wrap undotted index
        idx = spinor(:A)
        si = SpinIndex(idx, false)
        @test si.idx == idx
        @test si.dotted == false

        # Can wrap dotted index
        idx_dot = spinor_dot(:Bp)
        si_dot = SpinIndex(idx_dot, true)
        @test si_dot.idx == idx_dot
        @test si_dot.dotted == true

        # Equality
        @test SpinIndex(spinor(:A), false) == SpinIndex(spinor(:A), false)
        @test SpinIndex(spinor(:A), false) != SpinIndex(spinor(:B), false)
        @test SpinIndex(spinor_dot(:A), true) != SpinIndex(spinor(:A), false)

        # Validation: mismatched dotted flag and vbundle
        @test_throws ErrorException SpinIndex(spinor(:A), true)       # SL2C but dotted=true
        @test_throws ErrorException SpinIndex(spinor_dot(:A), false)  # SL2C_dot but dotted=false
        @test_throws ErrorException SpinIndex(up(:a), false)          # Tangent, not SL2C
    end

    @testset "is_dotted_spinor matches is_dotted" begin
        # is_dotted_spinor should agree with is_dotted for all spinor indices
        for name in [:A, :B, :C, :Ap, :Bp]
            idx_sl2c = TIndex(name, Up, :SL2C)
            @test is_dotted_spinor(idx_sl2c) == is_dotted(idx_sl2c)

            idx_sl2c_dot = TIndex(name, Up, :SL2C_dot)
            @test is_dotted_spinor(idx_sl2c_dot) == is_dotted(idx_sl2c_dot)
        end

        # Tangent: both should be false
        @test is_dotted_spinor(up(:a)) == false
        @test is_dotted(up(:a)) == false
    end

    @testset "spinor_dummy works with spinor_dummy_pairs" begin
        A_up, A_dn = spinor_dummy(:A)
        B_up, B_dn = spinor_dot_dummy(:Bp)

        # Mixed undotted + dotted contraction
        T1 = Tensor(:T, [A_up, B_up])
        T2 = Tensor(:S, [A_dn, B_dn])
        prod = T1 * T2

        sp = spinor_dummy_pairs(prod)
        @test length(sp) == 2

        vbundles = Set(p[1].vbundle for p in sp)
        @test :SL2C in vbundles
        @test :SL2C_dot in vbundles
    end

end
