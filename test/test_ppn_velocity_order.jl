@testset "PPN velocity-order expansion" begin

    function _ppn_reg()
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d,:e,:f,:g,:h,:i,:j,:k,:l]))
        define_ppn_potentials!(reg; manifold=:M4)
        reg
    end

    @testset "ppn_order: scalars" begin
        @test ppn_order(TScalar(1)) == 0
        @test ppn_order(TScalar(:x)) == 0
    end

    @testset "ppn_order: PPN potentials" begin
        @test ppn_order(Tensor(:U, TIndex[])) == 2
        @test ppn_order(Tensor(:Phi_W, TIndex[])) == 4
        @test ppn_order(Tensor(:Phi_1, TIndex[])) == 4
        @test ppn_order(Tensor(:V_ppn, [down(:a)])) == 3
        @test ppn_order(Tensor(:W_ppn, [down(:a)])) == 3
        @test ppn_order(Tensor(:A_ppn, TIndex[])) == 4
    end

    @testset "ppn_order: non-PPN tensor" begin
        @test ppn_order(Tensor(:T, [up(:a), down(:b)])) == 0
    end

    @testset "ppn_order: products" begin
        U = Tensor(:U, TIndex[])
        # U * U = O(2) + O(2) = O(4)
        @test ppn_order(U * U) == 4
        # 2 * U = O(2)
        @test ppn_order(tproduct(2 // 1, TensorExpr[U])) == 2
    end

    @testset "ppn_order: sums" begin
        U = Tensor(:U, TIndex[])
        Phi_W = Tensor(:Phi_W, TIndex[])
        # min(O(2), O(4)) = O(2)
        s = tsum(TensorExpr[U, Phi_W])
        @test ppn_order(s) == 2
    end

    @testset "ppn_max_order" begin
        U = Tensor(:U, TIndex[])
        Phi_W = Tensor(:Phi_W, TIndex[])
        s = tsum(TensorExpr[U, Phi_W])
        @test ppn_max_order(s) == 4
        @test ppn_max_order(U * U) == 4
    end

    @testset "truncate_ppn: sum" begin
        U = Tensor(:U, TIndex[])
        Phi_W = Tensor(:Phi_W, TIndex[])
        s = tsum(TensorExpr[
            tproduct(2 // 1, TensorExpr[U]),   # O(2)
            Phi_W                                # O(4)
        ])

        # Truncate at O(2): keep only the U term
        trunc2 = truncate_ppn(s, 2)
        @test trunc2 isa TensorExpr
        # The O(4) Phi_W term should be dropped
        if trunc2 isa TSum
            @test length(trunc2.terms) == 1
        else
            # Single term collapsed
            @test ppn_order(trunc2) <= 2
        end

        # Truncate at O(4): keep both
        trunc4 = truncate_ppn(s, 4)
        @test trunc4 isa TSum
        @test length(trunc4.terms) == 2
    end

    @testset "truncate_ppn: product" begin
        U = Tensor(:U, TIndex[])
        # U * U = O(4), truncate at O(2) should give zero
        uu = U * U
        @test truncate_ppn(uu, 2) == TScalar(0 // 1)
        @test truncate_ppn(uu, 4) == uu
    end

    @testset "ppn_truncate_metric: 1PN" begin
        reg = _ppn_reg()
        with_registry(reg) do
            mc = ppn_decompose(ppn_gr(), reg; order=2)
            mc_1pn = ppn_truncate_metric(mc, 1)

            # g0i should be zero at 1PN
            @test mc_1pn.g0i == TScalar(0 // 1)

            # g00 should have only O(2) terms (no O(4))
            @test ppn_max_order(mc_1pn.g00) <= 2
        end
    end

    @testset "ppn_truncate_metric: 2PN" begin
        reg = _ppn_reg()
        with_registry(reg) do
            mc = ppn_decompose(ppn_gr(), reg; order=2)
            mc_2pn = ppn_truncate_metric(mc, 2)

            # g0i should be nonzero at 2PN (has O(3) terms)
            @test mc_2pn.g0i != TScalar(0 // 1)

            # g00 should have terms up to O(4)
            @test ppn_max_order(mc_2pn.g00) <= 4
        end
    end

    @testset "ppn_truncate_metric: order validation" begin
        mc = PPNMetricComponents(TScalar(1), TScalar(0), TScalar(1))
        @test_throws ErrorException ppn_truncate_metric(mc, 3)
    end

    @testset "PPN_ORDER_TABLE completeness" begin
        # All standard PPN potentials should be in the table
        @test haskey(PPN_ORDER_TABLE, :U)
        @test haskey(PPN_ORDER_TABLE, :U_ppn)
        @test haskey(PPN_ORDER_TABLE, :Phi_W)
        @test haskey(PPN_ORDER_TABLE, :V_ppn)
        @test haskey(PPN_ORDER_TABLE, :W_ppn)
        @test haskey(PPN_ORDER_TABLE, :A_ppn)
        for k in 1:4
            @test haskey(PPN_ORDER_TABLE, Symbol(:Phi_, k))
        end
    end

end
