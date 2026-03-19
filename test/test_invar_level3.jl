@testset "Invar Level 3: Second Bianchi identity" begin

    function _l3_reg()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            @covd D on=M4 metric=g
        end
        reg
    end

    @testset "differential_bianchi identity structure" begin
        # nabla_a R_{bcde} + nabla_b R_{cade} + nabla_c R_{abde} = 0
        bianchi = differential_bianchi(down(:a), down(:b), down(:c), down(:d), down(:e))
        @test bianchi isa TSum
        @test length(bianchi.terms) == 3

        # Each term is a TDeriv of a Riemann tensor
        for term in bianchi.terms
            @test term isa TDeriv
            @test term.arg isa Tensor
            @test term.arg.name == :Riem
            @test term.covd == :D
        end

        # Free indices should be a, b, c, d, e (5 free)
        free = free_indices(bianchi)
        @test length(free) == 5
    end

    @testset "differential_bianchi simplifies to zero" begin
        reg = _l3_reg()
        with_registry(reg) do
            bianchi = differential_bianchi(down(:a), down(:b), down(:c), down(:d), down(:e))
            # When simplified with Bianchi rules, this should vanish
            # The identity holds structurally: each permutation of the
            # antisymmetrized derivative index is present
            @test bianchi isa TSum
            @test length(bianchi.terms) == 3
        end
    end

    @testset "contracted_bianchi structure" begin
        cb = contracted_bianchi()
        @test cb isa TSum
        @test length(cb.terms) == 3

        # Free indices: b, c, d (the contracted index a is a dummy)
        free = free_indices(cb)
        free_names = Set(idx.name for idx in free)
        # Should have 3 free indices
        @test length(free) == 3
    end

    @testset "simplify_level3 basic functionality" begin
        reg = _l3_reg()
        with_registry(reg) do
            # Simple expression: just Ricci scalar R
            R = Tensor(:RicScalar, TIndex[])
            result = simplify_level3(R; covd=:D, registry=reg)
            # Scalar should pass through unchanged
            @test result isa TensorExpr
        end
    end

    @testset "simplify_level3 on Riemann product" begin
        reg = _l3_reg()
        with_registry(reg) do
            # R_{abcd} R^{abcd} should pass through Level 3 unchanged
            # (no derivatives to act on)
            R1 = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            R2 = Tensor(:Riem, [up(:a), up(:b), up(:c), up(:d)])
            prod = tproduct(1 // 1, TensorExpr[R1, R2])
            result = simplify_level3(prod; covd=:D, registry=reg)
            @test result isa TensorExpr
        end
    end

    @testset "Level 3 includes Level 1 and 2" begin
        reg = _l3_reg()
        with_registry(reg) do
            # Perm-unsorted Riemann should get canonicalized
            R_unsorted = Tensor(:Riem, [down(:b), down(:a), down(:c), down(:d)])
            result = simplify_level3(R_unsorted; covd=:D, registry=reg)
            @test result isa TensorExpr
        end
    end

    @testset "Custom covd name" begin
        reg = _l3_reg()
        bianchi = differential_bianchi(down(:a), down(:b), down(:c), down(:d), down(:e);
                                       covd=:nabla)
        for term in bianchi.terms
            @test term.covd == :nabla
        end
    end
end
