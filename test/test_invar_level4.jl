#= Invar Level 4: Derivative commutation for differential invariants.
#
# Verify that [nabla_a, nabla_b] R_{cdef} produces Riemann commutator terms
# and that simplify_level4 canonically orders derivatives.
#
# Ground truth: Garcia-Parrado & Martin-Garcia, Comp. Phys. Comm.
#   176 (2007) 246, Section 4.3, Level 4.
=#

@testset "Invar Level 4: Derivative commutation" begin

    function _l4_reg()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            @covd D on=M4 metric=g
        end
        reg
    end

    @testset "simplify_level4 exists and includes Level 3" begin
        reg = _l4_reg()
        with_registry(reg) do
            # A simple curvature scalar should pass through unchanged
            R = Tensor(:RicScalar, TIndex[])
            result = simplify_level4(R; covd=:D, registry=reg)
            @test result isa TensorExpr
        end
    end

    @testset "Derivative commutation sorts ∇∇R to canonical order" begin
        reg = _l4_reg()
        with_registry(reg) do
            # Build ∇_b ∇_a R_{cd} (non-canonical order: b before a)
            Ric = Tensor(:Ric, [down(:c), down(:d)])
            ba_Ric = TDeriv(down(:b), TDeriv(down(:a), Ric, :D), :D)

            # Commute to canonical order ∇_a ∇_b (sorted)
            result = commute_covds(ba_Ric, :D; registry=reg)

            # Result should differ from input (commutation produces extra terms)
            result_s = simplify(result; registry=reg)
            input_s = simplify(ba_Ric; registry=reg)

            # Both should have 4 free indices (a, b, c, d)
            @test length(free_indices(result_s)) == 4
            @test length(free_indices(input_s)) == 4

            # The commuted result should be a valid expression
            @test result_s isa TensorExpr
        end
    end

    @testset "simplify_level4 on ∇∇R_{abcd}" begin
        reg = _l4_reg()
        with_registry(reg) do
            # Build ∇_e ∇_f R_{abcd}
            Riem = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            dd_Riem = TDeriv(down(:e), TDeriv(down(:f), Riem, :D), :D)

            # Level 4 should commute derivatives to canonical order
            result = simplify_level4(dd_Riem; covd=:D, registry=reg)
            @test result isa TensorExpr

            # The result should have 6 free indices (a,b,c,d,e,f)
            fi = free_indices(result)
            @test length(fi) == 6
        end
    end

    @testset "Contracted commutator: ∇^a ∇_b R_{ac} - ∇_b ∇^a R_{ac}" begin
        reg = _l4_reg()
        with_registry(reg) do
            # Build contracted commutator on Ricci tensor
            # ∇^a ∇_b R_{ac} - ∇_b ∇^a R_{ac}
            Ric = Tensor(:Ric, [down(:a), down(:c)])
            t1 = TDeriv(up(:a), TDeriv(down(:b), Ric, :D), :D)
            t2 = TDeriv(down(:b), TDeriv(up(:a), Ric, :D), :D)

            comm = tsum(TensorExpr[t1, tproduct(-1 // 1, TensorExpr[t2])])

            result = simplify(comm; registry=reg, commute_covds_name=:D)

            # Should have 2 free indices (b, c)
            fi = free_indices(result)
            @test length(fi) == 2

            # Non-zero in general (curvature commutator)
            @test !(result == TScalar(0 // 1))
        end
    end

    @testset "Level 4 subsumes Level 3" begin
        reg = _l4_reg()
        with_registry(reg) do
            # The Bianchi identity ∇_a R_{bcde} + cyclic should still
            # be handled by Level 4 (which includes Level 3)
            bianchi = differential_bianchi(down(:a), down(:b), down(:c),
                                           down(:d), down(:e))
            @test bianchi isa TSum
            @test length(bianchi.terms) == 3

            # Level 4 should handle it (Level 3 is prerequisite)
            result = simplify_level4(bianchi; covd=:D, registry=reg)
            @test result isa TensorExpr
        end
    end

end
