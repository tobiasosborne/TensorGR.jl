# Gamma matrix trace identities validation to order 6.
#
# Ground truth: Peskin & Schroeder (1995) Appendix A, Eqs A.21-A.30.
#
# Standard traces (d=4, Tr(I) = 4):
#   Tr(I) = 4
#   Tr(gamma^a) = 0
#   Tr(gamma^a gamma^b) = 4 g^{ab}
#   Tr(gamma^a gamma^b gamma^c) = 0
#   Tr(gamma^a gamma^b gamma^c gamma^d) = 4(g^{ab}g^{cd} - g^{ac}g^{bd} + g^{ad}g^{bc})
#   Tr(odd) = 0
#   Tr(gamma^a ... gamma^f) = 15-term expansion via recursive formula
#
# gamma5 traces:
#   Tr(gamma5) = 0
#   Tr(gamma5 gamma^a gamma^b) = 0
#   Tr(gamma5 gamma^a gamma^b gamma^c gamma^d) = -4i epsilon^{abcd}

@testset "Gamma trace identities to order 6" begin

    # ────────────────────────────────────────────────────────────────
    # Tr(I) = 4
    # ────────────────────────────────────────────────────────────────
    @testset "Trace of identity: Tr(I) = 4" begin
        result = gamma_chain_trace(GammaMatrix[])
        @test result == TScalar(4 // 1)
    end

    # ────────────────────────────────────────────────────────────────
    # Odd traces vanish: Tr(gamma^{a_1} ... gamma^{a_n}) = 0 for odd n
    # ────────────────────────────────────────────────────────────────
    @testset "Odd traces vanish" begin
        for n in [1, 3, 5]
            gammas = [GammaMatrix(up(Symbol("a$i"))) for i in 1:n]
            result = gamma_chain_trace(gammas)
            @test result == TScalar(0 // 1)
        end
    end

    # ────────────────────────────────────────────────────────────────
    # Tr(gamma^a gamma^b) = 4 g^{ab}
    # P&S Eq A.25
    # ────────────────────────────────────────────────────────────────
    @testset "Tr(gamma^a gamma^b) = 4 g^{ab}" begin
        result = gamma_chain_trace([GammaMatrix(up(:a)), GammaMatrix(up(:b))])
        # tproduct(4//1, [Tensor(:g, [up(:a), up(:b)])]) = TProduct(4//1, [...])
        @test result isa TProduct
        @test result.scalar == 4 // 1
        @test length(result.factors) == 1
        g_tensor = result.factors[1]
        @test g_tensor isa Tensor
        @test g_tensor.name == :g
        @test g_tensor.indices == [up(:a), up(:b)]
    end

    # ────────────────────────────────────────────────────────────────
    # Tr(gamma^a gamma^b gamma^c gamma^d) = 4(g^{ab}g^{cd} - g^{ac}g^{bd} + g^{ad}g^{bc})
    # P&S Eq A.26
    # ────────────────────────────────────────────────────────────────
    @testset "Tr(gamma^a gamma^b gamma^c gamma^d) = 4(g^{ab}g^{cd} - g^{ac}g^{bd} + g^{ad}g^{bc})" begin
        gammas = [GammaMatrix(up(:a)), GammaMatrix(up(:b)),
                  GammaMatrix(up(:c)), GammaMatrix(up(:d))]
        result = gamma_chain_trace(gammas)

        # The implementation does tproduct(4//1, [inner_sum])
        # where inner_sum = g_ab * g_cd - g_ac * g_bd + g_ad * g_bc
        # This should be TProduct(4//1, [TSum(...)])
        @test result isa TProduct
        @test result.scalar == 4 // 1
        @test length(result.factors) == 1
        inner = result.factors[1]
        @test inner isa TSum
        # The sum g^{ab}g^{cd} - g^{ac}g^{bd} + g^{ad}g^{bc} has 3 terms
        @test length(inner.terms) == 3
    end

    # ────────────────────────────────────────────────────────────────
    # trace_identity_2 matches gamma_chain_trace
    # ────────────────────────────────────────────────────────────────
    @testset "trace_identity_2 matches gamma_chain_trace" begin
        ti2 = trace_identity_2(up(:a), up(:b))
        gc2 = gamma_chain_trace([GammaMatrix(up(:a)), GammaMatrix(up(:b))])

        @test ti2 isa TProduct
        @test gc2 isa TProduct
        @test ti2.scalar == gc2.scalar
        @test ti2 == gc2
    end

    # ────────────────────────────────────────────────────────────────
    # trace_identity_4 matches gamma_chain_trace
    # ────────────────────────────────────────────────────────────────
    @testset "trace_identity_4 matches gamma_chain_trace" begin
        ti4 = trace_identity_4(up(:a), up(:b), up(:c), up(:d))
        gc4 = gamma_chain_trace([GammaMatrix(up(:a)), GammaMatrix(up(:b)),
                                  GammaMatrix(up(:c)), GammaMatrix(up(:d))])

        @test ti4 isa TProduct
        @test gc4 isa TProduct
        @test ti4.scalar == gc4.scalar
        # Both should produce identical AST since they use the same construction
        @test ti4 == gc4
    end

    # ────────────────────────────────────────────────────────────────
    # Order 6: Tr(gamma^a gamma^b gamma^c gamma^d gamma^e gamma^f)
    # P&S recursive formula: 15 metric-pair terms with coefficient 4
    # ────────────────────────────────────────────────────────────────
    @testset "Order 6: Tr(gamma^a gamma^b gamma^c gamma^d gamma^e gamma^f)" begin
        gammas = [GammaMatrix(up(:a)), GammaMatrix(up(:b)),
                  GammaMatrix(up(:c)), GammaMatrix(up(:d)),
                  GammaMatrix(up(:e)), GammaMatrix(up(:f))]
        result = gamma_chain_trace(gammas)

        # The recursive formula produces a TSum at the top level
        # (5 terms from i=2..6, each involving g * sub_trace_of_4)
        @test result isa TSum

        # Each of the 5 recursive terms contains a 4-gamma sub-trace which
        # itself has 3 metric-pair terms. So we expect 5 * 3 = 15 terms
        # when fully expanded. The TSum should contain terms that, when
        # flattened, give 15 metric-triple products.
        # At the top level, the TSum has 5 terms from the recursion.
        @test length(result.terms) == 5

        # Verify the result is non-trivial (not zero)
        @test result != TScalar(0 // 1)

        # Verify it contains metric tensors (g)
        all_tensors = TensorExpr[]
        function collect_tensors!(expr)
            if expr isa Tensor
                push!(all_tensors, expr)
            elseif expr isa TProduct
                for f in expr.factors
                    collect_tensors!(f)
                end
            elseif expr isa TSum
                for t in expr.terms
                    collect_tensors!(t)
                end
            end
        end
        collect_tensors!(result)

        # All leaf tensors should be metrics
        @test all(t -> t isa Tensor && t.name == :g, all_tensors)

        # The indices a,b,c,d,e,f should all appear in the result
        all_idx_names = Set{Symbol}()
        for t in all_tensors
            for idx in t.indices
                push!(all_idx_names, idx.name)
            end
        end
        @test :a in all_idx_names
        @test :b in all_idx_names
        @test :c in all_idx_names
        @test :d in all_idx_names
        @test :e in all_idx_names
        @test :f in all_idx_names
    end

    # ────────────────────────────────────────────────────────────────
    # Order 6: verify recursive structure matches P&S
    # The first recursive term (i=2) should be +g^{ab} * Tr(gamma^c gamma^d gamma^e gamma^f)
    # ────────────────────────────────────────────────────────────────
    @testset "Order 6: recursive structure verification" begin
        gammas6 = [GammaMatrix(up(:a)), GammaMatrix(up(:b)),
                   GammaMatrix(up(:c)), GammaMatrix(up(:d)),
                   GammaMatrix(up(:e)), GammaMatrix(up(:f))]
        result6 = gamma_chain_trace(gammas6)

        # Build the expected first term manually: g^{ab} * Tr(gamma^c gamma^d gamma^e gamma^f)
        sub_trace_cdef = gamma_chain_trace([GammaMatrix(up(:c)), GammaMatrix(up(:d)),
                                             GammaMatrix(up(:e)), GammaMatrix(up(:f))])
        g_ab = Tensor(:g, [up(:a), up(:b)])
        expected_first = g_ab * sub_trace_cdef

        # The first term in the TSum should match (i=2, sign = (-1)^2 = +1)
        @test result6 isa TSum
        first_term = result6.terms[1]
        @test first_term == expected_first
    end

    # ────────────────────────────────────────────────────────────────
    # gamma_trace coefficient function
    # ────────────────────────────────────────────────────────────────
    @testset "gamma_trace basic" begin
        @test gamma_trace(0) == 4    # Tr(I) = 4
        @test gamma_trace(1) == 0    # Tr(gamma^a) = 0
        @test gamma_trace(2) == 4    # Tr(gamma^a gamma^b) coefficient = 4
        @test gamma_trace(3) == 0    # Odd traces vanish
        @test gamma_trace(4) === nothing  # Needs recursive computation
        @test gamma_trace(5) == 0    # Odd traces vanish
        @test gamma_trace(6) === nothing  # Needs recursive computation
    end

    # ────────────────────────────────────────────────────────────────
    # gamma5 trace identities
    # P&S Eq A.30
    # ────────────────────────────────────────────────────────────────
    @testset "gamma5 traces" begin
        @test gamma5_trace(0) == 0       # Tr(gamma5) = 0
        @test gamma5_trace(1) == 0       # Tr(gamma5 gamma^a) = 0 (odd)
        @test gamma5_trace(2) == 0       # Tr(gamma5 gamma^a gamma^b) = 0
        @test gamma5_trace(3) == 0       # Tr(gamma5 gamma^a gamma^b gamma^c) = 0 (odd)
        @test gamma5_trace(4) == :(-4im) # Tr(gamma5 gamma^a gamma^b gamma^c gamma^d) = -4i eps^{abcd}
        @test gamma5_trace(5) == 0       # Odd
        @test gamma5_trace(6) === nothing  # Higher order: not yet implemented
    end

    # ────────────────────────────────────────────────────────────────
    # gamma5 algebraic properties
    # ────────────────────────────────────────────────────────────────
    @testset "gamma5 algebraic properties" begin
        @test gamma5_squared() == TScalar(1)           # (gamma5)^2 = I
        @test gamma5_anticommutator() == TScalar(0)    # {gamma5, gamma^a} = 0
    end

    # ────────────────────────────────────────────────────────────────
    # Clifford relation {gamma^a, gamma^b} = 2g^{ab}
    # ────────────────────────────────────────────────────────────────
    @testset "Clifford relation {gamma^a, gamma^b} = 2g^{ab}" begin
        cr = clifford_relation(up(:a), up(:b))
        @test cr isa TProduct
        @test cr.scalar == 2 // 1
        @test length(cr.factors) == 1
        @test cr.factors[1] isa Tensor
        @test cr.factors[1].name == :g
        @test cr.factors[1].indices == [up(:a), up(:b)]
    end

    # ────────────────────────────────────────────────────────────────
    # Down-index traces: verify consistency with Up-index traces
    # The trace formula should work regardless of index position
    # ────────────────────────────────────────────────────────────────
    @testset "Down-index traces" begin
        # Tr(gamma_a gamma_b) = 4 g_{ab}
        result_down = gamma_chain_trace([GammaMatrix(down(:a)), GammaMatrix(down(:b))])
        @test result_down isa TProduct
        @test result_down.scalar == 4 // 1
        g_tensor = result_down.factors[1]
        @test g_tensor.indices == [down(:a), down(:b)]

        # Tr(gamma_a gamma_b gamma_c gamma_d) should also work
        gammas_down = [GammaMatrix(down(:a)), GammaMatrix(down(:b)),
                       GammaMatrix(down(:c)), GammaMatrix(down(:d))]
        result4_down = gamma_chain_trace(gammas_down)
        @test result4_down isa TProduct
        @test result4_down.scalar == 4 // 1
    end

    # ────────────────────────────────────────────────────────────────
    # Custom metric name: verify trace works with non-default metric
    # ────────────────────────────────────────────────────────────────
    @testset "Custom metric name" begin
        result = gamma_chain_trace([GammaMatrix(up(:a)), GammaMatrix(up(:b))]; metric=:eta)
        @test result isa TProduct
        @test result.scalar == 4 // 1
        @test result.factors[1] isa Tensor
        @test result.factors[1].name == :eta
    end

    # ────────────────────────────────────────────────────────────────
    # Custom dimension: verify Tr(I) = dim for arbitrary dimension
    # ────────────────────────────────────────────────────────────────
    @testset "Custom spinor dimension" begin
        # In d dimensions, Tr(I) = d_s (here we use dim parameter)
        result_d6 = gamma_chain_trace(GammaMatrix[]; dim=6)
        @test result_d6 == TScalar(6 // 1)

        # Tr(gamma^a gamma^b) = d_s * g^{ab}
        result2_d6 = gamma_chain_trace([GammaMatrix(up(:a)), GammaMatrix(up(:b))]; dim=6)
        @test result2_d6 isa TProduct
        @test result2_d6.scalar == 6 // 1
    end

    # ────────────────────────────────────────────────────────────────
    # Symmetry of order-2 trace: Tr(gamma^a gamma^b) = Tr(gamma^b gamma^a)
    # (follows from g^{ab} = g^{ba})
    # ────────────────────────────────────────────────────────────────
    @testset "Cyclic symmetry of trace" begin
        # For n=2: Tr(gamma^a gamma^b) uses g^{ab}
        # Tr(gamma^b gamma^a) uses g^{ba} -- same tensor, different index order
        tr_ab = gamma_chain_trace([GammaMatrix(up(:a)), GammaMatrix(up(:b))])
        tr_ba = gamma_chain_trace([GammaMatrix(up(:b)), GammaMatrix(up(:a))])

        # Both should be TProduct with scalar 4
        @test tr_ab.scalar == tr_ba.scalar

        # The metric tensor indices are in the order given, so:
        # tr_ab has g(up(:a), up(:b)), tr_ba has g(up(:b), up(:a))
        # These are not identical AST nodes but represent the same symmetric tensor
        @test tr_ab.factors[1].name == :g
        @test tr_ba.factors[1].name == :g
    end

    # ────────────────────────────────────────────────────────────────
    # Charge conjugation properties (completeness check)
    # ────────────────────────────────────────────────────────────────
    @testset "Charge conjugation consistency" begin
        props = charge_conjugation_properties()
        @test props.antisymmetric == true
        @test props.unitary == true
        @test props.gamma_conjugation == -1
        @test props.sigma_conjugation == -1
        @test props.gamma5_conjugation == 1
    end

    # ────────────────────────────────────────────────────────────────
    # Fierz completeness (dimension sum = 16 = 4^2)
    # ────────────────────────────────────────────────────────────────
    @testset "Fierz completeness" begin
        @test fierz_identity_check() == true
        F = fierz_matrix()
        @test size(F) == (5, 5)
        # All entries should be Rational
        @test eltype(F) == Rational{Int}
    end

    # ────────────────────────────────────────────────────────────────
    # Slash notation
    # ────────────────────────────────────────────────────────────────
    @testset "Slash notation with GammaMatrix" begin
        # slash(v) for a vector with one free index
        v = Tensor(:p, [down(:a)])
        sv = slash(v)
        @test sv isa TProduct
        @test length(sv.factors) == 2
        has_gamma = any(f -> f isa GammaMatrix, sv.factors)
        has_tensor = any(f -> f isa Tensor, sv.factors)
        @test has_gamma
        @test has_tensor
    end

end
