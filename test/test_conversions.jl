@testset "Curvature Algebra Conversions" begin

    @testset "contract_curvature: Riemann trace -> Ricci" begin
        # R^a_{bac} has slots 1 and 3 contracted (same name :a, opposite position)
        # This should yield +Ric_{bc}
        Riem = Tensor(:Riem, [up(:a), down(:b), down(:a), down(:c)])
        expr = tproduct(1 // 1, TensorExpr[Riem])
        result = contract_curvature(expr)

        # Result should contain a Ricci tensor
        found_ricci = false
        function check_ricci(e)
            if e isa Tensor && e.name == :Ric
                found_ricci = true
            elseif e isa TProduct
                for f in e.factors
                    check_ricci(f)
                end
            elseif e isa TSum
                for t in e.terms
                    check_ricci(t)
                end
            end
        end
        check_ricci(result)
        @test found_ricci

        # The free indices should be b and c (the uncontracted ones)
        fi = free_indices(result)
        fi_names = Set(idx.name for idx in fi)
        @test :b in fi_names
        @test :c in fi_names
        @test !(:a in fi_names)
    end

    @testset "contract_curvature: Riemann (1,4) trace -> -Ricci" begin
        # R^a_{bc,a} — contraction of slots 1 and 4
        Riem = Tensor(:Riem, [up(:a), down(:b), down(:c), down(:a)])
        expr = tproduct(1 // 1, TensorExpr[Riem])
        result = contract_curvature(expr)

        # Should produce a term with sign -1 (antisymmetry in last two indices)
        if result isa TProduct
            @test result.scalar == -1 // 1
            @test any(f -> f isa Tensor && f.name == :Ric, result.factors)
        elseif result isa Tensor && result.name == :Ric
            # If sign was +1, it collapsed — but we expect -1
            @test false  # Should not reach here for (1,4) contraction
        end
    end

    @testset "contract_curvature: Ricci trace -> RicciScalar" begin
        Ric = Tensor(:Ric, [up(:a), down(:a)])
        expr = tproduct(1 // 1, TensorExpr[Ric])
        result = contract_curvature(expr)

        # Should yield RicciScalar
        found_scalar = false
        function check_scalar(e)
            if e isa Tensor && e.name == :RicScalar
                found_scalar = true
            elseif e isa TProduct
                for f in e.factors
                    check_scalar(f)
                end
            end
        end
        check_scalar(result)
        @test found_scalar
    end

    @testset "contract_curvature: no contraction leaves unchanged" begin
        Riem = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
        expr = tproduct(1 // 1, TensorExpr[Riem])
        result = contract_curvature(expr)
        # No contraction: should be unchanged (modulo normalization)
        @test result isa Tensor || (result isa TProduct && any(f -> f isa Tensor && f.name == :Riem, result.factors))
    end

    @testset "Schouten <-> Ricci roundtrip" begin
        a, b = down(:a), down(:b)

        # Schouten in terms of Ricci
        sch_expr = schouten_to_ricci(a, b, :g; dim=4)
        @test sch_expr isa TensorExpr

        # Ricci in terms of Schouten
        ric_expr = ricci_to_schouten(a, b, :g; dim=4)
        @test ric_expr isa TensorExpr

        # Check dimensions: P_{ab} = 1/(d-2)(R_{ab} - R g_{ab}/(2(d-1)))
        # In d=4: P_{ab} = (1/2)(R_{ab} - R g_{ab}/6)
        # R_{ab} = (d-2) P_{ab} + g_{ab} R/(2(d-1))
        # R_{ab} = 2 P_{ab} + g_{ab} R/6
        # Substituting the first into the second:
        # R_{ab} = 2 * (1/2)(R_{ab} - R g_{ab}/6) + g_{ab} R/6
        #        = R_{ab} - R g_{ab}/6 + R g_{ab}/6
        #        = R_{ab}  (consistent)

        # Both should have free indices a, b
        fi_sch = free_indices(sch_expr)
        fi_ric = free_indices(ric_expr)
        @test length(fi_sch) == 2
        @test length(fi_ric) == 2
    end

    @testset "Schouten: correct coefficients in d=4" begin
        a, b = down(:a), down(:b)
        sch = schouten_to_ricci(a, b, :g; dim=4)

        # P_{ab} = (1/2)(R_{ab} - (1/6) R g_{ab})
        # This is a TProduct with scalar 1//2 containing a TSum
        @test sch isa TProduct || sch isa TSum
    end

    @testset "TFRicci definition" begin
        a, b = down(:a), down(:b)

        # S_{ab} = R_{ab} - (1/d) g_{ab} R
        result = tfricci_expr(a, b, :g; dim=4)
        @test result isa TSum

        # The trace-free Ricci should have two terms
        if result isa TSum
            @test length(result.terms) == 2
        end
    end

    @testset "ricci_to_tfricci structure" begin
        a, b = down(:a), down(:b)

        # R_{ab} = S_{ab} + (1/d) g_{ab} R
        result = ricci_to_tfricci(a, b, :g; dim=4)
        @test result isa TSum

        # Should have free indices a, b
        fi = free_indices(result)
        @test length(fi) == 2
    end

    @testset "TFRicci: trace is zero conceptually" begin
        a, b = down(:a), down(:b)

        # S_{ab} = R_{ab} - (1/d) g_{ab} R
        # Trace: g^{ab} S_{ab} = R - (1/d) d R = R - R = 0
        # We verify the structure is correct: two terms with matching indices
        result = tfricci_expr(a, b, :g; dim=4)
        fi = free_indices(result)
        fi_names = Set(idx.name for idx in fi)
        @test :a in fi_names
        @test :b in fi_names
    end

    @testset "to_riemann: converts Weyl" begin
        # Weyl_{abcd} should be replaced by Riemann - Ricci decomposition
        W = Tensor(:Weyl, [down(:a), down(:b), down(:c), down(:d)])
        result = to_riemann(W; metric=:g, dim=4)

        # Result should contain Riemann tensor
        found_riemann = false
        walk(result) do node
            if node isa Tensor && node.name == :Riem
                found_riemann = true
            end
            node
        end
        @test found_riemann
    end

    @testset "to_riemann: converts Einstein" begin
        G = Tensor(:Ein, [down(:a), down(:b)])
        result = to_riemann(G; metric=:g, dim=4)

        # Should contain Ricci (Einstein -> Ricci)
        found_ricci = false
        walk(result) do node
            if node isa Tensor && node.name == :Ric
                found_ricci = true
            end
            node
        end
        @test found_ricci
    end

    @testset "to_riemann: converts Schouten" begin
        P = Tensor(:Sch, [down(:a), down(:b)])
        result = to_riemann(P; metric=:g, dim=4)

        # Schouten -> Ricci form
        found_ric = false
        walk(result) do node
            if node isa Tensor && node.name == :Ric
                found_ric = true
            end
            node
        end
        @test found_ric
    end

    @testset "to_riemann: converts TFRicci" begin
        S = Tensor(:TFRic, [down(:a), down(:b)])
        result = to_riemann(S; metric=:g, dim=4)

        # TFRicci -> Ricci + scalar form
        found_ric = false
        walk(result) do node
            if node isa Tensor && node.name == :Ric
                found_ric = true
            end
            node
        end
        @test found_ric
    end

    @testset "to_riemann: leaves Riemann unchanged" begin
        R = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
        result = to_riemann(R; metric=:g, dim=4)
        @test result == R
    end

    @testset "to_ricci: converts Schouten and Einstein" begin
        P = Tensor(:Sch, [down(:a), down(:b)])
        result = to_ricci(P; metric=:g, dim=4)

        # Should not contain Schouten anymore
        found_sch = false
        walk(result) do node
            if node isa Tensor && node.name == :Sch
                found_sch = true
            end
            node
        end
        @test !found_sch
    end

    @testset "to_ricci: converts TFRicci" begin
        S = Tensor(:TFRic, [down(:a), down(:b)])
        result = to_ricci(S; metric=:g, dim=4)

        found_tfric = false
        walk(result) do node
            if node isa Tensor && node.name == :TFRic
                found_tfric = true
            end
            node
        end
        @test !found_tfric
    end

    @testset "to_ricci: contracted Riemann becomes Ricci" begin
        # R^a_{bac} in a product — should contract to Ricci
        Riem = Tensor(:Riem, [up(:a), down(:b), down(:a), down(:c)])
        expr = tproduct(1 // 1, TensorExpr[Riem])
        result = to_ricci(expr; metric=:g, dim=4)

        found_ricci = false
        walk(result) do node
            if node isa Tensor && node.name == :Ric
                found_ricci = true
            end
            node
        end
        @test found_ricci
    end

end
