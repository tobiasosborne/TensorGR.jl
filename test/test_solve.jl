@testset "solve_tensors" begin
    reg = TensorRegistry()

    # Set up a 4D manifold with metric
    register_manifold!(reg, ManifoldProperties(:M, 4, :g, :∂, [:a, :b, :c, :d, :e, :f, :m, :n, :p, :q]))
    register_tensor!(reg, TensorProperties(
        name=:g, manifold=:M, rank=(0, 2),
        symmetries=SymmetrySpec[Symmetric(1, 2)],
        is_metric=true))
    register_tensor!(reg, TensorProperties(name=:δ, manifold=:M, rank=(1, 1), is_delta=true))

    # Define some tensors
    register_tensor!(reg, TensorProperties(
        name=:X, manifold=:M, rank=(0, 2),
        symmetries=SymmetrySpec[Symmetric(1, 2)]))
    register_tensor!(reg, TensorProperties(name=:R, manifold=:M, rank=(0, 0)))
    register_tensor!(reg, TensorProperties(
        name=:Ric, manifold=:M, rank=(0, 2),
        symmetries=SymmetrySpec[Symmetric(1, 2)]))
    register_tensor!(reg, TensorProperties(
        name=:T, manifold=:M, rank=(0, 2),
        symmetries=SymmetrySpec[Symmetric(1, 2)]))
    register_tensor!(reg, TensorProperties(
        name=:G, manifold=:M, rank=(0, 2),
        symmetries=SymmetrySpec[Symmetric(1, 2)]))
    register_tensor!(reg, TensorProperties(
        name=:A, manifold=:M, rank=(0, 2),
        symmetries=SymmetrySpec[Symmetric(1, 2)]))
    register_tensor!(reg, TensorProperties(
        name=:B, manifold=:M, rank=(0, 2),
        symmetries=SymmetrySpec[Symmetric(1, 2)]))

    with_registry(reg) do
        @testset "scalar equation: 2X = R" begin
            # Equation: 2R - X = 0  (i.e. R is rank-0)
            # Actually let's do a rank-0 scenario
            R_scalar = Tensor(:R, TIndex[])
            X_scalar = register_tensor!(reg, TensorProperties(
                name=:Xs, manifold=:M, rank=(0, 0)))
            Xs = Tensor(:Xs, TIndex[])

            # Equation: 2*Xs - R = 0 → Xs = R/2
            eq = 2 * Xs - R_scalar
            rules = solve_tensors(eq, :Xs; registry=reg)
            @test !isempty(rules)
            # Apply rule: Xs should become R/2
            result = apply_rules(Xs, rules)
            expected = tproduct(1 // 2, TensorExpr[R_scalar])
            result_s = simplify(result; registry=reg)
            expected_s = simplify(expected; registry=reg)
            @test result_s == expected_s
        end

        @testset "rank-2 symmetric: X_{ab} = Ric_{ab}" begin
            # Equation: X_{ab} - Ric_{ab} = 0
            X_ab = Tensor(:X, [down(:a), down(:b)])
            Ric_ab = Tensor(:Ric, [down(:a), down(:b)])
            eq = X_ab - Ric_ab
            rules = solve_tensors(eq, :X; registry=reg)
            @test !isempty(rules)
            # Apply: X_{ab} → Ric_{ab}
            result = apply_rules(X_ab, rules)
            @test result == Ric_ab
        end

        @testset "Einstein equation: G_{ab} - c*T_{ab} = 0" begin
            # G_{ab} - c * T_{ab} = 0 → T_{ab} = (1/c) G_{ab}
            G_ab = Tensor(:G, [down(:a), down(:b)])
            T_ab = Tensor(:T, [down(:a), down(:b)])
            c = TScalar(:c)
            eq = G_ab - c * T_ab
            rules = solve_tensors(eq, :T; registry=reg)
            @test !isempty(rules)
            result = apply_rules(T_ab, rules)
            # Result should contain G_{ab} and 1/c
            result_s = simplify(result; registry=reg)
            @test result_s isa TensorExpr
            # Verify it's nonzero (solved)
            @test result_s != TScalar(0 // 1)
        end

        @testset "unknown absent from equation" begin
            Ric_ab = Tensor(:Ric, [down(:a), down(:b)])
            eq = Ric_ab
            rules = @test_logs (:warn, r"does not appear") solve_tensors(eq, :X; registry=reg)
            @test isempty(rules)
        end

        @testset "derivative on unknown → error" begin
            X_ab = Tensor(:X, [down(:a), down(:b)])
            deriv_X = TDeriv(down(:c), X_ab)
            eq = deriv_X
            @test_throws ErrorException solve_tensors(eq, :X; registry=reg)
        end

        @testset "nonlinear → error" begin
            X_ab = Tensor(:X, [down(:a), down(:b)])
            X_cd = Tensor(:X, [down(:c), down(:d)])
            eq = X_ab * X_cd
            @test_throws ErrorException solve_tensors(eq, :X; registry=reg)
        end

        @testset "system of equations" begin
            A_ab = Tensor(:A, [down(:a), down(:b)])
            B_ab = Tensor(:B, [down(:a), down(:b)])
            Ric_ab = Tensor(:Ric, [down(:a), down(:b)])

            # A + B = Ric, A - B = 0
            eq1 = A_ab + B_ab - Ric_ab
            eq2 = A_ab - B_ab
            rules = solve_tensors([eq1, eq2], [:A, :B]; registry=reg)
            @test !isempty(rules)
        end

        @testset "roundtrip: solve then apply → zero" begin
            # X_{ab} - Ric_{ab} = 0
            X_ab = Tensor(:X, [down(:a), down(:b)])
            Ric_ab = Tensor(:Ric, [down(:a), down(:b)])
            eq = X_ab - Ric_ab
            rules = solve_tensors(eq, :X; registry=reg)
            # Substitute back into the equation
            substituted = apply_rules(eq, rules)
            result = simplify(substituted; registry=reg)
            @test result == TScalar(0 // 1)
        end

        @testset "make_rules=false returns pairs" begin
            X_ab = Tensor(:X, [down(:a), down(:b)])
            Ric_ab = Tensor(:Ric, [down(:a), down(:b)])
            eq = X_ab - Ric_ab
            pairs = solve_tensors(eq, :X; registry=reg, make_rules=false)
            @test pairs isa Vector{Pair{TensorExpr, TensorExpr}}
            @test length(pairs) == 1
            lhs, rhs = pairs[1]
            @test lhs == X_ab
            @test rhs == Ric_ab
        end

        @testset "rational coefficient" begin
            # (1//3) X_{ab} - Ric_{ab} = 0 → X_{ab} = 3 Ric_{ab}
            X_ab = Tensor(:X, [down(:a), down(:b)])
            Ric_ab = Tensor(:Ric, [down(:a), down(:b)])
            eq = (1 // 3) * X_ab - Ric_ab
            rules = solve_tensors(eq, :X; registry=reg)
            result = apply_rules(X_ab, rules)
            result_s = simplify(result; registry=reg)
            expected = simplify(3 * Ric_ab; registry=reg)
            @test result_s == expected
        end
    end
end
