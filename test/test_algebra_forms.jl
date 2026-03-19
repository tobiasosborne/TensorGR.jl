@testset "Algebra-valued differential forms" begin

    @testset "Construction" begin
        A = AlgValuedForm(1, :su2, Tensor(:A, [up(:I, :su2), down(:a)]))
        @test A.degree == 1
        @test A.algebra == :su2
        @test A.expr == Tensor(:A, [up(:I, :su2), down(:a)])

        # Zero-form (scalar-valued in algebra)
        phi = AlgValuedForm(0, :Adj, Tensor(:phi, [up(:I, :Adj)]))
        @test phi.degree == 0

        # Higher-degree form
        B = AlgValuedForm(2, :su3, Tensor(:B, [up(:I, :su3), down(:a), down(:b)]))
        @test B.degree == 2
    end

    @testset "Degree validation" begin
        @test_throws ArgumentError AlgValuedForm(-1, :su2, Tensor(:A, [up(:I, :su2), down(:a)]))
        @test_throws ArgumentError AlgValuedForm(-3, :Adj, TScalar(1))
    end

    @testset "Equality and hashing" begin
        A1 = AlgValuedForm(1, :su2, Tensor(:A, [up(:I, :su2), down(:a)]))
        A2 = AlgValuedForm(1, :su2, Tensor(:A, [up(:I, :su2), down(:a)]))
        A3 = AlgValuedForm(1, :su3, Tensor(:A, [up(:I, :su3), down(:a)]))

        @test A1 == A2
        @test hash(A1) == hash(A2)
        @test A1 != A3
    end

    @testset "AST integration: indices" begin
        A = AlgValuedForm(1, :su2, Tensor(:A, [up(:I, :su2), down(:a)]))

        fi = free_indices(A)
        @test length(fi) == 2
        @test up(:I, :su2) in fi
        @test down(:a) in fi

        all_idx = indices(A)
        @test length(all_idx) == 2
    end

    @testset "AST integration: children and walk" begin
        inner = Tensor(:A, [up(:I, :su2), down(:a)])
        A = AlgValuedForm(1, :su2, inner)

        @test children(A) == TensorExpr[inner]

        result = walk(identity, A)
        @test result == A
    end

    @testset "AST integration: derivative_order and is_constant" begin
        A = AlgValuedForm(1, :su2, Tensor(:A, [up(:I, :su2), down(:a)]))
        @test derivative_order(A) == 0
        @test !is_constant(A)

        s = AlgValuedForm(0, :su2, TScalar(42))
        @test is_constant(s)
    end

    @testset "AST integration: rename_dummy" begin
        A = AlgValuedForm(1, :su2, Tensor(:A, [up(:I, :su2), down(:a)]))
        renamed = rename_dummy(A, :a, :b)
        @test renamed isa AlgValuedForm
        @test renamed.degree == 1
        @test renamed.expr == Tensor(:A, [up(:I, :su2), down(:b)])
    end

    @testset "AST integration: rename_dummies" begin
        A = AlgValuedForm(1, :su2, Tensor(:A, [up(:I, :su2), down(:a)]))
        renamed = rename_dummies(A, Dict(:a => :c, :I => :J))
        @test renamed isa AlgValuedForm
        @test renamed.expr == Tensor(:A, [up(:J, :su2), down(:c)])
    end

    @testset "AST integration: _replace_index_name" begin
        A = AlgValuedForm(1, :su2, Tensor(:A, [up(:I, :su2), down(:a)]))
        replaced = TensorGR._replace_index_name(A, :a, :z)
        @test replaced isa AlgValuedForm
        @test replaced.expr == Tensor(:A, [up(:I, :su2), down(:z)])
    end

    @testset "AST integration: to_expr roundtrip" begin
        A = AlgValuedForm(1, :su2, Tensor(:A, [up(:I, :su2), down(:a)]))
        e = to_expr(A)
        @test e isa Expr
        @test e.head == :call
        @test e.args[1] == :AlgValuedForm
    end

    @testset "AST integration: is_sorted_covds" begin
        A = AlgValuedForm(1, :su2, Tensor(:A, [up(:I, :su2), down(:a)]))
        @test is_sorted_covds(A)
    end

    @testset "Display" begin
        A = AlgValuedForm(1, :su2, Tensor(:A, [up(:I, :su2), down(:a)]))

        s = sprint(show, A)
        @test occursin("AlgForm", s)
        @test occursin("su2", s)
        @test occursin("deg=1", s)

        latex = to_latex(A)
        @test occursin("su2", latex)

        uni = to_unicode(A)
        @test occursin("su2", uni)
        @test occursin("form", uni)
    end

    @testset "alg_exterior_d" begin
        A = AlgValuedForm(1, :su2, Tensor(:A, [up(:I, :su2), down(:a)]))

        dA = alg_exterior_d(A, down(:b))
        @test dA isa AlgValuedForm
        @test dA.degree == 2
        @test dA.algebra == :su2
        @test dA.expr isa TDeriv
        @test dA.expr.index == down(:b)
    end

    @testset "alg_exterior_d increases degree" begin
        phi = AlgValuedForm(0, :su2, Tensor(:phi, [up(:I, :su2)]))
        dphi = alg_exterior_d(phi, down(:a))
        @test dphi.degree == 1

        B = AlgValuedForm(2, :su2, Tensor(:B, [up(:I, :su2), down(:a), down(:b)]))
        dB = alg_exterior_d(B, down(:c))
        @test dB.degree == 3
    end

    @testset "connection_1form" begin
        A = connection_1form(:A, :su2, up(:I, :su2), down(:a))

        @test A isa AlgValuedForm
        @test A.degree == 1
        @test A.algebra == :su2
        @test A.expr isa Tensor
        @test A.expr.name == :A
        @test length(A.expr.indices) == 2
    end

    @testset "alg_wedge" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d]))

        with_registry(reg) do
            A = AlgValuedForm(1, :su2, Tensor(:A, [up(:I, :su2), down(:a)]))
            B = AlgValuedForm(1, :su2, Tensor(:B, [up(:J, :su2), down(:b)]))

            result = alg_wedge(A, B, :f; registry=reg)
            @test result isa AlgValuedForm
            @test result.degree == 2
            @test result.algebra == :su2
        end
    end

    @testset "alg_wedge algebra mismatch" begin
        A = AlgValuedForm(1, :su2, Tensor(:A, [up(:I, :su2), down(:a)]))
        B = AlgValuedForm(1, :su3, Tensor(:B, [up(:J, :su3), down(:b)]))
        @test_throws ArgumentError alg_wedge(A, B, :f)
    end

    @testset "curvature_2form" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d,:e,:f]))

        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))

            F = curvature_2form(A, :f; registry=reg)
            @test F isa AlgValuedForm
            @test F.degree == 2
            @test F.algebra == :su2
            # F = dA + (1/2)[A^A], should be a sum
            @test F.expr isa TSum
        end
    end

    @testset "curvature_2form requires 1-form" begin
        B = AlgValuedForm(2, :su2, Tensor(:B, [up(:I, :su2), down(:a), down(:b)]))
        @test_throws ArgumentError curvature_2form(B, :f)
    end

    @testset "field_strength alias" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d,:e,:f]))

        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
            F1 = curvature_2form(A, :f; registry=reg)
            F2 = field_strength(A, :f; registry=reg)
            @test F1.degree == F2.degree
            @test F1.algebra == F2.algebra
        end
    end

    @testset "free indices of curvature 2-form" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d,:e,:f]))

        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
            F = curvature_2form(A, :f; registry=reg)

            fi = free_indices(F)
            # Should have algebra index I^su2 and spacetime form indices
            alg_free = filter(idx -> idx.vbundle == :su2, fi)
            @test length(alg_free) >= 1
            @test any(idx -> idx.position == Up, alg_free)
        end
    end

    @testset "gauge_covd on 0-form" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d]))

        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
            phi = AlgValuedForm(0, :su2, Tensor(:phi, [up(:J, :su2)]))

            Dphi = gauge_covd(A, phi, :f; registry=reg)
            @test Dphi isa AlgValuedForm
            @test Dphi.degree == 1
            # For 0-forms, D_A phi = d(phi) -- no bracket term
            @test Dphi.expr isa TDeriv
        end
    end

    @testset "gauge_covd algebra mismatch" begin
        A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
        omega = AlgValuedForm(1, :su3, Tensor(:omega, [up(:J, :su3), down(:b)]))
        @test_throws ArgumentError gauge_covd(A, omega, :f)
    end

    @testset "gauge_covd requires 1-form connection" begin
        B = AlgValuedForm(2, :su2, Tensor(:B, [up(:I, :su2), down(:a), down(:b)]))
        omega = AlgValuedForm(1, :su2, Tensor(:omega, [up(:J, :su2), down(:c)]))
        @test_throws ArgumentError gauge_covd(B, omega, :f)
    end

    @testset "bianchi_identity structure" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d,:e,:f,:g,:h,:i,:j,:k,:l]))

        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
            DF = bianchi_identity(A, :f; registry=reg)
            @test DF isa AlgValuedForm
            @test DF.degree == 3
        end
    end

    @testset "dagger" begin
        A = AlgValuedForm(1, :su2, Tensor(:A, [up(:I, :su2), down(:a)]))
        Ad = dagger(A)
        @test Ad isa AlgValuedForm
        @test Ad.degree == 1
        @test Ad.algebra == :su2
    end

end
