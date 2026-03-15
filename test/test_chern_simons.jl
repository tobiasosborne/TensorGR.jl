@testset "Chern-Simons form (Nakahara Sec 11.5, Eq 11.106b)" begin

    # Shared setup: registry with manifold + vbundle for algebra indices
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
                                               [:a,:b,:c,:d,:e,:f,:m,:n,:p,:q,:r,:s]))
    define_vbundle!(reg, :su2; manifold=:M4, dim=3,
                    indices=[:I,:J,:K,:L,:M,:N])

    # Register structure constants
    register_tensor!(reg, TensorProperties(
        name=:f, manifold=:M4, rank=(1,2),
        symmetries=[AntiSymmetric(2,3)]))
    register_tensor!(reg, TensorProperties(
        name=:F, manifold=:M4, rank=(1,2),
        symmetries=[AntiSymmetric(2,3)]))

    @testset "chern_simons_form returns degree-3 AlgValuedForm" begin
        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
            CS = chern_simons_form(A, :f; registry=reg)

            @test CS isa AlgValuedForm
            @test CS.degree == 3
            @test CS.algebra == :su2
        end
    end

    @testset "chern_simons_form requires 1-form" begin
        B = AlgValuedForm(2, :su2, Tensor(:B, [up(:I, :su2), down(:a), down(:b)]))
        @test_throws ArgumentError chern_simons_form(B, :f)

        phi = AlgValuedForm(0, :su2, Tensor(:phi, [up(:I, :su2)]))
        @test_throws ArgumentError chern_simons_form(phi, :f)
    end

    @testset "chern_simons_form expression is a sum (A^dA + (2/3)A^A^A)" begin
        # Nakahara Eq 11.106b: CS = Tr(A dA + (2/3) A^3)
        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
            CS = chern_simons_form(A, :f; registry=reg)

            # The underlying expression should be a sum of two terms
            @test CS.expr isa TSum
            @test length(CS.expr.terms) == 2
        end
    end

    @testset "chern_simons_invariant is alias for chern_simons_form" begin
        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
            CS1 = chern_simons_form(A, :f; registry=reg)
            CS2 = chern_simons_invariant(A, :f; registry=reg)

            @test CS1.degree == CS2.degree
            @test CS1.algebra == CS2.algebra
            # Both should produce structurally equivalent expressions
            @test CS1.degree == 3
            @test CS2.degree == 3
        end
    end

    @testset "structural verification: d(CS) = Tr(F^F) degree check" begin
        # Nakahara Sec 11.5: d(CS) is a 4-form = Tr(F^F) which is also a 4-form
        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))

            # CS is a 3-form
            CS = chern_simons_form(A, :f; registry=reg)
            @test CS.degree == 3

            # d(CS) would be a 4-form
            d_idx = fresh_index(Set([:a,:b,:c,:d,:e,:f,:m,:n,:p,:q,:r,:s]))
            dCS = alg_exterior_d(CS, down(d_idx))
            @test dCS.degree == 4

            # Tr(F^F) is also a 4-form
            F = field_strength(A, :f; registry=reg)
            FF = instanton_density(F; registry=reg)
            @test FF.degree == 4

            # Both are degree-4 forms in the same algebra
            @test dCS.degree == FF.degree
            @test dCS.algebra == FF.algebra
        end
    end

    @testset "algebra index structure" begin
        # CS should carry contracted (traced) algebra indices
        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
            CS = chern_simons_form(A, :f; registry=reg)

            # The result should have indices (both free and dummy)
            all_idx = indices(CS)
            @test length(all_idx) > 0

            # Should have algebra indices (in :su2 vbundle)
            alg_indices = filter(idx -> idx.vbundle == :su2, all_idx)
            @test length(alg_indices) > 0
        end
    end

    @testset "CS contains both derivative and structure constant terms" begin
        # The expression should reference both A (connection) and f (structure constants)
        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
            CS = chern_simons_form(A, :f; registry=reg)

            # Collect all tensor names in the expression
            names = Set{Symbol}()
            walk(CS.expr) do node
                if node isa Tensor
                    push!(names, node.name)
                end
                node
            end

            # Should contain the connection tensor :A
            @test :A in names
            # Should contain the structure constants :f (from the A^A^A term)
            @test :f in names
        end
    end

    @testset "CS expression has derivatives (from A^dA term)" begin
        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
            CS = chern_simons_form(A, :f; registry=reg)

            # Check that the expression contains at least one TDeriv
            has_deriv = Ref(false)
            walk(CS.expr) do node
                if node isa TDeriv
                    has_deriv[] = true
                end
                node
            end
            @test has_deriv[]
        end
    end

    @testset "3D Chern-Simons: same algebra, degree 3 (Nakahara Ex 11.10)" begin
        # In 3D, CS is a 3-form = volume form, suitable for action
        reg3 = TensorRegistry()
        register_manifold!(reg3, ManifoldProperties(:M3, 3, :g3, :partial,
                                                    [:a,:b,:c,:d,:e,:f,:m,:n]))
        define_vbundle!(reg3, :su2; manifold=:M3, dim=3,
                        indices=[:I,:J,:K,:L,:M,:N])
        register_tensor!(reg3, TensorProperties(
            name=:f, manifold=:M3, rank=(1,2),
            symmetries=[AntiSymmetric(2,3)]))

        with_registry(reg3) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
            CS = chern_simons_form(A, :f; registry=reg3)

            @test CS.degree == 3
            @test CS.algebra == :su2
        end
    end

    @testset "Abelian case: A^A^A term structure" begin
        # For U(1) (Abelian), the structure constants vanish,
        # so CS reduces to A^dA. The cubic term should still be structurally present.
        reg_u1 = TensorRegistry()
        register_manifold!(reg_u1, ManifoldProperties(:M3, 3, :g3, :partial,
                                                      [:a,:b,:c,:d,:e,:f]))
        define_vbundle!(reg_u1, :u1; manifold=:M3, dim=1,
                        indices=[:I,:J,:K])
        register_tensor!(reg_u1, TensorProperties(
            name=:f_u1, manifold=:M3, rank=(1,2),
            symmetries=[AntiSymmetric(2,3)]))

        with_registry(reg_u1) do
            A = connection_1form(:A, :u1, up(:I, :u1), down(:a))
            CS = chern_simons_form(A, :f_u1; registry=reg_u1)

            @test CS isa AlgValuedForm
            @test CS.degree == 3
            # Even for Abelian, the formal expression has both terms
            @test CS.expr isa TSum
        end
    end

    @testset "Pipeline: connection -> field_strength -> instanton_density matches d(CS) degree" begin
        # Full pipeline consistency check
        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))

            # Chern-Simons 3-form
            CS = chern_simons_form(A, :f; registry=reg)

            # Field strength and instanton density
            F = field_strength(A, :f; registry=reg)
            FF = instanton_density(F; registry=reg)

            # d(CS) should match Tr(F^F) in degree
            @test CS.degree + 1 == FF.degree  # 3 + 1 == 4
        end
    end

end
