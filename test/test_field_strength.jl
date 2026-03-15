@testset "Field strength, Yang-Mills EOM, instanton density" begin

    # Shared setup: registry with manifold + vbundle for algebra indices
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
                                               [:a,:b,:c,:d,:e,:f,:m,:n,:p,:q]))
    define_vbundle!(reg, :su2; manifold=:M4, dim=3,
                    indices=[:I,:J,:K,:L,:M,:N])

    # Register structure constants and epsilon tensor
    register_tensor!(reg, TensorProperties(
        name=:f, manifold=:M4, rank=(1,2),
        symmetries=[AntiSymmetric(2,3)]))
    register_tensor!(reg, TensorProperties(
        name=:eps, manifold=:M4, rank=(0,4),
        symmetries=[AntiSymmetric(1,2), AntiSymmetric(2,3), AntiSymmetric(3,4)]))
    register_tensor!(reg, TensorProperties(
        name=:F, manifold=:M4, rank=(1,2),
        symmetries=[AntiSymmetric(2,3)]))

    @testset "field_strength is alias for curvature_2form (Nakahara Eq 10.38a)" begin
        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))

            F_curv = curvature_2form(A, :f; registry=reg)
            F_field = field_strength(A, :f; registry=reg)

            # Both should produce identical results
            @test F_curv == F_field
        end
    end

    @testset "field_strength returns degree-2 form (Nakahara Eq 10.38a)" begin
        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
            F = field_strength(A, :f; registry=reg)

            @test F isa AlgValuedForm
            @test F.degree == 2
            @test F.algebra == :su2
        end
    end

    @testset "field_strength requires 1-form" begin
        B = AlgValuedForm(2, :su2, Tensor(:B, [up(:I, :su2), down(:a), down(:b)]))
        @test_throws ArgumentError field_strength(B, :f)
    end

    @testset "bianchi_identity returns degree-3 form (Nakahara Eq 10.45)" begin
        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
            DAF = bianchi_identity(A, :f; registry=reg)

            @test DAF isa AlgValuedForm
            @test DAF.degree == 3
            @test DAF.algebra == :su2

            # D_A F should carry a free algebra index
            fi = free_indices(DAF)
            alg_free = filter(idx -> idx.vbundle == :su2, fi)
            @test length(alg_free) >= 1
            @test any(idx -> idx.position == Up, alg_free)
        end
    end

    @testset "yang_mills_eom returns correct degree (Nakahara Eq 10.109)" begin
        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))

            # In 4D: F is a 2-form, *F is a (4-2)=2-form, D_A(*F) is a 3-form
            eom = yang_mills_eom(A, :f, :eps, :g, 4; registry=reg)

            @test eom isa AlgValuedForm
            @test eom.degree == 3  # (dim - 2) + 1 = 3 in 4D
            @test eom.algebra == :su2
        end
    end

    @testset "yang_mills_eom requires 1-form connection" begin
        B = AlgValuedForm(2, :su2, Tensor(:B, [up(:I, :su2), down(:a), down(:b)]))
        @test_throws ArgumentError yang_mills_eom(B, :f, :eps, :g, 4)
    end

    @testset "yang_mills_eom in 3D: D_A(*F) is degree 2" begin
        # In 3D: F is 2-form, *F is (3-2)=1-form, D_A(*F) is 2-form
        reg3 = TensorRegistry()
        register_manifold!(reg3, ManifoldProperties(:M3, 3, :g3, :partial,
                                                    [:a,:b,:c,:d,:e,:f,:m,:n]))
        define_vbundle!(reg3, :su2; manifold=:M3, dim=3,
                        indices=[:I,:J,:K,:L,:M,:N])
        register_tensor!(reg3, TensorProperties(
            name=:f, manifold=:M3, rank=(1,2),
            symmetries=[AntiSymmetric(2,3)]))
        register_tensor!(reg3, TensorProperties(
            name=:eps3, manifold=:M3, rank=(0,3),
            symmetries=[AntiSymmetric(1,2), AntiSymmetric(2,3)]))
        register_tensor!(reg3, TensorProperties(
            name=:F, manifold=:M3, rank=(1,2),
            symmetries=[AntiSymmetric(2,3)]))

        with_registry(reg3) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
            eom = yang_mills_eom(A, :f, :eps3, :g3, 3; registry=reg3)

            @test eom isa AlgValuedForm
            @test eom.degree == 2  # (3-2) + 1 = 2 in 3D
        end
    end

    @testset "yang_mills_eom has algebra index" begin
        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
            eom = yang_mills_eom(A, :f, :eps, :g, 4; registry=reg)

            fi = free_indices(eom)
            alg_free = filter(idx -> idx.vbundle == :su2, fi)
            @test length(alg_free) >= 1
        end
    end

    @testset "instanton_density: F^F is degree 4 (Nakahara Sec 10.5.5)" begin
        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
            F = field_strength(A, :f; registry=reg)

            FF = instanton_density(F; registry=reg)

            @test FF isa AlgValuedForm
            @test FF.degree == 4  # 2 + 2 = 4
            @test FF.algebra == :su2
        end
    end

    @testset "instanton_density requires 2-form" begin
        A = AlgValuedForm(1, :su2, Tensor(:A, [up(:I, :su2), down(:a)]))
        @test_throws ArgumentError instanton_density(A)

        B = AlgValuedForm(3, :su2, Tensor(:B, [up(:I, :su2), down(:a), down(:b), down(:c)]))
        @test_throws ArgumentError instanton_density(B)
    end

    @testset "instanton_density from explicit F tensor" begin
        with_registry(reg) do
            # Build F directly as AlgValuedForm
            F = AlgValuedForm(2, :su2, Tensor(:F, [up(:I, :su2), down(:a), down(:b)]))
            FF = instanton_density(F; registry=reg)

            @test FF.degree == 4
            @test FF.algebra == :su2
            # The expression should contain a product of two F tensors
            @test FF.expr isa TProduct
        end
    end

    @testset "Consistency: field_strength + instanton_density pipeline" begin
        with_registry(reg) do
            # Full pipeline: A -> F -> Tr(F^F)
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
            F = field_strength(A, :f; registry=reg)
            FF = instanton_density(F; registry=reg)

            @test FF.degree == 4
            # The underlying expression should be non-trivial (a sum or product)
            @test !(FF.expr isa TScalar)
        end
    end

    @testset "Consistency: field_strength + yang_mills_eom pipeline" begin
        with_registry(reg) do
            # Full pipeline: A -> D_A(*F)
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
            eom = yang_mills_eom(A, :f, :eps, :g, 4; registry=reg)

            @test eom.degree == 3
            # EOM expression should be a sum (d(*F) + [A^*F])
            @test eom.expr isa TSum
        end
    end

end
