@testset "Gauge-covariant exterior derivative" begin

    # Shared setup: registry with manifold + vbundle for algebra indices
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
                                               [:a,:b,:c,:d,:e,:f,:m,:n,:p,:q]))
    define_vbundle!(reg, :su2; manifold=:M4, dim=3,
                    indices=[:I,:J,:K,:L,:M,:N])

    @testset "Construction: gauge_covd returns AlgValuedForm of degree p+1" begin
        # Ref: Nakahara (2003) Eq 11.5 -- D_A omega is a (p+1)-form
        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
            omega = AlgValuedForm(1, :su2, Tensor(:omega, [up(:J, :su2), down(:b)]))

            result = gauge_covd(A, omega, :f; registry=reg)
            @test result isa AlgValuedForm
            @test result.degree == 2
            @test result.algebra == :su2
        end
    end

    @testset "A must be degree 1" begin
        # A 2-form is not a valid connection
        B = AlgValuedForm(2, :su2, Tensor(:B, [up(:I, :su2), down(:a), down(:b)]))
        omega = AlgValuedForm(1, :su2, Tensor(:omega, [up(:J, :su2), down(:c)]))
        @test_throws ArgumentError gauge_covd(B, omega, :f)
    end

    @testset "D_A on 0-form: just exterior derivative" begin
        # Ref: Nakahara (2003) Eq 11.3 -- D_A phi = d phi for algebra-valued 0-form
        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
            phi = AlgValuedForm(0, :su2, Tensor(:phi, [up(:J, :su2)]))

            result = gauge_covd(A, phi, :f; registry=reg)
            @test result isa AlgValuedForm
            @test result.degree == 1
            @test result.algebra == :su2
            # Should be pure derivative (no bracket term for 0-forms)
            @test result.expr isa TDeriv
        end
    end

    @testset "D_A on 1-form: dw + [A^w]" begin
        # Ref: Nakahara (2003) Eq 11.5 -- D_A omega = d omega + [A ^ omega]
        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
            omega = AlgValuedForm(1, :su2, Tensor(:omega, [up(:J, :su2), down(:b)]))

            result = gauge_covd(A, omega, :f; registry=reg)
            @test result.degree == 2
            # Should be a sum of dw and bracket term
            @test result.expr isa TSum
            @test length(result.expr.terms) == 2
        end
    end

    @testset "Curvature 2-form: F = dA + 1/2[A^A]" begin
        # Ref: Nakahara (2003) Eq 11.7 -- F is a 2-form
        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
            F = curvature_2form(A, :f; registry=reg)

            @test F isa AlgValuedForm
            @test F.degree == 2
            @test F.algebra == :su2
        end
    end

    @testset "D_A F structure (Bianchi): degree-3 form" begin
        # Ref: Nakahara (2003) Eq 11.12 -- D_A F is a 3-form
        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))

            DAF = bianchi_identity(A, :f; registry=reg)
            @test DAF isa AlgValuedForm
            @test DAF.degree == 3
            @test DAF.algebra == :su2
        end
    end

    @testset "bianchi_identity requires 1-form" begin
        B = AlgValuedForm(2, :su2, Tensor(:B, [up(:I, :su2), down(:a), down(:b)]))
        @test_throws ArgumentError bianchi_identity(B, :f)
    end

    @testset "Algebra consistency: A and omega must be in same algebra" begin
        # Ref: Nakahara (2003) Ch 11 -- forms must be valued in the same algebra
        A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
        omega_su3 = AlgValuedForm(1, :su3, Tensor(:omega, [up(:J, :su3), down(:b)]))
        @test_throws ArgumentError gauge_covd(A, omega_su3, :f)
    end

    @testset "D_A on 2-form: degree increases to 3" begin
        # Ref: Eguchi, Gilkey & Hanson (1980) Eq 2.22 -- D_A on p-form gives (p+1)-form
        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
            B2 = AlgValuedForm(2, :su2, Tensor(:B, [up(:J, :su2), down(:b), down(:c)]))

            result = gauge_covd(A, B2, :f; registry=reg)
            @test result.degree == 3
            @test result.expr isa TSum
        end
    end

    @testset "Algebra index preserved in result" begin
        # The result should carry a free algebra index in the :su2 vbundle
        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
            omega = AlgValuedForm(1, :su2, Tensor(:omega, [up(:J, :su2), down(:b)]))

            result = gauge_covd(A, omega, :f; registry=reg)
            fi = free_indices(result)
            alg_free = filter(idx -> idx.vbundle == :su2, fi)
            @test length(alg_free) >= 1
            @test any(idx -> idx.position == Up, alg_free)
        end
    end

    @testset "Bianchi identity has algebra index" begin
        # Ref: Nakahara (2003) Eq 11.12 -- D_A F carries algebra index
        with_registry(reg) do
            A = connection_1form(:A, :su2, up(:I, :su2), down(:a))
            DAF = bianchi_identity(A, :f; registry=reg)

            fi = free_indices(DAF)
            alg_free = filter(idx -> idx.vbundle == :su2, fi)
            @test length(alg_free) >= 1
        end
    end

end
