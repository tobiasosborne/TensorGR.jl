@testset "Bianchi cosmology" begin

    @testset "BianchiIBackground construction" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d]))

        with_registry(reg) do
            b = define_bianchi_I!(reg, :BI)
            @test b isa BianchiIBackground
            @test b.name == :BI
            @test b.manifold == :M4
            @test b.scale_factors == (:a1, :a2, :a3)
        end
    end

    @testset "BianchiI registers tensors" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d]))

        with_registry(reg) do
            b = define_bianchi_I!(reg, :BI)

            # Scale factors registered
            @test has_tensor(reg, :a1)
            @test has_tensor(reg, :a2)
            @test has_tensor(reg, :a3)

            # Hubble rates registered
            @test has_tensor(reg, :H_a1)
            @test has_tensor(reg, :H_a2)
            @test has_tensor(reg, :H_a3)
        end
    end

    @testset "BianchiI creates foliation" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d]))

        with_registry(reg) do
            b = define_bianchi_I!(reg, :BI)
            @test has_foliation(reg, b.foliation)
            fol = get_foliation(reg, b.foliation)
            @test fol.temporal_component == 0
            @test fol.spatial_components == [1,2,3]
        end
    end

    @testset "BianchiI custom scale factors" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial,
            [:a,:b,:c,:d]))

        with_registry(reg) do
            b = define_bianchi_I!(reg, :BI2; scale_factors=(:alpha, :beta, :gamma_sf))
            @test b.scale_factors == (:alpha, :beta, :gamma_sf)
            @test has_tensor(reg, :alpha)
            @test has_tensor(reg, :beta)
            @test has_tensor(reg, :gamma_sf)
        end
    end

    @testset "BianchiI display" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b]))
        b = define_bianchi_I!(reg, :BI)
        s = sprint(show, b)
        @test occursin("BianchiI", s)
        @test occursin("a1", s)
    end

    @testset "is_isotropic" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b]))
        b = define_bianchi_I!(reg, :BI)
        @test !is_isotropic(b)
    end

    # ── Bianchi structure constants ──

    @testset "Named constructors" begin
        @test bianchi_I().type == :I
        @test bianchi_I().n == (0,0,0)
        @test bianchi_II().type == :II
        @test bianchi_II().n == (1,0,0)
        @test bianchi_VI0().type == :VI0
        @test bianchi_VI0().n == (1,-1,0)
        @test bianchi_VII0().type == :VII0
        @test bianchi_VII0().n == (1,1,0)
        @test bianchi_VIII().type == :VIII
        @test bianchi_VIII().n == (1,1,-1)
        @test bianchi_IX().type == :IX
        @test bianchi_IX().n == (1,1,1)
    end

    @testset "bianchi_type classification" begin
        @test bianchi_type((0,0,0)) == :I
        @test bianchi_type((1,0,0)) == :II
        @test bianchi_type((0,1,0)) == :II
        @test bianchi_type((1,-1,0)) == :VI0
        @test bianchi_type((1,1,0)) == :VII0
        @test bianchi_type((1,1,-1)) == :VIII
        @test bianchi_type((1,1,1)) == :IX
    end

    @testset "structure_constant: Bianchi I (trivial)" begin
        bI = bianchi_I()
        for i in 1:3, j in 1:3, k in 1:3
            @test structure_constant(bI, i, j, k) == 0
        end
    end

    @testset "structure_constant: Bianchi IX (SU(2))" begin
        bIX = bianchi_IX()
        # C^1_{23} = ε_{231} n^{11} = +1 * 1 = 1
        @test structure_constant(bIX, 1, 2, 3) == 1
        # Antisymmetry: C^1_{32} = -C^1_{23} = -1
        @test structure_constant(bIX, 1, 3, 2) == -1
        # C^2_{31} = ε_{312} n^{22} = +1 * 1 = 1
        @test structure_constant(bIX, 2, 3, 1) == 1
        # C^3_{12} = ε_{123} n^{33} = +1 * 1 = 1
        @test structure_constant(bIX, 3, 1, 2) == 1
    end

    @testset "verify_jacobi: all class A types" begin
        for bsc in [bianchi_I(), bianchi_II(), bianchi_VI0(),
                    bianchi_VII0(), bianchi_VIII(), bianchi_IX()]
            @test verify_jacobi(bsc)
        end
    end

    @testset "is_class_A: all types" begin
        for bsc in [bianchi_I(), bianchi_II(), bianchi_VI0(),
                    bianchi_VII0(), bianchi_VIII(), bianchi_IX()]
            @test is_class_A(bsc)
        end
    end

    @testset "display" begin
        s = sprint(show, bianchi_IX())
        @test occursin("IX", s)
    end

end
