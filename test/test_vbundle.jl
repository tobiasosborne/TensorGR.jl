@testset "VBundle Support" begin

    @testset "Default tangent bundle" begin
        idx = TIndex(:a, Up)
        @test idx.vbundle == :Tangent

        idx2 = up(:a)
        @test idx2.vbundle == :Tangent

        idx3 = down(:b)
        @test idx3.vbundle == :Tangent
    end

    @testset "Explicit vbundle construction" begin
        idx = TIndex(:A, Up, :SU2)
        @test idx.name == :A
        @test idx.position == Up
        @test idx.vbundle == :SU2

        idx2 = up(:B, :SU2)
        @test idx2.vbundle == :SU2

        idx3 = down(:C, :SU2)
        @test idx3.vbundle == :SU2
    end

    @testset "Equality respects vbundle" begin
        # Same name/position, different vbundle => not equal
        t_idx = TIndex(:a, Up, :Tangent)
        g_idx = TIndex(:a, Up, :SU2)
        @test t_idx != g_idx
        @test hash(t_idx) != hash(g_idx)

        # Same everything => equal
        @test TIndex(:a, Up, :SU2) == TIndex(:a, Up, :SU2)
        @test hash(TIndex(:a, Up, :SU2)) == hash(TIndex(:a, Up, :SU2))
    end

    @testset "VBundle registry" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g

            # Tangent bundle auto-registered
            @test has_vbundle(reg, :Tangent)
            tb = get_vbundle(reg, :Tangent)
            @test tb.manifold == :M4
            @test tb.dim == 4

            # Define a gauge bundle
            define_vbundle!(reg, :SU2; manifold=:M4, dim=3,
                           indices=[:A, :B, :C, :D, :E])
            @test has_vbundle(reg, :SU2)
            su2 = get_vbundle(reg, :SU2)
            @test su2.dim == 3
            @test su2.manifold == :M4
            @test su2.indices == [:A, :B, :C, :D, :E]
        end
    end

    @testset "Mixed-bundle tensor" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_vbundle!(reg, :SU2; manifold=:M4, dim=3,
                           indices=[:A, :B, :C])

            # Field strength F^A_{mu nu}: gauge index A, spacetime mu,nu
            F = Tensor(:F, [up(:A, :SU2), down(:mu), down(:nu)])
            @test F.indices[1].vbundle == :SU2
            @test F.indices[2].vbundle == :Tangent
            @test F.indices[3].vbundle == :Tangent
        end
    end

    @testset "Cross-bundle contraction refused" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_vbundle!(reg, :SU2; manifold=:M4, dim=3,
                           indices=[:A, :B, :C])

            # Same index name 'a' but different vbundles should NOT be
            # recognized as a dummy pair
            T1 = Tensor(:T, [up(:a, :Tangent), down(:b)])
            T2 = Tensor(:S, [down(:a, :SU2), up(:c)])
            prod = T1 * T2

            # free_indices should contain both 'a' indices (different vbundles)
            fi = free_indices(prod)
            a_indices = filter(i -> i.name == :a, fi)
            @test length(a_indices) == 2  # not contracted

            # dummy_pairs should not pair them
            dp = dummy_pairs(prod)
            a_pairs = filter(p -> p[1].name == :a || p[2].name == :a, dp)
            @test isempty(a_pairs)
        end
    end

    @testset "Same-bundle contraction works" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g

            # Standard contraction: same vbundle (Tangent)
            T = Tensor(:T, [up(:a), down(:b)])
            S = Tensor(:S, [down(:a), up(:c)])
            prod = T * S

            dp = dummy_pairs(prod)
            a_pairs = filter(p -> p[1].name == :a || p[2].name == :a, dp)
            @test length(a_pairs) == 1
        end
    end

    @testset "Dagger preserves vbundle" begin
        t = Tensor(:F, [up(:A, :SU2), down(:mu)])
        d = dagger(t)
        @test d.indices[1].vbundle == :SU2
        @test d.indices[2].vbundle == :Tangent
    end

    @testset "Rename dummy preserves vbundle" begin
        t = Tensor(:T, [up(:a, :SU2), down(:a, :SU2)])
        renamed = rename_dummy(t, :a, :b)
        @test renamed.indices[1].vbundle == :SU2
        @test renamed.indices[2].vbundle == :SU2
        @test renamed.indices[1].name == :b
    end

    @testset "Canonicalize preserves vbundle" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            @define_tensor T on=M4 rank=(0,2) symmetry=Symmetric(1,2)

            # T_{ba} should canonicalize to T_{ab}
            T1 = Tensor(:T, [down(:b), down(:a)])
            result = canonicalize(T1)
            @test result isa Tensor
            @test result.indices[1].vbundle == :Tangent
            @test result.indices[2].vbundle == :Tangent
        end
    end

    @testset "Metric contraction preserves vbundle" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g

            # g^{ab} g_{bc} = delta^a_c
            result = simplify(Tensor(:g, [up(:a), up(:b)]) *
                             Tensor(:g, [down(:b), down(:c)]))
            @test result isa Tensor
            @test all(idx -> idx.vbundle == :Tangent, result.indices)
        end
    end

    @testset "Delta self-trace with vbundle" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g

            # delta^a_a = 4 (same vbundle)
            result = simplify(Tensor(:delta, [up(:a), down(:a)]))
            # Note: delta^a_a where delta is not registered just stays as-is
            # Use the registered delta
            result = simplify(Tensor(:g, [up(:a), up(:b)]) *
                             Tensor(:g, [down(:a), down(:b)]))
            @test result == TScalar(4 // 1)
        end
    end
end
