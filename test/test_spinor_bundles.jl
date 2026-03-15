@testset "Spinor Bundles (SL2C / SL2C_dot)" begin

    @testset "define_spinor_bundles! registers both bundles" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_spinor_bundles!(reg; manifold=:M4)

            @test has_vbundle(reg, :SL2C)
            @test has_vbundle(reg, :SL2C_dot)

            sl2c = get_vbundle(reg, :SL2C)
            @test sl2c.manifold == :M4
            @test sl2c.dim == 2
            @test sl2c.indices == [:A, :B, :C, :D, :E, :F]

            sl2c_dot = get_vbundle(reg, :SL2C_dot)
            @test sl2c_dot.manifold == :M4
            @test sl2c_dot.dim == 2
            @test sl2c_dot.indices == [:Ap, :Bp, :Cp, :Dp, :Ep, :Fp]
        end
    end

    @testset "Conjugation metadata round-trip" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_spinor_bundles!(reg; manifold=:M4)

            sl2c = get_vbundle(reg, :SL2C)
            sl2c_dot = get_vbundle(reg, :SL2C_dot)

            @test sl2c.conjugate_bundle === :SL2C_dot
            @test sl2c_dot.conjugate_bundle === :SL2C
        end
    end

    @testset "Convenience constructors" begin
        @test spin_up(:A) == TIndex(:A, Up, :SL2C)
        @test spin_down(:B) == TIndex(:B, Down, :SL2C)
        @test spin_dot_up(:Ap) == TIndex(:Ap, Up, :SL2C_dot)
        @test spin_dot_down(:Bp) == TIndex(:Bp, Down, :SL2C_dot)

        # Verify vbundle field
        @test spin_up(:A).vbundle == :SL2C
        @test spin_dot_down(:Ap).vbundle == :SL2C_dot
    end

    @testset "is_dotted" begin
        @test is_dotted(spin_dot_up(:Ap)) == true
        @test is_dotted(spin_dot_down(:Bp)) == true
        @test is_dotted(spin_up(:A)) == false
        @test is_dotted(spin_down(:B)) == false
        @test is_dotted(up(:a)) == false  # tangent index
    end

    @testset "is_spinor_index" begin
        @test is_spinor_index(spin_up(:A)) == true
        @test is_spinor_index(spin_down(:B)) == true
        @test is_spinor_index(spin_dot_up(:Ap)) == true
        @test is_spinor_index(spin_dot_down(:Bp)) == true
        @test is_spinor_index(up(:a)) == false
        @test is_spinor_index(down(:b)) == false
    end

    @testset "conjugate_index" begin
        # Undotted -> dotted
        idx = spin_up(:A)
        conj = conjugate_index(idx)
        @test conj == TIndex(:A, Up, :SL2C_dot)
        @test conj.name == :A
        @test conj.position == Up
        @test conj.vbundle == :SL2C_dot

        # Dotted -> undotted
        idx2 = spin_dot_down(:Bp)
        conj2 = conjugate_index(idx2)
        @test conj2 == TIndex(:Bp, Down, :SL2C)
        @test conj2.vbundle == :SL2C

        # Double conjugation is identity
        @test conjugate_index(conjugate_index(spin_up(:C))) == spin_up(:C)
        @test conjugate_index(conjugate_index(spin_dot_down(:Dp))) == spin_dot_down(:Dp)

        # Non-spinor index errors
        @test_throws ErrorException conjugate_index(up(:a))
    end

    @testset "Different vbundles are distinct" begin
        # SL2C vs SL2C_dot with same name
        idx_undotted = TIndex(:A, Up, :SL2C)
        idx_dotted = TIndex(:A, Up, :SL2C_dot)
        @test idx_undotted != idx_dotted
        @test hash(idx_undotted) != hash(idx_dotted)
    end

    @testset "Backward compat: VBundleProperties 4-arg constructor" begin
        vb = VBundleProperties(:Foo, :M4, 3, [:x, :y, :z])
        @test vb.name == :Foo
        @test vb.manifold == :M4
        @test vb.dim == 3
        @test vb.indices == [:x, :y, :z]
        @test vb.conjugate_bundle === nothing
    end

    @testset "Backward compat: define_vbundle! without conjugate_bundle" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_vbundle!(reg, :SU2; manifold=:M4, dim=3,
                            indices=[:I, :J, :K])
            su2 = get_vbundle(reg, :SU2)
            @test su2.conjugate_bundle === nothing
        end
    end

    @testset "Mixed spinor-spacetime tensor" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_spinor_bundles!(reg; manifold=:M4)

            # Infeld-van der Waerden symbol sigma^a_{A Ap}
            sigma = Tensor(:sigma, [up(:a), spin_down(:A), spin_dot_down(:Ap)])
            @test sigma.indices[1].vbundle == :Tangent
            @test sigma.indices[2].vbundle == :SL2C
            @test sigma.indices[3].vbundle == :SL2C_dot
        end
    end

    @testset "Cross-bundle spinor contraction refused" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_spinor_bundles!(reg; manifold=:M4)

            # Same name 'A' in SL2C vs SL2C_dot should NOT form a dummy pair
            T1 = Tensor(:psi, [spin_up(:A)])
            T2 = Tensor(:chi, [spin_dot_down(:A)])
            prod = T1 * T2

            dp = dummy_pairs(prod)
            a_pairs = filter(p -> p[1].name == :A || p[2].name == :A, dp)
            @test isempty(a_pairs)
        end
    end

    @testset "Same-bundle spinor contraction works" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_spinor_bundles!(reg; manifold=:M4)

            # psi^A chi_A should form a dummy pair (both SL2C)
            T1 = Tensor(:psi, [spin_up(:A)])
            T2 = Tensor(:chi, [spin_down(:A)])
            prod = T1 * T2

            dp = dummy_pairs(prod)
            a_pairs = filter(p -> p[1].name == :A || p[2].name == :A, dp)
            @test length(a_pairs) == 1
        end
    end

    @testset "Manifold must exist" begin
        reg = TensorRegistry()
        @test_throws ErrorException define_spinor_bundles!(reg; manifold=:M4)
    end

    @testset "Cannot register twice" begin
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_spinor_bundles!(reg; manifold=:M4)
            @test_throws ErrorException define_spinor_bundles!(reg; manifold=:M4)
        end
    end

end
