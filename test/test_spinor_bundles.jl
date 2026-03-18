@testset "Spinor VBundle registration (TGR-oun)" begin

    # ── Test 1: define_spinor_bundles! registers both bundles ──
    @testset "registration" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d,:e,:f]))

        define_spinor_bundles!(reg; manifold=:M4)

        @test has_vbundle(reg, :SL2C)
        @test has_vbundle(reg, :SL2C_dot)

        sl2c = get_vbundle(reg, :SL2C)
        sl2c_dot = get_vbundle(reg, :SL2C_dot)

        @test sl2c.dim == 2
        @test sl2c_dot.dim == 2
        @test sl2c.manifold == :M4
        @test sl2c_dot.manifold == :M4
        @test sl2c.indices == [:A, :B, :C, :D, :E, :F]
        @test sl2c_dot.indices == [:Ap, :Bp, :Cp, :Dp, :Ep, :Fp]
    end

    # ── Test 2: convenience constructors ──
    @testset "convenience constructors" begin
        @test spin_up(:A) == TIndex(:A, Up, :SL2C)
        @test spin_down(:B) == TIndex(:B, Down, :SL2C)
        @test spin_dot_up(:Ap) == TIndex(:Ap, Up, :SL2C_dot)
        @test spin_dot_down(:Bp) == TIndex(:Bp, Down, :SL2C_dot)
    end

    # ── Test 3: conjugation metadata round-trip ──
    @testset "conjugation metadata" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d]))
        define_spinor_bundles!(reg; manifold=:M4)

        @test conjugate_vbundle(reg, :SL2C) == :SL2C_dot
        @test conjugate_vbundle(reg, :SL2C_dot) == :SL2C
        # Tangent bundle has no conjugate
        @test conjugate_vbundle(reg, :Tangent) === nothing
    end

    # ── Test 4: different vbundles are distinct ──
    @testset "vbundle distinction" begin
        idx_a = TIndex(:A, Up, :SL2C)
        idx_a_dot = TIndex(:A, Up, :SL2C_dot)
        idx_a_tang = TIndex(:A, Up, :Tangent)

        # Same name and position but different vbundles => NOT equal
        @test idx_a != idx_a_dot
        @test idx_a != idx_a_tang
        @test idx_a_dot != idx_a_tang
    end

    # ── Test 5: predicates ──
    @testset "predicates" begin
        @test is_spinor_index(spin_up(:A))
        @test is_spinor_index(spin_dot_down(:Bp))
        @test !is_spinor_index(up(:a))
        @test !is_spinor_index(down(:b))

        @test !is_dotted(spin_up(:A))
        @test is_dotted(spin_dot_up(:Ap))
    end

    # ── Test 6: conjugate_index ──
    @testset "conjugate_index" begin
        idx = spin_up(:A)
        conj = conjugate_index(idx)
        @test conj == TIndex(:A, Up, :SL2C_dot)
        @test conj.position == Up  # position preserved

        idx2 = spin_dot_down(:Bp)
        conj2 = conjugate_index(idx2)
        @test conj2 == TIndex(:Bp, Down, :SL2C)

        # Double conjugation is identity
        @test conjugate_index(conjugate_index(spin_down(:C))) == spin_down(:C)

        # Non-spinor index errors
        @test_throws ErrorException conjugate_index(up(:a))
    end

    # ── Test 7: registry-aware conjugation ──
    @testset "registry-aware conjugate_index" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d]))
        define_spinor_bundles!(reg; manifold=:M4)

        with_registry(reg) do
            idx = spin_up(:A)
            conj = conjugate_index(idx, reg)
            @test conj.vbundle == :SL2C_dot
            @test conj.name == :A
            @test conj.position == Up
        end
    end

    # ── Test 8: backward compat — existing VBundleProperties 4-arg constructor ──
    @testset "backward compat" begin
        vb = VBundleProperties(:TestBundle, :M, 3, [:x, :y, :z])
        @test vb.name == :TestBundle
        @test vb.dim == 3
        @test vb.indices == [:x, :y, :z]
        @test isempty(vb.options)
    end

    # ── Test 9: existing define_vbundle! still works without conjugate_bundle ──
    @testset "define_vbundle! backward compat" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d]))
        define_vbundle!(reg, :SU2; manifold=:M4, dim=3, indices=[:i,:j,:k])

        su2 = get_vbundle(reg, :SU2)
        @test su2.dim == 3
        @test conjugate_vbundle(reg, :SU2) === nothing
    end

    # ── Test 10: tensors with spinor indices ──
    @testset "tensors with spinor indices" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :partial, [:a,:b,:c,:d]))
        define_spinor_bundles!(reg; manifold=:M4)

        with_registry(reg) do
            # A spinor with one undotted and one dotted index
            psi = Tensor(:psi, [spin_down(:A), spin_dot_down(:Bp)])
            @test length(psi.indices) == 2
            @test psi.indices[1].vbundle == :SL2C
            @test psi.indices[2].vbundle == :SL2C_dot

            # Free indices should work
            fi = free_indices(psi)
            @test length(fi) == 2
        end
    end

    # ── Test 11: error on unregistered manifold ──
    @testset "error on missing manifold" begin
        reg = TensorRegistry()
        @test_throws ErrorException define_spinor_bundles!(reg; manifold=:M4)
    end
end
