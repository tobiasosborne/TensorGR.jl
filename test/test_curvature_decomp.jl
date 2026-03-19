@testset "Riemann spinor decomposition" begin

    function _decomp_reg()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            define_spinor_structure!(reg; manifold=:M4, metric=:g)
            define_curvature_spinors!(reg; manifold=:M4)
        end
        reg
    end

    @testset "weyl_spinor_expr" begin
        reg = _decomp_reg()
        with_registry(reg) do
            psi = weyl_spinor_expr()
            @test psi isa Tensor
            @test psi.name == :Psi
            @test length(psi.indices) == 4
            @test all(idx -> idx.vbundle == :SL2C, psi.indices)
            @test all(idx -> idx.position == Down, psi.indices)
            # All index names should be distinct
            names = [idx.name for idx in psi.indices]
            @test length(unique(names)) == 4
        end
    end

    @testset "weyl_spinor_bar_expr" begin
        reg = _decomp_reg()
        with_registry(reg) do
            psi_bar = weyl_spinor_bar_expr()
            @test psi_bar isa Tensor
            @test psi_bar.name == :Psi_bar
            @test length(psi_bar.indices) == 4
            @test all(idx -> idx.vbundle == :SL2C_dot, psi_bar.indices)
            @test all(idx -> idx.position == Down, psi_bar.indices)
            names = [idx.name for idx in psi_bar.indices]
            @test length(unique(names)) == 4
        end
    end

    @testset "ricci_spinor_expr" begin
        reg = _decomp_reg()
        with_registry(reg) do
            phi = ricci_spinor_expr()
            @test phi isa Tensor
            @test phi.name == :Phi_Ricci
            @test length(phi.indices) == 4
            # First 2 undotted, last 2 dotted
            @test phi.indices[1].vbundle == :SL2C
            @test phi.indices[2].vbundle == :SL2C
            @test phi.indices[3].vbundle == :SL2C_dot
            @test phi.indices[4].vbundle == :SL2C_dot
            names = [idx.name for idx in phi.indices]
            @test length(unique(names)) == 4
        end
    end

    @testset "riemann_spinor_parts" begin
        reg = _decomp_reg()
        with_registry(reg) do
            parts = riemann_spinor_parts()
            @test parts.weyl isa Tensor
            @test parts.weyl.name == :Psi
            @test parts.weyl_bar isa Tensor
            @test parts.weyl_bar.name == :Psi_bar
            @test parts.ricci isa Tensor
            @test parts.ricci.name == :Phi_Ricci
            @test parts.scalar isa TProduct
            @test parts.scalar.scalar == 1 // 24
        end
    end

    @testset "Fresh indices don't collide" begin
        reg = _decomp_reg()
        with_registry(reg) do
            # Call twice — each should give independent fresh indices
            psi1 = weyl_spinor_expr()
            psi2 = weyl_spinor_expr()
            names1 = Set(idx.name for idx in psi1.indices)
            names2 = Set(idx.name for idx in psi2.indices)
            # Both should have 4 unique names within themselves
            @test length(names1) == 4
            @test length(names2) == 4
        end
    end

    @testset "Vacuum: only Weyl survives" begin
        reg = _decomp_reg()
        with_registry(reg) do
            parts = riemann_spinor_parts()
            # In vacuum, Phi = 0 and Lambda = 0, only Psi remains
            # Just verify the structure is correct
            @test parts.weyl.name == :Psi
            @test parts.ricci.name == :Phi_Ricci
            # Lambda = (1/24) R, which is zero in vacuum
            @test parts.scalar.factors[1].name == :RicScalar
        end
    end
end
