@testset "Multi-field kernel extraction" begin

    function _mk_reg()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            @covd D on=M4 metric=g

            # Register two fields: h_{ab} (rank-2 symmetric) and phi (scalar)
            register_tensor!(reg, TensorProperties(
                name=:h, manifold=:M4, rank=(0, 2),
                symmetries=SymmetrySpec[Symmetric(1, 2)]))
            register_tensor!(reg, TensorProperties(
                name=:phi_field, manifold=:M4, rank=(0, 0),
                symmetries=SymmetrySpec[]))
        end
        reg
    end

    @testset "Single field diagonal extraction" begin
        reg = _mk_reg()
        with_registry(reg) do
            # h_{ab} h^{ab} — simple self-coupling
            h_dn = Tensor(:h, [down(:a), down(:b)])
            h_up = Tensor(:h, [up(:c), up(:d)])
            g_ac = Tensor(:g, [down(:a), up(:c)])
            g_bd = Tensor(:g, [down(:b), up(:d)])
            expr = tproduct(1 // 1, TensorExpr[h_dn, g_ac, g_bd, h_up])

            mk = extract_kernel_multi(expr, [:h]; registry=reg)
            @test mk isa MultiFieldKernels
            @test length(mk.fields) == 1
            @test haskey(mk.diagonal, :h)
            @test isempty(mk.cross)
        end
    end

    @testset "Two fields, no cross-coupling" begin
        reg = _mk_reg()
        with_registry(reg) do
            # h_{ab} h^{ab} + phi * phi — diagonal only
            h_dn = Tensor(:h, [down(:a), down(:b)])
            h_up = Tensor(:h, [up(:a), up(:b)])
            hh = tproduct(1 // 1, TensorExpr[h_dn, h_up])

            phi1 = Tensor(:phi_field, TIndex[])
            phi2 = Tensor(:phi_field, TIndex[])
            pp = tproduct(1 // 1, TensorExpr[phi1, phi2])

            expr = tsum(TensorExpr[hh, pp])
            mk = extract_kernel_multi(expr, [:h, :phi_field]; registry=reg)

            @test length(mk.fields) == 2
            # At least one diagonal block should exist
            @test length(mk.diagonal) >= 1
        end
    end

    @testset "Cross-coupling detection" begin
        reg = _mk_reg()
        with_registry(reg) do
            # h_{ab} g^{ab} phi — cross-coupling term
            h_dn = Tensor(:h, [down(:a), down(:b)])
            g_up = Tensor(:g, [up(:a), up(:b)])
            phi = Tensor(:phi_field, TIndex[])
            cross_term = tproduct(1 // 1, TensorExpr[h_dn, g_up, phi])

            mk = extract_kernel_multi(cross_term, [:h, :phi_field]; registry=reg)

            # Should detect cross-coupling
            @test haskey(mk.cross, (:h, :phi_field))
            cross_K = mk.cross[(:h, :phi_field)]
            @test !isempty(cross_K.terms)
            @test cross_K.field == :h_phi_field
        end
    end

    @testset "Mixed diagonal + cross" begin
        reg = _mk_reg()
        with_registry(reg) do
            # h_{ab}h^{ab} + h_{ab}g^{ab}phi + phi*phi
            h_dn = Tensor(:h, [down(:a), down(:b)])
            h_up = Tensor(:h, [up(:a), up(:b)])
            hh = tproduct(1 // 1, TensorExpr[h_dn, h_up])

            g_up = Tensor(:g, [up(:c), up(:d)])
            h_dn2 = Tensor(:h, [down(:c), down(:d)])
            phi = Tensor(:phi_field, TIndex[])
            hp = tproduct(1 // 1, TensorExpr[h_dn2, g_up, phi])

            phi1 = Tensor(:phi_field, TIndex[])
            phi2 = Tensor(:phi_field, TIndex[])
            pp = tproduct(1 // 1, TensorExpr[phi1, phi2])

            expr = tsum(TensorExpr[hh, hp, pp])
            mk = extract_kernel_multi(expr, [:h, :phi_field]; registry=reg)

            @test length(mk.fields) == 2
            # Should have cross-coupling
            @test haskey(mk.cross, (:h, :phi_field))
        end
    end

    @testset "Display" begin
        reg = _mk_reg()
        with_registry(reg) do
            h_dn = Tensor(:h, [down(:a), down(:b)])
            h_up = Tensor(:h, [up(:a), up(:b)])
            expr = tproduct(1 // 1, TensorExpr[h_dn, h_up])
            mk = extract_kernel_multi(expr, [:h]; registry=reg)
            s = sprint(show, mk)
            @test occursin("MultiFieldKernels", s)
        end
    end

    @testset "Empty fields list" begin
        reg = _mk_reg()
        with_registry(reg) do
            h_dn = Tensor(:h, [down(:a), down(:b)])
            h_up = Tensor(:h, [up(:a), up(:b)])
            expr = tproduct(1 // 1, TensorExpr[h_dn, h_up])
            mk = extract_kernel_multi(expr, Symbol[]; registry=reg)
            @test isempty(mk.diagonal)
            @test isempty(mk.cross)
        end
    end
end
