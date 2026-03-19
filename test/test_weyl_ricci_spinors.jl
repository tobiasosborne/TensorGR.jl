@testset "Weyl and Ricci curvature spinors" begin

    function _curv_spin_reg()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg, :M4, :g)
            define_spinor_structure!(reg; manifold=:M4, metric=:g)
        end
        reg
    end

    @testset "Weyl spinor Psi_{ABCD}" begin
        reg = _curv_spin_reg()
        define_weyl_spinor!(reg; manifold=:M4)
        @test has_tensor(reg, :Psi)
        @test has_tensor(reg, :Psi_bar)

        # Check properties
        psi_props = get_tensor(reg, :Psi)
        @test psi_props.rank == (0, 4)
        @test psi_props.options[:spinor_type] == :weyl
        @test psi_props.options[:index_vbundles] == [:SL2C, :SL2C, :SL2C, :SL2C]

        # Conjugate properties
        psi_bar_props = get_tensor(reg, :Psi_bar)
        @test psi_bar_props.rank == (0, 4)
        @test psi_bar_props.options[:spinor_type] == :weyl_conjugate
        @test psi_bar_props.options[:index_vbundles] == [:SL2C_dot, :SL2C_dot, :SL2C_dot, :SL2C_dot]

        # Total symmetry: Psi_{ABCD} = Psi_{BACD} after canonicalize
        psi_ABCD = Tensor(:Psi, [spin_down(:A), spin_down(:B), spin_down(:C), spin_down(:D)])
        psi_BACD = Tensor(:Psi, [spin_down(:B), spin_down(:A), spin_down(:C), spin_down(:D)])
        with_registry(reg) do
            c1 = canonicalize(psi_ABCD)
            c2 = canonicalize(psi_BACD)
            @test to_latex(c1) == to_latex(c2)
        end

        # Total symmetry: Psi_{ABCD} = Psi_{DCBA} after canonicalize
        psi_DCBA = Tensor(:Psi, [spin_down(:D), spin_down(:C), spin_down(:B), spin_down(:A)])
        with_registry(reg) do
            c1 = canonicalize(psi_ABCD)
            c3 = canonicalize(psi_DCBA)
            @test to_latex(c1) == to_latex(c3)
        end

        # Idempotency
        with_registry(reg) do
            c = canonicalize(psi_ABCD)
            cc = canonicalize(c)
            @test to_latex(c) == to_latex(cc)
        end
    end

    @testset "Ricci spinor Phi_{ABA'B'}" begin
        reg = _curv_spin_reg()
        define_ricci_spinor!(reg; manifold=:M4)
        @test has_tensor(reg, :Phi_Ricci)

        phi_props = get_tensor(reg, :Phi_Ricci)
        @test phi_props.rank == (0, 4)
        @test phi_props.options[:spinor_type] == :ricci
        @test phi_props.options[:index_vbundles] == [:SL2C, :SL2C, :SL2C_dot, :SL2C_dot]

        # Symmetric in undotted pair: Phi_{ABA'B'} = Phi_{BAA'B'}
        phi = Tensor(:Phi_Ricci, [spin_down(:A), spin_down(:B), spin_dot_down(:Ap), spin_dot_down(:Bp)])
        phi_swap_undotted = Tensor(:Phi_Ricci, [spin_down(:B), spin_down(:A), spin_dot_down(:Ap), spin_dot_down(:Bp)])
        with_registry(reg) do
            c1 = canonicalize(phi)
            c2 = canonicalize(phi_swap_undotted)
            @test to_latex(c1) == to_latex(c2)
        end

        # Symmetric in dotted pair: Phi_{ABA'B'} = Phi_{ABB'A'}
        phi_swap_dotted = Tensor(:Phi_Ricci, [spin_down(:A), spin_down(:B), spin_dot_down(:Bp), spin_dot_down(:Ap)])
        with_registry(reg) do
            c1 = canonicalize(phi)
            c3 = canonicalize(phi_swap_dotted)
            @test to_latex(c1) == to_latex(c3)
        end
    end

    @testset "define_curvature_spinors! convenience" begin
        reg2 = TensorRegistry()
        with_registry(reg2) do
            @manifold M4 dim=4 metric=g
            define_curvature_tensors!(reg2, :M4, :g)
            define_spinor_structure!(reg2; manifold=:M4, metric=:g)
            define_curvature_spinors!(reg2; manifold=:M4)
        end

        @test has_tensor(reg2, :Psi)
        @test has_tensor(reg2, :Psi_bar)
        @test has_tensor(reg2, :Phi_Ricci)
        @test has_tensor(reg2, :Lambda_spin)
    end

    @testset "Idempotent registration" begin
        reg = _curv_spin_reg()
        define_weyl_spinor!(reg; manifold=:M4)
        @test has_tensor(reg, :Psi)
        # Second call should not error
        define_weyl_spinor!(reg; manifold=:M4)
        @test has_tensor(reg, :Psi)

        define_ricci_spinor!(reg; manifold=:M4)
        @test has_tensor(reg, :Phi_Ricci)
        define_ricci_spinor!(reg; manifold=:M4)
        @test has_tensor(reg, :Phi_Ricci)
    end
end
