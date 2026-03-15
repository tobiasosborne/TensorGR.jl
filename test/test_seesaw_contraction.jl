@testset "See-saw contraction rule (Penrose-Rindler Sec 2.5)" begin

    # Helper: fresh registry with manifold + spinor bundles + spin metric + test spinors
    function _seesaw_registry()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_spinor_bundles!(reg; manifold=:M4)
            define_spin_metric!(reg; manifold=:M4)
            # Register test spinor fields
            register_tensor!(reg, TensorProperties(
                name=:psi, manifold=:M4, rank=(0, 1),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}()))
            register_tensor!(reg, TensorProperties(
                name=:chi, manifold=:M4, rank=(0, 1),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}()))
            # A mixed tensor with both spacetime and spinor indices: T_{aB}
            register_tensor!(reg, TensorProperties(
                name=:T, manifold=:M4, rank=(0, 2),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}()))
            # A rank-2 spinor tensor T_{BD}
            register_tensor!(reg, TensorProperties(
                name=:S, manifold=:M4, rank=(0, 2),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}()))
        end
        reg
    end

    @testset "eps^{AB} psi_B -> psi^A (contract on second slot, no sign)" begin
        reg = _seesaw_registry()
        with_registry(reg) do
            eps = Tensor(:eps_spin, [spin_up(:A), spin_up(:B)])
            psi = Tensor(:psi, [spin_down(:B)])
            prod = eps * psi

            result = contract_metrics(prod)
            @test result isa Tensor
            @test result.name == :psi
            @test length(result.indices) == 1
            @test result.indices[1].position == Up
            @test result.indices[1].name == :A
        end
    end

    @testset "eps^{AB} psi_A -> -psi^B (contract on first slot, see-saw sign)" begin
        reg = _seesaw_registry()
        with_registry(reg) do
            eps = Tensor(:eps_spin, [spin_up(:A), spin_up(:B)])
            psi = Tensor(:psi, [spin_down(:A)])
            prod = eps * psi

            result = contract_metrics(prod)
            # Should be -1 * psi^B
            @test result isa TProduct
            @test result.scalar == -1 // 1
            @test length(result.factors) == 1
            @test result.factors[1] isa Tensor
            @test result.factors[1].name == :psi
            @test result.factors[1].indices[1].position == Up
            @test result.factors[1].indices[1].name == :B
        end
    end

    @testset "See-saw identity via epsilon: eps^{AB}psi_B chi_A + eps^{AB}psi_A chi_B = 0" begin
        reg = _seesaw_registry()
        with_registry(reg) do
            # Demonstrate see-saw: eps^{AB} psi_B chi_A = psi^A chi_A
            # and eps^{AB} psi_A chi_B = -psi^B chi_B = -psi^A chi_A (dummy rename)
            # So their sum = 0.
            eps1 = Tensor(:eps_spin, [spin_up(:A), spin_up(:B)])
            psi1 = Tensor(:psi, [spin_down(:B)])
            chi1 = Tensor(:chi, [spin_down(:A)])
            term1 = eps1 * psi1 * chi1  # eps^{AB} psi_B chi_A -> psi^A chi_A

            eps2 = Tensor(:eps_spin, [spin_up(:A), spin_up(:B)])
            psi2 = Tensor(:psi, [spin_down(:A)])
            chi2 = Tensor(:chi, [spin_down(:B)])
            term2 = eps2 * psi2 * chi2  # eps^{AB} psi_A chi_B -> -psi^B chi_B

            s = simplify(term1 + term2; registry=reg)
            @test s == TScalar(0 // 1)
        end
    end

    @testset "eps^{AC} eps_{CB} -> delta^A_B (second slot both, no sign)" begin
        reg = _seesaw_registry()
        with_registry(reg) do
            eps_up = Tensor(:eps_spin, [spin_up(:A), spin_up(:C)])
            eps_dn = Tensor(:eps_spin, [spin_down(:C), spin_down(:B)])
            prod = eps_up * eps_dn

            result = contract_metrics(prod)
            @test result isa Tensor
            @test result.name == :delta_spin
            # Verify the delta has one Up one Down index
            positions = Set(idx.position for idx in result.indices)
            @test Up in positions
            @test Down in positions
        end
    end

    @testset "eps^{AC} eps_{BC} -> -delta^A_B (partner slot 2 swap)" begin
        reg = _seesaw_registry()
        with_registry(reg) do
            # eps^{AC} eps_{BC}: C at slot 2 of eps^, C at slot 2 of eps_
            # Standard pairing for metric x metric is (slot 2, slot 1).
            # Here partner has C at slot 2, not slot 1, so partner_swap = -1.
            # metric_swap: mi=2 -> +1. Combined: +1 * -1 = -1.
            # Result: -delta^A_B.
            # Equivalently: eps_{BC} = -eps_{CB}, so
            #   eps^{AC} eps_{BC} = -eps^{AC} eps_{CB} = -delta^A_B
            eps_up = Tensor(:eps_spin, [spin_up(:A), spin_up(:C)])
            eps_dn = Tensor(:eps_spin, [spin_down(:B), spin_down(:C)])
            prod = eps_up * eps_dn

            result = simplify(prod; registry=reg)
            # Check: result + delta^A_B = 0
            delta_AB = Tensor(:delta_spin, [spin_up(:A), spin_down(:B)])
            check = simplify(result + delta_AB; registry=reg)
            @test check == TScalar(0 // 1)
        end
    end

    @testset "Double raise: eps^{AB} eps^{CD} S_{BD}" begin
        reg = _seesaw_registry()
        with_registry(reg) do
            eps1 = Tensor(:eps_spin, [spin_up(:A), spin_up(:B)])
            eps2 = Tensor(:eps_spin, [spin_up(:C), spin_up(:D)])
            S_BD = Tensor(:S, [spin_down(:B), spin_down(:D)])
            prod = eps1 * eps2 * S_BD

            result = contract_metrics(prod)
            # Both contractions are on second slot (B at slot 2 of eps1, D at slot 2 of eps2)
            # -> no extra sign from either
            @test result isa Tensor
            @test result.name == :S
            @test length(result.indices) == 2
            @test result.indices[1].position == Up
            @test result.indices[1].name == :A
            @test result.indices[2].position == Up
            @test result.indices[2].name == :C
        end
    end

    @testset "Dotted see-saw: eps^{A'B'} psi_{B'} -> psi^{A'}" begin
        reg = _seesaw_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:phi, manifold=:M4, rank=(0, 1),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}()))

            eps_dot = Tensor(:eps_spin_dot, [spin_dot_up(:Ap), spin_dot_up(:Bp)])
            phi = Tensor(:phi, [spin_dot_down(:Bp)])
            prod = eps_dot * phi

            result = contract_metrics(prod)
            @test result isa Tensor
            @test result.name == :phi
            @test length(result.indices) == 1
            @test result.indices[1].position == Up
            @test result.indices[1].name == :Ap
            @test result.indices[1].vbundle == :SL2C_dot
        end
    end

    @testset "Dotted see-saw sign: eps^{A'B'} psi_{A'} -> -psi^{B'}" begin
        reg = _seesaw_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:phi, manifold=:M4, rank=(0, 1),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}()))

            eps_dot = Tensor(:eps_spin_dot, [spin_dot_up(:Ap), spin_dot_up(:Bp)])
            phi = Tensor(:phi, [spin_dot_down(:Ap)])
            prod = eps_dot * phi

            result = contract_metrics(prod)
            @test result isa TProduct
            @test result.scalar == -1 // 1
            @test result.factors[1].name == :phi
            @test result.factors[1].indices[1].position == Up
            @test result.factors[1].indices[1].name == :Bp
            @test result.factors[1].indices[1].vbundle == :SL2C_dot
        end
    end

    @testset "Mixed: eps^{AB} T_{aB} contracts spinor only" begin
        reg = _seesaw_registry()
        with_registry(reg) do
            eps = Tensor(:eps_spin, [spin_up(:A), spin_up(:B)])
            # T_{aB}: spacetime index a (Tangent, Down) + spinor index B (SL2C, Down)
            T_aB = Tensor(:T, [down(:a), spin_down(:B)])
            prod = eps * T_aB

            result = contract_metrics(prod)
            # Should contract B, leaving T^A_a (spinor index raised, spacetime untouched)
            @test result isa Tensor
            @test result.name == :T
            @test length(result.indices) == 2
            # One tangent index (down), one spinor index (up)
            tangent_idx = filter(i -> i.vbundle == :Tangent, result.indices)
            spinor_idx = filter(i -> i.vbundle == :SL2C, result.indices)
            @test length(tangent_idx) == 1
            @test length(spinor_idx) == 1
            @test tangent_idx[1].position == Down
            @test tangent_idx[1].name == :a
            @test spinor_idx[1].position == Up
            @test spinor_idx[1].name == :A
        end
    end

    @testset "contract_spin_metrics convenience wrapper" begin
        reg = _seesaw_registry()
        with_registry(reg) do
            eps = Tensor(:eps_spin, [spin_up(:A), spin_up(:B)])
            psi = Tensor(:psi, [spin_down(:B)])
            prod = eps * psi

            result = contract_spin_metrics(prod; registry=reg)
            @test result isa Tensor
            @test result.name == :psi
            @test result.indices[1].position == Up
            @test result.indices[1].name == :A
        end
    end

    @testset "Spacetime metric unaffected by see-saw changes" begin
        reg = _seesaw_registry()
        with_registry(reg) do
            # g^{ab} T_{bc} -> T^a_c (no sign change for symmetric metric)
            g_up = Tensor(:g, [up(:a), up(:b)])
            T_bc = Tensor(:T, [down(:b), down(:c)])
            prod = g_up * T_bc

            result = contract_metrics(prod)
            @test result isa Tensor
            @test result.name == :T
            # Verify no spurious sign: the product scalar should be +1
            # (If it were a TProduct, scalar would need to be checked)
            @test result isa Tensor  # collapsed single-factor product = bare Tensor
        end
    end

end
