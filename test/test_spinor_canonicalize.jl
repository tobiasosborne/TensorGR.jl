@testset "Spinor Canonicalization (TGR-6cn)" begin

    # Helper: fresh registry with manifold + spinor bundles + spin metric
    function _spinor_canon_registry()
        reg = TensorRegistry()
        with_registry(reg) do
            @manifold M4 dim=4 metric=g
            define_spinor_bundles!(reg; manifold=:M4)
            define_spin_metric!(reg; manifold=:M4)
        end
        reg
    end

    @testset "Psi_{ABCD} = Psi_{BACD}: fully symmetric Weyl spinor" begin
        reg = _spinor_canon_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:Psi, manifold=:M4, rank=(0, 4),
                symmetries=SymmetrySpec[FullySymmetric(1,2,3,4)],
                options=Dict{Symbol,Any}()))

            psi_ABCD = Tensor(:Psi, [spin_down(:A), spin_down(:B), spin_down(:C), spin_down(:D)])
            psi_BACD = Tensor(:Psi, [spin_down(:B), spin_down(:A), spin_down(:C), spin_down(:D)])

            # Canonicalize should map BACD -> ABCD
            result = canonicalize(psi_BACD)
            @test result == psi_ABCD

            # Via simplify: difference should vanish
            diff = simplify(psi_ABCD - psi_BACD; registry=reg)
            @test diff == TScalar(0 // 1)
        end
    end

    @testset "Psi_{DCBA} = Psi_{ABCD}: full slot permutation" begin
        reg = _spinor_canon_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:Psi, manifold=:M4, rank=(0, 4),
                symmetries=SymmetrySpec[FullySymmetric(1,2,3,4)],
                options=Dict{Symbol,Any}()))

            psi_ABCD = Tensor(:Psi, [spin_down(:A), spin_down(:B), spin_down(:C), spin_down(:D)])
            psi_DCBA = Tensor(:Psi, [spin_down(:D), spin_down(:C), spin_down(:B), spin_down(:A)])

            diff = simplify(psi_ABCD - psi_DCBA; registry=reg)
            @test diff == TScalar(0 // 1)
        end
    end

    @testset "Phi_{ABA'B'} = Phi_{BAA'B'}: undotted pair symmetry" begin
        reg = _spinor_canon_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:Phi, manifold=:M4, rank=(0, 4),
                symmetries=SymmetrySpec[Symmetric(1,2), Symmetric(3,4)],
                options=Dict{Symbol,Any}()))

            phi_ABAaBa = Tensor(:Phi, [spin_down(:A), spin_down(:B),
                                        spin_dot_down(:Ap), spin_dot_down(:Bp)])
            phi_BAAaBa = Tensor(:Phi, [spin_down(:B), spin_down(:A),
                                        spin_dot_down(:Ap), spin_dot_down(:Bp)])

            result = canonicalize(phi_BAAaBa)
            @test result == phi_ABAaBa

            diff = simplify(phi_ABAaBa - phi_BAAaBa; registry=reg)
            @test diff == TScalar(0 // 1)
        end
    end

    @testset "Phi_{ABA'B'} = Phi_{ABB'A'}: dotted pair symmetry" begin
        reg = _spinor_canon_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:Phi, manifold=:M4, rank=(0, 4),
                symmetries=SymmetrySpec[Symmetric(1,2), Symmetric(3,4)],
                options=Dict{Symbol,Any}()))

            phi_ABAaBa = Tensor(:Phi, [spin_down(:A), spin_down(:B),
                                        spin_dot_down(:Ap), spin_dot_down(:Bp)])
            phi_ABBaAa = Tensor(:Phi, [spin_down(:A), spin_down(:B),
                                        spin_dot_down(:Bp), spin_dot_down(:Ap)])

            result = canonicalize(phi_ABBaAa)
            @test result == phi_ABAaBa

            diff = simplify(phi_ABAaBa - phi_ABBaAa; registry=reg)
            @test diff == TScalar(0 // 1)
        end
    end

    @testset "Mixed T_{a}^{A} sigma^{BB'}_b: tangent not swapped with spinor" begin
        reg = _spinor_canon_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:T, manifold=:M4, rank=(0, 2),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}()))
            register_tensor!(reg, TensorProperties(
                name=:sigma, manifold=:M4, rank=(0, 3),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}()))

            T_aA = Tensor(:T, [down(:a), spin_up(:A)])
            sig = Tensor(:sigma, [spin_up(:B), spin_dot_up(:Bp), down(:b)])
            prod = T_aA * sig

            result = canonicalize(prod)
            idxs = indices(result)

            tangent_idxs = filter(i -> i.vbundle == :Tangent, idxs)
            sl2c_idxs = filter(i -> i.vbundle == :SL2C, idxs)
            sl2c_dot_idxs = filter(i -> i.vbundle == :SL2C_dot, idxs)

            @test length(tangent_idxs) == 2
            @test length(sl2c_idxs) == 2
            @test length(sl2c_dot_idxs) == 1
        end
    end

    @testset "epsilon_{AB} antisymmetry: eps_{AB} + eps_{BA} = 0" begin
        reg = _spinor_canon_registry()
        with_registry(reg) do
            eps_AB = Tensor(:eps_spin, [spin_down(:A), spin_down(:B)])
            eps_BA = Tensor(:eps_spin, [spin_down(:B), spin_down(:A)])

            result = simplify(eps_AB + eps_BA; registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

    @testset "Antisymmetric Y: X_{AB} Y^{BA} = -X_{AB} Y^{AB}" begin
        reg = _spinor_canon_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:X, manifold=:M4, rank=(0, 2),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}()))
            register_tensor!(reg, TensorProperties(
                name=:Yanti, manifold=:M4, rank=(0, 2),
                symmetries=SymmetrySpec[AntiSymmetric(1,2)],
                options=Dict{Symbol,Any}()))

            X_AB = Tensor(:X, [spin_down(:A), spin_down(:B)])
            Yanti_AB = Tensor(:Yanti, [spin_up(:A), spin_up(:B)])
            Yanti_BA = Tensor(:Yanti, [spin_up(:B), spin_up(:A)])

            # Yanti_{BA} = -Yanti_{AB}, so sum should vanish
            s = simplify(X_AB * Yanti_AB + X_AB * Yanti_BA; registry=reg)
            @test s == TScalar(0 // 1)
        end
    end

    @testset "Symmetric Y: X_{AB} Y^{BA} = X_{AB} Y^{AB}" begin
        reg = _spinor_canon_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:X, manifold=:M4, rank=(0, 2),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}()))
            register_tensor!(reg, TensorProperties(
                name=:Ysym, manifold=:M4, rank=(0, 2),
                symmetries=SymmetrySpec[Symmetric(1,2)],
                options=Dict{Symbol,Any}()))

            X_AB = Tensor(:X, [spin_down(:A), spin_down(:B)])
            Ysym_AB = Tensor(:Ysym, [spin_up(:A), spin_up(:B)])
            Ysym_BA = Tensor(:Ysym, [spin_up(:B), spin_up(:A)])

            diff = simplify(X_AB * Ysym_AB - X_AB * Ysym_BA; registry=reg)
            @test diff == TScalar(0 // 1)
        end
    end

    @testset "VBundle boundary: same name, different bundles are NOT dummies" begin
        reg = _spinor_canon_registry()
        with_registry(reg) do
            # An index named :A in Tangent and :A in SL2C should not be
            # treated as a dummy pair
            register_tensor!(reg, TensorProperties(
                name=:V, manifold=:M4, rank=(1, 0),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}()))
            register_tensor!(reg, TensorProperties(
                name=:W, manifold=:M4, rank=(0, 1),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}()))

            # V^A (Tangent) * W_A (SL2C) -- same name, different vbundles
            V_up = Tensor(:V, [up(:A)])  # :A in Tangent
            W_dn = Tensor(:W, [spin_down(:A)])  # :A in SL2C

            prod = V_up * W_dn
            dp = dummy_pairs(prod)
            # Should NOT find a dummy pair since vbundles differ
            @test isempty(dp)

            # Canonicalization should leave the product unchanged (no dummies to contract)
            result = canonicalize(prod)
            idxs = indices(result)
            @test length(idxs) == 2
            tangent_count = count(i -> i.vbundle == :Tangent, idxs)
            sl2c_count = count(i -> i.vbundle == :SL2C, idxs)
            @test tangent_count == 1
            @test sl2c_count == 1
        end
    end

    @testset "Product with spinor dummies: T_{AB} U^{AB}" begin
        reg = _spinor_canon_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:T, manifold=:M4, rank=(0, 2),
                symmetries=SymmetrySpec[Symmetric(1,2)],
                options=Dict{Symbol,Any}()))
            register_tensor!(reg, TensorProperties(
                name=:U, manifold=:M4, rank=(0, 2),
                symmetries=SymmetrySpec[Symmetric(1,2)],
                options=Dict{Symbol,Any}()))

            # T_{AB} U^{AB} and T_{AB} U^{BA}: with symmetric U, both should
            # canonicalize to the same expression
            T_AB = Tensor(:T, [spin_down(:A), spin_down(:B)])
            U_AB = Tensor(:U, [spin_up(:A), spin_up(:B)])
            U_BA = Tensor(:U, [spin_up(:B), spin_up(:A)])

            c1 = canonicalize(T_AB * U_AB)
            c2 = canonicalize(T_AB * U_BA)
            diff = simplify(T_AB * U_AB - T_AB * U_BA; registry=reg)
            @test diff == TScalar(0 // 1)
        end
    end

    @testset "Factor sort key includes vbundle" begin
        reg = _spinor_canon_registry()
        with_registry(reg) do
            register_tensor!(reg, TensorProperties(
                name=:P, manifold=:M4, rank=(0, 1),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}()))
            register_tensor!(reg, TensorProperties(
                name=:Q, manifold=:M4, rank=(0, 1),
                symmetries=SymmetrySpec[],
                options=Dict{Symbol,Any}()))

            # P_a (Tangent) vs P_A (SL2C): factor sort should distinguish
            P_tang = Tensor(:P, [down(:a)])
            P_spin = Tensor(:P, [spin_down(:A)])

            key_tang = TensorGR._factor_sort_key(P_tang)
            key_spin = TensorGR._factor_sort_key(P_spin)
            @test key_tang != key_spin
        end
    end

    @testset "Riemann + spinor in same expression: no cross-contamination" begin
        reg = _spinor_canon_registry()
        with_registry(reg) do
            define_curvature_tensors!(reg, :M4, :g)
            register_tensor!(reg, TensorProperties(
                name=:Psi, manifold=:M4, rank=(0, 4),
                symmetries=SymmetrySpec[FullySymmetric(1,2,3,4)],
                options=Dict{Symbol,Any}()))

            # R_{abcd} Psi_{ABCD} -- mixed Tangent + SL2C product
            R = Tensor(:Riem, [down(:a), down(:b), down(:c), down(:d)])
            Psi = Tensor(:Psi, [spin_down(:A), spin_down(:B), spin_down(:C), spin_down(:D)])

            prod = R * Psi
            result = canonicalize(prod)
            idxs = indices(result)

            tangent_idxs = filter(i -> i.vbundle == :Tangent, idxs)
            sl2c_idxs = filter(i -> i.vbundle == :SL2C, idxs)

            @test length(tangent_idxs) == 4
            @test length(sl2c_idxs) == 4

            # Riemann antisymmetry should still work
            R_bacd = Tensor(:Riem, [down(:b), down(:a), down(:c), down(:d)])
            diff = simplify(R * Psi + R_bacd * Psi; registry=reg)
            @test diff == TScalar(0 // 1)

            # Psi full symmetry should still work
            Psi_BACD = Tensor(:Psi, [spin_down(:B), spin_down(:A), spin_down(:C), spin_down(:D)])
            diff2 = simplify(R * Psi - R * Psi_BACD; registry=reg)
            @test diff2 == TScalar(0 // 1)
        end
    end

    @testset "Epsilon self-contraction: eps^{AB} eps_{AB} = -2" begin
        reg = _spinor_canon_registry()
        with_registry(reg) do
            eps_up = Tensor(:eps_spin, [spin_up(:A), spin_up(:B)])
            eps_dn = Tensor(:eps_spin, [spin_down(:A), spin_down(:B)])
            prod = eps_up * eps_dn

            result = simplify(prod; registry=reg)
            @test result == TScalar(-2 // 1)
        end
    end

    @testset "Dotted epsilon antisymmetry via canonicalize" begin
        reg = _spinor_canon_registry()
        with_registry(reg) do
            eps_AaBa = Tensor(:eps_spin_dot, [spin_dot_down(:Ap), spin_dot_down(:Bp)])
            eps_BaAa = Tensor(:eps_spin_dot, [spin_dot_down(:Bp), spin_dot_down(:Ap)])

            result = simplify(eps_AaBa + eps_BaAa; registry=reg)
            @test result == TScalar(0 // 1)
        end
    end

end
