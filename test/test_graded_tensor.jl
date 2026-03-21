@testset "Grassmann parity (graded tensor)" begin
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing,
                                                [:a, :b, :c, :d, :e, :f]))

    # ── register_grassmann_field! sets the option ──
    with_registry(reg) do
        register_grassmann_field!(reg, :psi; manifold=:M4, rank=(0, 1))
        @test has_tensor(reg, :psi)
        tp = get_tensor(reg, :psi)
        @test tp.options[:is_grassmann] === true
    end

    # ── is_grassmann: true for fermionic ──
    @test is_grassmann(reg, :psi) === true

    # ── is_grassmann: false for bosonic ──
    register_tensor!(reg, TensorProperties(
        name=:A, manifold=:M4, rank=(0, 1)))
    @test is_grassmann(reg, :A) === false

    # ── is_grassmann: false for unregistered ──
    @test is_grassmann(reg, :unknown) === false

    # ── Register a second Grassmann field for product tests ──
    register_grassmann_field!(reg, :chi; manifold=:M4, rank=(0, 1))

    with_registry(reg) do
        # ── grassmann_parity: single fermion = 1 ──
        psi_a = Tensor(:psi, [down(:a)])
        @test grassmann_parity(psi_a) == 1

        # ── grassmann_parity: boson = 0 ──
        A_a = Tensor(:A, [down(:a)])
        @test grassmann_parity(A_a) == 0

        # ── grassmann_parity: TScalar = 0 ──
        @test grassmann_parity(TScalar(42)) == 0

        # ── grassmann_parity: TDeriv of fermion = 1 ──
        dpsi = TDeriv(down(:b), psi_a)
        @test grassmann_parity(dpsi) == 1

        # ── grassmann_parity: TDeriv of boson = 0 ──
        dA = TDeriv(down(:b), A_a)
        @test grassmann_parity(dA) == 0

        # ── grassmann_parity: product of two fermions = 0 (even) ──
        chi_b = Tensor(:chi, [down(:b)])
        prod_ff = TProduct(1 // 1, [psi_a, chi_b])
        @test grassmann_parity(prod_ff) == 0

        # ── grassmann_parity: product of fermion and boson = 1 (odd) ──
        prod_fb = TProduct(1 // 1, [psi_a, A_a])
        @test grassmann_parity(prod_fb) == 1

        # ── grassmann_parity: product of boson and boson = 0 ──
        register_tensor!(reg, TensorProperties(
            name=:B, manifold=:M4, rank=(0, 1)))
        B_a = Tensor(:B, [down(:a)])
        prod_bb = TProduct(1 // 1, [A_a, B_a])
        @test grassmann_parity(prod_bb) == 0

        # ── grassmann_parity: product of three fermions = 1 ──
        register_grassmann_field!(reg, :eta; manifold=:M4, rank=(0, 1))
        eta_c = Tensor(:eta, [down(:c)])
        prod_fff = TProduct(1 // 1, [psi_a, chi_b, eta_c])
        @test grassmann_parity(prod_fff) == 1

        # ── grassmann_parity: TSum (all terms same parity) ──
        s = TSum([psi_a, chi_b])
        @test grassmann_parity(s) == 1

        s_even = TSum([prod_ff])
        @test grassmann_parity(s_even) == 0

        # ── grassmann_parity: empty TSum = 0 ──
        @test grassmann_parity(TSum(TensorExpr[])) == 0

        # ── grassmann_sign: identity permutation gives +1 ──
        factors = TensorExpr[psi_a, chi_b]
        @test grassmann_sign([1, 2], factors) == 1

        # ── grassmann_sign: swapping two fermions gives -1 ──
        @test grassmann_sign([2, 1], factors) == -1

        # ── grassmann_sign: swapping fermion and boson gives +1 ──
        factors_fb = TensorExpr[psi_a, A_a]
        @test grassmann_sign([2, 1], factors_fb) == 1

        # ── grassmann_sign: swapping two bosons gives +1 ──
        factors_bb = TensorExpr[A_a, B_a]
        @test grassmann_sign([2, 1], factors_bb) == 1

        # ── grassmann_sign: cyclic permutation of three fermions ──
        # (1,2,3) -> (2,3,1) is two transpositions of odd factors = +1
        factors3 = TensorExpr[psi_a, chi_b, eta_c]
        @test grassmann_sign([2, 3, 1], factors3) == 1

        # ── grassmann_sign: single swap in three fermions gives -1 ──
        # (1,2,3) -> (2,1,3) is one swap of two odd factors
        @test grassmann_sign([2, 1, 3], factors3) == -1

        # ── grassmann_sign: GammaMatrix has parity 0 ──
        gm = GammaMatrix(up(:a))
        @test grassmann_parity(gm) == 0
    end
end
