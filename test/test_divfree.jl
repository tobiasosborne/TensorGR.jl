using TensorGR: enforce_divfree

@testset "Divergence-free enforcement" begin
    @testset "Direct divergence of divfree tensor -> zero" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0,2), symmetries=SymmetrySpec[Symmetric(1,2)],
            is_metric=true, options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M4, rank=(1,1), symmetries=SymmetrySpec[],
            is_delta=true, options=Dict{Symbol,Any}(:is_delta => true)))
        define_covd!(reg, :D; manifold=:M4, metric=:g)
        # Symmetric rank-2 tensor T^{ab}, divergence-free w.r.t. D on index 1
        register_tensor!(reg, TensorProperties(
            name=:T, manifold=:M4, rank=(2,0),
            symmetries=SymmetrySpec[Symmetric(1,2)],
            options=Dict{Symbol,Any}()))
        set_divfree!(reg, :T; covd=:D, index=1)

        with_registry(reg) do
            # D_a T^{ab} -> 0
            expr = TDeriv(down(:a), Tensor(:T, [up(:a), up(:b)]), :D)
            result = enforce_divfree(expr)
            @test result isa TScalar
            @test result.val == 0
        end
    end

    @testset "Non-divergence contraction does NOT zero" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0,2), symmetries=SymmetrySpec[Symmetric(1,2)],
            is_metric=true, options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M4, rank=(1,1), symmetries=SymmetrySpec[],
            is_delta=true, options=Dict{Symbol,Any}(:is_delta => true)))
        define_covd!(reg, :D; manifold=:M4, metric=:g)
        register_tensor!(reg, TensorProperties(
            name=:T, manifold=:M4, rank=(2,0),
            symmetries=SymmetrySpec[Symmetric(1,2)],
            options=Dict{Symbol,Any}()))
        set_divfree!(reg, :T; covd=:D, index=1)

        with_registry(reg) do
            # D_c T^{ab} — derivative index c does not contract with any T index -> unchanged
            expr = TDeriv(down(:c), Tensor(:T, [up(:a), up(:b)]), :D)
            result = enforce_divfree(expr)
            @test result isa TDeriv
            @test result.arg isa Tensor
            @test result.arg.name == :T
        end
    end

    @testset "Einstein tensor divergence-free via simplify" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0,2), symmetries=SymmetrySpec[Symmetric(1,2)],
            is_metric=true, options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M4, rank=(1,1), symmetries=SymmetrySpec[],
            is_delta=true, options=Dict{Symbol,Any}(:is_delta => true)))
        define_covd!(reg, :D; manifold=:M4, metric=:g)
        # G^{ab} divergence-free on both indices
        register_tensor!(reg, TensorProperties(
            name=:Ein, manifold=:M4, rank=(2,0),
            symmetries=SymmetrySpec[Symmetric(1,2)],
            options=Dict{Symbol,Any}()))
        set_divfree!(reg, :Ein; covd=:D, index=1)

        with_registry(reg) do
            # D_a G^{ab} -> 0 via simplify
            expr = TDeriv(down(:a), Tensor(:Ein, [up(:a), up(:b)]), :D)
            result = simplify(expr; registry=reg)
            @test result isa TScalar
            @test result.val == 0
        end
    end

    @testset "Regular tensor is NOT zeroed" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0,2), symmetries=SymmetrySpec[Symmetric(1,2)],
            is_metric=true, options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M4, rank=(1,1), symmetries=SymmetrySpec[],
            is_delta=true, options=Dict{Symbol,Any}(:is_delta => true)))
        define_covd!(reg, :D; manifold=:M4, metric=:g)
        # S^{ab} is NOT marked divergence-free
        register_tensor!(reg, TensorProperties(
            name=:S, manifold=:M4, rank=(2,0),
            symmetries=SymmetrySpec[Symmetric(1,2)],
            options=Dict{Symbol,Any}()))

        with_registry(reg) do
            # D_a S^{ab} should NOT be zero
            expr = TDeriv(down(:a), Tensor(:S, [up(:a), up(:b)]), :D)
            result = simplify(expr; registry=reg)
            @test !(result isa TScalar && result.val == 0)
        end
    end

    @testset "Wrong covd does NOT trigger divfree" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0,2), symmetries=SymmetrySpec[Symmetric(1,2)],
            is_metric=true, options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M4, rank=(1,1), symmetries=SymmetrySpec[],
            is_delta=true, options=Dict{Symbol,Any}(:is_delta => true)))
        define_covd!(reg, :D; manifold=:M4, metric=:g)
        register_tensor!(reg, TensorProperties(
            name=:T, manifold=:M4, rank=(2,0),
            symmetries=SymmetrySpec[Symmetric(1,2)],
            options=Dict{Symbol,Any}()))
        set_divfree!(reg, :T; covd=:D, index=1)

        with_registry(reg) do
            # partial_a T^{ab} — partial, not D -> should NOT zero
            expr = TDeriv(down(:a), Tensor(:T, [up(:a), up(:b)]), :partial)
            result = enforce_divfree(expr)
            @test result isa TDeriv
        end
    end

    @testset "Divfree through sums and products" begin
        reg = TensorRegistry()
        register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d,:e,:f]))
        register_tensor!(reg, TensorProperties(
            name=:g, manifold=:M4, rank=(0,2), symmetries=SymmetrySpec[Symmetric(1,2)],
            is_metric=true, options=Dict{Symbol,Any}(:is_metric => true)))
        register_tensor!(reg, TensorProperties(
            name=:delta, manifold=:M4, rank=(1,1), symmetries=SymmetrySpec[],
            is_delta=true, options=Dict{Symbol,Any}(:is_delta => true)))
        define_covd!(reg, :D; manifold=:M4, metric=:g)
        register_tensor!(reg, TensorProperties(
            name=:T, manifold=:M4, rank=(2,0),
            symmetries=SymmetrySpec[Symmetric(1,2)],
            options=Dict{Symbol,Any}()))
        set_divfree!(reg, :T; covd=:D, index=1)
        register_tensor!(reg, TensorProperties(
            name=:V, manifold=:M4, rank=(0,1), symmetries=SymmetrySpec[],
            options=Dict{Symbol,Any}()))
        register_tensor!(reg, TensorProperties(
            name=:S, manifold=:M4, rank=(2,0),
            symmetries=SymmetrySpec[Symmetric(1,2)],
            options=Dict{Symbol,Any}()))

        with_registry(reg) do
            # Product: V_c * D_a T^{ab} -> 0
            div_T = TDeriv(down(:a), Tensor(:T, [up(:a), up(:b)]), :D)
            V = Tensor(:V, [down(:c)])
            expr = V * div_T
            result = enforce_divfree(expr)
            @test result isa TScalar
            @test result.val == 0

            # Sum: D_a T^{ab} + D_a S^{ab} -> 0 + D_a S^{ab}
            div_S = TDeriv(down(:a), Tensor(:S, [up(:a), up(:b)]), :D)
            sum_expr = div_T + div_S
            result2 = enforce_divfree(sum_expr)
            @test !(result2 isa TScalar && result2.val == 0)
        end
    end
end
