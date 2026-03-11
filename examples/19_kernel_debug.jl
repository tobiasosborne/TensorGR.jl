#= Debug: check what extract_kernel misses =#

using TensorGR

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    @define_tensor k on=M4 rank=(0,1)

    mp = define_metric_perturbation!(reg, :g, :h)
    set_vanishing!(reg, :Riem)
    set_vanishing!(reg, :Ric)
    set_vanishing!(reg, :RicScalar)

    # Compute δ²R, expand derivatives, simplify
    δ2R = simplify(δricci_scalar(mp, 2); registry=reg)
    δ2R = expand_derivatives(δ2R)
    δ2R = simplify(δ2R; registry=reg)

    # Fourier transform
    δ2R_f = to_fourier(δ2R)
    δ2R_f = simplify(δ2R_f; registry=reg)
    δ2R_f = fix_dummy_positions(δ2R_f)

    # Manually check each term for h count
    terms = δ2R_f isa TSum ? δ2R_f.terms : [δ2R_f]
    println("Total Fourier terms: $(length(terms))")

    bilinear_count = 0
    missed_count = 0
    for (i, t) in enumerate(terms)
        sc, factors = if t isa TProduct
            (t.scalar, collect(t.factors))
        elseif t isa Tensor
            (1//1, TensorExpr[t])
        else
            (1//1, TensorExpr[t])
        end

        h_count = count(f -> f isa Tensor && f.name == :h, factors)
        if h_count == 2
            bilinear_count += 1
        elseif h_count != 2
            missed_count += 1
            # Check for h inside non-Tensor structures
            has_hidden_h = any(f -> !(f isa Tensor) && !(f isa TScalar) && occursin("h[", string(f)), factors)
            println("  term $i: h_count=$h_count, factors=$(length(factors)), hidden_h=$has_hidden_h")
            if length(string(t)) < 200
                println("    $t")
            else
                println("    (too long to print)")
            end
        end
    end
    println("\nBilinear (2 h's): $bilinear_count")
    println("Missed (≠2 h's): $missed_count")

    # Also check: is the full expression correct by evaluating on a specific configuration?
    # Build FP in position space and compare with δ²R + ½h·δR
    println("\n── Direct comparison: δ²R + ½h·δR vs FP ──")
    δR = simplify(expand_perturbation(Tensor(:RicScalar, TIndex[]), mp, 1); registry=reg)
    δR = expand_derivatives(δR)
    δR = simplify(δR; registry=reg)

    h_trace = Tensor(:g, [up(:z1), up(:z2)]) * Tensor(:h, [down(:z1), down(:z2)])
    half_h_δR = tproduct(1//2, TensorExpr[h_trace]) * δR
    half_h_δR = expand_derivatives(half_h_δR)
    half_h_δR = simplify(half_h_δR; registry=reg)

    L2 = δ2R + half_h_δR
    L2 = expand_derivatives(L2)
    L2 = simplify(L2; registry=reg)

    # Build FP in position space
    t1 = tproduct(1//2, TensorExpr[
        TDeriv(down(:c), Tensor(:h, [down(:a), down(:b)])),
        TDeriv(up(:c), Tensor(:h, [up(:a), up(:b)]))])
    t2 = tproduct(-1//1, TensorExpr[
        TDeriv(down(:b), Tensor(:h, [up(:a), up(:b)])),
        TDeriv(down(:c), Tensor(:h, [up(:c), down(:a)]))])
    t3 = tproduct(1//1, TensorExpr[
        Tensor(:g, [up(:c), up(:d)]),
        TDeriv(down(:a), Tensor(:h, [down(:c), down(:d)])),
        TDeriv(down(:b), Tensor(:h, [up(:a), up(:b)]))])
    t4 = tproduct(-1//2, TensorExpr[
        Tensor(:g, [up(:c), up(:d)]),
        TDeriv(down(:a), Tensor(:h, [down(:c), down(:d)])),
        Tensor(:g, [up(:e), up(:f)]),
        TDeriv(up(:a), Tensor(:h, [down(:e), down(:f)]))])
    FP_pos = t1 + t2 + t3 + t4
    FP_pos = simplify(FP_pos; registry=reg)

    # Check difference: L2 - FP should be zero
    diff = L2 - FP_pos
    diff = simplify(diff; registry=reg)
    diff = expand_derivatives(diff)
    diff = simplify(diff; registry=reg)

    n_diff = diff isa TSum ? length(diff.terms) : (diff == TScalar(0//1) ? 0 : 1)
    println("  L₂ terms: $(L2 isa TSum ? length(L2.terms) : 1)")
    println("  FP terms: $(FP_pos isa TSum ? length(FP_pos.terms) : 1)")
    println("  diff = L₂ - FP: $n_diff terms")

    if n_diff > 0 && n_diff <= 10
        if diff isa TSum
            for (i, t) in enumerate(diff.terms)
                println("    diff term $i: $t")
            end
        else
            println("    diff: $diff")
        end
    end
end
