#= Diagnose WHY simplify changes the Fourier transform result.

   Raw δ²R (3 terms) gives different spin projections than simplified δ²R (10 terms).
   This means simplify is changing the derivative structure in a way that
   doesn't preserve the Fourier transform.
=#

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

    # Raw δ²R
    δ2R = δricci_scalar(mp, 2)
    println("═══ Raw δ²R: $(δ2R isa TSum ? length(δ2R.terms) : 1) terms ═══\n")

    if δ2R isa TSum
        for (i, t) in enumerate(δ2R.terms)
            println("  Term $i ($(typeof(t).name.name)):")
            println("    $(string(t))")
            println()
        end
    else
        println("  Single term: $(string(δ2R))")
    end

    # What does simplify do step by step?
    println("\n═══ Simplify pipeline steps ═══\n")

    step1 = TensorGR.expand_products(δ2R)
    println("After expand_products: $(step1 isa TSum ? length(step1.terms) : 1) terms")
    if step1 isa TSum
        for (i, t) in enumerate(step1.terms)
            println("  $i: $(string(t)[1:min(end,150)])")
        end
    end

    step2 = contract_metrics(step1)
    println("\nAfter contract_metrics: $(step2 isa TSum ? length(step2.terms) : 1) terms")
    if step2 isa TSum
        for (i, t) in enumerate(step2.terms)
            println("  $i: $(string(t)[1:min(end,150)])")
        end
    end

    step3 = TensorGR.canonicalize(step2)
    println("\nAfter canonicalize: $(step3 isa TSum ? length(step3.terms) : 1) terms")
    if step3 isa TSum
        for (i, t) in enumerate(step3.terms)
            println("  $i: $(string(t)[1:min(end,150)])")
        end
    end

    step4 = TensorGR.collect_terms(step3)
    println("\nAfter collect_terms: $(step4 isa TSum ? length(step4.terms) : 1) terms")
    if step4 isa TSum
        for (i, t) in enumerate(step4.terms)
            println("  $i: $(string(t)[1:min(end,150)])")
        end
    end

    step5 = TensorGR.apply_rules(step4; registry=reg)
    println("\nAfter apply_rules: $(step5 isa TSum ? length(step5.terms) : 1) terms")
    if step5 isa TSum
        for (i, t) in enumerate(step5.terms)
            println("  $i: $(string(t)[1:min(end,150)])")
        end
    end

    # Now Fourier-transform at different stages and compare
    println("\n═══ Fourier at each stage ═══\n")

    function fourier_and_count(expr, label)
        f = to_fourier(expr)
        f = simplify(f; registry=reg)
        f = fix_dummy_positions(f)
        n = f isa TSum ? length(f.terms) : 1
        println("$label: $n Fourier terms")
        f
    end

    f_raw   = fourier_and_count(δ2R, "Fourier(raw)")
    f_step1 = fourier_and_count(step1, "Fourier(expand_products)")
    f_step2 = fourier_and_count(step2, "Fourier(contract_metrics)")
    f_step3 = fourier_and_count(step3, "Fourier(canonicalize)")
    f_step4 = fourier_and_count(step4, "Fourier(collect_terms)")
    f_step5 = fourier_and_count(step5, "Fourier(apply_rules)")

    # Spin projection comparison
    k2 = 1.7
    function spin_all(K)
        s2  = _eval_spin_scalar(spin_project(K, :spin2;  registry=reg), k2)
        s1  = _eval_spin_scalar(spin_project(K, :spin1;  registry=reg), k2)
        s0s = _eval_spin_scalar(spin_project(K, :spin0s; registry=reg), k2)
        (s2, s1, s0s)
    end

    function do_spin(f, label)
        K = extract_kernel(f, :h; registry=reg)
        sp = spin_all(K)
        println("  $label: spin2=$(round(sp[1];digits=4)), spin1=$(round(sp[2];digits=4)), spin0s=$(round(sp[3];digits=4))")
        sp
    end

    println("\n═══ Spin projections at each stage ═══\n")
    sp_raw   = do_spin(f_raw, "raw              ")
    sp_step1 = do_spin(f_step1, "expand_products  ")
    sp_step2 = do_spin(f_step2, "contract_metrics ")
    sp_step3 = do_spin(f_step3, "canonicalize     ")
    sp_step4 = do_spin(f_step4, "collect_terms    ")
    sp_step5 = do_spin(f_step5, "apply_rules      ")

    # Identify where the change happens
    println("\n═══ Where does the spin change? ═══\n")
    stages = [("raw", sp_raw), ("expand_products", sp_step1),
              ("contract_metrics", sp_step2), ("canonicalize", sp_step3),
              ("collect_terms", sp_step4), ("apply_rules", sp_step5)]
    for i in 2:length(stages)
        prev_label, prev_sp = stages[i-1]
        curr_label, curr_sp = stages[i]
        changed = !all(isapprox(a, b; atol=1e-10) for (a,b) in zip(prev_sp, curr_sp))
        if changed
            println("  ✗ CHANGE between $prev_label → $curr_label")
            println("    Δspin2=$(round(curr_sp[1]-prev_sp[1];digits=4)), " *
                    "Δspin1=$(round(curr_sp[2]-prev_sp[2];digits=4)), " *
                    "Δspin0s=$(round(curr_sp[3]-prev_sp[3];digits=4))")
        else
            println("  ✓ same: $prev_label → $curr_label")
        end
    end
end
