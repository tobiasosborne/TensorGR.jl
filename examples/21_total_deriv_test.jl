#= Test: do total derivatives vanish under spin projection?
# Build a known total derivative in Fourier space, extract kernel, spin-project.
# If spin projections are zero → pipeline handles total derivs correctly.
# If nonzero → pipeline bug.
=#

using TensorGR

reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    @define_tensor h on=M4 rank=(0,2) symmetry=Symmetric(1,2)
    @define_tensor k on=M4 rank=(0,1)

    k2 = 1.7

    # ─── Total derivative 1: ∂_a(h_{bc} ∂^a h^{bc}) ───
    # Position space: TDeriv(∂_a, h_{bc} × TDeriv(∂^a, h^{bc}))
    # After Fourier: k_a × h_{bc} × k^a × h^{bc} = k² h_{bc} h^{bc}
    # But this is just a regular k² h·h term, not recognizable as a total derivative.
    # In the FP kernel, this term has coefficient 1/2. If we add coefficient 1, we get 3/2.

    # Instead, let's build the total derivative in expanded form:
    # ∂_a(h_{bc} ∂^a h^{bc}) = (∂_a h_{bc})(∂^a h^{bc}) + h_{bc}(∂_a ∂^a h^{bc})
    # = (∂h)(∂h) + h(□h)
    # After Fourier with our convention (∂→k):
    # = k_a h_{bc} k^a h^{bc} + h_{bc} k² h^{bc} = k²hh + k²hh = 2k²hh
    # This is NOT a recognizable total derivative anymore — it's just 2k²hh.
    # Since the FP has ½k²hh, adding this total derivative changes the coefficient.

    # The KEY question: does this total derivative contribute zero to spin projections?
    # Total derivative in Fourier space = k_μ × (something).
    # Let's build: TD₁ = k_a × (h_{bc} × k^a × h^{bc})
    # This has the structure k_a × X^a where X^a = k^a h_{bc} h^{bc}
    # In the kernel: k_a k^a h h = k² hh
    TD1 = tproduct(1//1, TensorExpr[
        Tensor(:k, [down(:a)]),
        Tensor(:h, [down(:b), down(:c)]),
        Tensor(:k, [up(:a)]),
        Tensor(:h, [up(:b), up(:c)])])
    # = k_a k^a h_{bc} h^{bc} = k² h_{bc} h^{bc}

    K_TD1 = extract_kernel(TD1, :h; registry=reg)
    println("── Total derivative 1: k²·h_{bc}h^{bc} ──")
    println("  Kernel: $(length(K_TD1.terms)) terms")
    s2  = _eval_spin_scalar(spin_project(K_TD1, :spin2;  registry=reg), k2)
    s1  = _eval_spin_scalar(spin_project(K_TD1, :spin1;  registry=reg), k2)
    s0s = _eval_spin_scalar(spin_project(K_TD1, :spin0s; registry=reg), k2)
    s0w = _eval_spin_scalar(spin_project(K_TD1, :spin0w; registry=reg), k2)
    println("  spin-2=$s2, spin-1=$s1, spin-0s=$s0s, spin-0w=$s0w")
    println("  (These should NOT be zero — k²hh is a physical term, not a total deriv in k-space)")

    # The issue: a total derivative ∂_μ(stuff) in position space, after Fourier,
    # becomes a REGULAR bilinear term. It's no longer recognizable as a total derivative.
    # The antisymmetry is lost.
    #
    # To recover the antisymmetry, we'd need to track that the k came from ∂
    # acting on the WHOLE product, not on individual factors.

    # ─── Test: build the antisymmetric part of a bilinear kernel ───
    # Antisymmetric kernel: K^A_{μν,ρσ} = K_{μν,ρσ} - K_{ρσ,μν}
    # For a term: c × h_{μν} × h_{ρσ}, the antisym is:
    # c_{μν,ρσ} h_{μν} h_{ρσ} - c_{ρσ,μν} h_{ρσ} h_{μν}
    # In our representation, this means:
    # (c, L=[μ,ν], R=[ρ,σ]) and -(c', L=[ρ,σ], R=[μ,ν])
    # where c' has indices relabeled.

    # Simple antisymmetric kernel: k_a h^{ab} h_{b}^c k_c - k_c h^{c}_{b} h^{ab} k_a
    # = k_a k_c h^{ab} h_b^c - k_c k_a h^{cb} h_b^a = 0 (by relabeling a↔c)
    # So this IS zero.

    # But what about: k_a k_b h^{ab} h_trace - k_a k_b h_trace h^{ab}?
    # = k_a k_b h^{ab} h_trace - k_a k_b h^{ab} h_trace = 0 ✓

    # The antisymmetric kernel is IDENTICALLY zero when expressed in tensor notation.
    # But in our bilinear decomposition, the "left" and "right" h's have specific
    # index positions (Up/Down), and swapping them may not give zero due to index handling.

    # ─── Critical test: same kernel term with h's swapped ───
    println("\n── Test: same term with h's swapped ──")
    # Term: k_a k_b h^{ab} h_trace
    t_orig = tproduct(1//1, TensorExpr[
        Tensor(:k, [down(:a)]), Tensor(:k, [down(:b)]),
        Tensor(:h, [up(:a), up(:b)]),
        Tensor(:g, [up(:c), up(:d)]), Tensor(:h, [down(:c), down(:d)])])
    # Swapped: k_a k_b h_trace h^{ab} — but with FRESH indices for consistency
    t_swap = tproduct(1//1, TensorExpr[
        Tensor(:k, [down(:e)]), Tensor(:k, [down(:f)]),
        Tensor(:g, [up(:e), up(:f)]),   # h_trace side gets k indices
        Tensor(:h, [down(:c), down(:d)]),  # this is now "left"
        Tensor(:h, [up(:c), up(:d)])])  # this is now "right" (was trace)

    # Hmm, this isn't right. Let me think more carefully.
    # The original term has:
    #   coeff = k_a k_b g^{cd}, left = h^{ab}, right = h_{cd}
    # The swapped term should have:
    #   coeff = k_a k_b g^{cd} (SAME coeff, just left↔right)
    #   left = h_{cd}, right = h^{ab}
    # But the spin projection contracts P^{μν,ρσ} with left=(μν) and right=(ρσ).
    # For the original: P contracts with (Up(a),Up(b)), (Down(c),Down(d))
    # For the swapped: P contracts with (Down(c),Down(d)), (Up(a),Up(b))
    # Since P is symmetric: P^{cd,ab} = P^{ab,cd}. So the result should be the same!

    K_orig = extract_kernel(t_orig, :h; registry=reg)
    K_swap = extract_kernel(t_swap, :h; registry=reg)

    for spin in [:spin2, :spin1, :spin0s, :spin0w]
        v1 = _eval_spin_scalar(spin_project(K_orig, spin; registry=reg), k2)
        v2 = _eval_spin_scalar(spin_project(K_swap, spin; registry=reg), k2)
        println("  $spin: orig=$(round(v1;digits=6)), swap=$(round(v2;digits=6)), diff=$(round(v1-v2;digits=6))")
    end

    println("\n── If orig ≠ swap for any sector, the pipeline breaks antisymmetry! ──")
end
