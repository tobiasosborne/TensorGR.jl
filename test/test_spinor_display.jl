# Tests for spinor display with dotted (primed) notation.
# Penrose & Rindler Vol 1, Section 2.5: primed/dotted index conventions.

@testset "Spinor display: show" begin
    # Undotted indices display normally (no vbundle annotation)
    @test sprint(show, spin_up(:A)) == "A"
    @test sprint(show, spin_down(:A)) == "-A"
    @test sprint(show, spin_down(:B)) == "-B"

    # Dotted indices: :Ap displayed as A' (strip trailing 'p', add prime)
    @test sprint(show, spin_dot_up(:Ap)) == "A'"
    @test sprint(show, spin_dot_down(:Ap)) == "-A'"
    @test sprint(show, spin_dot_down(:Bp)) == "-B'"

    # Tensor with undotted spinor indices
    psi = Tensor(:psi, [spin_down(:A)])
    @test sprint(show, psi) == "psi[-A]"

    # Tensor with dotted spinor index
    chi = Tensor(:chi, [spin_dot_down(:Ap)])
    @test sprint(show, chi) == "chi[-A']"

    # Mixed tensor: spacetime + undotted + dotted spinor indices
    sigma = Tensor(:sigma, [up(:a), spin_down(:A), spin_dot_down(:Ap)])
    @test sprint(show, sigma) == "sigma[a, -A, -A']"
end

@testset "Spinor display: to_latex" begin
    # Undotted spinor index (SL2C) -- same as regular indices
    @test to_latex(spin_up(:A)) == "^{A}"
    @test to_latex(spin_down(:A)) == "_{A}"

    # Dotted spinor index (SL2C_dot) -- \dot{} notation
    @test to_latex(spin_dot_up(:Ap)) == "^{\\dot{A}}"
    @test to_latex(spin_dot_down(:Ap)) == "_{\\dot{A}}"
    @test to_latex(spin_dot_down(:Bp)) == "_{\\dot{B}}"

    # Tensor with only undotted spinor indices
    psi = Tensor(:psi, [spin_up(:A)])
    @test to_latex(psi) == "psi^{A}"

    # Tensor with dotted spinor index
    phi = Tensor(:phi, [spin_dot_down(:Ap)])
    @test to_latex(phi) == "phi_{\\dot{A}}"

    # Mixed tensor: T^a_{A \dot{A}}
    T = Tensor(:T, [up(:a), spin_down(:A), spin_dot_down(:Ap)])
    @test to_latex(T) == "T^{a}_{A \\dot{A}}"

    # sigma^a_{A A'} -- Infeld-van der Waerden symbol
    sigma = Tensor(:sigma, [up(:a), spin_down(:A), spin_dot_down(:Ap)])
    @test to_latex(sigma) == "sigma^{a}_{A \\dot{A}}"

    # Tensor with multiple dotted indices (both up and down)
    W = Tensor(:W, [spin_dot_up(:Ap), spin_dot_down(:Bp)])
    @test to_latex(W) == "W^{\\dot{A}}_{\\dot{B}}"
end

@testset "Spinor display: to_unicode" begin
    # Undotted spinor indices -- normal display
    @test to_unicode(spin_up(:A)) == "^A"
    @test to_unicode(spin_down(:A)) == "_A"

    # Dotted spinor indices -- prime notation A'
    @test to_unicode(spin_dot_up(:Ap)) == "^A'"
    @test to_unicode(spin_dot_down(:Ap)) == "_A'"
    @test to_unicode(spin_dot_down(:Bp)) == "_B'"

    # Tensor with mixed indices
    sigma = Tensor(:sigma, [up(:a), spin_down(:A), spin_dot_down(:Ap)])
    @test to_unicode(sigma) == "sigma^a_A_A'"

    # Tensor with only dotted indices
    chi = Tensor(:chi, [spin_dot_up(:Ap), spin_dot_down(:Bp)])
    @test to_unicode(chi) == "chi^A'_B'"
end

@testset "Spinor display: _spinor_base_name edge cases" begin
    # Single character -- no stripping
    @test TensorGR._spinor_base_name(:A) == "A"
    @test TensorGR._spinor_base_name(:B) == "B"

    # Two characters ending in 'p' -- strip
    @test TensorGR._spinor_base_name(:Ap) == "A"
    @test TensorGR._spinor_base_name(:Bp) == "B"

    # Extended names
    @test TensorGR._spinor_base_name(:Ap1) == "Ap1"  # only strip trailing 'p'
end

@testset "Spinor display: derivatives with spinor indices" begin
    # Partial derivative with undotted spinor index
    psi = Tensor(:psi, [spin_up(:B)])
    d1 = TDeriv(spin_down(:A), psi)
    @test sprint(show, d1) == "∂[-A](psi[B])"
    @test to_latex(d1) == "\\partial_{A} psi^{B}"

    # Partial derivative with dotted spinor index
    d2 = TDeriv(spin_dot_down(:Ap), psi)
    @test sprint(show, d2) == "∂[-A'](psi[B])"
    @test to_latex(d2) == "\\partial_{\\dot{A}} psi^{B}"
    @test to_unicode(d2) == "∂_A'(psi^B)"
end
