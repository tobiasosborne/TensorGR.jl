@testset "Display: to_latex" begin
    a = TIndex(:a, Up)
    b = TIndex(:b, Down)
    mu = TIndex(:mu, Up)
    nu = TIndex(:nu, Down)

    # TIndex
    @test to_latex(a) == "^{a}"
    @test to_latex(b) == "_{b}"

    # Tensor - no indices
    T0 = Tensor(:phi, TIndex[])
    @test to_latex(T0) == "phi"

    # Tensor - mixed indices, grouped by position
    T1 = Tensor(:T, [a, b])
    @test to_latex(T1) == "T^{a}_{b}"

    # Tensor - only up indices
    T2 = Tensor(:g, [TIndex(:a, Up), TIndex(:b, Up)])
    @test to_latex(T2) == "g^{a b}"

    # Tensor - only down indices
    T3 = Tensor(:g, [TIndex(:a, Down), TIndex(:b, Down)])
    @test to_latex(T3) == "g_{a b}"

    # TScalar
    @test to_latex(TScalar(42)) == "42"

    # TProduct - unit coefficient
    p1 = TProduct(1//1, [T1])
    @test to_latex(p1) == "T^{a}_{b}"

    # TProduct - minus one coefficient
    p2 = TProduct(-1//1, [T1])
    @test to_latex(p2) == "-T^{a}_{b}"

    # TProduct - integer coefficient
    p3 = TProduct(3//1, [T1])
    @test to_latex(p3) == "3 T^{a}_{b}"

    # TProduct - rational coefficient with frac
    p4 = TProduct(1//2, [T1])
    @test to_latex(p4) == "\\frac{1}{2} T^{a}_{b}"

    # TProduct - negative rational coefficient
    p5 = TProduct(-3//4, [T1])
    @test to_latex(p5) == "-\\frac{3}{4} T^{a}_{b}"

    # TProduct - no factors (pure scalar)
    p6 = TProduct(5//1, TensorExpr[])
    @test to_latex(p6) == "5"

    # TProduct - multiple factors
    A = Tensor(:A, [a])
    B = Tensor(:B, [b])
    p7 = TProduct(1//1, [A, B])
    @test to_latex(p7) == "A^{a} B_{b}"

    # TSum - simple sum
    s1 = TSum([TProduct(1//1, [A]), TProduct(1//1, [B])])
    @test to_latex(s1) == "A^{a} + B_{b}"

    # TSum - subtraction (negative coefficient)
    s2 = TSum([TProduct(1//1, [A]), TProduct(-1//1, [B])])
    @test to_latex(s2) == "A^{a} - B_{b}"

    # TSum - empty
    s3 = TSum(TensorExpr[])
    @test to_latex(s3) == "0"

    # TDeriv
    d1 = TDeriv(nu, T1)
    @test to_latex(d1) == "\\partial_{nu} T^{a}_{b}"
end

@testset "Display: to_unicode" begin
    a = TIndex(:a, Up)
    b = TIndex(:b, Down)

    # TIndex - letter indices use ^/_ notation
    @test to_unicode(a) == "^a"
    @test to_unicode(b) == "_b"

    # TIndex - numeric indices use Unicode super/subscripts
    i1 = TIndex(Symbol("0"), Up)
    i2 = TIndex(Symbol("1"), Down)
    @test to_unicode(i1) == "\u2070"       # superscript 0
    @test to_unicode(i2) == "\u2081"       # subscript 1

    # Multi-digit numeric index
    i3 = TIndex(Symbol("23"), Down)
    @test to_unicode(i3) == "\u2082\u2083" # subscript 23

    # Tensor - no indices
    T0 = Tensor(:phi, TIndex[])
    @test to_unicode(T0) == "phi"

    # Tensor - mixed indices in order
    T1 = Tensor(:T, [a, b])
    @test to_unicode(T1) == "T^a_b"

    # TScalar
    @test to_unicode(TScalar(7)) == "7"

    # TProduct - unit coefficient
    p1 = TProduct(1//1, [T1])
    @test to_unicode(p1) == "T^a_b"

    # TProduct - negative
    p2 = TProduct(-1//1, [T1])
    @test to_unicode(p2) == "-T^a_b"

    # TProduct - rational coefficient
    p3 = TProduct(2//3, [T1])
    @test to_unicode(p3) == "(2//3) T^a_b"

    # TSum - with subtraction
    A = Tensor(:A, [a])
    B = Tensor(:B, [b])
    s1 = TSum([TProduct(1//1, [A]), TProduct(-1//1, [B])])
    @test to_unicode(s1) == "A^a - B_b"

    # TDeriv
    d1 = TDeriv(b, T1)
    @test to_unicode(d1) == "\u2202_b(T^a_b)"  # partial with subscript b
end
