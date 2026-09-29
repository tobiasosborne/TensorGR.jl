using TensorGR
include(joinpath(@__DIR__, "..", "07_probes", "setup.jl"))
reg = setup()
with_registry(reg) do
    R(i...) = Tensor(:Riem, collect(TIndex, i))
    # Riem^3 cubic invariant R_ab^cd R_cd^ef R_ef^ab  (3 identical factors, 12 slots)
    cub = TProduct(1//1, TensorExpr[R(down(:a),down(:b),up(:c),up(:d)), R(down(:c),down(:d),up(:e),up(:f)), R(down(:e),down(:f),up(:a),up(:b))])
    # 4 Riemanns, 16 slots
    q = TProduct(1//1, TensorExpr[R(down(:a),down(:b),up(:c),up(:d)), R(down(:c),down(:d),up(:e),up(:f)), R(down(:e),down(:f),up(:g),up(:h)), R(down(:g),down(:h),up(:a),up(:b))])
    for (lbl, x) in [("Riem^3 (n=12)", cub), ("Riem^4 (n=16)", q)]
        canonicalize(x); N = 2000
        t = @elapsed for _ in 1:N; canonicalize(x); end
        a = @allocated canonicalize(x)
        println(rpad(lbl, 16), "  ", round(t/N*1e6, digits=1), " µs/call   ", a, " bytes allocated/call")
    end
end
