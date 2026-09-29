function setup()
    reg = TensorRegistry()
    register_manifold!(reg, ManifoldProperties(:M4, 4, :g, :dM4, [:a,:b,:c,:d,:e,:f]))
    register_tensor!(reg, TensorProperties(name=:g, manifold=:M4, rank=(0,2),
        symmetries=SymmetrySpec[Symmetric(1,2)],
        options=Dict{Symbol,Any}(:is_metric => true)))
    register_tensor!(reg, TensorProperties(name=:dM4, manifold=:M4, rank=(1,1),
        options=Dict{Symbol,Any}(:is_delta => true)))
    define_curvature_tensors!(reg, :M4, :g)
    define_covd!(reg, :D; manifold=:M4, metric=:g)
    register_tensor!(reg, TensorProperties(name=:S, manifold=:M4, rank=(0,2),
        symmetries=SymmetrySpec[Symmetric(1,2)]))
    register_tensor!(reg, TensorProperties(name=:A, manifold=:M4, rank=(0,2),
        symmetries=SymmetrySpec[AntiSymmetric(1,2)]))
    register_tensor!(reg, TensorProperties(name=:V, manifold=:M4, rank=(1,0)))
    reg
end
DaDbS_(S) = TDeriv(down(:a), TDeriv(down(:b), S, :D), :D)
