# Static checks for reviews/07_julia_best_practices_review.md (Aqua.jl + JET.jl).
# Uses a temporary environment; does not touch the project's Manifest.
#
#     julia reviews/07_probes/aqua_jet.jl            # ~2-3 min
#
using Pkg
Pkg.activate(; temp=true)
Pkg.develop(path=joinpath(@__DIR__, "..", ".."); io=devnull)
Pkg.add(["Aqua", "JET"]; io=devnull)
using TensorGR, Aqua, JET, Test

@testset "Aqua" begin
    @testset "ambiguities" Aqua.test_ambiguities(TensorGR)
    @testset "unbound args" Aqua.test_unbound_args(TensorGR)
    @testset "undefined exports" Aqua.test_undefined_exports(TensorGR)
    @testset "piracy" Aqua.test_piracies(TensorGR)
    @testset "stale deps" Aqua.test_stale_deps(TensorGR)
    @testset "deps compat" Aqua.test_deps_compat(TensorGR)
    @testset "persistent tasks" Aqua.test_persistent_tasks(TensorGR)
end

r = JET.report_package(TensorGR; target_modules=(TensorGR,), toplevel_logger=nothing)
println("JET reports: ", length(JET.get_reports(r)))
out = joinpath(@__DIR__, "jet_report.txt")
open(io -> show(io, MIME"text/plain"(), r), out, "w")
println("Full JET report written to ", out)
