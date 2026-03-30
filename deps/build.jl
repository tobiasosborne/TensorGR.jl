const srcdir = @__DIR__
const xperm_src = joinpath(srcdir, "xperm.c")

# Determine shared library extension per platform
const dlext = Sys.isapple() ? "dylib" : Sys.iswindows() ? "dll" : "so"
const libname = joinpath(srcdir, "libxperm." * dlext)

if !isfile(xperm_src)
    @warn "xperm.c not found in $srcdir — canonicalization will not be available"
    return
end

# Find a C compiler
compiler = nothing
for cc in ["gcc", "cc", "clang"]
    try
        run(pipeline(`$cc --version`; stdout=devnull, stderr=devnull))
        compiler = cc
        break
    catch
    end
end

if compiler === nothing
    @warn "No C compiler found (tried gcc, cc, clang). " *
          "Install gcc or clang to enable xperm index canonicalization."
    return
end

try
    run(`$compiler -shared -fPIC -O2 -o $libname $xperm_src`)
    @info "Built $libname using $compiler"
catch e
    @warn "Failed to build xperm.c" exception=e
end
