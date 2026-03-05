const srcdir = @__DIR__
const xperm_src = joinpath(srcdir, "xperm.c")

# Determine shared library extension per platform
const dlext = Sys.isapple() ? "dylib" : Sys.iswindows() ? "dll" : "so"
const libname = joinpath(srcdir, "libxperm." * dlext)

if !isfile(xperm_src)
    error("xperm.c not found in $srcdir")
end

run(`gcc -shared -fPIC -O2 -o $libname $xperm_src`)

@info "Built $libname"
