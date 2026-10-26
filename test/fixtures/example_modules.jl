# Reuse an example module across test files, including standalone invocations.
# This avoids replacing modules and recompiling the same builders in one suite.
if !isdefined(@__MODULE__, :_load_example)
    function _load_example(name::Symbol, relative_path::AbstractString)
        scope = @__MODULE__
        if !isdefined(scope, name)
            Base.include(scope, normpath(joinpath(@__DIR__, "..", "..", "examples", relative_path)))
        end
        # include may have just created this global binding in a newer world.
        result = Base.invokelatest(getfield, scope, name)
        result isa Module || error("Example $relative_path must define module $name")
        result
    end
end
