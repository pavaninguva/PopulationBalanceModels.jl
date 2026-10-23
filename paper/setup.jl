using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=dirname(@__DIR__))
Pkg.instantiate()
println("Paper environment uses checkout ", dirname(@__DIR__))
