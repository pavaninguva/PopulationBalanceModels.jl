function _context(op,N,c,t,system=nothing)
    localfields=op.fields(c,t)
    system===nothing && return (grid=op.grid,operator=op,N=view(N,:,c),cell=c,fields=localfields,
        bulk=NamedTuple(),populations=NamedTuple(),system=nothing)
    (grid=op.grid,operator=op,N=view(N,:,c),cell=c,fields=merge(system.fields,localfields),
        bulk=system.bulk,populations=system.populations,system=system)
end
