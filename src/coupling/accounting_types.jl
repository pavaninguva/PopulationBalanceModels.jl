"""
    CumulativeAccounting(population; properties=(material=3,))

Append historical removal states to a PBMSystem. Each mechanism has a particle
number ledger and the declared raw-power properties (e.g. L³ per process volume).
Multiply material by the same density/shape factor as the bulk exchange to obtain
material concentration. Ledgers have no subsequent kinetics and are not advected.
Collision conservation requires the property to be proportional to the collision's
additive coordinate. Defaults without accounting retain their original behavior.
"""
struct CumulativeAccounting{P}
    population::Symbol
    properties::P
    function CumulativeAccounting(population::Symbol;properties=(material=3,))
        properties isa NamedTuple && all(k->k isa Integer && k>=1,values(properties)) ||
            throw(ArgumentError("accounted properties must be named positive integer powers"))
        :number in keys(properties) && throw(ArgumentError("number is already accounted"))
        new{typeof(properties)}(population,properties)
    end
end
const _LOSS_MECHANISMS=(:extinction,:lower_escape,:upper_escape,
    :aggregation_underflow,:aggregation_overflow,:breakage_underflow,:breakage_overflow,
    :nucleation_underflow,:nucleation_overflow)
