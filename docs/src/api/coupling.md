# Coupling and accounting

`PBMSystem` combines named populations and shared bulk balances. Prefer named
population, bulk and cumulative accessors to packed-state offsets. Instantaneous
accounting rates, historical loss ledgers and physical reservoirs have distinct
meanings.

See [Coupled systems](../coupled_systems.md) for callback, exchange and tensor
contracts, and [Axially dispersed reactors](../axial_reactors.md) for physical
transport. Axial domain declarations and compiled adapters also appear under
[Models](models.md) and [Compilation](numerics.md).

```@autodocs
Modules = [PopulationBalanceModels]
Pages = Main.API_REFERENCE_SOURCES[:coupling]
Public = true
Private = false
Order = [:constant, :type, :function, :macro]
```
