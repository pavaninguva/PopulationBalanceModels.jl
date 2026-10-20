# Contributing

Read the [developer guide](docs/src/development.md) for source organization,
extension contracts, verification, and documentation conventions. The scientific
manual lives in `docs/src/`; edit those pages directly. Preserve explicit units,
conservation assumptions, and qualifications when changing examples.

Run `julia --project=. -e 'using Pkg; Pkg.test()'` and build the documentation
before proposing a change. Optional integration checks use a separate environment,
as described in the guide. Follow the existing author attribution and MIT license.
