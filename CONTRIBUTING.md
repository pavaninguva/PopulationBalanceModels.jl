# Contributing

Read the [developer guide](docs/src/development.md) for source organization,
extension contracts, verification, and documentation conventions. The scientific
manual lives in `docs/src/`; edit those pages directly. Preserve explicit units,
conservation assumptions, and qualifications when changing examples.

Run `julia --project=. -e 'using Pkg; Pkg.test()'` and build the documentation
before proposing a change. Optional integration checks use a separate environment,
as described in the guide. Follow the existing author attribution and MIT license.

Review `git status --short` and `git diff`, run the checks relevant to the change,
and include source, tests and documentation together in a focused pull request.
Use isolated environments for optional dependencies; the [verification guide](docs/src/verification.md)
provides commands. Generated builds and local paper PDFs are not source files.
Release and GitHub Pages instructions are in the [publishing guide](docs/src/publishing.md).
