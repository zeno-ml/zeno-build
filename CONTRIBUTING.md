# Contributing to zeno-build

If you're reading this, you're probably interested in contributing to
`zeno-build`. Thank you for your interest!

## Developer Installation

If you're a developer, you should install pre-commit hooks before developing.

```bash
pip install .[dev]
pre-commit install
```

These will do things like ensuring code formatting, linting, and type checking.

You'll also want to run tests to make sure your code is working by running the
following:

```bash
pytest
```

## Types of Contributions

We welcome:

- **New examples**, which you can create by navigating to the [examples/]
  directory and adding a new sub-directory implementing a new example
  (see the examples directory for details).
- **New reports**, which you can create by modifying the configuration and/or
  modeling files under [examples/] and sending us the a link of the resulting
  report.
- **New core functionality**, making `zeno-build` more feature-rich or easier to
  use.

## Contribution Guide

If you want to make a contribution you can:

1. Browse existing issues and select one to work on.
2. Create a new issue to discuss a feature that you might want to contribute
3. Send a PR directly

We'd recommend the first two to increase the chances of your PR being accepted,
but if you're confident in your contribution, you can go ahead and send a PR
directly.

## Making a Release

If you are an admin of the repository, you can make a new release of the
library.

We are using the [hatchling](https://github.com/pypa/hatch) build system, which
makes it easy to make new library releases. In order to do so, just create a new
version tag on github (it has to be a valid [semantic
version](https://semver.org/)) and the CI will automatically build and publish
the new version to PyPI.
