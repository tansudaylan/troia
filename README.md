# troia

## Introduction
Troia is a pipeline to search for and characterize compact objects with stellar companions.

## Installation

Clone the repository and install the package in editable mode:

```
git clone https://github.com/<user>/troia.git
cd troia
pip install -e .
```

The pipeline relies on several optional science packages.  Refer to the
documentation of `tdpy`, `miletos` and related projects for detailed
dependencies.

## Usage

Example configurations are provided in the `examples` module.  To run a
configuration from the command line, use:

```
python -m examples.examples <configuration>
```

For instance, to execute the `cnfg_prev` example:

```
python -m examples.examples cnfg_prev
```

Each example function can be inspected to see the parameters passed to
`troia.init`.

