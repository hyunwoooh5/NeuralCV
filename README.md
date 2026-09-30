# Neural Control Variates for Lattice Field Theory

This repository contains lattice models, Monte Carlo configuration generators,
neural control-variate training scripts, and analysis utilities. The Python and
C++ samplers are separate tools with different output formats; see
[Configuration formats](#configuration-formats) before connecting a sampler to
a training script.

## Repository Layout

- `src/models/`: scalar, gauge, transmon, and Thirring model definitions.
- `src/mc/`: Python Metropolis, HMC, replica-exchange, and U(1) heat-bath
  samplers, plus a scalar cluster sampler.
- `src/mc/cpp/`: standalone C++ samplers and their Makefile.
- `src/cv_*.py`: control-variate model definitions and training command-line
	programs for scalar, gauge, and contour-deformation workflows.
- `src/gevp_utils.py`, `src/fitting.py`, `src/util.py`, and
	`src/util_pytree.py`: correlator/GEVP analysis, fitting, statistics, and JAX
	PyTree helpers.
- `pub/`: publication code and notebooks. This directory is maintained
	separately from the installable `src/` package.

## Setup

Use a virtual environment, then install the pinned dependencies and this
repository in editable mode:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Editable installation exposes the `src/` modules under their existing import
names, including `models`, `mc`, and `util`. Python sampler and training
commands below are run from the repository root.

## Model Files

Several Python command-line tools take a model-expression file. For example,
save this expression as `data/model.dat` for the scalar model:

```python
scalar.Model(
    geom=(4, 4),
    m2=0.01,
    lamda=0.01,
)
```

The command loads the expression with Python `eval`; use only trusted model
files. The expression must name a model class imported by that command and
provide the arguments expected by its constructor.

## Python Workflow

`src/mc/sample.py` generates configurations with a model that supplies an
action, a one-argument observable, and any optional local action required by
the selected sampler. Its positional arguments are the model-expression file
and output configuration file. This command form assumes a model compatible
with that interface; the scalar model expression above is a constructor
example, not a promise that every sampler/trainer accepts that model:

```sh
mkdir -p data
python src/mc/sample.py data/model.dat data/configs.pkl \
    --samples 2000 --skip 100 --seed 1
```

The sampler supports global Metropolis by default, with `--local`, `--hmc`, or
`--replica` selecting other chain modes. Configuration output is a pickle of a
JAX array and is checkpointed every 1,000 samples.

The `src/cv_*.py` programs take three positional inputs: a model-expression
file, a control-variate output path, and a configuration file. A scalar
training invocation has this form; supply model and configuration files whose
observable and data formats match the selected trainer:

```sh
python src/cv_scalar.py data/compatible_model.dat data/control_variate.pkl \
	data/compatible_configs.pkl --init --layers 1 --width 8 --learningrate 1e-3
```

Training runs continuously and prints periodic diagnostics; stop it with
Ctrl-C. Existing options include model initialization/loading, architecture
size, optimizer and learning-rate settings, train/test sample counts, and
regularization. Use `--help` for a script's full option list. Select a model
and training script whose action and observable interfaces match; the scripts
cover distinct research workflows and are not interchangeable for every model.

## Configuration Formats

- `src/mc/sample.py` writes a Python pickle containing its collected
	configurations. The `src/cv_*.py` training scripts load their configuration
	input with `pickle`.
- `src/mc/heatbath_u1.py` and `src/mc/scalar_brower-tamayo.py` write NumPy `.npy`
	arrays.
- The C++ programs in `src/mc/cpp/` write a custom binary format: two native
	integers (`dof` and sample count), followed by configuration doubles in
	Eigen column-major order. This is not a pickle or a `.npy` file and cannot be
	passed directly to a Python training script expecting a pickle.

## C++ Samplers

Build an individual executable with its Makefile target. For example, the
samplers share RNG, acceptance, calibration, thermalization, sample collection,
and binary output helpers in the header-only `monte_carlo.hpp`; each source
keeps its own model action and proposal. The periodic 2D scalar sampler takes
`nt`, `nx`, mass-squared, quartic coupling, decorrelation sweeps, sample count,
and output path:

```sh
brew install cli11 eigen
make -C src/mc/cpp sample_scalar_2d \
	EIGEN_INCLUDE="$(brew --prefix eigen)/include/eigen3" \
	CLI11_INCLUDE="$(brew --prefix cli11)/include"
src/mc/cpp/sample_scalar_2d --nt 4 --nx 4 --mass-squared 0.01 --lambda 0.01 \
	--decorrelation-sweeps 100 --thermalization-sweeps 10000 --samples 2000 \
	--output data/scalar2d.bin --seed 12345
```

`EIGEN_INCLUDE` defaults to `$HOME/eigen-3.3.9`; `CLI11_INCLUDE` defaults to
`$HOME/local/include` and must contain `CLI/CLI.hpp`.
Options can appear in any order; all model inputs and `--output` are required,
while `--seed` defaults to `42` and `--thermalization-sweeps` defaults to
`10000`. SU(2) samplers accept an optional `--decorrelation-sweeps` value,
defaulting to `1`. Every executable supports `--help`. Other executables cover scalar fields in 3D/4D, open- and
defaulting to `1`. Every executable supports `--help`. Other executables cover
scalar fields in 3D/4D, open- and periodic-boundary U(1), open-boundary SU(2),
and the transmon model; use their `--help` output for model-specific options.

## Papers

If you use this code or a derivative of it, please consider citing one or more
of the following papers:

- [Leveraging neural control variates for enhanced precision in lattice field theory](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.109.094519)
- [Control variates with neural networks](https://arxiv.org/abs/2501.14614)
- [Training neural control variates using correlated configurations](https://arxiv.org/abs/2505.07719)
