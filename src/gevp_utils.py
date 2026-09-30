"""Stable GEVP analysis utilities for transmon correlators."""

import jax
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_enable_x64", True)


DEFAULT_CORRELATORS = ["observe", "observe_square", "observe_cubic", "observe_o1o3"]
DEFAULT_OPERATORS = DEFAULT_CORRELATORS


def _as_batch_conf(conf):
    """Convert configurations to a two-dimensional float64 JAX array."""
    conf = jnp.asarray(conf, dtype=jnp.float64)
    if conf.ndim == 1:
        conf = conf[None, :]
    if conf.ndim != 2:
        raise ValueError("conf must have shape (n_configurations, n_sites)")
    return conf


def _evaluate_operator_series_batch(model, conf, name):
    """Evaluate one named transmon observable for a batch using JAX ``vmap``."""
    conf = jnp.asarray(conf)
    if name == "observe_square":
        av = jnp.mean(jnp.sin(conf) ** 2, axis=1)
        values = jax.vmap(model.observe_square)(conf, av)
    else:
        values = jax.vmap(getattr(model, name))(conf)
    return values


def _operator_samples(model, conf, operators, square_average=None):
    """Evaluate channels for ``t=0..Nt//2`` with shape ``(config, channel, time)``."""
    if square_average is None:
        square_average = jnp.mean(jax.vmap(model.observe)(conf[-100000:])[:, 0])
    return jnp.stack(
        [(
            jax.vmap(lambda item: model.observe_square(item, square_average))(conf)
            if name == "observe_square"
            else jax.vmap(getattr(model, name))(conf)
        )[:, :model.Nt // 2 + 1] for name in operators],
        axis=1,
    )


def calculate_correlator_samples(model, conf, correlators=None, chunk_size=1024):
    """Calculate reusable per-configuration correlator channels.

    Each model method already returns one correlator ``C_ij(t)``. The returned
    cache has shape ``(n_configurations, n_channels, n_times)`` and is not
    multiplied by another channel at ``t=0``.
    ``chunk_size`` bounds device memory during the initial calculation.
    """
    correlators = DEFAULT_CORRELATORS if correlators is None else list(correlators)
    conf = _as_batch_conf(conf)
    square_average = jnp.mean(jax.vmap(model.observe)(conf[-100000:])[:, 0])
    chunks = []
    for start in range(0, len(conf), chunk_size):
        chunks.append(np.asarray(_operator_samples(
            model, conf[start:start + chunk_size], correlators, square_average,
        )))
    if not chunks:
        raise ValueError("At least one configuration is required")
    return {"values": np.concatenate(chunks, axis=0).astype(np.float64),
            "correlators": np.asarray(correlators, dtype=str),
            "operators": np.asarray(correlators, dtype=str),
            "n_configurations": len(conf), "n_times": chunks[0].shape[-1]}


def calculate_correlator_samples_with_control_variates(
    model, conf, cv_path, correlators=None, cv_correlator="observe_square",
    chunk_size=1024,
):
    """Calculate direct correlator channels using saved Stein control variates.

    ``cv_path`` must contain the ``params_list`` produced by the transmon
    control-variate training workflow. Only the time slices represented by
    that list are returned.
    """
    import pickle
    import sys
    from itertools import product
    import flax.linen as nn

    correlators = DEFAULT_CORRELATORS if correlators is None else list(correlators)
    if correlators != DEFAULT_CORRELATORS or cv_correlator not in correlators:
        raise ValueError("Control-variate caches require the four default correlators")
    conf = _as_batch_conf(conf)
    main_module = sys.modules["__main__"]
    main_module.MLP = getattr(main_module, "MLP", type("MLP", (), {}))
    main_module.CV_MLP = getattr(main_module, "CV_MLP", type("CV_MLP", (), {}))
    if isinstance(cv_path, dict):
        cv_paths = cv_path
    else:
        cv_paths = {cv_correlator: cv_path}

    class MLP(nn.Module):
        volume: int
        features: tuple

        @nn.compact
        def __call__(self, values):
            for features in self.features:
                values = nn.Dense(features, use_bias=False)(values)
                values = jnp.arcsinh(values)
            return nn.Dense(1, use_bias=False)(values)

    class CV_MLP(nn.Module):
        volume: int
        features: tuple

        @nn.compact
        def __call__(self, values):
            values = MLP(self.volume, self.features)(values)
            bias = self.param("bias", nn.initializers.zeros, (1,))
            return values, bias

    volume = model.dof
    cv_model = CV_MLP(volume, (32, 32))
    index = jnp.asarray([
        -jnp.asarray(i) for i in product(*[range(size) for size in model.shape])
    ])
    dS = jax.grad(lambda values: model.action(values).real)
    square_average = jnp.mean(jax.vmap(model.observe)(conf[-100000:])[:, 0])
    n_times = model.Nt // 2 + 1

    params_by_correlator = {}
    for name in correlators:
        if name not in cv_paths:
            raise ValueError(f"Missing control-variate pickle for {name}")
        with open(cv_paths[name], "rb") as handle:
            _, params_list = pickle.load(handle)
        if len(params_list) < n_times:
            raise ValueError(
                f"Control-variate parameter list for {name} has {len(params_list)} slices; "
                f"expected at least {n_times} for Nt={model.Nt}"
            )
        params_by_correlator[name] = params_list[:n_times]

    def control_variate(values, params):
        def g(single):
            def translated(shift):
                rolled = jnp.roll(single.reshape(model.shape), shift, axis=(0,))
                return cv_model.apply(params, rolled.reshape(volume))[0]
            return jnp.ravel(jax.vmap(translated)(index).T)

        def one(single):
            def diagonal(shift):
                rolled = jnp.roll(single.reshape(model.shape), shift, axis=(0,)).reshape(volume)
                direction = jnp.zeros_like(single).at[0].set(1.0)
                _, jvp_value = jax.jvp(
                    lambda item: cv_model.apply(params, item)[0],
                    (rolled,),
                    (direction,),
                )
                return jvp_value[0]
            return jax.vmap(diagonal)(index).sum() - g(single) @ dS(single)

        return jax.vmap(one)(values)

    chunks = []
    for start in range(0, len(conf), chunk_size):
        values = conf[start:start + chunk_size]
        raw = _operator_samples(model, values, correlators, square_average)
        for channel_index, name in enumerate(correlators):
            control_variates = jnp.stack([
                control_variate(values, params)
                for params in params_by_correlator[name]
            ], axis=1)
            raw = raw.at[:, channel_index, :].add(-control_variates)
        chunks.append(np.asarray(raw))
    if not chunks:
        raise ValueError("At least one configuration is required")
    values = np.concatenate(chunks, axis=0).astype(np.float64)
    return {
        "values": values,
        "correlators": np.asarray(correlators, dtype=str),
        "operators": np.asarray(correlators, dtype=str),
        "n_configurations": len(conf),
        "n_times": values.shape[-1],
    }


def calculate_operator_samples(model, conf, operators=None, chunk_size=1024):
    """Backward-compatible alias for ``calculate_correlator_samples``."""
    return calculate_correlator_samples(model, conf, operators, chunk_size)


def save_operator_samples(path, operator_samples):
    """Save a correlator-channel cache as a compressed NumPy archive."""
    cache = _normalise_operator_samples(operator_samples)
    np.savez_compressed(
        path,
        format="direct-correlators-v2",
        values=cache["values"],
        correlators=cache["correlators"],
        operators=cache["correlators"],
        n_configurations=cache["values"].shape[0],
        n_times=cache["values"].shape[2],
    )


def load_operator_samples(path):
    """Load and validate a correlator-channel cache saved by ``save_operator_samples``."""
    with np.load(path, allow_pickle=False) as archive:
        if "format" not in archive or str(archive["format"]) != "direct-correlators-v2":
            raise ValueError("Cache uses an old correlator format; recalculate direct correlators")
        names = archive["correlators"] if "correlators" in archive else archive["operators"]
        cache = {"values": archive["values"], "correlators": names}
    return _normalise_operator_samples(cache)


save_correlator_samples = save_operator_samples
load_correlator_samples = load_operator_samples


def _normalise_operator_samples(operator_samples):
    """Validate a direct correlator-channel cache."""
    if isinstance(operator_samples, dict):
        values = operator_samples["values"]
        operators = operator_samples.get("correlators", operator_samples.get("operators", DEFAULT_CORRELATORS))
    else:
        values = operator_samples
        operators = DEFAULT_OPERATORS
    values = np.asarray(values, dtype=np.float64)
    operators = np.asarray(operators, dtype=str)
    if values.ndim != 3:
        raise ValueError("correlator cache values must have shape (n_configurations, n_channels, n_times)")
    if len(operators) != values.shape[1]:
        raise ValueError("correlator cache names do not match the channel dimension")
    if not np.isfinite(values).all():
        raise ValueError("operator cache contains non-finite values")
    if values.shape[1] != 4 or list(operators) != DEFAULT_CORRELATORS:
        raise ValueError("Expected the four direct channels: observe, observe_square, observe_cubic, observe_o1o3")
    return {"values": values, "correlators": operators, "operators": operators}


def _correlator_matrix_from_operator_values(op_values, use_tmax=None):
    """Map direct channels to per-configuration GEVP matrices without products."""
    values = jnp.asarray(op_values, dtype=jnp.float64)
    if values.ndim != 3:
        raise ValueError("Expected correlator channels with shape (n_conf, n_channels, n_t)")
    if use_tmax is not None:
        values = values[:, :, :use_tmax]
    if values.shape[1] < 4:
        raise ValueError("Expected channels observe, observe_square, observe_cubic, observe_o1o3")
    n_conf, _, n_times = values.shape
    corr = jnp.zeros((n_conf, 3, 3, n_times), dtype=jnp.float64)
    corr = corr.at[:, 0, 0, :].set(values[:, 0, :])
    corr = corr.at[:, 1, 1, :].set(values[:, 1, :])
    corr = corr.at[:, 2, 2, :].set(values[:, 2, :])
    corr = corr.at[:, 0, 2, :].set(values[:, 3, :])
    corr = corr.at[:, 2, 0, :].set(values[:, 3, :])
    return corr


def _correlator_sum_in_chunks(model, conf, operators, use_tmax=None,
                              chunk_size=1024, block_labels=None, n_blocks=0):
    """Accumulate correlator sums in bounded GPU-memory chunks."""
    conf = _as_batch_conf(conf)
    n_samples = len(conf)
    square_average = jnp.mean(jax.vmap(model.observe)(conf[-100000:])[:, 0])
    total = None
    block_sums = [None] * n_blocks
    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        samples = _operator_samples(
            model, conf[start:stop], operators, square_average,
        )
        corr_samples = _correlator_matrix_from_operator_values(samples, use_tmax)
        chunk_sum = np.asarray(jnp.sum(corr_samples, axis=0))
        total = chunk_sum if total is None else total + chunk_sum
        if block_labels is not None:
            for block_index in np.unique(block_labels[start:stop]):
                local = block_labels[start:stop] == block_index
                block_sum = np.asarray(jnp.sum(corr_samples[local], axis=0))
                previous = block_sums[block_index]
                block_sums[block_index] = block_sum if previous is None else previous + block_sum
    return total, block_sums


def correlator_samples_from_operator_samples(operator_samples, use_tmax=None):
    """Build per-configuration GEVP matrices from saved correlator channels.

    The result has shape ``(n_configurations, 3, 3, n_times)``.
    """
    cache = _normalise_operator_samples(operator_samples)
    return np.asarray(_correlator_matrix_from_operator_values(cache["values"], use_tmax))


def correlator_samples_from_correlator_samples(correlator_samples, use_tmax=None):
    """Build direct GEVP matrices from cached correlator channels."""
    return correlator_samples_from_operator_samples(correlator_samples, use_tmax)


def build_correlator_matrix_from_operator_samples(operator_samples, use_tmax=None):
    """Average cached per-configuration correlators over configurations."""
    return np.mean(correlator_samples_from_operator_samples(operator_samples, use_tmax), axis=0)


def _correlator_sums_from_operator_samples(operator_samples, use_tmax=None,
                                           n_blocks=20, chunk_size=1024):
    """Accumulate full and block correlator sums from cached values in chunks."""
    cache = _normalise_operator_samples(operator_samples)
    values = cache["values"]
    n_samples = values.shape[0]
    blocks = np.array_split(np.arange(n_samples), n_blocks)
    labels = np.empty(n_samples, dtype=int)
    for block_index, block in enumerate(blocks):
        labels[block] = block_index
    total = None
    block_sums = [None] * n_blocks
    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        corr_samples = _correlator_matrix_from_operator_values(values[start:stop], use_tmax)
        chunk_sum = np.asarray(jnp.sum(corr_samples, axis=0))
        total = chunk_sum if total is None else total + chunk_sum
        for block_index in np.unique(labels[start:stop]):
            local = labels[start:stop] == block_index
            block_sum = np.asarray(jnp.sum(corr_samples[local], axis=0))
            block_sums[block_index] = (
                block_sum if block_sums[block_index] is None
                else block_sums[block_index] + block_sum
            )
    return total, block_sums, blocks


def build_correlator_matrix_from_observables(model, conf, operators=None, use_tmax=None):
    """Build the configuration-averaged correlator matrix used by the GEVP."""
    operators = DEFAULT_OPERATORS if operators is None else operators
    total, _ = _correlator_sum_in_chunks(model, conf, operators, use_tmax)
    return total / len(_as_batch_conf(conf))


def build_operator_correlator_matrix(model, conf, operators=None, use_tmax=None):
    """Backward-compatible alias for ``build_correlator_matrix_from_observables``."""
    return build_correlator_matrix_from_observables(model, conf, operators, use_tmax)


def _matrix_sqrt_inverse(matrix, rcond=1e-10):
    """Return a truncated symmetric inverse square root for a reference matrix."""
    matrix = 0.5 * (np.asarray(matrix) + np.asarray(matrix).T)
    values, vectors = np.linalg.eigh(matrix)
    scale = max(np.max(np.abs(values)), 1.0)
    keep = values > rcond * scale
    if not np.any(keep):
        raise np.linalg.LinAlgError("C(t0) has no positive eigenvalues")
    return (vectors[:, keep] / np.sqrt(values[keep])) @ vectors[:, keep].T


def solve_gevp(C_matrix, t, t0=1, n_states=1, rcond=1e-10):
    """Solve ``C(t)v=lambda C(t0)v`` with symmetric whitening and PSD regularization."""
    C_matrix = np.asarray(C_matrix, dtype=float)
    if C_matrix.ndim != 3 or C_matrix.shape[0] != C_matrix.shape[1]:
        raise ValueError("C_matrix must have shape (n_ops, n_ops, n_t)")
    whitener = _matrix_sqrt_inverse(C_matrix[:, :, t0], rcond)
    Ct = 0.5 * (C_matrix[:, :, t] + C_matrix[:, :, t].T)
    ct_values, ct_vectors = np.linalg.eigh(Ct)
    ct_floor = max(np.max(np.abs(ct_values)) * rcond, np.finfo(float).eps)
    Ct = (ct_vectors * np.maximum(ct_values, ct_floor)) @ ct_vectors.T
    values, vectors = np.linalg.eigh(whitener @ Ct @ whitener)
    scale = max(np.max(np.abs(values)), np.finfo(float).eps)
    values = np.maximum(np.nan_to_num(values, nan=rcond * scale), rcond * scale)
    indices = np.argsort(values)[::-1][:n_states]
    return values[indices], whitener @ vectors[:, indices]


def gevp_effective_masses(C_matrix, t0=1, t_init=None, n_states=1, t_max=None, rcond=1e-10):
    """Return times, log-ratio effective energies, and principal correlators."""
    if t_init is None:
        t_init = t0
    if t_max is None:
        t_max = np.asarray(C_matrix).shape[-1] - 1
    if t_max <= t_init + 1:
        raise ValueError("Need at least two adjacent times after t_init")
    lambda_times = np.arange(t_init, t_max + 1)
    lambdas = np.asarray([solve_gevp(C_matrix, time, t0, n_states, rcond)[0] for time in lambda_times])
    return lambda_times[1:], np.log(lambdas[:-1] / lambdas[1:]), lambdas


def _jackknife_errors(samples):
    """Compute the mean and standard block-jackknife error of an array of estimates."""
    samples = np.asarray(samples, dtype=float)
    mean = np.mean(samples, axis=0)
    n = samples.shape[0]
    return mean, np.sqrt((n - 1) / n * np.sum((samples - mean) ** 2, axis=0))


def gevp_with_jackknife_from_operator_samples(operator_samples, use_tmax=None,
                                              t0=1, t_init=None, t_max=None,
                                              n_states=3, n_blocks=20,
                                              rcond=1e-10, chunk_size=1024,
                                              conf_slice=None):
    """Compute GEVP energies from a saved operator cache.

    Operator evaluation is skipped entirely. The cache is contracted in
    chunks, then NumPy performs the small per-time eigensolves and jackknife
    reductions. This is the main reusable analysis entry point. ``conf_slice``
    optionally selects configurations before the GEVP, for example
    ``slice(len(operator_samples["values"]) // 2, None)`` for the last half.
    """
    cache = _normalise_operator_samples(operator_samples)
    if conf_slice is not None:
        cache["values"] = cache["values"][conf_slice]
    n_samples = cache["values"].shape[0]
    if n_samples < 2:
        raise ValueError("At least two configurations are required")
    n_blocks = min(max(2, int(n_blocks)), n_samples)
    total_sum, block_sums, blocks = _correlator_sums_from_operator_samples(
        cache, use_tmax, n_blocks, chunk_size)

    def analyze(corr_sum, count):
        corr = corr_sum / count
        return gevp_effective_masses(corr, t0, t_init, n_states, t_max, rcond)

    times, masses, lambdas = analyze(total_sum, n_samples)
    jk_masses, jk_lambdas = [], []
    for block_index, block in enumerate(blocks):
        jk_times, jk_mass, jk_lambda = analyze(
            total_sum - block_sums[block_index], n_samples - len(block))
        if not np.array_equal(times, jk_times):
            raise RuntimeError("Jackknife samples produced different time ranges")
        jk_masses.append(jk_mass)
        jk_lambdas.append(jk_lambda)
    _, masses_err = _jackknife_errors(jk_masses)
    lambda_mean, lambda_err = _jackknife_errors(jk_lambdas)
    return {"times": times, "masses_mean": masses, "masses_err": masses_err,
            "lambdas_mean": lambda_mean, "lambdas_err": lambda_err,
            "lambdas_jk": np.asarray(jk_lambdas), "n_blocks": n_blocks}


def gevp_with_jackknife(model, conf, operators=None, use_tmax=None, t0=1, t_init=None,
                        t_max=None, n_states=3, n_blocks=20, rcond=1e-10,
                        chunk_size=1024):
    """Calculate operator samples once, then run the cached GEVP jackknife."""
    operator_samples = calculate_operator_samples(model, conf, operators, chunk_size)
    return gevp_with_jackknife_from_operator_samples(
        operator_samples, use_tmax, t0, t_init, t_max, n_states, n_blocks,
        rcond, chunk_size)


def run_gevp_from_configurations(model, conf, **kwargs):
    """Run the full jackknife GEVP workflow on already loaded configurations."""
    return gevp_with_jackknife(model, conf, **kwargs)


def _resolve_model_class(model_cls=None):
    """Resolve an explicitly supplied model class or the project transmon model."""
    if model_cls is not None:
        return model_cls
    from models import transmon
    return transmon.Model


def _load_sample(Nt, t, E_C, E_J, data_dir="."):
    """Load one Fortran-order transmon sample file from ``data_dir``."""
    path = f"{data_dir}/sample_1site_Nt{Nt}_t{t}_EC{E_C}_EJ{E_J}.bin"
    with open(path, "rb") as sample:
        m = np.fromfile(sample, dtype=np.int32, count=1)[0]
        n = np.fromfile(sample, dtype=np.int32, count=1)[0]
        return np.fromfile(sample, dtype=np.float64, count=m * n).reshape((m, n), order="F").T


def load_transmon_configurations(Nt, t=1, E_C=1, E_J=50, data_dir="."):
    """Load the project binary sample for one ``Nt`` as ``(configuration, site)``."""
    return _load_sample(Nt, t, E_C, E_J, data_dir)


def run_gevp_for_nt(Nt, t=1, E_C=1, E_J=50, operators=None, t0=1, t_init=5,
                    t_max=None, n_states=3, n_blocks=20, use_tmax=None,
                    model_cls=None, data_dir=".", rcond=1e-10):
    """Load and analyze one lattice size, returning energies and jackknife errors."""
    model = _resolve_model_class(model_cls)(Nt=Nt, t=t, E_C=E_C, E_J=E_J)
    return gevp_with_jackknife(
        model, _load_sample(Nt, t, E_C, E_J, data_dir), operators, use_tmax,
        t0, t_init, t_max, n_states, n_blocks, rcond)


def extract_plateau_mass(res, state_idx=0, fit_window=None):
    """Return a weighted constant fit to one effective-energy plateau in lattice units."""
    times = np.asarray(res["times"])
    values = np.asarray(res["masses_mean"])[:, state_idx]
    errors = np.maximum(np.asarray(res["masses_err"])[:, state_idx], 1e-14)
    if fit_window is None:
        selected = (times >= times[len(times) // 4]) & (times <= times[len(times) // 2])
    elif isinstance(fit_window, slice):
        selected = np.zeros(len(times), dtype=bool)
        selected[fit_window] = True
    else:
        selected = np.isin(times, np.asarray(fit_window))
    if np.count_nonzero(selected) < 2:
        raise ValueError("fit_window must contain at least two time points")
    weights = 1.0 / errors[selected] ** 2
    return float(np.sum(weights * values[selected]) / np.sum(weights)), float(np.sqrt(1.0 / np.sum(weights)))


def fit_continuum(points, model="a2"):
    """Fit ``E(a) = E0 + c*a^2`` or ``E(a) = E0 + c1*a + c2*a^2`` with errors."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must contain columns [a, energy, error]")
    x, y, sigma = points[:, 0], points[:, 1], np.maximum(points[:, 2], 1e-14)
    design = np.column_stack([np.ones_like(x), x ** 2] if model == "a2" else [np.ones_like(x), x, x ** 2])
    if len(y) < design.shape[1]:
        raise ValueError("Not enough Nt values for the requested continuum model")
    scaled, target = design / sigma[:, None], y / sigma
    parameters = np.linalg.lstsq(scaled, target, rcond=None)[0]
    covariance = np.linalg.pinv(scaled.T @ scaled)
    residuals = (design @ parameters - y) / sigma
    return {"model": model, "parameters": parameters,
            "errors": np.sqrt(np.maximum(np.diag(covariance), 0.0)),
            "covariance": covariance, "chi2": float(residuals @ residuals),
            "dof": len(y) - len(parameters)}


def gevp_continuum_points(Nts, t=1, E_C=1, E_J=50, operators=None, t0=1,
                          t_init=5, t_max=None, n_states=3, n_blocks=20,
                          state_idx=0, fit_window=None, use_tmax=None,
                          model_cls=None, data_dir=".", rcond=1e-10):
    """Extract one physical energy per ``Nt`` and return points plus raw results."""
    points, results = [], []
    for Nt in Nts:
        result = run_gevp_for_nt(Nt, t, E_C, E_J, operators, t0, t_init,
                                 t_max, n_states, n_blocks, use_tmax,
                                 model_cls, data_dir, rcond)
        mass, mass_error = extract_plateau_mass(result, state_idx, fit_window)
        scale = Nt / (2.0 * np.pi * t)
        points.append([1.0 / Nt, scale * mass, scale * mass_error])
        results.append(result)
    return {"points": np.asarray(points), "results": results}


def gevp_continuum_points_from_operator_samples(operator_samples_by_nt, Nts=None,
                                                t=1, t0=1, t_init=5, t_max=None,
                                                n_states=3, n_blocks=20,
                                                state_idx=0, fit_window=None,
                                                rcond=1e-10, chunk_size=1024,
                                                conf_slice=None):
    """Build continuum points from cached operator samples keyed by ``Nt``.

    ``conf_slice`` may be one slice used for every ``Nt`` or a mapping from
    ``Nt`` to a slice, allowing each cache to select its own last half.
    """
    if Nts is None:
        Nts = sorted(operator_samples_by_nt)
    points, results = [], []
    for Nt in Nts:
        if isinstance(conf_slice, dict):
            if Nt not in conf_slice:
                raise ValueError(f"Missing configuration slice for Nt={Nt}")
            nt_conf_slice = conf_slice[Nt]
        else:
            nt_conf_slice = conf_slice
        result = gevp_with_jackknife_from_operator_samples(
            operator_samples_by_nt[Nt],
            t0=t0,
            t_init=t_init,
            t_max=t_max,
            n_states=n_states,
            n_blocks=n_blocks,
            rcond=rcond,
            chunk_size=chunk_size,
            conf_slice=nt_conf_slice,
        )
        mass, mass_error = extract_plateau_mass(result, state_idx, fit_window)
        scale = Nt / (2.0 * np.pi * t)
        points.append([1.0 / Nt, scale * mass, scale * mass_error])
        results.append(result)
    return {"points": np.asarray(points), "results": results}


gevp_with_jackknife_from_correlator_samples = gevp_with_jackknife_from_operator_samples
gevp_continuum_points_from_correlator_samples = gevp_continuum_points_from_operator_samples