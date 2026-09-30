#!/usr/bin/env python
"""Train contour-deformation control variates for scalar or gauge models.

The command-line inputs are a model-expression file, a pickled configuration
array, and a destination for the control-variate pickle, plus training options.
The script prints training estimates and periodically writes the trained model
and parameters to the destination.
"""

from models import scalar, gauge
import pickle
import sys
import time
from typing import Callable, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import sympy
import optax
import optuna
from util import *

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)


@jax.jit
def arcsinh(x: any) -> any:
    """Apply the inverse hyperbolic sine to a JAX value."""
    return jnp.arcsinh(x)


@jax.jit
def sinh(x: any) -> any:
    """Apply the hyperbolic sine to a JAX value."""
    return jnp.sinh(x)


class MLP(nn.Module):
    """Dense network mapping a configuration vector to a contour shift."""

    volume: int
    features: Sequence[int]
    kernel_init: Callable = nn.initializers.variance_scaling(
        2, "fan_in", "truncated_normal")  # for ReLU / CELU
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        """Map the input vector through CELU hidden layers to a shift vector."""
        for feat in self.features:
            x = nn.Dense(feat, use_bias=False,
                         kernel_init=self.kernel_init,
                         bias_init=self.bias_init)(x)
            x = nn.celu(x)
        x = nn.Dense(self.volume, use_bias=False,
                     kernel_init=self.kernel_init)(x)
        return x


class ConstantShift(nn.Module):
    """Learnable configuration-independent shift and scalar offset."""

    volume: int

    @nn.compact
    def __call__(self, x):
        """Return a learned shift shaped like ``x`` and a scalar bias parameter."""
        shift = self.param('shift', nn.initializers.zeros, x.shape)
        y = self.param('bias', nn.initializers.zeros, (1,))
        return shift, y


class CV_MLP(nn.Module):
    """Dense neural control variate returning a vector field and scalar bias."""

    volume: int
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        """Evaluate the learned vector field and return it with its bias."""
        x = MLP(self.volume, self.features)(x)
        y = self.param('bias', nn.initializers.zeros, (1,))
        return x, y


class CV_MLP_Periodic(nn.Module):
    """Periodic dense control variate using sine-transformed input coordinates."""

    volume: int
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        """Evaluate a periodic vector field and return it with its bias parameter."""
        input = jnp.sin(x)

        x = MLP(self.volume, self.features)(input)
        y = self.param('bias', nn.initializers.ones, (1,))
        return x, y


class CNN(nn.Module):
    """Convolutional feature extractor followed by a dense shift projection."""

    volume: int
    features: Sequence[int]
    kernel_init: Callable = nn.initializers.variance_scaling(
        2, "fan_in", "truncated_normal")  # for ReLU / CELU
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        """Map a spatial input through circular convolutions to a shift vector."""
        for feat in self.features:
            x = nn.Conv(feat, kernel_size=(3, 3), kernel_init=self.kernel_init,
                        bias_init=self.bias_init, padding='CIRCULAR')(x)  # Periodic boundary
            x = nn.celu(x)
            x = nn.max_pool(x, window_shape=(2, 2),
                            strides=(1, 1))  # max or avg

        x = jnp.ravel(x)
        x = nn.Dense(self.volume, kernel_init=self.kernel_init,
                     use_bias=False)(x)
        x = nn.celu(x)
        return x


class CV_CNN(nn.Module):
    """Two-dimensional convolutional control-variate model."""

    volume: int
    length: int
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        """Reshape a flat configuration to a square image and evaluate the CNN."""
        x = x.reshape(self.length, self.length, 1)
        x = CNN(self.volume, self.features)(x)
        return x


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Train g",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@')
    parser.add_argument('model', type=str, help="model filename")
    parser.add_argument('cv', type=str, help="cv filename")
    parser.add_argument('cf', type=str, help="configurations file name")
    parser.add_argument('-i', '--init', action='store_true',
                        help="re-initialize even if cv already exists")
    parser.add_argument('-f', '--from', dest='fromfile', type=str,
                        help="initialize from other file")
    parser.add_argument('-c', '--constant', action='store_true',
                        help="constant shift")
    parser.add_argument('-l', '--layers', type=int, default=0,
                        help='number of (hidden) layers')
    parser.add_argument('-w', '--width', type=int, default=1,
                        help='width (scaling)')
    parser.add_argument('-r', '--replica', action='store_true',
                        help="use replica exchange")
    parser.add_argument('-nrep', '--nreplicas', type=int, default=30,
                        help="number of replicas (with -r)")
    parser.add_argument('-maxh', '--max-hbar', type=float, default=10.,
                        help="maximum hbar (with -r)")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--seed-time', action='store_true',
                        help="seed PRNG with current time")
    parser.add_argument('--dp', action='store_true',
                        help="turn on double precision")
    parser.add_argument('-lr', '--learningrate', type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument('-N', '--nstochastic', default=1, type=int,
                        help="number of samples to estimate gradient")
    parser.add_argument('-o',  '--optimizer', choices=['adam', 'sgd', 'yogi'], default='adam',
                        help='optimizer to use')
    parser.add_argument('-s', '--schedule', action='store_true',
                        help="scheduled learning rate")
    parser.add_argument('-C', '--care', type=float, default=1,
                        help='scaling for learning schedule')
    parser.add_argument('--b1', type=float, default=0.9,
                        help="b1 parameter for adam")
    parser.add_argument('--b2', type=float, default=0.999,
                        help="b2 parameter for adam")
    parser.add_argument('-nt', '--n_train', type=int, default=1000,
                        help="number of training set")
    parser.add_argument('-ns', '--n_test', type=int, default=1000,
                        help="number of test set")
    parser.add_argument('--l2', type=float, default=0.0,
                        help="l2 regularization")
    parser.add_argument('-opt', '--optuna', action='store_true',
                        help="Use optuna")
    parser.add_argument('--cnn',  action='store_true',
                        help="Use CNN")
    parser.add_argument('--wilson', type=int, default=1,
                        help="size of wilson loop")

    args = parser.parse_args()

    if args.dp:
        jax.config.update('jax_enable_x64', True)

    seed = args.seed
    if args.seed_time:
        seed = time.time_ns()
    key = jax.random.PRNGKey(seed)

    with open(args.model, 'rb') as f:
        model = eval(f.read())
    V = model.dof

    g_ikey, chain_key = jax.random.split(key, 2)

    # define the function g
    loaded = False
    if not args.init and not args.fromfile:
        try:
            with open(args.cv, 'rb') as f:
                g, g_params = pickle.load(f)
            loaded = True
        except FileNotFoundError:
            pass
    if args.fromfile:
        with open(args.fromfile, 'rb') as f:
            g, g_params = pickle.load(f)
        loaded = True
    if not loaded:
        if args.constant:
            g = ConstantShift(V)
            g_params = g.init(g_ikey, jnp.zeros(V))
        elif args.cnn:
            g = CV_CNN(V, int(jnp.sqrt(V)), [args.width]*args.layers)
            g_params = g.init(g_ikey, jnp.zeros(V))

        else:
            if model.periodic:
                g = CV_MLP_Periodic(V, [args.width]*args.layers)
                g_params = g.init(g_ikey, jnp.zeros(V))
            else:
                g = CV_MLP(V, [args.width]*args.layers)
                g_params = g.init(g_ikey, jnp.zeros(V))

    @jax.jit
    def Seff(x, p):
        imag, _ = g.apply(p, x)
        Seff = model.action(x+1j*imag)
        return Seff

    # define subtraction function
    @jax.jit
    def f(x, p):
        imag, _ = g.apply(p, x)
        shift = x+1j*imag

        return model.observe(x, args.wilson) - model.observe(shift, args.wilson) * jnp.exp(-Seff(x, p)+model.action(x))

    # define loss function
    @jax.jit
    def Loss(x, p):
        _, y = g.apply(p, x)

        # shift is not regularized
        return jnp.abs(model.observe(x, args.wilson) - f(x, p))**2 + sum(l2_loss(w, alpha=args.l2) for w in jax.tree_util.tree_leaves(p["params"])) - args.l2 * y[0]**2

    Loss_grad = jax.jit(jax.grad(lambda x, p: Loss(x, p), argnums=1))

    if args.schedule:
        sched = optax.exponential_decay(
            init_value=args.learningrate,
            transition_steps=int(args.care),
            decay_rate=0.1,
            end_value=1e-5)
    else:
        sched = optax.constant_schedule(args.learningrate)
    opt = getattr(optax, args.optimizer)(sched, args.b1, args.b2)
    opt_state = opt.init(g_params)
    opt_update_jit = jax.jit(opt.update)

    def save():
        with open(args.cv, 'wb') as aa:
            pickle.dump((g, g_params), aa)

    # measurement
    with open(args.cf, 'rb') as aa:  # variable name aa should be different from f
        configs = pickle.load(aa)

    # separate configurations for training and test
    configs = jnp.array(configs)
    configs_test = configs[:args.n_test]
    configs = configs[-args.n_train:]

    obs = jax.vmap(lambda x: model.observe(x, args.wilson))(configs_test)
    obs_av = jackknife(np.array(obs))

    # Training
    for epochs in range(10**10):
        if epochs % 100 == 0:
            # Reduce memory usage
            fs, ls = [], []
            for i in range(args.n_test//50):
                fs.append(jax.vmap(lambda x: f(x, g_params))(
                    configs_test[50*i: 50*(i+1)]))
                ls.append(jax.vmap(lambda x: Loss(x, g_params))(
                    configs_test[50*i: 50*(i+1)]))

            fs = jnp.ravel(jnp.array(fs))
            ls = jnp.mean(jnp.array(ls))

            print(
                f"Epoch {epochs}: {obs_av} {jackknife(np.array(obs-fs))} {jackknife(np.array(fs))} {ls}", flush=True)

        g_ikey, subkey = jax.random.split(g_ikey)
        configs = jax.random.permutation(subkey, configs)
        for s in range(args.n_train//args.nstochastic):  # one epoch
            grads = jax.vmap(lambda y: Loss_grad(y, g_params))(
                configs[args.nstochastic*s: args.nstochastic*(s+1)])

            grad = jax.tree_util.tree_map(
                lambda x: jnp.mean(x, axis=0), grads)
            updates, opt_state = opt_update_jit(grad, opt_state)
            g_params = optax.apply_updates(g_params, updates)

        save()
