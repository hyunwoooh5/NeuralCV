#!/usr/bin/env python

from models import scalar
import pickle
import sys
import time
from typing import Callable, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from util import *
from itertools import product

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)


def arcsinh(x: any) -> any:
    return jnp.arcsinh(x)


def sinh(x: any) -> any:
    return jnp.sinh(x)


class MLP(nn.Module):
    volume: int
    features: Sequence[int]
    kernel_init: Callable = nn.initializers.variance_scaling(
        2, "fan_in", "truncated_normal")  # for ReLU / CELU
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Dense(feat, use_bias=False,
                         kernel_init=self.kernel_init,
                         bias_init=self.bias_init)(x)
            x = arcsinh(x)
        x = nn.Dense(1, use_bias=False,
                     kernel_init=self.kernel_init)(x)
        return x


class CV_MLP(nn.Module):
    volume: int
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = MLP(self.volume, self.features)(x)
        y = self.param('bias', nn.initializers.zeros, (1,))
        return x, y


class CNN(nn.Module):
    volume: int
    features: Sequence[int]
    kernel_init: Callable = nn.initializers.variance_scaling(
        2, "fan_in", "truncated_normal")  # for ReLU / CELU
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Conv(feat, kernel_size=(3, 3), use_bias=False, kernel_init=self.kernel_init,
                        bias_init=self.bias_init, padding='CIRCULAR')(x)  # Periodic boundary
            x = arcsinh(x)
        x = nn.Conv(1, kernel_size=(3, 3), use_bias=False, kernel_init=self.kernel_init,
                    bias_init=self.bias_init, padding='CIRCULAR')(x)

        y = self.param('bias', nn.initializers.zeros, (1,))
        return jnp.ravel(x), y


class CV_CNN(nn.Module):
    volume: int
    features: Sequence[int]

    @nn.compact
    def __call__(self, x, shape):
        x = x.reshape(shape)
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
    parser.add_argument('-l', '--layers', type=int, default=0,
                        help='number of (hidden) layers')
    parser.add_argument('-w', '--width', type=int, default=1,
                        help='width')
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--seed-time', action='store_true',
                        help="seed PRNG with current time")
    parser.add_argument('--dp', action='store_true',
                        help="turn on double precision")
    parser.add_argument('-lr', '--learningrate', type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument('-N', '--n_stochastic', default=1, type=int,
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
    parser.add_argument('--cnn',  action='store_true',
                        help="Use CNN")

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

    # define control variates
    loaded = False
    if not args.init and not args.fromfile:
        try:
            with open(args.cv, 'rb') as f:
                g1, g_params = pickle.load(f)
            loaded = True
        except FileNotFoundError:
            pass
    if args.fromfile:
        with open(args.fromfile, 'rb') as f:
            g1, g_params = pickle.load(f)
        loaded = True
    if not loaded:
        if args.cnn:
            shape = list(model.shape)
            shape.append(1)

            g1 = CV_CNN(V, [args.width]*args.layers)
            g_params = g1.init(g_ikey, jnp.zeros(V), shape)
        else:
            g1 = CV_MLP(V, [args.width]*args.layers)
            g_params = g1.init(g_ikey, jnp.zeros(V))

    if type(g1) == CV_CNN:
        dS = jax.grad(lambda y: model.action(y).real)
        j = jax.jacfwd(lambda x, p: g1.apply(p, x, shape)[0], argnums=0)

        @jax.jit
        def f(x, p):
            dg = jnp.trace(j(x, p))
            ds = dS(x)
            gx, _ = g1.apply(p, x, shape)

            return dg - gx@ds

        # define loss function
        @jax.jit
        def Loss(x, p):
            _, y = g1.apply(p, x, shape)

            # shift is not regularized
            return jnp.abs(model.observe(x) - f(x, p) - y[0])**2 + sum(l2_loss(w, alpha=args.l2) for w in jax.tree_util.tree_leaves(p["params"])) - args.l2 * y[0]**2

    else:
        # g(Tx) = Tg(x)
        index = jnp.array(
            [(-i, -j) for i, j in product(*list(map(lambda y: range(y), model.shape)))])

        def g(x, p):
            def g_(x, p, ind):
                return g1.apply(p, jnp.roll(x.reshape(model.shape), ind, axis=(0, 1)).reshape(V))[0]
            return jnp.ravel(jax.vmap(lambda ind: g_(x, p, ind))(index).T)

        dS = jax.grad(lambda y: model.action(y).real)

        # define subtraction function
        @jax.jit
        def f(x, p):
            # diagonal sum (Stein's identity)
            def diag_(ind):
                rolled_x = jnp.roll(x.reshape(model.shape), ind, axis=(0, 1)).reshape(V)

                ei = jnp.zeros_like(x).at[0].set(1.0)
                _, jvp_val = jax.jvp(lambda y: g1.apply(p, y)[0], (rolled_x,), (ei,))

                return jvp_val[0]
            j = jax.vmap(diag_)(index).sum()
            return j - g(x, p)@dS(x)

        # define loss function
        @jax.jit
        def Loss(x, p):
            _, y = g1.apply(p, x)

            # shift is not regularized
            return jnp.abs(model.observe(x) - f(x, p) - y[0])**2 + sum(l2_loss(w, alpha=args.l2) for w in jax.tree_util.tree_leaves(p["params"])) - args.l2 * y[0]**2

    Loss_grad = jax.grad(lambda x, p: Loss(x, p), argnums=1)

    if args.schedule:
        sched = optax.exponential_decay(
            init_value=args.learningrate,
            transition_steps=int(args.care),
            decay_rate=0.99,
            end_value=1e-6)
    else:
        sched = optax.constant_schedule(args.learningrate)
    opt = getattr(optax, args.optimizer)(sched, args.b1, args.b2)
    opt_state = opt.init(g_params)

    @jax.jit
    def train(x, p, opt_state):
        grads = jax.vmap(lambda y: Loss_grad(y, p))(x)
        grad = jax.tree_util.tree_map(lambda y: jnp.mean(y, axis=0), grads)
        updates, opt_state = opt.update(grad, opt_state)
        p = optax.apply_updates(p, updates)
        return p, opt_state

    # measurement
    with open(args.cf, 'rb') as aa:  # variable name aa should be different from f
        configs = pickle.load(aa)

    # separate configurations for training and test
    configs = jnp.array(configs)
    conf_test = configs[:args.n_test]
    conf_train = configs[-args.n_train:]

    obs = jax.vmap(model.observe)(conf_test)
    obs_av = jackknife(np.array(obs))

    def save():
        with open(args.cv, 'wb') as aa:
            pickle.dump((g1, g_params), aa)

    ns=50

    # Training
    for epochs in range(10**10):
        key, _ = jax.random.split(key)
        conf_train = jax.random.permutation(key, conf_train)

        if epochs % 100 == 0:
            # Reduce memory usage
            fs, ls = [], []
            for i in range(args.n_test//ns):
                fs.append(jax.vmap(lambda x: f(x, g_params))(
                    conf_test[ns*i: ns*(i+1)]))
                ls.append(jax.vmap(lambda x: Loss(x, g_params))(
                    conf_test[ns*i: ns*(i+1)]))
            fs = jnp.ravel(jnp.array(fs))
            ls = jnp.mean(jnp.array(ls))

            print(
                f"Epoch {epochs}: {obs_av} {jackknife(np.array(obs-fs))} {jackknife(np.array(fs))} {ls}", flush=True)

        for s in range(args.n_train//args.n_stochastic):  # one epoch
            g_params, opt_state = train(
                conf_train[args.n_stochastic*s: args.n_stochastic*(s+1)], g_params, opt_state)

        save()
