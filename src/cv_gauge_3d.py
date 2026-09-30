#!/usr/bin/env python

from models import gauge
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

import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)
os.environ.update({
    "NCCL_LL128_BUFFSIZE": "-2",
    "NCCL_LL_BUFFSIZE": "-2",
    "NCCL_PROTO": "SIMPLE,LL,LL128",
})

jax.config.update("jax_default_matmul_precision", "highest")


def arcsinh(x: any) -> any:
    return jnp.arcsinh(x)


def sinh(x: any) -> any:
    return jnp.sinh(x)


class MLP(nn.Module):
    volume: int
    features: Sequence[int]
    bias: bool
    kernel_init: Callable = nn.initializers.variance_scaling(
        2, "fan_in", "truncated_normal")  # for ReLU / CELU
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Dense(feat, use_bias=self.bias,
                         kernel_init=self.kernel_init,
                         bias_init=self.bias_init)(x)
            x = nn.celu(x)
            # x = arcsinh(x)
            # x = nn.leaky_relu(x, negative_slope=0.01) #nn.relu jnp.tan nn.tanh #nn.celu
        x = nn.Dense(2, use_bias=self.bias,
                     kernel_init=self.bias_init, bias_init=self.bias_init)(x)  # adfasdfadsf
        return x


class CV_MLP_Periodic(nn.Module):
    volume: int
    features: Sequence[int]
    n: int

    @nn.compact
    def __call__(self, x):
        pl = model.plaquette(x)

        powers = jnp.array([pl**i for i in range(1, self.n+1)])
        sinx = jnp.hstack((powers.imag, powers.real))[0]

        g0 = MLP(self.volume, self.features, True)(sinx)
        # g1 = MLP(self.volume, self.features, True)(sinx)

        y = self.param('bias', nn.initializers.zeros, (1,))
        return g0, y


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
    parser.add_argument('-n', '--nx', type=int, default=1,
                        help='sin(nx), cos(nx)')
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--seed-time', action='store_true',
                        help="seed PRNG with current time")
    parser.add_argument('--dp', action='store_true',
                        help="turn on double precision")
    parser.add_argument('-lr', '--learningrate', type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument('-nt', '--n_train', type=int, default=1000,
                        help="number of training set")
    parser.add_argument('-ns', '--n_test', type=int, default=1000,
                        help="number of test set")
    parser.add_argument('-btrain', '--batch_train', type=int, default=32,
                        help='minibatch size for training')
    parser.add_argument('-btest', '--batch_test', type=int, default=32,
                        help='minibatch size for test')
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
    parser.add_argument('--l2', type=float, default=0.0,
                        help="l2 regularization")

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

    # define control variates
    loaded = False
    if not args.init and not args.fromfile:
        try:
            with open(args.cv+'.pkl', 'rb') as f:
                g1, g_params,  opt_state, track_red, track_ltest, track_ltrain = pickle.load(
                    f)
            loaded = True
        except FileNotFoundError:
            pass
    if args.fromfile:
        with open(args.fromfile, 'rb') as f:
            g1, g_params, _, _, _, _ = pickle.load(f)
        loaded = True
    if not loaded:
        g1 = CV_MLP_Periodic(V, [args.width]*args.layers, args.nx)
        g_params = g1.init(key, jnp.zeros(V))
        track_ltrain, track_ltest, track_red = [], [], []

    reduction_best = 0 if len(track_red) == 0 else track_red[-1][1]

    # g(Tx) = Tg(x)
    index = jnp.array([(-i, -j, -k) for i, j, k in product(*
                      list(map(lambda y: range(y), model.shape[:-1])))])

    def g(x, p):
        def g_(x, p, ind):
            a, b = g1.apply(p, jnp.roll(x.reshape(model.shape),
                            ind, axis=(0, 1, 2)).reshape(V))[0]
            return jnp.array([a, b, jnp.array(0.)])
        return jnp.ravel(jax.vmap(lambda ind: g_(x, p, ind))(index))

    dS = jax.grad(lambda y: model.action(y).real)

    # Stein control variates
    @jax.jit
    def f(x, p):
        def f_(x, p):
            def diag_(ind):
                rolled_x = jnp.roll(x.reshape(model.shape),
                                    ind, axis=(0, 1, 2)).reshape(V)

                trace = 0.0

                ei = jnp.zeros_like(x).at[0].set(1.0)
                _, jvp_val = jax.jvp(lambda y: g1.apply(p, y)[
                                     0], (rolled_x,), (ei,))
                trace += jvp_val[0]

                ei = jnp.zeros_like(x).at[1].set(1.0)
                _, jvp_val = jax.jvp(lambda y: g1.apply(p, y)[
                                     0], (rolled_x,), (ei,))
                trace += jvp_val[1]

                return trace
            j = jax.vmap(diag_)(index).sum()
            return j - g(x, p)@dS(x)
        return f_(x, p)

    # define loss function
    @jax.jit
    def Loss(x, p):
        _, y = g1.apply(p, x)
        al = 0

        return jnp.abs(model.correlation(x, model.shape[2]//2, av).real - f(x, p) - y[0])**2 + al * l2_regularization(p)

    def Loss_batch(batch, params):
        # Compute the per-sample losses
        per_sample_losses = jax.vmap(Loss, in_axes=(0, None))(batch, params)
        # Return the average loss over the batch
        return jnp.mean(per_sample_losses)

    Loss_batch_grad = jax.jit(jax.grad(Loss_batch, argnums=1))

    if args.schedule:
        sched = optax.exponential_decay(
            init_value=args.learningrate,
            transition_steps=int(args.care),
            decay_rate=0.1,
            end_value=1e-5)
    else:
        sched = optax.constant_schedule(args.learningrate)

    opt = optax.chain(
        # Clip by the gradient by the global norm.
        optax.clip_by_global_norm(1.0),
        # optax.adamw(1e-3)
        # Use the updates from adam.
        optax.scale_by_adam(b1=args.b1, b2=args.b2),
        # Use the learning rate from the scheduler.
        optax.scale_by_schedule(sched),
        # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
        optax.scale(-1.0))
    opt_state = opt.init(g_params)

    @jax.jit
    def train_batch_shard(x, p, opt_state):
        grad = Loss_batch_grad(x, p)
        updates, opt_state = opt.update(grad, opt_state)
        p = optax.apply_updates(p, updates)
        return p, opt_state

    # separate configurations for training and test
    conf = np.load(args.cf, mmap_mode='r')

    conf_train = conf[:args.n_train]
    conf_test = conf[-args.n_test:]

    plaq_av_jit = jax.jit(jax.vmap(model.plaq_av))

    av = []
    for i in range(len(conf)//args.batch_test):
        batch = jax.device_put(conf[i*args.batch_test:(i+1)*args.batch_test])
        av.append(plaq_av_jit(batch))
    av = jnp.mean(jnp.asarray(av), axis=(0, 1))
    print(f"plaq_av: {av}")

    obs = jax.vmap(lambda y: model.correlation(
        y, model.shape[2]//2, av).real)(conf_test)[:(args.n_test//args.batch_test)*args.batch_test]

    f_vmap = jax.vmap(f, in_axes=(0, None))
    Loss_vmap = jax.vmap(Loss, in_axes=(0, None))

    def save():
        with open(args.cv+'.pkl', 'wb') as aa:
            pickle.dump((g1, g_params, opt_state, track_red,
                        track_ltest, track_ltrain), aa)

    def save_best():
        with open(args.cv+'_best.pkl', 'wb') as aa:
            pickle.dump((g1, g_params), aa)

    num_devices = jax.local_device_count()
    print(jax.devices())

    from jax.sharding import PartitionSpec as P, NamedSharding, Mesh

    mesh = jax.make_mesh((num_devices,), ('batch',))

    sharding = NamedSharding(mesh, P('batch'))
    replicated_sharding = NamedSharding(mesh, P())

    g_params = jax.device_put(g_params, replicated_sharding)

    # Training
    start = time.time()
    epochs = 0
    while True:
        key, _ = jax.random.split(key)
        perm = jax.random.permutation(key, args.n_train)

        if epochs % 100 == 0:
            # Reduce memory usage
            fs, ls, ltrain = [], [], []
            for i in range(args.n_test//args.batch_test):
                minibatch = jax.device_put(
                    conf_test[args.batch_test*i: args.batch_test*(i+1)], sharding)
                fs.append(f_vmap(minibatch, g_params))
                ls.append(Loss_vmap(minibatch, g_params))

            for i in range(args.n_train//args.batch_train):
                minibatch = jax.device_put(
                    conf_train[args.batch_train*i: args.batch_train*(i+1)], sharding)
                ltrain.append(Loss_vmap(minibatch, g_params))

            fs = jnp.ravel(jnp.array(fs))
            ls = jnp.mean(jnp.array(ls))
            ltrain = jnp.mean(jnp.array(ltrain))

            print(
                f"Epoch {epochs}:  <Test Loss>: {ls} <Train Loss>: {ltrain} <f>: {jackknife(fs)} <f^2>: {jackknife(fs**2)} <Of>: {jackknife(obs*fs)} <O-f>:{jackknife(obs-fs)}, time={time.time()-start}", flush=True)

            track_red.append([epochs, jackknife(obs)[1]/jackknife(obs-fs)[1]])
            track_ltrain.append([epochs, ltrain])
            track_ltest.append([epochs, ls])

            save()
            if reduction_best < track_red[-1][1]:
                save_best()
                reduction_best = track_red[-1][1]

        for i in range(args.n_train//args.batch_train):
            minibatch = jax.device_put(
                conf_train[perm[i*args.batch_train: (i+1)*args.batch_train]], sharding)
            g_params, opt_state = train_batch_shard(
                minibatch, g_params, opt_state)

        epochs += 1
