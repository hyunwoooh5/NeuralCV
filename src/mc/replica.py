"""Replica-exchange wrapper around compatible Monte Carlo chains.

``ReplicaExchange`` takes an action, initial state, PRNG key, chain class, and
replica-temperature settings. It advances and exchanges replicas; ``iter``
yields the cold-replica configuration and ``acceptance_rate`` returns within-
replica and exchange acceptance statistics. No files are read or written.
"""

import jax
import jax.numpy as jnp
import numpy as np

from mc import metropolis


class ReplicaExchange:
    """Run multiple temperature-scaled chains and exchange their configurations."""

    def __init__(self, action, x0, key, Chain=metropolis.Chain, max_hbar=10., Nreplicas=30):
        """Create ``Nreplicas`` chains spanning unit to ``max_hbar`` temperature."""
        self.Nreplicas = Nreplicas

        loghbars = jnp.linspace(0, jnp.log(max_hbar), Nreplicas)
        hbars = jnp.exp(loghbars)
        key, self.key = jax.random.split(key, 2)
        keys = jax.random.split(key, Nreplicas)
        self.replicas = [Chain(action, x0, key, temperature=hbar)
                         for hbar, key in zip(hbars, keys)]

        for replica in self.replicas:
            replica._track = [False]

        def _propose(key):
            """Draw two replica indices and a uniform acceptance variate."""
            kswap, kacc, key = jax.random.split(key, 3)
            i, j = jax.random.randint(kswap, shape=(
                [2]), minval=0, maxval=self.Nreplicas)
            x = jax.random.uniform(kacc)
            return i, j, x, key

        self._propose = jax.jit(_propose)

    def step(self, N=1):
        """Advance each replica by ``N`` steps, attempt exchanges, and update cold state."""
        for replica in self.replicas:
            replica.step(N=N)
            replica._track = replica._track[-100:]
        self.exchange()
        self.x = self.replicas[0].x

    def exchange(self):
        """Attempt random pairwise replica exchanges and record their outcomes."""
        if self.Nreplicas == 1:
            return
        for _ in range(self.Nreplicas):  # change to jax loop?
            i, j, rand, self.key = self._propose(self.key)
            if i == j:
                continue
            hbari, hbarj = self.replicas[i].temperature, self.replicas[j].temperature
            Si, Sj = self.replicas[i].S, self.replicas[j].S
            S = Si/hbari + Sj/hbarj
            Sp = Sj/hbari + Si/hbarj
            Sdiff = Sp - S
            if Sdiff < 0 or rand < jnp.exp(-Sdiff):
                self._swap(i, j)
                self.replicas[i]._track.append(True)
                self.replicas[j]._track.append(True)
            else:
                self.replicas[i]._track.append(False)
                self.replicas[j]._track.append(False)

    def _swap(self, i, j):
        """Swap replica objects and their temperatures at indices ``i`` and ``j``."""
        temperaturei, temperaturej = self.replicas[i].temperature, self.replicas[j].temperature
        self.replicas[i].temperature, self.replicas[j].temperature = temperaturej, temperaturei
        self.replicas[i], self.replicas[j] = self.replicas[j], self.replicas[i]

    def calibrate(self):
        """Calibrate the proposal settings of every contained replica chain."""
        for replica in self.replicas:
            replica.calibrate()

    def iter(self, skip=1):
        """Yield the coldest replica configuration after each ``skip``-step advance."""
        while True:
            self.step(N=skip)
            yield self.replicas[0].x

    def acceptance_rate(self):
        """Return mean within-chain acceptance and per-replica exchange acceptance."""
        return [np.mean([r.acceptance_rate() for r in self.replicas]), [sum(r._track)/len(r._track) for r in self.replicas]]
