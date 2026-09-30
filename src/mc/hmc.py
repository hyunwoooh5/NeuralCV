"""Hamiltonian Monte Carlo chain for a differentiable action.

``Chain`` takes an action callable, initial JAX state, PRNG key, and integration
and temperature settings. Its methods update chain state; ``iter`` yields
successive states and ``acceptance_rate`` returns the recent acceptance ratio.
The module does not perform file I/O.
"""

import jax
import jax.numpy as jnp
from functools import partial


class Chain:
    """Stateful HMC sampler using leapfrog proposals and Metropolis acceptance."""

    def __init__(self, action, x0, key, L=10, dt=0.3, temperature=1.):
        """Initialize from an action, state, PRNG key, leapfrog length/step, and temperature."""
        self.action = jax.jit(lambda y: action(y).real)
        self._grad = jax.jit(jax.grad(lambda y: action(y).real))
        self.x = x0
        self.L = L
        self.dt = dt
        self.temperature = temperature
        self._key = key
        self._recent = [False]

        @partial(jax.jit, static_argnums=2)
        def _propose(key, x, L, dt):
            """Propose a state with ``L`` leapfrog steps and return old/new energies."""
            kstep, key = jax.random.split(key, 2)
            p = jax.random.normal(kstep, x.shape)

            # initial hamiltonian
            x0 = x
            h0 = jnp.sum(p**2)/2+self.action(x)

            # Leapfrog integration
            for _ in range(L):
                p -= dt/2*self._grad(x)
                x += dt * p
                p -= dt/2*self._grad(x)

            # final hamiltonian
            xp = x
            hp = jnp.sum(p**2)/2+self.action(xp)

            return x0, h0, xp, hp

        def _acceptreject(key, temperature, x, h, xp, hp):
            """Accept or reject a proposal and return its updated key and state."""
            key, kacc = jax.random.split(key, 2)
            hdiff = hp - h

            def accept():
                """Select the proposed state and mark the trajectory accepted."""
                return xp, True

            def reject():
                """Keep the current state and mark the trajectory rejected."""
                return x, False

            acc = jax.random.uniform(kacc) < jnp.exp(-hdiff/temperature)
            x, accepted = jax.lax.cond(acc, accept, reject)

            return key, x, accepted

        self._propose = _propose
        self._acceptreject = jax.jit(_acceptreject)

    def step(self, N=1):
        """Advance the chain by ``N`` trajectories, mutating state and recent history."""
        for _ in range(N):
            x, h, xp, hp = self._propose(self._key, self.x, self.L, self.dt)
            self._key, self.x, accepted = self._acceptreject(
                self._key, self.temperature, x, h, xp, hp)
            self._recent.append(accepted)
        self._recent = self._recent[-100:]

    def calibrate(self):
        """Adjust leapfrog length until recent acceptance is between 0.6 and 0.9."""
        # Adjust leapfrog steps
        self.step(N=100)
        while self.acceptance_rate() < 0.6 or self.acceptance_rate() > 0.9:
            if self.acceptance_rate() < 0.6:
                self.L += 1
            if self.acceptance_rate() > 0.9:
                self.L -= 1
            self.step(N=100)

    def acceptance_rate(self):
        """Return the fraction of accepted proposals in recent chain history."""
        return sum(self._recent) / len(self._recent)

    def iter(self, skip=1):
        """Yield the current configuration after each group of ``skip`` trajectories."""
        while True:
            self.step(N=skip)
            yield self.x
