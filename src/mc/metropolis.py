"""Global and site-local Metropolis chains for JAX-valued configurations.

Each chain takes an action callable, initial state, PRNG key, proposal scale,
and optional temperature. ``step`` mutates the chain state, ``iter`` yields
successive configurations, and ``acceptance_rate`` returns the recent
acceptance fraction. This module does not read or write files.
"""

import jax
import jax.numpy as jnp


class Chain:
    """Global-proposal Metropolis sampler for a JAX configuration and action."""

    def __init__(self, action, x0, key, delta=1., temperature=1.):
        """Initialize action, starting state, PRNG key, proposal scale, and temperature."""
        self.action = action
        self.x = x0
        self.S = self.action(self.x)
        self.delta = delta
        self.temperature = temperature
        self._key = key
        self._recent = [False]

        def _propose(key, x, delta):
            """Draw a Gaussian global proposal and return it."""
            kstep, key = jax.random.split(key, 2)
            xp = x + delta*jax.random.normal(kstep, x.shape)

            return xp

        def _acceptreject(key, temperature, x, S, xp, Sp):
            """Apply the Metropolis test and return key, selected state/action, and flag."""
            key, kacc = jax.random.split(key, 2)
            Sdiff = Sp - S

            acc = jax.random.uniform(kacc) < jnp.exp(-Sdiff/temperature)
            x, S, accepted = jax.lax.cond(acc, lambda: (xp, Sp, True),
                                          lambda: (x, S, False))
            return key, x, S, accepted

        self._propose = jax.jit(_propose)
        self._acceptreject = jax.jit(_acceptreject)

    def _action(self, key, x, delta):
        """Generate one proposal and evaluate its real action."""
        xp = self._propose(key, x, delta)
        Sp = self.action(xp).real
        return xp, Sp

    def step(self, N=1):
        """Advance the chain by ``N`` proposals and update its state and history."""
        self.S = self.action(self.x).real
        for _ in range(N):
            xp, Sp = self._action(self._key, self.x, self.delta)
            self._key, self.x, self.S, accepted = self._acceptreject(
                self._key, self.temperature, self.x, self.S, xp, Sp)
            self._recent.append(accepted)
        self._recent = self._recent[-100:]

    def calibrate(self):
        """Adjust the global proposal scale until acceptance lies in [0.3, 0.55]."""
        # Adjust delta.
        self.step(N=100)
        while self.acceptance_rate() < 0.3 or self.acceptance_rate() > 0.55:
            if self.acceptance_rate() < 0.3:
                self.delta *= 0.98
            if self.acceptance_rate() > 0.55:
                self.delta *= 1.02
            self.step(N=100)

    def acceptance_rate(self):
        """Return the fraction of accepted proposals in recent chain history."""
        return sum(self._recent) / len(self._recent)

    def iter(self, skip=1):
        """Yield the configuration after each group of ``skip`` proposals."""
        while True:
            self.step(N=skip)
            yield self.x


class Chain_Local:
    """Site-by-site Metropolis sampler using a local action difference."""

    def __init__(self, action_local, x0, key, delta=1., temperature=1.):
        """Initialize the local action, starting state, PRNG key, scale, and temperature."""
        self.action_local = action_local
        self.x = x0
        self.delta = delta
        self.temperature = temperature
        self._key = key
        self._recent = [False]
        self.dof = len(self.x)

        def _propose(key, x, n, delta):
            """Perturb only component ``n`` and return the proposed configuration."""
            kstep, key = jax.random.split(key, 2)
            x = x.at[n].add(delta*jax.random.normal(kstep, x[n].shape))
            return x

        def _acceptreject(key, temperature, x, xp, dS):
            """Apply Metropolis acceptance using action difference ``dS``."""
            key, kacc = jax.random.split(key, 2)

            def accept():
                """Select the proposed configuration and mark the update accepted."""
                return xp, True

            def reject():
                """Keep the current configuration and mark the update rejected."""
                return x, False

            acc = jax.random.uniform(kacc) < jnp.exp(-dS/temperature)
            x, accepted = jax.lax.cond(acc, accept, reject)

            return key, x, accepted

        self._propose = jax.jit(_propose)
        self._acceptreject = jax.jit(_acceptreject)

    def _action_local(self, key, x, n, delta):
        """Propose a change at site ``n`` and evaluate its local action difference."""
        xp = self._propose(key, x, n, delta)
        # calling function twice: slow
        dS = self.action_local(xp, n).real - self.action_local(x, n).real
        return xp, dS

    def step(self, N=1):
        """Perform ``N`` full site sweeps, mutating the state and acceptance history."""
        for _ in range(N):
            for n in range(self.dof):
                xp, dS = self._action_local(self._key, self.x, n, self.delta)
                self._key, self.x, accepted = self._acceptreject(
                    self._key, self.temperature, self.x, xp, dS)
                self._recent.append(accepted)
        self._recent = self._recent[-100:]

    def calibrate(self):
        """Adjust the local proposal scale until acceptance lies in [0.3, 0.55]."""
        # Adjust delta.
        self.step(N=100)
        while self.acceptance_rate() < 0.3 or self.acceptance_rate() > 0.55:
            if self.acceptance_rate() < 0.3:
                self.delta *= 0.98
            if self.acceptance_rate() > 0.55:
                self.delta *= 1.02
            self.step(N=100)

    def acceptance_rate(self):
        """Return the fraction of accepted site proposals in recent history."""
        return sum(self._recent) / len(self._recent)

    def iter(self, skip=1):
        """Yield the configuration after each group of ``skip`` full site sweeps."""
        while True:
            self.step(N=skip)
            yield self.x
