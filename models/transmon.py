from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp
import numpy as np


@dataclass
class Model:
    # geom: Tuple[int]
    Nt: int
    t: float  # ns
    E_C: float  # h * GHz
    E_J: float  # h * GHz

    def __post_init__(self):
        self.shape = (self.Nt, )
        # self.D = len(self.geom)
        # self.dof = np.prod(self.geom, dtype=int)

        self.dof = self.Nt

        self.E_C = 2*jnp.pi * 10**9 * self.E_C  # Hz
        self.E_J = 2*jnp.pi * 10**9 * self.E_J  # Hz

        self.dt = 1e-9 * 1./self.Nt  # Hz

        self.C = 1./(2.*self.E_C)

        # Backwards compatibility
        self.periodic = False

    def action(self, phi):
        # phi = phi.reshape(self.shape)

        kin = 1/8 * self.C * jnp.sum((jnp.roll(phi, -1)-phi)**2) / self.dt
        pot = -self.E_J * jnp.sum(jnp.cos(phi)) * self.dt

        return kin+pot

    def action_separate(self, phi):
        kin = 1/8 * self.C * jnp.sum((jnp.roll(phi, -1)-phi)**2) / self.dt
        pot = -self.E_J * jnp.sum(jnp.cos(phi)) * self.dt

        return kin, pot

    def observe(self, phi):
        # phi_re = phi.reshape(self.shape)
        return jnp.asarray([jnp.mean(jnp.sin(jnp.roll(phi, -i))*jnp.sin(phi)) for i in range(self.Nt)])

    def observe_square(self, phi, av):
        # phi_re = phi.reshape(self.shape)
        return jnp.asarray([jnp.mean((jnp.sin(jnp.roll(phi, -i))**2 - av) * (jnp.sin(phi)**2 - av)) for i in range(self.Nt)])

    def observe_cubic(self, phi):
        # phi_re = phi.reshape(self.shape)
        return jnp.asarray([jnp.mean(jnp.sin(jnp.roll(phi, -i))**3 * jnp.sin(phi)**3) for i in range(self.Nt)])

    def observe_o1o3(self, phi):
        # phi_re = phi.reshape(self.shape)
        return jnp.asarray([jnp.mean(jnp.sin(jnp.roll(phi, -i))**1 * jnp.sin(phi)**3) for i in range(self.Nt)])
