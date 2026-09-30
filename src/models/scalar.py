from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp
import numpy as np


@dataclass
class Model:
    geom: Tuple[int]
    m2: float
    lamda: float

    def __post_init__(self):
        self.shape = self.geom
        self.D = len(self.geom)
        self.dof = np.prod(self.geom, dtype=int)

        # Backwards compatibility
        self.periodic = False

    def action(self, phi):
        phi = phi.reshape(self.shape)

        pot = jnp.sum(self.m2/2*phi**2 + self.lamda*phi**4/24)
        kin = jnp.array([0.5 * jnp.sum((jnp.roll(phi, -1, axis=d) - phi)**2) for d in range(self.D)])

        return pot + jnp.sum(kin)

    def observe(self, phi, i):
        phi_re = phi.reshape(self.shape)

        phi_av = np.mean(phi_re, axis=tuple(i for i in range(phi_re.ndim) if i != 0))
        return jnp.mean(phi_av * jnp.roll(phi_av, -i))
        return jnp.array([jnp.mean(phi_av * jnp.roll(phi_av, -i)) for i in range(self.geom[0])])
