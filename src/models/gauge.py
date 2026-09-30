"""Lattice geometry and U(1)/SU(2) gauge-theory model definitions.

Construct a model with lattice geometry and its coupling, then pass a flattened
gauge-field array to its action or observable methods. These return JAX scalar
actions, plaquettes, Wilson loops, correlations, or averages; this module does
not read or write configuration files.
"""

from dataclasses import dataclass
from typing import Tuple
from functools import reduce
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class Lattice:
    """Periodic lattice index helper for a shape whose last axis is directions."""

    shape: Tuple

    def __post_init__(self):
        """Compute total link degrees of freedom and lattice volume."""
        self.dof = np.prod(self.shape, dtype=int)
        self.V = self.dof//self.shape[-1]

    def idx(self, *args):
        """Convert trailing lattice coordinates to a wrapped flattened index."""
        n = len(args)
        return jnp.ravel_multi_index(args, self.shape[-n:], mode='wrap')


@dataclass
class U1_2D_OBC:
    """Two-dimensional open-boundary U(1) plaquette model."""

    geom: Tuple[int]
    beta: float

    def __post_init__(self):
        """Derive the link count and array shape from lattice geometry."""
        self.dof = np.prod(self.geom, dtype=int)
        self.shape = (self.geom[0], self.geom[1], 1)

    def action(self, phi):
        """Return the negative-cosine action for the supplied plaquette angles."""
        return -self.beta*jnp.cos(phi).sum()

    def observe(self, phi, i):
        """Return the complex product of exponentiated angles in the first ``i`` entries."""
        # phi = phi.reshape(self.shape)
        # return jnp.array([jnp.prod(jnp.exp(1j*phi[:k, :k])) for k in range(1, self.shape[0]+1)])
        # return jnp.array([jnp.mean(jnp.array([jnp.prod(jnp.exp(1j*(jnp.roll(phi, (i, j), axis=(0, 1))[:k, :k])))
        #                                     for i in range(self.shape[0]) for j in range(self.shape[1])])) for k in range(1, self.shape[0]+1)])
        return jnp.prod(jnp.exp(1j*phi[:i]))
        # Move the area and take average, full area
        # obs = jnp.array([jnp.mean(jnp.array([jnp.prod(jnp.exp(1j*(jnp.roll(phi, (i, j), axis=(0, 1))[:k, :k])))
        #                                     for i in range(self.shape[0]) for j in range(self.shape[1])])) for k in range(self.shape[0])])
        return obs


@dataclass
class U1_2D_PBC:
    """Two-dimensional periodic U(1) gauge model with two links per site."""

    geom: Tuple[int]
    beta: float

    def __post_init__(self):
        """Build lattice metadata and derive degrees of freedom and link shape."""
        self.shape = (self.geom[0], self.geom[1], 2)

        self.lattice = Lattice(self.shape)
        self.dof = self.lattice.dof
        self.V = self.lattice.V

    def plaquette(self, phi):
        """Return the complex plaquette field for flattened link angles ``phi``."""
        phi = jnp.exp(1j*phi).reshape(self.shape)

        plaqs = jnp.array([phi[:, :, 0] * jnp.roll(phi[:, :, 1], -1, axis=0) *
                           jnp.roll(phi[:, :, 0].conj(), -1, axis=1) * phi[:, :, 1].conj()])

        return plaqs[0]  # plaqs shape is (1, L, L)

    def action(self, phi):
        """Return the Wilson plaquette action for link angles ``phi``."""
        return self.beta*jnp.sum(1-self.plaquette(phi)).real

    def wilsonloop_single(self, phi, i):
        """Return the product of the first ``i`` plaquettes in the flattened field."""
        x = self.plaquette(phi)
        return jnp.prod(x[:i])

    def wilsonloop_average(self, phi, i):
        """Average length-``i`` Wilson-loop products over periodic translations."""
        plaqs = self.plaquette(phi)
        index = jnp.array(
            [(-i, -j) for i, j in product(*list(map(lambda y: range(y), self.shape[:-1])))])

        def loop(ind):
            """Evaluate one translated loop after rolling the plaquette field."""
            plaqs_rolled = jnp.roll(plaqs, ind, axis=(0, 1)).ravel()
            return jnp.prod(plaqs_rolled[:i])

        return jax.vmap(loop)(index).mean()


@dataclass
class SU2_2D_OBC_Bronzan:
    """Open-boundary two-dimensional SU(2) model in Bronzan coordinates."""

    geom: Tuple[int]
    g: float

    def __post_init__(self):
        """Derive the number of plaquettes and configuration shape."""
        self.dof = np.prod(self.geom, dtype=int)
        self.shape = self.geom

    def action(self, phi):
        """Return the Bronzan-coordinate action for the SU(2) angle array."""
        phi = phi.reshape([self.dof, 2**2-1])

        return jnp.sum(-4./(self.g**2)*jnp.sin(phi[:, 0])*jnp.cos(phi[:, 1])-jnp.log(jnp.sin(2*phi[:, 0])))

    def observe(self, phi, i):
        """Return half the trace of the ordered product of the first ``i`` plaquettes."""
        phi = phi.reshape([self.dof, 2**2-1])
        plaq = jnp.array([[jnp.sin(phi[:, 0])*jnp.exp(1j*phi[:, 1]),
                         jnp.cos(phi[:, 0])*jnp.exp(1j*phi[:, 2])], [-jnp.cos(phi[:, 0])*jnp.exp(-1j*phi[:, 2]), jnp.sin(phi[:, 0])*jnp.exp(-1j*phi[:, 1])]]).transpose(2, 0, 1)
        return 0.5*jnp.trace(reduce(jnp.matmul, plaq[:i]))
        return reduce(jnp.matmul, plaq[:i])[0, 0]


@dataclass
class SU2_2D_OBC_Euler:
    """Open-boundary two-dimensional SU(2) model in Euler-angle coordinates."""

    geom: Tuple[int]
    g: float

    def __post_init__(self):
        """Derive the number of plaquettes and configuration shape."""
        self.dof = np.prod(self.geom, dtype=int)
        self.shape = self.geom

    def action(self, phi):
        """Return the Euler-coordinate action for the SU(2) angle array."""
        phi = phi.reshape([self.dof, 2**2-1])

        return jnp.sum(-4./(self.g**2)*jnp.cos(phi[:, 0]/2) - jnp.log(jnp.sin(phi[:, 0]/2)**2 * jnp.sin(phi[:, 1])))

    def observe(self, phi, i):
        """Return half the trace of the ordered product of the first ``i`` plaquettes."""
        phi = phi.reshape([self.dof, 2**2-1])

        plaq = jnp.array([[jnp.cos(phi[:, 0]/2) + 1j*jnp.sin(phi[:, 0]/2) * jnp.cos(phi[:, 1]), jnp.sin(phi[:, 0]/2) * jnp.sin(phi[:, 1]) * (1j * jnp.cos(phi[:, 2]) + jnp.sin(phi[:, 2]))], [jnp.sin(phi[:, 0]/2) * jnp.sin(phi[:, 1]) * (1j * jnp.cos(
            phi[:, 2]) - jnp.sin(phi[:, 2])), jnp.cos(phi[:, 0]/2) - 1j*jnp.sin(phi[:, 1]) * jnp.cos(phi[:, 1])]]).transpose(2, 0, 1)
        return 0.5*jnp.trace(reduce(jnp.matmul, plaq[:i]))
        return reduce(jnp.matmul, plaq[:i])[0, 0]


@dataclass
class U1_3D_PBC:
    """Three-dimensional periodic U(1) gauge model with three links per site."""

    geom: Tuple[int]
    beta: float

    def __post_init__(self):
        """Build lattice metadata and derive degrees of freedom and link shape."""
        self.shape = (self.geom[0], self.geom[1], self.geom[2], 3)

        self.lattice = Lattice(self.shape)
        self.dof = self.lattice.dof
        self.V = self.lattice.V

    def plaquette(self, phi):
        """Return flattened complex plaquettes for the three plane orientations."""
        phi = jnp.exp(1j*phi).reshape(self.shape)

        plaqs = jnp.stack([phi[:, :, :, mu] * jnp.roll(phi[:, :, :, nu], -1, axis=mu) *
                           jnp.roll(phi[:, :, :, mu].conj(), -1, axis=nu) *
                           phi[:, :, :, nu].conj()
                           for mu, nu in [(0, 1), (1, 2), (2, 0)]]).ravel()

        return plaqs

    def action(self, phi):
        """Return the Wilson plaquette action for flattened link angles ``phi``."""
        return self.beta*jnp.sum(1-self.plaquette(phi)).real

    def wilsonloop12(self, phi, i):
        """Return a length-``i`` loop formed from the second plaquette orientation."""
        # first 3: direction of plaquettes xy, yz, zx # lattice increases with z -> y -> x
        x = self.plaquette(phi).reshape([3, self.V])
        return jnp.prod(x[1, :i])

    def wilsonloop01(self, phi, i):
        """Return a length-``i`` loop in the 0-1 plane, averaged in its flattened order."""
        # first 3: direction of plaquettes xy, yz, zx # lattice increases with z -> y -> x
        x = self.plaquette(phi).reshape([3, self.V])[0]
        x = x.reshape(self.shape[:-1]).transpose([2, 0, 1]).reshape(self.V)
        return jnp.prod(x[:i])

    def wilsonloop20(self, phi, i):
        """Return a length-``i`` loop in the 2-0 plane, averaged in its flattened order."""
        # first 3: direction of plaquettes xy, yz, zx # lattice increases with z -> y -> x
        x = self.plaquette(phi).reshape([3, self.V])[2]
        x = x.reshape(self.shape[:-1]).transpose([1, 2, 0]).reshape(self.V)
        return jnp.prod(x[:i])

    # z direction is the time direction in this convention
    def correlation(self, phi, i, av):
        """Return the shifted connected correlation of xy-plane plaquettes at separation ``i``."""
        pl = self.plaquette(phi).reshape(self.shape[-1:]+self.shape[:-1])
        o = jnp.mean(pl[0], axis=(0, 1))  # plaquettes on xy-plane
        return jnp.sum(jnp.roll(o-av, -i) * (o-av))

    def plaq_av(self, phi):
        """Return the mean xy-plane plaquette for each slice along the third axis."""
        pl = self.plaquette(phi).reshape(self.shape[-1:]+self.shape[:-1])
        return jnp.mean(pl[0], axis=(0, 1))  # plaquettes on xy-plane
