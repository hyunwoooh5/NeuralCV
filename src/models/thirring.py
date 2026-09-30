from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class Lattice:
    L: int
    beta: int

    def __post_init__(self):
        self.V = self.L * self.beta
        self.dof = 2*self.V
        self.periodic_contour = True

    def idx(self, t, x):
        return (t % self.beta)*self.L + (x % self.L)

    def sites(self):
        """ Return a list of all sites.
        """
        return jnp.indices((self.beta, self.L))

    def coords(self, i):
        t = i//self.L
        x = i % self.L
        return t, x


@dataclass
class StaggeredModel:
    L: int
    nt: int
    m: float
    g2: float
    mu: float

    def __post_init__(self):
        self.lattice = Lattice(self.L, self.nt)

        # backward compatibility
        self.dof = self.lattice.dof
        self.periodic_contour = self.lattice.periodic_contour
        self.shape = (self.nt, self.L, 2)

    def M_old(self, A):
        idx = self.lattice.idx
        A = A.reshape((self.lattice.beta, self.lattice.L, 2))
        M = self.m*jnp.eye(self.lattice.beta*self.lattice.L) + 0j

        def update_at(M, t, x):
            eta0 = jax.lax.cond(t == self.lattice.beta-1,
                                lambda x: -x, lambda x: x, 1.)  # Anti-periodic BC for time direction
            eta1 = (-1)**t
            M = M.at[idx(t, x), idx(t, x+1)].add(-eta1 /
                                                 2 * jnp.exp(-1j*A[t, x, 1]))
            M = M.at[idx(t, x+1), idx(t, x)].add(eta1 /
                                                 2 * jnp.exp(1j*A[t, x, 1]))
            M = M.at[idx(t, x), idx(t+1, x)].add(-eta0/2 *
                                                 jnp.exp(-self.mu - 1j*A[t, x, 0]))
            M = M.at[idx(t+1, x), idx(t, x)].add(eta0/2 *
                                                 jnp.exp(self.mu + 1j*A[t, x, 0]))
            return M

        ts, xs = self.lattice.sites()
        ts = jnp.ravel(ts)
        xs = jnp.ravel(xs)

        def update_at_i(i, M):
            return update_at(M, ts[i], xs[i])
        M = jax.lax.fori_loop(0, len(ts), update_at_i, M)
        if False:
            for t in range(self.lattice.beta):
                for x in range(self.lattice.L):
                    K = update_at(K, t, x)
        return M

    # Not implemented yet
    def M_component(self, A, t, x, tp, xp):
        A = A.reshape((self.lattice.beta, self.lattice.L, 2))

        def diag(t, x):
            return self.m + 0j

        def t_p(t, x):
            eta0 = jax.lax.cond(t == self.lattice.beta-1,
                                lambda x: -x, lambda x: x, 1.)
            return -eta0/2 * jnp.exp(-self.mu - 1j*A[t, x, 0])

        def t_m(t, x):
            eta0 = jax.lax.cond(t == self.lattice.beta-1,
                                lambda x: -x, lambda x: x, 1.)
            return eta0/2 * jnp.exp(self.mu + 1j*A[t, x, 0])

        def x_p(t, x):
            eta1 = (-1)**t
            return -eta1/2 * jnp.exp(-1j*A[t, x, 1])

        def x_m(t, x):
            eta1 = (-1)**t
            return eta1/2 * jnp.exp(1j*A[t, x, 1])

        def nada(t, x):
            return 0.j

        dt = (tp-t) % self.lattice.beta
        dx = (xp-x) % self.lattice.L

        ret = 0.j
        ret += jax.lax.cond(jnp.logical_and(dt == 0,
                            dx == 0), diag, nada, t, x)
        ret += jax.lax.cond(jnp.logical_and(dt == 1, dx == 0), t_p, nada, t, x)
        ret += jax.lax.cond(jnp.logical_and(dt == -1 %
                            self.lattice.beta, dx == 0), t_m, nada, tp, x)
        ret += jax.lax.cond(jnp.logical_and(dt == 0, dx == 1), x_p, nada, t, x)
        ret += jax.lax.cond(jnp.logical_and(dt == 0, dx == -1 %
                            self.lattice.L), x_m, nada, t, xp)

        return ret

    def M(self, A):
        return self.M_old(A)
        t, x = jnp.indices((self.lattice.beta, self.lattice.L))
        t, x = t.ravel(), x.ravel()
        return jax.vmap(lambda tp, xp: jax.vmap(lambda t, x: self.K_component(A, tp, xp, t, x))(t, x))(t, x)

    def action(self, A):
        s, logdet = jnp.linalg.slogdet(self.M(A))
        return 2./(self.g2) * jnp.sum(1-jnp.cos(A)) - jnp.log(s) - logdet

    def density(self, A):
        idx = self.lattice.idx
        Minv = jnp.linalg.inv(self.M(A))
        A = A.reshape((self.lattice.beta, self.lattice.L, 2))

        def n_at(t, x):
            eta0 = jax.lax.cond(t == self.lattice.beta-1,
                                lambda x: -x, lambda x: x, 1.)
            n = eta0/2 * jnp.exp(self.mu + 1j * A[t, x, 0]) * \
                Minv[idx(t, x), idx(t+1, x)]
            n -= -eta0/2 * jnp.exp(-self.mu - 1j * A[t, x, 0]) * \
                Minv[idx(t+1, x), idx(t, x)]
            return n

        t, x = jnp.indices((self.lattice.beta, self.lattice.L))
        t, x = t.ravel(), x.ravel()
        dens = jnp.sum(jax.vmap(n_at)(t, x))
        return dens / (self.lattice.beta*self.lattice.L)

    def chiral_condensate(self, A):
        Minv = jnp.linalg.inv(self.M(A))
        A = A.reshape((self.lattice.beta, self.lattice.L, 2))

        return jnp.trace(Minv) / (self.lattice.beta*self.lattice.L)

    def correlator_f(self, A):
        idx = self.lattice.idx
        Minv = jnp.linalg.inv(self.M(A))

        tp, x, xp = jnp.indices((self.nt, self.L, self.L))
        tp, x, xp = tp.ravel(), x.ravel(), xp.ravel()

        return jnp.array([jnp.sum(jax.vmap(lambda a, y, yp: jax.lax.select(t+a >= self.nt, -1, 1) * Minv[idx(t+a, y), idx(0+a, yp)])(tp, x, xp)) for t in range(self.nt)])

    def correlator_b(self, A):
        idx = self.lattice.idx
        Minv = jnp.linalg.inv(self.M(A))

        tp, x, xp = jnp.indices((self.nt, self.L, self.L))
        tp, x, xp = tp.ravel(), x.ravel(), xp.ravel()

        return jnp.array([jnp.sum(jax.vmap(lambda a, y, yp: (-1)**(t+y+0+yp) * Minv[idx(t+a, y), idx(0+a, yp)] * Minv[idx(0+a, yp), idx(t+a, y)])(tp, x, xp)) for t in range(self.nt)])

    def observe(self, A):
        return jnp.array([self.density(A)])


@dataclass
class WilsonModel:
    L: int
    nt: int
    m: float
    g2: float
    mu: float

    def __post_init__(self):
        self.lattice = Lattice(self.L, self.nt)

        # backward compatibility
        self.dof = self.lattice.dof
        self.periodic_contour = self.lattice.periodic_contour
        self.shape = (self.nt, self.L, 2)
        
        self.kappa = 1./(2*self.m + 4)
        self.pm = jnp.array([[[1., -1], [-1, 1]], [[0., 0], [0, 2]]])
        self.pm = self.pm.at[0].multiply(jnp.exp(self.mu))
        self.pp = jnp.array([[[1., 1], [1, 1]], [[2., 0], [0, 0]]])
        self.pp = self.pp.at[0].multiply(jnp.exp(-self.mu))

    def M_old(self, A):
        idx = self.lattice.idx
        A = A.reshape((self.lattice.beta, self.lattice.L, 2))
        M = jnp.eye(2*self.lattice.beta*self.lattice.L) + 0j

        def update_at(M, t, x):
            bc = jax.lax.cond(t == self.lattice.beta-1,
                              lambda x: -x, lambda x: x, 1.)  # Anti-periodic BC for time direction
            M = jax.lax.dynamic_update_slice(
                M, -self.kappa * self.pm[0] * bc * jnp.exp(1j*A[t, x, 0]), (2*idx(t+1, x), 2*idx(t, x)))
            M = jax.lax.dynamic_update_slice(
                M, -self.kappa * self.pm[1] * jnp.exp(1j*A[t, x, 1]), (2*idx(t, x+1), 2*idx(t, x)))
            M = jax.lax.dynamic_update_slice(
                M, -self.kappa * self.pp[0] * bc * jnp.exp(-1j*A[t, x, 0]), (2*idx(t, x), 2*idx(t+1, x)))
            M = jax.lax.dynamic_update_slice(
                M, -self.kappa * self.pp[1] * jnp.exp(-1j*A[t, x, 1]), (2*idx(t, x), 2*idx(t, x+1)))

            return M

        ts, xs = self.lattice.sites()
        ts = jnp.ravel(ts)
        xs = jnp.ravel(xs)

        def update_at_i(i, K):
            return update_at(K, ts[i], xs[i])
        M = jax.lax.fori_loop(0, len(ts), update_at_i, M)

        return M

    # different convention
    def M_old2(self, A):
        idx = self.lattice.idx
        A = A.reshape((self.lattice.beta, self.lattice.L, 2))
        A = A.at[-1, :, 0].add(jnp.pi)  # Anti-periodic BC for time direction
        M = jnp.eye(2*self.lattice.beta*self.lattice.L) + 0j

        def update_at(M, t, x):
            M = jax.lax.dynamic_update_slice(
                M, -self.kappa * self.pm[0] * jnp.exp(1j*A[t, x, 0]), (2*idx(t, x), 2*idx(t+1, x)))
            M = jax.lax.dynamic_update_slice(
                M, -self.kappa * self.pm[1] * jnp.exp(1j*A[t, x, 1]), (2*idx(t, x), 2*idx(t, x+1)))
            M = jax.lax.dynamic_update_slice(
                M, -self.kappa * self.pp[0] * jnp.exp(-1j*A[(t-1) % self.nt, x, 0]), (2*idx(t, x), 2*idx(t-1, x)))
            M = jax.lax.dynamic_update_slice(
                M, -self.kappa * self.pp[1] * jnp.exp(-1j*A[t, (x-1) % self.L, 1]), (2*idx(t, x), 2*idx(t, x-1)))

            return M

        ts, xs = self.lattice.sites()
        ts = jnp.ravel(ts)
        xs = jnp.ravel(xs)

        def update_at_i(i, M):
            return update_at(M, ts[i], xs[i])
        M = jax.lax.fori_loop(0, len(ts), update_at_i, M)

        return M

    # Not implemented yet
    def M_component(self, A, t, x, tp, xp):
        A = A.reshape((self.lattice.beta, self.lattice.L, 2))

        def diag(t, x):
            return 1.0

        def t_p(t, x):
            eta0 = jax.lax.cond(t == self.lattice.beta-1,
                                lambda x: -x, lambda x: x, 1.)
            return -eta0/2 * jnp.exp(-self.mu - 1j*A[t, x, 0])

        def t_m(t, x):
            eta0 = jax.lax.cond(t == self.lattice.beta-1,
                                lambda x: -x, lambda x: x, 1.)
            return eta0/2 * jnp.exp(self.mu + 1j*A[t, x, 0])

        def x_p(t, x):
            eta1 = (-1)**t
            return -eta1/2 * jnp.exp(-1j*A[t, x, 1])

        def x_m(t, x):
            eta1 = (-1)**t
            return eta1/2 * jnp.exp(1j*A[t, x, 1])

        def nada(t, x):
            return 0.j

        dt = (tp-t) % self.lattice.beta
        dx = (xp-x) % self.lattice.L

        ret = 0.j
        ret += jax.lax.cond(jnp.logical_and(dt == 0,
                            dx == 0), diag, nada, t, x)
        ret += jax.lax.cond(jnp.logical_and(dt == 1, dx == 0), t_p, nada, t, x)
        ret += jax.lax.cond(jnp.logical_and(dt == -1 %
                            self.lattice.beta, dx == 0), t_m, nada, tp, x)
        ret += jax.lax.cond(jnp.logical_and(dt == 0, dx == 1), x_p, nada, t, x)
        ret += jax.lax.cond(jnp.logical_and(dt == 0, dx == -1 %
                            self.lattice.L), x_m, nada, t, xp)

        return ret

    def M(self, A):
        return self.M_old(A)
        t, x = jnp.indices((self.lattice.beta, self.lattice.L))
        t, x = t.ravel(), x.ravel()
        return jax.vmap(lambda tp, xp: jax.vmap(lambda t, x: self.M_component(A, tp, xp, t, x))(t, x))(t, x)

    def action(self, A):
        s, logdet = jnp.linalg.slogdet(self.M(A))
        return 2. * (1./(self.g2) * jnp.sum(1-jnp.cos(A)) - jnp.log(s) - logdet)

    def chiral_condensate(self, A):
        Minv = jnp.linalg.inv(self.M(A))

        return Minv.trace()/self.lattice.V

    def density(self, A):
        idx = self.lattice.idx
        Minv = jnp.linalg.inv(self.M(A))
        gamma_0 = jnp.array([[0, 1], [1, 0]])

        t, x = self.lattice.sites()
        t, x = t.ravel(), x.ravel()

        return jnp.mean(jax.vmap(lambda a, b: jnp.trace(jax.lax.dynamic_slice(
            Minv, (2*idx(a, b), 2*idx(a, b)), (2, 2))@gamma_0))(t, x))

    def correlator_f(self, A):
        idx = self.lattice.idx
        Minv = jnp.linalg.inv(self.M(A))

        tp, x, xp = jnp.indices((self.nt, self.L, self.L))
        tp, x, xp = tp.ravel(), x.ravel(), xp.ravel()

        return jnp.array([jnp.sum(jax.vmap(lambda a, y, yp: jax.lax.select(t+a >= self.nt, -1, 1) * jnp.trace(jax.lax.dynamic_slice(Minv, (2*idx(t+a, y), 2*idx(0+a, yp)), (2, 2))))(tp, x, xp)) for t in range(self.nt)])

    def correlator_b(self, A):
        idx = self.lattice.idx
        Minv = jnp.linalg.inv(self.M(A))
        gamma_5 = jnp.array([[0, -1j], [1j, 0]])

        tp, x, xp = jnp.indices((self.nt, self.L, self.L))
        tp, x, xp = tp.ravel(), x.ravel(), xp.ravel()

        return jnp.array([jnp.sum(jax.vmap(lambda a, y, yp: jnp.trace(gamma_5 @ jax.lax.dynamic_slice(Minv, (2*idx(t+a, y), 2*idx(0+a, yp)), (2, 2)) @ gamma_5 @ jax.lax.dynamic_slice(Minv, (2*idx(0+a, yp), 2*idx(t+a, y)), (2, 2))))(tp, x, xp)) for t in range(self.nt)])

    def observe(self, A):
        return jnp.array([self.density(A)])
