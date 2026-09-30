#!/usr/bin/env python

import numpy as np
import argparse
import sys
sys.path.append('../CVML')


# Global counters for rejection sampling acceptance
global_total_proposals = 0
global_total_accepts = 0


def sample_angle(a, rng):
    """
    Sample a new angle theta from the distribution
       P(theta) ~ exp(a*cos(theta))
    using a variant of the Hattori–Nakajima rejection sampling method.

    Args:
        a (float): The effective parameter, a = beta * |staple|
        rng (np.random.Generator): NumPy random number generator.

    Returns:
        float: A sampled angle theta (in radians).
    """
    global global_total_proposals, global_total_accepts

    if a == 0:
        return rng.uniform(-np.pi, np.pi)

    eps = 1e-3
    astar = 0.798953686083986

    amax = np.maximum(0, a-astar)
    delta = 0.35 * amax + 1.03 * np.sqrt(amax)

    alpha = np.minimum(np.sqrt(a * (2.0 - eps)),
                       np.maximum(np.sqrt(eps * a), delta))

    beta_tilde = np.maximum(
        alpha**2/a, (np.cosh(np.pi*alpha)-1)/(np.exp(2*a)-1))-1
    beta_s = np.sqrt((1 + beta_tilde) / (1 - beta_tilde))
    tmp = np.arctan(np.tanh(0.5 * np.pi * alpha) / beta_s)

    def h(x):
        # Map x in [0,1] to an angle.
        return (2.0 / alpha) * np.arctanh(beta_s * np.tan((2 * x - 1) * tmp))

    def G(x):
        return 1.0 - np.cos(x) - (1.0 / a) * np.log(1 + (np.cosh(alpha * x) - 1) / (1 + beta_tilde))

    def gg(x):
        return np.exp(-a * G(h(x)))

    # Worst acceptance ratio is 0.88 when a -> \infty
    while True:
        global_total_proposals += 1
        x1 = rng.uniform(0, 1)
        x2 = rng.uniform(0, 1)
        theta = h(x1)
        if x2 < gg(x1):
            global_total_accepts += 1
            return theta


def compute_staple(U, x, mu):
    """
    Compute the staple for the link U(x, mu) on a d-dimensional lattice.
    The staple is the sum over all directions nu != mu of the two plaquette contributions.

    Args:
        U (ndarray): Lattice gauge links with shape (l1, l2, ..., ld, d)
                     (each link is a U(1) element stored as a complex number).
        x (tuple): A d-tuple of lattice indices.
        mu (int): Direction index to update (0 <= mu < d).

    Returns:
        complex: The staple (a complex number).
    """
    dims = U.shape[:-1]  # lattice dimensions, a tuple (l1, l2, ..., ld)
    d = len(dims)
    staple = 0.0 + 0j
    for nu in range(d):
        if nu == mu:
            continue
        # --- Forward plaquette contribution ---
        xp = list(x)
        xp[nu] = (xp[nu] + 1) % dims[nu]
        xp = tuple(xp)
        xmu = list(x)
        xmu[mu] = (xmu[mu] + 1) % dims[mu]
        xmu = tuple(xmu)
        # U(x, nu) * U(x+e_nu, mu) * conj(U(x+e_mu, nu))
        staple += U[x][nu] * U[xp][mu] * np.conj(U[xmu][nu])

        # --- Backward plaquette contribution ---
        xm = list(x)
        xm[nu] = (xm[nu] - 1) % dims[nu]
        xm = tuple(xm)
        xmn = list(xm)
        xmn[mu] = (xmn[mu] + 1) % dims[mu]
        xmn = tuple(xmn)
        # conj(U(x-e_nu, nu)) * U(x-e_nu, mu) * U(x-e_nu+e_mu, nu)
        staple += np.conj(U[xm][nu]) * U[xm][mu] * U[xmn][nu]
    return staple


def update_link(U, x, mu, beta, rng):
    """
    Update the link variable U(x, mu) using the heat bath algorithm.

    Args:
        U (ndarray): Lattice gauge links, shape (l1, l2, ..., ld, d).
        x (tuple): Lattice site indices (d-tuple).
        mu (int): Direction index to update.
        beta (float): Gauge coupling.
        rng (np.random.Generator): Random number generator.

    Returns:
        None; U is updated in place.
    """
    staple = compute_staple(U, x, mu)
    a = beta * abs(staple)
    phase_staple = np.angle(staple)
    if a > 0:
        theta = sample_angle(a, rng)
    else:
        theta = rng.uniform(-np.pi, np.pi)
    # New link: U(x, mu) = exp[i*(theta + phase_staple)]
    U[x][mu] = np.exp(1j * (theta + phase_staple))


def heat_bath_sweep(U, beta, rng):
    """
    Perform a full heat bath sweep over the entire d-dimensional lattice.

    Args:
        U (ndarray): Lattice gauge links with shape (l1, l2, ..., ld, d).
        beta (float): Gauge coupling.
        rng (np.random.Generator): Random number generator.

    Returns:
        ndarray: Updated U.
    """
    dims = U.shape[:-1]
    for x in np.ndindex(dims):
        for mu in range(len(dims)):
            update_link(U, x, mu, beta, rng)
    return U


def compute_average_plaquette(U):
    """
    Compute the average plaquette for a d-dimensional U(1) lattice gauge field.

    For each site and for each pair of directions (mu, nu) with mu < nu,
    the plaquette is given by
      P_mu,nu(x) = U(x, mu) * U(x+e_mu, nu) * conj(U(x+e_nu, mu)) * conj(U(x, nu)).

    Args:
        U (ndarray): Lattice gauge links with shape (l1, l2, ..., ld, d).

    Returns:
        float: The average plaquette value.
    """
    dims = U.shape[:-1]
    d = len(dims)
    total = 0.0
    count = 0
    for x in np.ndindex(dims):
        for mu in range(d - 1):
            for nu in range(mu + 1, d):
                # Compute x + e_mu and x + e_nu (with periodic boundary conditions)
                x_plus_mu = list(x)
                x_plus_mu[mu] = (x_plus_mu[mu] + 1) % dims[mu]
                x_plus_mu = tuple(x_plus_mu)
                x_plus_nu = list(x)
                x_plus_nu[nu] = (x_plus_nu[nu] + 1) % dims[nu]
                x_plus_nu = tuple(x_plus_nu)
                U1 = U[x][mu]
                U2 = U[x_plus_mu][nu]
                U3 = np.conj(U[x_plus_nu][mu])
                U4 = np.conj(U[x][nu])
                plaq = np.real(U1 * U2 * U3 * U4)
                total += plaq
                count += 1
    return total / count if count > 0 else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples")
    parser.add_argument('model', type=str, help="model file name")
    parser.add_argument('cf', type=str, help="configurations file name")
    parser.add_argument('-N', '--samples', default=1000, type=int,
                        help='number of samples before termination')
    parser.add_argument('-S', '--skip', default=1, type=int,
                        help='number of sweeps to skip')
    parser.add_argument('-T', '--thermalize', default=100,
                        type=int, help="number of sweeps to thermalize")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    args = parser.parse_args()

    with open(args.model, 'rb') as aa:
        model = eval(aa.read())

    rng = np.random.default_rng(seed=args.seed)

    # Lattice parameters
    # Define an arbitrary lattice shape in d dimensions.
    dims = model.geom
    d = len(dims)

    configs = np.zeros([args.samples, model.dof])

    # Initialize U(1) gauge links.
    # U has shape dims + (d,): each site has d link variables.
    U = np.exp(1j * 2 * np.pi * rng.uniform(0, 1, size=dims + (d,)))

    for sweep in range(args.thermalize):
        U = heat_bath_sweep(U, model.beta, rng)
        avg_plaq = compute_average_plaquette(U)
        print(
            f"Sweep {sweep+1:3d}: Average Plaquette = {avg_plaq:.6f}, Overall Acceptance Ratio = {global_total_accepts / global_total_proposals:.4f}", flush=True)

    for sweep in range(args.samples):
        for _ in range(args.skip):
            U = heat_bath_sweep(U, model.beta, rng)
        configs[sweep, :] = np.log(np.copy(U)).imag.ravel()
        if sweep % 100 == 100-1:
            avg_plaq = compute_average_plaquette(U)
            print(
                f"Sweep {sweep+1:3d}: Average Plaquette = {avg_plaq:.6f}, Overall Acceptance Ratio = {global_total_accepts / global_total_proposals:.4f}", flush=True)

    np.save(args.cf, configs)
