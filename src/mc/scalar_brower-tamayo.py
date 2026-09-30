#!/usr/bin/env python

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import argparse
import ast
import sys
sys.path.append('../CVML')


class ScalarFieldTheory:
    """
    A class to simulate n-dimensional scalar phi-4 lattice field theory.
    Uses a hybrid Monte Carlo approach with Metropolis and Brower-Tamayo updates.
    """

    def __init__(self, shape, m2, lamda, delta=1.):
        """
        Initializes the lattice and simulation parameters.

        Args:
            shape (tuple): The dimensions of the lattice (e.g., (32, 32) for 2D).
            m2 (float): The squared mass parameter.
            lamda (float): The self-interaction coupling constant.
        """
        self.shape = shape
        self.dim = len(shape)
        self.dof = np.prod(shape)
        self.m2 = m2
        self.lamda = lamda

        self.delta = delta

        # Initialize with a "hot" start (random configuration)
        self.phi = np.random.uniform(-1.0, 1.0, self.shape)

    def calculate_local_action(self, phi_val, site_coords):
        """
        Calculates the action contribution from a single site, given a proposed
        field value. This is used for the Metropolis update.
        """
        # On-site potential term
        potential_action = 0.5 * self.m2 * phi_val**2 + self.lamda / 24.0 * phi_val**4

        # Kinetic (neighbor interaction) term
        kinetic_action = 0.0
        for d in range(self.dim):
            # Sum over positive and negative neighbors in each dimension
            for shift in [-1, 1]:
                neighbor_coords = list(site_coords)
                neighbor_coords[d] = (
                    neighbor_coords[d] + shift) % self.shape[d]
                neighbor_phi = self.phi[tuple(neighbor_coords)]
                kinetic_action += 0.5 * (phi_val - neighbor_phi)**2

        return potential_action + kinetic_action

    def metropolis_sweep(self):
        """
        Performs one sweep of local Metropolis updates over the entire lattice.
        """
        accepted_count = 0
        for site_coords in np.ndindex(self.shape):
            phi_old = self.phi[site_coords]

            # Propose a new value
            phi_new = phi_old + np.random.uniform(-self.delta, self.delta)

            # Calculate action for old and new values
            action_old = self.calculate_local_action(phi_old, site_coords)
            action_new = self.calculate_local_action(phi_new, site_coords)

            # Acceptance step
            if np.random.rand() < np.exp(action_old - action_new):
                self.phi[site_coords] = phi_new
                accepted_count += 1
        return accepted_count / self.dof

    def brower_tamayo_update(self):
        """
        Performs one global update using the Brower-Tamayo cluster algorithm.
        This is an adaptation of the Swendsen-Wang algorithm for continuous fields.
        """
        # --- 1. Form Bonds ---
        # Iterate over each dimension to form bonds between neighbors.
        # We store the indices of bonded sites for building a graph.
        row_indices = []
        col_indices = []

        site_indices = np.arange(self.dof).reshape(self.shape)

        for d in range(self.dim):
            # Get neighbor field values by rolling the array
            phi_neighbor = np.roll(self.phi, shift=-1, axis=d)

            # Condition 1: Neighbors must have the same sign
            same_sign = (self.phi * phi_neighbor) > 0

            # Condition 2: Probabilistic bond formation
            # The interaction strength 'beta' is analogous to J/kT in the Ising model
            beta = 2.0 * self.phi * phi_neighbor
            p_bond = 1.0 - np.exp(-beta)
            rand_nums = np.random.rand(*self.shape)

            # A bond is formed if both conditions are met
            bonds_formed = same_sign & (rand_nums < p_bond)

            # Get the flat indices of the sites and their neighbors where bonds formed
            current_sites = site_indices[bonds_formed]
            neighbor_sites = np.roll(
                site_indices, shift=-1, axis=d)[bonds_formed]

            row_indices.extend(current_sites)
            col_indices.extend(neighbor_sites)

        # --- 2. Identify Clusters ---
        # Build a sparse adjacency matrix for the graph of bonded sites.
        # An undirected graph is needed, so add edges in both directions.
        all_rows = row_indices + col_indices
        all_cols = col_indices + row_indices
        data = np.ones(len(all_rows), dtype=bool)

        graph = coo_matrix((data, (all_rows, all_cols)),
                           shape=(self.dof, self.dof))

        # Find connected components (the clusters)
        n_clusters, labels = connected_components(
            csgraph=graph, directed=False, return_labels=True)

        # Reshape the flat labels array back to the lattice shape
        cluster_map = labels.reshape(self.shape)

        # --- 3. Flip Clusters ---
        # Decide randomly whether to flip each cluster (sign -> -sign)
        # 1.0 means no flip, -1.0 means flip
        flip_decisions = np.random.choice([1.0, -1.0], size=n_clusters)

        # Create a map of the flip decision for every site on the lattice
        flip_map = flip_decisions[cluster_map]

        # Apply the flips to the field configuration
        self.phi *= flip_map

    def calibrate(self):
        # Adjust delta
        acceptance_rate = self.metropolis_sweep()
        while acceptance_rate < 0.3 or acceptance_rate > 0.55:
            if acceptance_rate < 0.3:
                self.delta *= 0.98
            if acceptance_rate > 0.55:
                self.delta *= 1.02
            acceptance_rate = self.metropolis_sweep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples")
    parser.add_argument('shape', type=str, help="model file name")
    parser.add_argument('m2', type=float, help="m2")
    parser.add_argument('lamda', type=float, help="lamda")
    parser.add_argument('cf', type=str, help="configurations file name")
    parser.add_argument('-N', '--samples', default=1000, type=int,
                        help='number of samples before termination')
    parser.add_argument('-S', '--skip', default=1, type=int,
                        help='number of sweeps to skip')
    parser.add_argument('-T', '--thermalize', default=100,
                        type=int, help="number of sweeps to thermalize")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(seed=args.seed)

    model = ScalarFieldTheory(ast.literal_eval(
        args.shape), args.m2, args.lamda)

    configs = np.zeros((args.samples,) + model.shape)

    model.calibrate()
    for sweep in range(args.thermalize):
        model.metropolis_sweep()
        model.brower_tamayo_update()
    model.calibrate()


    for sweep in range(args.samples):
        for _ in range(args.skip):
            model.metropolis_sweep()
            model.brower_tamayo_update()

        configs[sweep, :] = model.phi.copy()

    np.save(args.cf, configs)
