"""Small utilities for inspecting and transforming JAX parameter pytrees.

Inputs are JAX pytrees (nested containers of array leaves), and where needed a
vector for leaf-wise scaling. Functions return a parameter count, a tree of
comparisons/errors, a tree-vector product, or a copied pytree; they do not use
files or produce console output.
"""

import jax
import jax.numpy as jnp


def count_parameters(pytree):
    """Return the total number of scalar entries across all pytree leaves."""
    return sum(x.size for x in jax.tree_util.tree_leaves(pytree))


def compare_pytrees(tree1, tree2):
    """Compare corresponding leaves and return a pytree of elementwise equality arrays."""
    return jax.tree_util.tree_map(lambda x, y: jnp.array_equal(x, y), tree1, tree2)


def error_pytrees(tree1, tree2):
    """Return corresponding leaf-wise relative differences ``(tree1-tree2)/tree1``."""
    return jax.tree_util.tree_map(lambda x, y: (x-y)/x, tree1, tree2)


def tree_dot(tree, vec):
    """Contract each pytree leaf with the leading axis of ``vec`` and sum that axis."""

    def vector_broadcasting(leaf, m):
        """Reshape the vector for broadcasting over one parameter leaf."""
        # Create a new shape for m: (n, 1, 1, ..., 1) where the number of 1's equals leaf.ndim - 1.
        new_shape = (m.shape[0],) + (1,) * (leaf.ndim - 1)
        return leaf * m.reshape(new_shape)

    dots = jax.tree_util.tree_map(
        lambda leaf: vector_broadcasting(leaf, vec), tree)
    return jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0), dots)


def copy_pytree(tree):
    """Return a tree with a shallow array copy of every leaf."""
    def copy_leaf(x):
        """Copy one array leaf."""
        return x.copy()

    return jax.tree_util.tree_map(copy_leaf, tree)
