import jax
import jax.numpy as jnp


def count_parameters(pytree):
    """Counts the total number of parameters in a JAX PyTree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(pytree))


def compare_pytrees(tree1, tree2):
    """Compare two pytrees."""
    return jax.tree_util.tree_map(lambda x, y: jnp.array_equal(x, y), tree1, tree2)


def error_pytrees(tree1, tree2):
    """Calculate errors to compare two pytrees."""
    return jax.tree_util.tree_map(lambda x, y: (x-y)/x, tree1, tree2)


def tree_dot(tree, vec):
    """dot product of a tree and a vector."""

    def vector_broadcasting(leaf, m):
        # Create a new shape for m: (n, 1, 1, ..., 1) where the number of 1's equals leaf.ndim - 1.
        new_shape = (m.shape[0],) + (1,) * (leaf.ndim - 1)
        return leaf * m.reshape(new_shape)

    dots = jax.tree_util.tree_map(
        lambda leaf: vector_broadcasting(leaf, vec), tree)
    return jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0), dots)


def copy_pytree(tree):
    """Copy a pytree."""
    def copy_leaf(x):
        return x.copy()

    return jax.tree_util.tree_map(copy_leaf, tree)
