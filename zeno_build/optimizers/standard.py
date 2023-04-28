"""The standard optimizer used by default."""

from zeno_build.optimizers.random import RandomOptimizer


class StandardOptimizer(RandomOptimizer):
    """The Standard optimizer used by default.

    See zeno_build.optimizer.base for details of the interface.
    In the current version, the standard optimizer uses random
    search.
    """
