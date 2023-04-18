"""The standard optimizer used by default."""

from llm_compare.optimizers.vizier import VizierOptimizer


class StandardOptimizer(VizierOptimizer):
    """The Standard optimizer used by default.

    See llm_compare.optimizer.base for details of the interface.
    In the current version, the standard optimizer uses Vizier.
    """
