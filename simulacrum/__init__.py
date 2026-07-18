"""simulacrum: batched RL environment development framework.

Two implementations per environment, one spec. See docs/architecture.md.
"""

from simulacrum.reference import ReferenceEnv
from simulacrum.batched import BatchedEnv, invariant

__all__ = ["ReferenceEnv", "BatchedEnv", "invariant"]

__version__ = "0.1.0"
