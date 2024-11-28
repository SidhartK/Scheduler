from abc import ABC, abstractmethod

class AbstractSolver(ABC):
    """
    Abstract base class for solvers in the RL environment.
    """
    def __init__(self, environment):
        """
        Initialize the solver with the given environment.

        Args:
            environment: An instance of RLProjectEnv.
        """
        self.environment = environment

    @abstractmethod
    def solve(self):
        """Solve the given environment."""
        pass
