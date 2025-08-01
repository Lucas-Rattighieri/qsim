from ..operators import Operators


class Hamiltonian(ABC, Operators):
    """
    Abstract base class for quantum Hamiltonians.

    Inherits from Operators and ABC to define a common interface for Hamiltonian
    operators and their time evolution on quantum state vectors.

    Attributes:
        tmppsi1 (torch.Tensor): Temporary tensor for intermediate quantum state calculations.
        tmppsi2 (torch.Tensor): Temporary tensor for intermediate quantum state calculations.
    """


    def __init__(self, L: int, device = "cpu"):
        super().__init__(L, device)
        self.tmppsi1 = torch.zeros(self.dim, dtype=self.dtype, device=device)
        self.tmppsi2 = torch.zeros(self.dim, dtype=self.dtype, device=device)

    
    @abstractmethod
    def hamiltonian(self, psi, out=None):
        """
        Abstract method to apply the Hamiltonian operator on a quantum state vector.

        Args:
            psi (torch.Tensor): Input quantum state vector.
            out (torch.Tensor, optional): Output tensor to store the result. If None, a new tensor is created.

        Returns:
            torch.Tensor: Result of applying the Hamiltonian to `psi`.
        """
        pass
    
    @abstractmethod
    def evolution(self, psi, time, out=None):
        """
        Abstract method to evolve a quantum state vector under the Hamiltonian for a given time.

        Args:
            psi (torch.Tensor): Input quantum state vector.
            time (float): Evolution time parameter.
            out (torch.Tensor, optional): Output tensor to store the evolved state. If None, a new tensor is created.

        Returns:
            torch.Tensor: Quantum state vector after evolution.
        """

        pass
