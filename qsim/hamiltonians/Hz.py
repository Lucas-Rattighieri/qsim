import torch
from ..operators import Operators
from ..bitops import BitOps
from abc import ABC, abstractmethod
from .base import Hamiltonian



class Hz(Hamiltonian):
    """
    Quantum Hamiltonian representing a sum of Pauli-Z operators on each qubit.

    This class implements both the application of the Hamiltonian to a quantum state
    and the corresponding unitary evolution, simulating a transverse field in the Z direction.

    Methods:
        hamiltonian(psi, out=None): Applies the sum of Pauli-Z operators to the input state.
        evolution(psi, time, out=None): Evolves the state `psi` under the Hamiltonian
                                       for a given time.
    """

    def __init__(self, L: int, device="cpu", bitops: BitOps = None, indices: torch.Tensor = None, tmp: torch.Tensor = None):
        """
        Initializes the Hz Hamiltonian for a system with L qubits.

        Args:
            L (int): Number of qubits.
            device (str, optional): Device used for tensor computations (default is "cpu").
        """
        super().__init__(L, device, bitops, indices, tmp)


    def hamiltonian(self, psi, out=None):
        """
        Applies the Hamiltonian (sum of Pauli-Z) to the quantum state vector `psi`.

        Args:
            psi (torch.Tensor): Input quantum state vector.
            out (torch.Tensor, optional): Tensor to store the result. If None, a new tensor is created.

        Returns:
            torch.Tensor: Result of the Hamiltonian applied to `psi`.
        """
        if out is None:
            out = torch.zeros_like(psi)
        else:
            out.zero_()

        for qubit in range(self.L):
            self.Z(psi, qubit, out=self.tmppsi1)
            out.add_(self.tmppsi1)

        return out


    def evolution(self, psi, time, out=None):
        """
        Evolves the quantum state `psi` under the Hz Hamiltonian for a duration `time`.

        Args:
            psi (torch.Tensor): Input quantum state vector.
            time (float): Time evolution parameter.
            out (torch.Tensor, optional): Tensor to store the evolved state. If None, a new tensor is returned.

        Returns:
            torch.Tensor: Evolved quantum state.
        """
        self.tmppsi1.copy_(psi)

        for qubit in range(self.L):
            self.Rz(self.tmppsi1, 2 * time, qubit, out=self.tmppsi2)
            self.tmppsi2, self.tmppsi1 = self.tmppsi1, self.tmppsi2

        if out is None:
            out = self.tmppsi1.clone()
        else:
            out.copy_(self.tmppsi1)        

        if self.L & 1:
            self.tmppsi2, self.tmppsi1 = self.tmppsi1, self.tmppsi2

        return out
