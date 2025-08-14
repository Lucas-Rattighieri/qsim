import torch
from .base import Hamiltonian


class Hy(Hamiltonian):
    """
    Quantum Hamiltonian representing a sum of Pauli-Y operators on each qubit.

    Implements the Hamiltonian and its time evolution for a transverse field
    in the Y direction.

    Methods:
        hamiltonian(psi, out=None): Applies the sum of Pauli-Y operators to state `psi`.
        evolution(psi, time, out=None): Evolves the state `psi` under the Hamiltonian
                                       for a given time.
    """


    def __init__(self, L: int, device="cpu", bitops: BitOps = None, indices: torch.Tensor = None, tmp: torch.Tensor = None):
        """
        Initializes the Hy Hamiltonian for a system of L qubits.

        Args:
            L (int): Number of qubits.
            device (str, optional): Device for tensor operations (default 'cpu').
        """
        super().__init__(L, device, bitops, indices, tmp)


    def hamiltonian(self, psi, out=None):
        """
        Applies the Hamiltonian (sum of Pauli-Y) to the quantum state vector `psi`.

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
            self.Y(psi, qubit, out=self.tmppsi1)
            out.add_(self.tmppsi1)

        return out


    def evolution(self, psi, time, out=None):
        """
        Evolves the quantum state `psi` under the Hy Hamiltonian for a time `time`.

        Args:
            psi (torch.Tensor): Initial quantum state vector.
            time (float): Evolution time parameter.
            out (torch.Tensor, optional): Tensor to store the evolved state.
                                          If None, a new tensor is created.

        Returns:
            torch.Tensor: Quantum state after evolution.
        """
        self.tmppsi1.copy_(psi)

        for qubit in range(self.L):
            self.Ry(self.tmppsi1, 2 * time, qubit, out=self.tmppsi2)
            self.tmppsi2, self.tmppsi1 = self.tmppsi1, self.tmppsi2

        if out is None:
            out = self.tmppsi1.clone()
        else:
            out.copy_(self.tmppsi1)        

        if self.L & 1:
            self.tmppsi2, self.tmppsi1 = self.tmppsi1, self.tmppsi2

        return out
