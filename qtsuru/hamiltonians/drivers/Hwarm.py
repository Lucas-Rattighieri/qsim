import torch
from ...operators import Operators
from ..base import Hamiltonian
from ...buffermanager import BufferManager

# Implements the drive Hamiltonian for warm-starting
class Hwarm(Hamiltonian):


    def __init__(self, L: int, cs, epsilon =  0, device="cpu"):

        super().__init__(L, device)

        self.ops = Operators(L, device)

        self.epsilon = torch.tensor(epsilon, device = device)
        self.cs = cs
        self.thetas = self.calcular_thetas()


    def calcular_thetas(self):

        thetas = [0] * len(self.cs)

        for i, ci in enumerate(self.cs):
            if ci <= self.epsilon:
                pi = self.epsilon
            elif ci < 1 - self.epsilon:
                pi = torch.tensor(ci, device=self.device) if not isinstance(ci, torch.Tensor) else ci
            else:
                pi = 1 - self.epsilon

            thetas[i] = torch.asin(torch.sqrt(pi))
        return thetas



    def hamiltonian(self, psi, out=None):

        if out is None:
            out = torch.zeros_like(psi)
        else:
            out.zero_()

        tmppsi = self.manager.get()

        for qubit in range(self.L):
            mcthetai = -torch.cos(self.thetas[qubit])
            msthetai = -torch.sin(self.thetas[qubit])

            self.ops.X(psi, qubit, out=tmppsi)
            out.add_(tmppsi, alpha=msthetai)
            self.ops.Z(psi, qubit, out=tmppsi)
            out.add_(tmppsi, alpha=mcthetai)

        self.manager.release(tmppsi)
        return out


    def evolution(self, psi, time, out=None):


        tmppsi1 = self.manager.get()
        tmppsi2 = self.manager.get()

        tmppsi1.copy_(psi)

        for qubit in range(self.L):
            self.ops.Ry(tmppsi1, -self.thetas[qubit], qubit, out=tmppsi2)
            self.ops.Rz(tmppsi2, -2 * time, qubit, out=tmppsi1)
            self.ops.Ry(tmppsi1, self.thetas[qubit], qubit, out=tmppsi2)
            tmppsi2, tmppsi1 = tmppsi1, tmppsi2

        if out is None:
            out = tmppsi1.clone()
        else:
            out.copy_(tmppsi1)

        self.manager.release(tmppsi1)
        self.manager.release(tmppsi2)
        return out
