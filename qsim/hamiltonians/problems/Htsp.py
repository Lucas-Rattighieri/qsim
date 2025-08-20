import torch
from ..base import Hamiltonian
from ...diagonalops import DiagonalOps
from ...buffermanager import BufferManager


class Htsp(Hamiltonian):

    def __init__(self, 
                 num_cities: int,
                 adjacency_matrix, 
                 penalty_weigth:float, 
                 cost_weigth: float,
                 fix_city: bool = False,
                 device="cpu"):
                     
        self.num_cities = num_cities
        self.fix_city = fix_city
        self.adjacency_matrix = adjacency_matrix
        self.cost_weigth = cost_weigth
        self.penalty_weigth = penalty_weigth

        L = (num_cities - 1) ** 2 if self.fix_city else num_cities ** 2

        super().__init__(L, device)

        self.diag = DiagonalOps(L, self.device)
        self.diag_hamiltonian = torch.empty(2**L, device=self.device, dtype=self.dtype)
        self.create_diagonal_hamiltonian()


    def _index(self, city, position):

        if self.fix_city:
            if city == 0 or position == 0:
                return -1
            return (city - 1) * (self.num_cities - 1) + (position - 1)
        else:
            return city * self.num_cities + position


    def create_diagonal_hamiltonian(self):

        self.diag_hamiltonian.zero_()

        term = self.manager.get()
        out_chain = self.manager.get()

        start = 1 if self.fix_city else 0


        for city in range(start, self.num_cities):
            term.fill_(1)
            for position in range(start, self.num_cities):
                index = self._index(city, position)  
                self.diag.number_chain([index], out = out_chain)
                term.sub_(out_chain)
            term.pow_(2)
            self.diag_hamiltonian.add_(term, alpha=self.penalty_weigth)


        for position in range(start, self.num_cities):
            term.fill_(1)
            for city in range(start, self.num_cities):
                index = self._index(city, position)
                self.diag.number_chain([index], out = out_chain)
                term.sub_(out_chain)
            term.pow_(2)
            self.diag_hamiltonian.add_(term, alpha=self.penalty_weigth)


        for city_i in range(self.num_cities):
            for city_j in range(self.num_cities):
                if city_i != city_j:
                    for position in range(self.num_cities):
                        index_i = self._index(city_i, position)
                        index_j = self._index(city_j, (position + 1) % self.num_cities)

                        weigth = self.adjacency_matrix[city_i, city_j] * self.cost_weigth
                        if self.fix_city:
                            
                            if city_i == 0 or position == 0:
                                self.diag.number_chain([index_j], weigth, out = out_chain)
                                self.diag_hamiltonian.add_(out_chain)

                            elif city_j == 0 or (position + 1) % self.num_cities == 0:
                                self.diag.number_chain([index_i], weigth, out = out_chain)
                                self.diag_hamiltonian.add_(out_chain)

                            elif (city_i != 0 and city_j != 0 and position != 0
                                  and (position + 1) % self.num_cities != 0):
                                self.diag.number_chain([index_i, index_j], weigth, out = out_chain)
                                self.diag_hamiltonian.add_(out_chain)
                        else:
                            self.diag.number_chain([index_i, index_j], weigth, out = out_chain)
                            self.diag_hamiltonian.add_(out_chain)

        self.manager.release(term)
        self.manager.release(out_chain)
        return self.diag_hamiltonian


    def hamiltonian(self, psi, out = None):
        if out is None:
            out = torch.empty_like(psi)

        torch.mul(psi, self.diag_hamiltonian, out = out)

        return out


    def evolution(self, psi, time, out=None):
        if out is None:
            out = torch.empty_like(psi)

        torch.mul(self.diag_hamiltonian, -1j * time, out=out)
        out.exp_()
        out.mul_(psi)

        return out



