import torch

class DiagonalOps:
    """
    Class for construction of diagonal operators in quantum systems
    composed of L qubits, using Kronecker products with predefined vectors.

    This implementation allows fast generation of operators such as Z chains or
    number operators, as well as arbitrary diagonal operators in the computational basis.

    Parameters
    ----------
    L : int
        Number of qubits in the system.
    dtype : torch.dtype, optional
        Data type of the tensor elements (default: torch.complex128).
    device : str or torch.device, optional
        Device where the tensors will be allocated (default: 'cpu').
    tmpdiag : torch.Tensor, optional
        Reusable temporary vector for storing intermediate results.
        Must have size `2**L`, type `dtype`, and be located on the same `device` as the class.

    Attributes
    ----------
    dim : int
        Dimension of the Hilbert space (2**L).
    tmpdiag : torch.Tensor
        Temporary vector used in internal computations.
    ones : torch.Tensor
        Vector filled with ones, used as the identity in Kronecker products.
    diag_z : torch.Tensor
        Diagonal representation of the Pauli-Z operator for a single qubit.
    diag_number : torch.Tensor
        Diagonal representation of the number operator (|1><1|) for a single qubit.

    Methods
    -------
    validate_vector(tmpdiag) -> bool
        Checks if the provided tensor is compatible with the current configuration.
    operator(op, pos, coef=1, out=None) -> torch.Tensor
        Constructs a diagonal operator as a Kronecker product of `op` on the
        specified positions, multiplied by a coefficient `coef`.
    z_chain(pos, coef=1, out=None) -> torch.Tensor
        Generates a chain of Pauli-Z operators on the specified positions.
    number_chain(pos, coef=1, out=None) -> torch.Tensor
        Generates a chain of number operators on the specified positions.
    """

    def __init__(self, L: int, dtype=torch.complex128, device='cpu', tmpdiag = None, tmp = None):

        self.device = device
        self.L = L
        self.dim = 2 ** L
        self.dtype = dtype

        if self.validate_vector(tmpdiag):
            self.tmpdiag = tmpdiag
        else:
            self.tmpdiag = torch.zeros(self.dim, dtype=self.dtype, device=device)

        if validate_tmp(tmp):
            self.ones = tmp
            self.ones.ones_()
        else:
            self.ones = torch.ones(self.dim, dtype = torch.int32 if L < 32 else torch.int64, device = device)

        self.diag_z = torch.tensor([1.0, -1.0], dtype=self.dtype, device=self.device)
        self.diag_number = torch.tensor([0.0, 1.0], dtype=self.dtype, device=self.device)


    def validate_tmp(self, tmp: torch.Tensor) -> bool:
        """
        Checks whether the input tensor `tmp` is valid for this system.

        A valid index tensor must:
        - Be on the correct device;
        - Have the correct dtype;
        - Have length equal to 2 ** L.

        Parameters:
        - indices (torch.Tensor): The tensor to be validated.

        Returns:
        - bool: True if the tensor is valid, False otherwise.
        """
        if isinstance(tmp, torch.Tensor):
            return (tmp.device == torch.device(self.device)
                and ((tmp.dtype == tmp.int32 and L < 32) or (tmp.dtype == tmp.int64))
                and tmp.numel() == self.dim
            )
        else:
            return False

    def validate_vector(self, tmpdiag) -> bool:
        """
        Validate whether a given vector matches the expected device, data type, and dimension.

        Args:
            tmpdiag (torch.Tensor): The vector to be validated.

        Returns:
            bool: True if the vector is valid, False otherwise.
        """
        if tmpdiag is None:
            return False
        return (
            tmpdiag.device == torch.device(self.device)
            and tmpdiag.dtype == self.dtype
            and tmpdiag.numel() == self.dim
        )


    def operator(self, op: torch.Tensor, pos: list[int], coef: complex = 1, out = None):
        """
        Build a diagonal operator as the Kronecker product of `op` applied to
        the specified qubit positions, scaled by a coefficient.

        Parameters
        ----------
        op : torch.Tensor
            1-qubit diagonal operator to be applied at the given positions.
        pos : list[int]
            List of qubit positions (0-based, leftmost qubit is position 0).
        coef : complex, optional
            Scalar factor multiplying the resulting operator (default: 1).
        out : torch.Tensor, optional
            Preallocated output tensor to store the result. If None, a new tensor is created.

        Returns
        -------
        torch.Tensor
            The resulting diagonal operator as a 1D tensor of length `2**L`.
        """

        if out is None:
            out = torch.zeros(self.dim, dtype=self.dtype, device=self.device)
        else:
            out.zeros_()

        if coef == 0:
            return out

        pos = [self.L - k - 1 for k in pos]
        pos = sorted(pos)
        resultado = self.tmpdiag
        resultado[0] = coef
        nout = out

        j = 0
        size = 1

        for i in pos:
            if i > j:
                ident = self.ones[:2 ** (i-j)]

                torch.kron(resultado[:size], ident, out=nout[:size * 2 ** (i-j)])
                nout, resultado = resultado, nout
                size *= 2 ** (i-j)

            torch.kron(resultado[:size], op, out=nout[:size*2])
            nout, resultado = resultado, nout

            size *= 2
            j = i + 1


        if j < self.L:
            ident = self.ones[:2 ** (self.L - j)]
            torch.kron(resultado, ident, out=nout[:size * (2 ** (self.L - j))])
            nout, resultado = resultado, nout

        if out is not resultado:
            out.copy_(resultado)

        return out


    def z_chain(self, pos: list[int], coef: complex = 1, out = None):
        """
        Generate a diagonal operator consisting of a chain of Pauli-Z operators
        applied at the specified qubit positions.

        Parameters
        ----------
        pos : list[int]
            List of qubit positions (0-based, leftmost qubit is position 0).
        coef : complex, optional
            Scalar factor multiplying the resulting operator (default: 1).
        out : torch.Tensor, optional
            Preallocated output tensor to store the result. If None, a new tensor is created.

        Returns
        -------
        torch.Tensor
            The diagonal representation of the Z-chain operator as a 1D tensor.
        """
        return self.operator(self.diag_z, pos, coef, out)


    def number_chain(self, pos: list[int], coef: complex = 1, out = None):
        """
        Generate a diagonal operator consisting of a chain of number operators
        (|1><1| projectors) applied at the specified qubit positions.

        Parameters
        ----------
        pos : list[int]
            List of qubit positions (0-based, leftmost qubit is position 0).
        coef : complex, optional
            Scalar factor multiplying the resulting operator (default: 1).
        out : torch.Tensor, optional
            Preallocated output tensor to store the result. If None, a new tensor is created.

        Returns
        -------
        torch.Tensor
            The diagonal representation of the number-chain operator as a 1D tensor.
        """
        return self.operator(self.diag_number, pos, coef, out)
