import torch

class BufferManager:
    """
    Manages a set of reusable tensors (buffers) with a global registry to allow
    sharing across multiple instances with the same dimension, device, and dtype.

    Each buffer has a flag indicating whether it is in use. If all buffers are
    occupied, a new one is created on demand. The global registry ensures that
    multiple classes or simulators requesting buffers with the same properties
    will reuse the same BufferManager instance.
    """
    
    _registry = {}
    _index_registry = {}

    def __init__(self, dim, device="cpu", dtype=torch.complex64):
        """
        Initializes the buffer manager for a given dimension, device, and dtype.

        Args:
            dim (int): dimension of each tensor.
            device (str or torch.device, optional): device where tensors will be allocated.
            dtype (torch.dtype, optional): data type of the tensors.
        """
        self.dim = dim
        self.device = device
        self.dtype = dtype
        self.buffers = []
        self.in_use = []


    @classmethod
    def get_manager(cls, dim, device, dtype):
        """
        Returns a BufferManager instance from the global registry. Creates a new
        one if no manager exists for the given properties.

        Args:
            dim (int): dimension of each tensor.
            device (str or torch.device, optional): device where tensors will be allocated.
            dtype (torch.dtype, optional): data type of the tensors.

        Returns:
            BufferManager: an instance corresponding to the requested properties.
        """
        key = (dim, str(device), dtype)
        if key not in cls._registry:
            cls._registry[key] = cls(dim, device, dtype)
        return cls._registry[key]

    @classmethod
    def delete_manager(cls, dim, device="cpu", dtype=torch.complex64):
        """
        Delete a BufferManager from the registry for the given key.

        Parameters
        ----------
        dim : int
            Dimension of the buffers handled by the manager.
        device : str or torch.device
            Device where the buffers are allocated.
        dtype : torch.dtype
            Data type of the buffers.
        """
        key = (dim, str(device), dtype)
        if key in cls._registry:
            del cls._registry[key]
        else:
            raise KeyError(f"No BufferManager found for key {key}")


    @classmethod
    def get_index(cls, dim, device="cpu"):
        """
        Returns an immutable index tensor [0, 1, ..., dim-1] from the registry.
        If it does not exist, creates it.

        Args:
            dim (int): length of the index tensor.
            device (str or torch.device): device for tensor allocation.
            dtype (torch.dtype): data type of the tensor.

        Returns:
            torch.Tensor: immutable index tensor.
        """
        dtype = torch.int32 if dim < 2 ** 32 else torch.int64
        key = (dim, str(device))
        if key not in cls._index_registry:
            cls._index_registry[key] = torch.arange(dim, device=device, dtype=dtype)
        return cls._index_registry[key]

    
    def get(self):
        """
        Returns an available buffer. If none is free, creates a new one.

        Returns:
            torch.Tensor: a tensor ready for use.
        """
        for i, used in enumerate(self.in_use):
            if not used:
                self.in_use[i] = True
                return self.buffers[i]

        buf = torch.empty(self.dim, device=self.device, dtype=self.dtype)
        self.buffers.append(buf)
        self.in_use.append(True)
        return buf

    def release(self, buf):
        """
        Marks a buffer as available, allowing it to be reused.

        Args:
            buf (torch.Tensor): tensor previously obtained via `get()`.

        Raises:
            ValueError: if the buffer does not belong to this manager.
        """
        for i, b in enumerate(self.buffers):
            if b is buf:
                self.in_use[i] = False
                return
        raise ValueError("Buffer does not belong to this manager")
