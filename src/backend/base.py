"""
Backend abstraction for numerical computations.

This module provides a unified interface for numerical operations,
allowing switching between NumPy (CPU) and JAX (GPU) backends.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Callable
import numpy as np
from scipy import sparse


class Backend(ABC):
    """Abstract base class for numerical backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return backend name."""
        pass

    @abstractmethod
    def array(self, data, dtype=None):
        """Create array."""
        pass

    @abstractmethod
    def zeros(self, shape, dtype=None):
        """Create zeros array."""
        pass

    @abstractmethod
    def ones(self, shape, dtype=None):
        """Create ones array."""
        pass

    @abstractmethod
    def clip(self, x, lower, upper):
        """Clip array values."""
        pass

    @abstractmethod
    def sum(self, x, axis=None):
        """Sum array."""
        pass

    @abstractmethod
    def mean(self, x, axis=None):
        """Mean of array."""
        pass

    @abstractmethod
    def dot(self, a, b):
        """Dot product."""
        pass

    @abstractmethod
    def norm(self, x, ord=None):
        """Vector/matrix norm."""
        pass

    @abstractmethod
    def random_seed(self, seed: int):
        """Set random seed."""
        pass

    @abstractmethod
    def random_normal(self, shape, dtype=None):
        """Sample from normal distribution."""
        pass

    @abstractmethod
    def random_uniform(self, shape, dtype=None):
        """Sample from uniform distribution."""
        pass


class NumPyBackend(Backend):
    """NumPy CPU backend."""

    def __init__(self):
        self.rng = np.random.RandomState()

    @property
    def name(self) -> str:
        return "numpy"

    def array(self, data, dtype=None):
        return np.array(data, dtype=dtype if dtype else np.float64)

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype if dtype else np.float64)

    def ones(self, shape, dtype=None):
        return np.ones(shape, dtype=dtype if dtype else np.float64)

    def clip(self, x, lower, upper):
        return np.clip(x, lower, upper)

    def sum(self, x, axis=None):
        return np.sum(x, axis=axis)

    def mean(self, x, axis=None):
        return np.mean(x, axis=axis)

    def dot(self, a, b):
        return np.dot(a, b)

    def norm(self, x, ord=None):
        return np.linalg.norm(x, ord=ord)

    def random_seed(self, seed: int):
        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)

    def random_normal(self, shape, dtype=None):
        return self.rng.randn(*shape).astype(dtype if dtype else np.float64)

    def random_uniform(self, shape, dtype=None):
        return self.rng.uniform(0, 1, shape).astype(dtype if dtype else np.float64)


class JAXBackend(Backend):
    """JAX GPU/CPU backend."""

    def __init__(self, use_gpu: bool = True):
        import jax
        import jax.numpy as jnp
        import jax.random as random

        self.jax = jax
        self.jnp = jnp
        self.random = random
        self._use_gpu = use_gpu

        # Set device
        if use_gpu:
            try:
                devices = jax.devices()
                if any(d.platform == 'gpu' for d in devices):
                    pass  # GPU available
                else:
                    self._use_gpu = False
            except:
                self._use_gpu = False

        # Random key
        self._key = random.PRNGKey(0)

    @property
    def name(self) -> str:
        return "jax_gpu" if self._use_gpu else "jax_cpu"

    def array(self, data, dtype=None):
        return self.jnp.array(data, dtype=dtype if dtype else self.jnp.float64)

    def zeros(self, shape, dtype=None):
        return self.jnp.zeros(shape, dtype=dtype if dtype else self.jnp.float64)

    def ones(self, shape, dtype=None):
        return self.jnp.ones(shape, dtype=dtype if dtype else self.jnp.float64)

    def clip(self, x, lower, upper):
        return self.jnp.clip(x, lower, upper)

    def sum(self, x, axis=None):
        return self.jnp.sum(x, axis=axis)

    def mean(self, x, axis=None):
        return self.jnp.mean(x, axis=axis)

    def dot(self, a, b):
        return self.jnp.dot(a, b)

    def norm(self, x, ord=None):
        return self.jnp.linalg.norm(x, ord=ord)

    def random_seed(self, seed: int):
        self._key = self.random.PRNGKey(seed)

    def random_normal(self, shape, dtype=None):
        self._key, subkey = self.random.split(self._key)
        return self.random.normal(subkey, shape, dtype=dtype if dtype else self.jnp.float64)

    def random_uniform(self, shape, dtype=None):
        self._key, subkey = self.random.split(self._key)
        return self.random.uniform(subkey, shape, dtype=dtype if dtype else self.jnp.float64)

    def jit(self, fn):
        """JIT compile a function."""
        return self.jax.jit(fn)

    def vmap(self, fn):
        """Vectorize a function."""
        return self.jax.vmap(fn)


# Global backend instance
_backend: Optional[Backend] = None


def get_backend(name: str = "numpy") -> Backend:
    """Get the global backend instance.

    Args:
        name: Backend name ('numpy' or 'jax')

    Returns:
        Backend instance
    """
    global _backend

    if _backend is not None and _backend.name.startswith(name):
        return _backend

    if name == "numpy":
        _backend = NumPyBackend()
    elif name == "jax":
        try:
            _backend = JAXBackend()
        except ImportError:
            print("JAX not available, falling back to NumPy")
            _backend = NumPyBackend()
    else:
        raise ValueError(f"Unknown backend: {name}")

    return _backend


def set_backend(name: str) -> Backend:
    """Set the global backend.

    Args:
        name: Backend name ('numpy' or 'jax')

    Returns:
        New backend instance
    """
    global _backend

    if name == "numpy":
        _backend = NumPyBackend()
    elif name == "jax":
        _backend = JAXBackend()
    else:
        raise ValueError(f"Unknown backend: {name}")

    return _backend


# Convenience functions
def xp():
    """Get the array module for current backend."""
    return get_backend()


if __name__ == "__main__":
    print("Testing Backend Abstraction")
    print("=" * 50)

    # Test NumPy backend
    print("\n[1] NumPy Backend:")
    np_backend = NumPyBackend()
    np_backend.random_seed(42)

    x = np_backend.random_normal((5, 3))
    print(f"  Random array shape: {x.shape}")
    print(f"  Mean: {np_backend.mean(x):.4f}")
    print(f"  Norm: {np_backend.norm(x):.4f}")

    # Test JAX backend
    print("\n[2] JAX Backend:")
    try:
        jax_backend = JAXBackend()
        jax_backend.random_seed(42)

        x = jax_backend.random_normal((5, 3))
        print(f"  Backend name: {jax_backend.name}")
        print(f"  Random array shape: {x.shape}")
        print(f"  Mean: {jax_backend.mean(x):.4f}")
        print(f"  Norm: {jax_backend.norm(x):.4f}")
    except ImportError as e:
        print(f"  JAX not available: {e}")

    print("\n[3] Global Backend:")
    backend = get_backend("numpy")
    print(f"  Current backend: {backend.name}")
