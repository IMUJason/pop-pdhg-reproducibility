"""
Metal Device Management for Apple Silicon

Handles GPU/CPU device selection and unified memory allocation.
"""

import numpy as np
import warnings

# Try to import Metal via pyobjc
try:
    import Metal
    import objc
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False
    warnings.warn(
        "Metal not available. Install pyobjc-framework-Metal for GPU acceleration. "
        "Falling back to CPU (numpy) implementation.",
        RuntimeWarning
    )


class MetalDevice:
    """Manages Metal GPU device and command queue."""

    _instance = None
    _device = None
    _command_queue = None

    def __new__(cls):
        """Singleton pattern - only one Metal device per process."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_device()
        return cls._instance

    def _init_device(self):
        """Initialize Metal device."""
        if not METAL_AVAILABLE:
            self.device = None
            self.command_queue = None
            self.is_available = False
            return

        try:
            # Get default Metal device (GPU on Apple Silicon)
            self.device = Metal.MTLCreateSystemDefaultDevice()
            if self.device is None:
                raise RuntimeError("No Metal device available")

            # Create command queue
            self.command_queue = self.device.newCommandQueue()
            if self.command_queue is None:
                raise RuntimeError("Failed to create command queue")

            self.is_available = True

            # Log device info
            self.name = self.device.name()
            self.has_unified_memory = self.device.hasUnifiedMemory()

            print(f"Metal device initialized: {self.name}")
            print(f"Unified memory: {self.has_unified_memory}")

        except Exception as e:
            warnings.warn(f"Failed to initialize Metal: {e}. Using CPU fallback.")
            self.device = None
            self.command_queue = None
            self.is_available = False

    @classmethod
    def default(cls):
        """Get default Metal device instance."""
        return cls()

    def create_buffer(self, numpy_array):
        """Create Metal buffer from numpy array (zero-copy with unified memory).

        Args:
            numpy_array: numpy array with float32 dtype

        Returns:
            Metal buffer sharing memory with numpy array
        """
        if not self.is_available:
            raise RuntimeError("Metal not available")

        # Ensure float32 for Metal
        if numpy_array.dtype != np.float32:
            numpy_array = numpy_array.astype(np.float32)

        # Create buffer with shared storage mode (unified memory)
        buffer = self.device.newBufferWithBytesNoCopy_length_options_deallocator_(
            numpy_array.ctypes.data,
            numpy_array.nbytes,
            Metal.MTLResourceStorageModeShared,
            None
        )

        if buffer is None:
            raise RuntimeError("Failed to create Metal buffer")

        return buffer

    def create_command_buffer(self):
        """Create a new command buffer."""
        if not self.is_available:
            raise RuntimeError("Metal not available")
        return self.command_queue.commandBuffer()


class UnifiedMemoryBuffer:
    """Manages numpy array with Metal unified memory backing.

    This allows zero-copy data transfer between CPU and GPU.
    """

    def __init__(self, shape, dtype=np.float32):
        """Create unified memory buffer.

        Args:
            shape: Shape of the array
            dtype: Data type (must be float32 for Metal)
        """
        self.shape = shape
        self.dtype = dtype

        # Allocate numpy array with page-aligned memory
        self.numpy_array = np.zeros(shape, dtype=dtype)

        # Create Metal buffer if available
        self.metal_device = MetalDevice.default()
        self.metal_buffer = None

        if self.metal_device.is_available:
            try:
                self.metal_buffer = self.metal_device.create_buffer(self.numpy_array)
            except Exception as e:
                warnings.warn(f"Failed to create Metal buffer: {e}")

    def __getitem__(self, key):
        """Allow direct numpy indexing."""
        return self.numpy_array[key]

    def __setitem__(self, key, value):
        """Allow direct numpy assignment."""
        self.numpy_array[key] = value

    @property
    def data(self):
        """Get numpy array data."""
        return self.numpy_array

    def sync_to_gpu(self):
        """Ensure CPU writes are visible to GPU (no-op for unified memory)."""
        # With unified memory, no explicit sync needed
        # But we might want to add memory barriers in the future
        pass

    def sync_to_cpu(self):
        """Ensure GPU writes are visible to CPU (no-op for unified memory)."""
        pass


def get_compute_device(prefer_gpu=True, problem_size=None):
    """Get appropriate compute device based on problem characteristics.

    Args:
        prefer_gpu: Whether to prefer GPU if available
        problem_size: Dict with 'n_vars', 'n_constrs' for size-based decision

    Returns:
        'gpu' or 'cpu'
    """
    device = MetalDevice.default()

    if not device.is_available:
        return 'cpu'

    if not prefer_gpu:
        return 'cpu'

    # Size-based decision rules
    if problem_size:
        n = problem_size.get('n_vars', 0)
        m = problem_size.get('n_constrs', 0)

        # Small problems: CPU might be faster (avoid GPU launch overhead)
        if n < 100 and m < 100:
            return 'cpu'

        # Large sparse: GPU is better
        if n > 1000 or m > 1000:
            return 'gpu'

    return 'gpu'


if __name__ == "__main__":
    # Test Metal device initialization
    device = MetalDevice.default()

    if device.is_available:
        print("\nMetal device test passed!")

        # Test unified memory buffer
        buf = UnifiedMemoryBuffer((100, 100))
        buf.numpy_array[:] = np.random.randn(100, 100).astype(np.float32)
        print(f"Unified memory buffer created: shape={buf.shape}")
        print(f"Metal buffer: {buf.metal_buffer is not None}")
    else:
        print("\nMetal not available - using CPU fallback")
