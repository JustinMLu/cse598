# validate_jax.py
import jax
import jax.numpy as jp
import numpy as np

print("\n================ Validating GPU... ================")
try:
    # First check JAX default backend (should be GPU)
    backend = jax.default_backend()
    
    # If not GPU then error out immediately
    if backend.lower() != 'gpu':
        print("\n❌ ERROR: JAX is not using the GPU backend.")
        print("This means JAX was not installed with CUDA support or cannot find the necessary libraries.")
    else:
        print(f"✅ JAX default backend: '{backend}'\n")

        # List all devices seen by JAX
        devices = jax.devices()
        print(f"✅ Found {len(devices)} JAX device(s):")
        for i, device in enumerate(devices):
            print(f"  - Device {i}: {device.device_kind} (platform: {device.platform})")

        # Ensure we have at least 1 actual GPU
        gpu_devices = [i for i in devices if i.platform.lower() == 'gpu']
        if not gpu_devices:
            print("\n❌ ERROR: No GPU devices were found by JAX, despite the backend being set to 'gpu'.")
        else:
            print(f"\n✅ Successfully found {len(gpu_devices)} GPU device(s).")
            
            # 3. Run a sample computation on the GPU
            print("\nTesting GPU: (2000 x 2000) matmul operation...")
            
            # Create a JIT-compiled function for matrix multiplication
            @jax.jit
            def multiply_matrices(a, b):
                return jp.dot(a, b)

            # Create random matrices on the GPU
            key = jax.random.PRNGKey(0)
            a = jax.random.normal(key, (2000, 2000))
            b = jax.random.normal(key, (2000, 2000))

            # Run the computation and wait for it to complete
            result = multiply_matrices(a, b).block_until_ready()

            print("Matrix multiplication finished succesfully.\n")
            print("✅ Validation Complete! JAX can be GPU-accelerated.")

except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")
    print("Validation script did not finish successfully.")
print("===================================================\n")