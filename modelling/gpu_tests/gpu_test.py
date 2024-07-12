import torch
import time

# Function to perform matrix multiplication and measure time
def measure_time(device, size=10000):
    # Generate random matrices
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Ensure tensors are on the specified device
    a = a.to(device)
    b = b.to(device)
    
    # Start the timer
    start_time = time.time()
    
    # Perform matrix multiplication
    c = torch.mm(a, b)
    
    # Wait for the operation to complete (important for GPU)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Stop the timer
    end_time = time.time()
    
    return end_time - start_time

# Measure time for CPU
cpu_device = torch.device('cpu')
cpu_time = measure_time(cpu_device)
print(f"CPU time: {cpu_time:.6f} seconds")

# Measure time for GPU, if available
if torch.cuda.is_available():
    gpu_device = torch.device('cuda')
    gpu_time = measure_time(gpu_device)
    print(f"GPU time: {gpu_time:.6f} seconds")
else:
    print("CUDA is not available on this device.")