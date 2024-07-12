import torch
import time
import logging

# Set up logging
logging.basicConfig(filename='gpu_test_log.log', level=logging.INFO, format='%(asctime)s - %(message)s')

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
logging.info(f"CPU time: {cpu_time:.6f} seconds")

# Measure time for single GPU, if available
if torch.cuda.is_available():
    gpu_device = torch.device('cuda')
    gpu_time = measure_time(gpu_device)
    logging.info(f"Single GPU time: {gpu_time:.6f} seconds")
else:
    logging.info("CUDA is not available on this device.")

# Measure time for multiple GPUs, if available
if torch.cuda.device_count() > 1:
    multi_gpu_device = torch.device('cuda')
    
    # Function to perform matrix multiplication on multiple GPUs
    def measure_time_multi_gpu(device, size=10000):
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        a = a.to(device)
        b = b.to(device)
        
        # Use DataParallel to parallelize the computation across multiple GPUs
        a = torch.nn.DataParallel(a)
        b = torch.nn.DataParallel(b)
        
        start_time = time.time()
        c = torch.mm(a.module, b.module)
        torch.cuda.synchronize()
        end_time = time.time()
        
        return end_time - start_time
    
    multi_gpu_time = measure_time_multi_gpu(multi_gpu_device)
    logging.info(f"Multi-GPU time: {multi_gpu_time:.6f} seconds")
else:
    logging.info("Multiple GPUs are not available on this device.")
