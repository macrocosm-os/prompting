import re
import torch
import bittensor as bt

def contains_gpu_index_in_device(device: str) -> bool:
    pattern = r'^cuda:\d+$'
    return bool(re.match(pattern, device))


def calculate_gpu_requirements(device: str, max_allowed_memory_allocation_in_bytes: int = 20e9) -> float:        
    """Calculates the memory utilization requirements for the model to be loaded on the device.
    Args:
        device (str): The device to load the model to.
        max_allowed_memory_allocation_in_bytes (int, optional): The maximum allowed memory allocation in bytes. Defaults to 20e9 (20GB).
    """
    if contains_gpu_index_in_device(device):
        device_with_gpu_index = device
    else:
        device_with_gpu_index = torch.cuda.current_device()
        
    torch.cuda.synchronize()
    global_free, total_gpu_memory = torch.cuda.mem_get_info(device=device_with_gpu_index)

    bt.logging.info(f'Available free memory: {round(global_free / 10e8, 2)} GB')
    bt.logging.info(f'Total gpu memory {round(total_gpu_memory / 10e8, 2)} GB')

    if global_free < max_allowed_memory_allocation_in_bytes:
        raise torch.cuda.CudaError(f'Not enough memory to allocate for the model. Please ensure you have at least {max_allowed_memory_allocation_in_bytes / 10e8} GB of free GPU memory.')

    gpu_utilization = round(max_allowed_memory_allocation_in_bytes / global_free, 2)
    bt.logging.info(f'{gpu_utilization * 100}% of the GPU memory will be utilized for loading the model to device "{device}".')

    return gpu_utilization
