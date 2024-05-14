import re
import torch
import bittensor as bt


def contains_gpu_index_in_device(device: str) -> bool:
    pattern = r"^cuda:\d+$"
    return bool(re.match(pattern, device))

def calculate_single_gpu_requirements(device: str, max_allowed_memory_allocation_in_bytes: int):
    if contains_gpu_index_in_device(device):
        device_with_gpu_index = device
    else:
        device_with_gpu_index = torch.cuda.current_device()

    torch.cuda.synchronize()
    global_free, total_gpu_memory = torch.cuda.mem_get_info(
        device=device_with_gpu_index
    )

    bt.logging.info(f"Available free memory: {round(global_free / 10e8, 2)} GB")
    bt.logging.info(f"Total gpu memory {round(total_gpu_memory / 10e8, 2)} GB")

    if global_free < max_allowed_memory_allocation_in_bytes:
        raise torch.cuda.CudaError(
            f"Not enough memory to allocate for the model. Please ensure you have at least {max_allowed_memory_allocation_in_bytes / 10e8} GB of free GPU memory."
        )

    gpu_utilization = round(max_allowed_memory_allocation_in_bytes / global_free, 2)
    bt.logging.info(
        f'{gpu_utilization * 100}% of the GPU memory will be utilized for loading the model to device "{device}".'
    )

    return gpu_utilization    

def calculate_multiple_gpu_requirements(device: str, gpus: int, max_allowed_memory_allocation_in_bytes: int):     
    torch.cuda.synchronize()
    total_free_memory = 0
    total_gpu_memory = 0
    
    for i in range(gpus):
        gpu_device = f"cuda:{i}"
        global_free, total_memory = torch.cuda.mem_get_info(device=gpu_device)
        total_free_memory += global_free
        total_gpu_memory += total_memory

    bt.logging.info(f"Total available free memory across all visible {gpus} GPUs: {round(total_free_memory / 10e8, 2)} GB")
    bt.logging.info(f"Total GPU memory across all visible GPUs: {gpus} {round(total_gpu_memory / 10e8, 2)} GB")

    if total_free_memory < max_allowed_memory_allocation_in_bytes:
        raise torch.cuda.CudaError(
            f"Not enough memory across all specified {gpus} GPUs to allocate for the model. Please ensure you have at least {max_allowed_memory_allocation_in_bytes / 10e8} GB of free GPU memory."
        )

    gpu_utilization = round(max_allowed_memory_allocation_in_bytes / total_free_memory, 2)
    bt.logging.info(
        f"{gpu_utilization * 100}% of the total GPU memory across all GPUs will be utilized for loading the model."
    )

    return gpu_utilization


def calculate_gpu_requirements(
    device: str, gpus: int, max_allowed_memory_allocation_in_bytes: float,
) -> float:
    """Calculates the memory utilization requirements for the model to be loaded on the device.
    Args:
        device (str): The device to load the model to.
        max_allowed_memory_allocation_in_bytes (int, optional): The maximum allowed memory allocation in bytes. Defaults to 20e9 (20GB).
    """
    if gpus == 1:
        return calculate_single_gpu_requirements(device, max_allowed_memory_allocation_in_bytes) 
    else:    
        return calculate_multiple_gpu_requirements(device, gpus, max_allowed_memory_allocation_in_bytes=max_allowed_memory_allocation_in_bytes)
