
import sys
import platform
import torch

def check_gpu():
    print("="*40)
    print("PyTorch GPU Availability Check")
    print("="*40)
    
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    try:
        import torchvision
        print(f"Torchvision Version: {torchvision.__version__}")
    except ImportError:
        print("Torchvision Version: Not Installed")
        
    print(f"OS: {platform.system()} {platform.release()}")
    
    is_cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {is_cuda_available}")
    
    if is_cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"Number of GPUs: {device_count}")
        
        for i in range(device_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
        print("\nTesting Tensor Operation on GPU...")
        try:
            x = torch.rand(5, 3).cuda()
            print("Tensor successfully created on GPU:")
            print(x)
        except Exception as e:
            print(f"Error moving tensor to GPU: {e}")
    else:
        print("\nWARNING: CUDA is not available. PyTorch is running on CPU.")
        print("This might be due to:")
        print("1. No GPU installed.")
        print("2. Incorrect PyTorch version installed (CPU-only build).")
        print("3. Missing CUDA drivers.")

    print("="*40)

if __name__ == "__main__":
    check_gpu()
