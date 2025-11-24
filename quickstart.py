"""
Quick Start Script for Phi-4 Agent
This script helps you test your setup and ensures everything is working.
"""

import sys


def check_dependencies():
    """Check if all required packages are installed."""
    print("Checking dependencies...\n")
    
    required_packages = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "langchain": "LangChain",
        "langchain_huggingface": "LangChain HuggingFace",
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print("\n⚠️  Missing packages detected!")
        print("Please run: .\.venv\Scripts\python.exe -m pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed!\n")
    return True


def check_hardware():
    """Check hardware capabilities."""
    print("Checking hardware...\n")
    
    import torch
    import psutil
    
    # Check RAM
    ram_gb = psutil.virtual_memory().total / 1e9
    print(f"System RAM: {ram_gb:.1f} GB")
    
    if ram_gb < 16:
        print("⚠️  Warning: Less than 16GB RAM. You may experience issues.")
    else:
        print("✓ Sufficient RAM")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 8:
            print("⚠️  Warning: Less than 8GB VRAM. Consider using CPU mode.")
        else:
            print("✓ GPU has sufficient VRAM")
        
        print(f"CUDA Version: {torch.version.cuda}")
        recommended_device = "cuda"
        recommended_dtype = "torch.float16"
    else:
        print("\nNo GPU detected")
        print("ℹ️  Inference will run on CPU (slower but will work)")
        recommended_device = "cpu"
        recommended_dtype = "torch.float32"
    
    print(f"\nRecommended configuration:")
    print(f"  device='{recommended_device}'")
    print(f"  torch_dtype={recommended_dtype}")
    
    return recommended_device


def check_disk_space():
    """Check available disk space."""
    print("\nChecking disk space...\n")
    
    import shutil
    
    # Check cache directory
    from pathlib import Path
    cache_dir = Path.home() / ".cache" / "huggingface"
    
    if cache_dir.exists():
        total, used, free = shutil.disk_usage(cache_dir)
    else:
        total, used, free = shutil.disk_usage(Path.home())
    
    free_gb = free / 1e9
    print(f"Free disk space: {free_gb:.1f} GB")
    
    if free_gb < 20:
        print("⚠️  Warning: Less than 20GB free. Model download may fail.")
        return False
    else:
        print("✓ Sufficient disk space")
        return True
    

def test_basic_inference():
    """Test basic model inference."""
    print("\n" + "="*60)
    print("Testing Phi-4 Agent (this may take a few minutes)...")
    print("="*60 + "\n")
    
    try:
        from phi4_agent import Phi4Agent
        
        print("Creating agent (model will download if not cached)...\n")
        agent = Phi4Agent(verbose=False, max_new_tokens=128)
        
        print("Running test query...\n")
        response = agent.run("What is 5 + 7?")
        
        print("="*60)
        print("TEST RESULT")
        print("="*60)
        print(f"Response: {response}\n")
        
        if "12" in response:
            print("✓ Agent is working correctly!")
        else:
            print("⚠️  Unexpected response. Agent may need adjustment.")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def main():
    """Run all checks."""
    print("\n" + "="*60)
    print("PHI-4 AGENT - QUICK START")
    print("="*60 + "\n")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check hardware
    try:
        check_hardware()
    except ImportError:
        print("Cannot check hardware - torch not installed properly")
        sys.exit(1)
    
    # Check disk space
    if not check_disk_space():
        print("\n⚠️  Proceeding with caution due to low disk space...")
    
    # Ask user if they want to test
    print("\n" + "="*60)
    user_input = input("\nRun test inference? This will download the model (~15GB) [y/N]: ")
    
    if user_input.lower() in ['y', 'yes']:
        test_basic_inference()
    else:
        print("\nSkipping test. You can run examples.py when ready.")
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run examples: .\.venv\Scripts\python.exe examples.py")
    print("2. Import in your code: from phi4_agent import Phi4Agent")
    print("3. Check config.py for performance presets")
    print("\n")


if __name__ == "__main__":
    main()
