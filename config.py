"""
Configuration settings for Phi-4 Agent
Adjust these settings based on your hardware capabilities.
"""

import torch

# Model configuration
MODEL_ID = "microsoft/phi-4"

# Device settings
# Options: "cuda" (GPU), "cpu", or None (auto-detect)
DEVICE = None  # Auto-detect

# Precision settings
# For GPU: torch.float16 (faster, less memory) or torch.float32 (slower, more accurate)
# For CPU: torch.float32 (recommended)
TORCH_DTYPE = None  # Auto-detect based on device

# Generation parameters
TEMPERATURE = 0.7  # Creativity (0.0 = deterministic, 1.0 = very creative)
MAX_NEW_TOKENS = 512  # Maximum response length
TOP_P = 0.95  # Nucleus sampling
REPETITION_PENALTY = 1.1  # Penalize repetition

# Agent settings
VERBOSE = True  # Show agent reasoning steps
MAX_ITERATIONS = 5  # Maximum agent iterations

# Performance presets
PRESETS = {
    "fast": {
        "max_new_tokens": 256,
        "temperature": 0.5,
    },
    "balanced": {
        "max_new_tokens": 512,
        "temperature": 0.7,
    },
    "creative": {
        "max_new_tokens": 1024,
        "temperature": 0.9,
    },
    "deterministic": {
        "max_new_tokens": 512,
        "temperature": 0.0,
    }
}


def get_preset(preset_name: str) -> dict:
    """
    Get a configuration preset.
    
    Args:
        preset_name: One of "fast", "balanced", "creative", "deterministic"
        
    Returns:
        Dictionary of configuration parameters
    """
    return PRESETS.get(preset_name, PRESETS["balanced"])


def detect_optimal_device():
    """Detect the best available device for inference."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
        
        if gpu_memory < 8:
            print("Warning: GPU has less than 8GB VRAM. Consider using CPU.")
            return "cpu"
        return "cuda"
    else:
        print("No GPU detected. Using CPU (inference will be slower)")
        return "cpu"


def get_optimal_dtype(device: str):
    """Get optimal dtype for the given device."""
    if device == "cuda":
        return torch.float16  # Faster and uses less VRAM
    else:
        return torch.float32  # CPU requires float32


if __name__ == "__main__":
    # Test device detection
    print("Testing device detection...")
    device = detect_optimal_device()
    dtype = get_optimal_dtype(device)
    print(f"\nRecommended settings:")
    print(f"  Device: {device}")
    print(f"  Dtype: {dtype}")
    
    print("\nAvailable presets:")
    for name, settings in PRESETS.items():
        print(f"  {name}: {settings}")
