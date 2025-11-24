# Getting Started with Phi-4 Local Agent

## Quick Start Guide

### Step 1: Install Dependencies

First, install all required packages:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**For GPU support** (recommended for faster inference):
```powershell
# Check your CUDA version first: nvidia-smi
# Then install appropriate PyTorch version

# For CUDA 11.8:
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 2: Verify Setup

Run the quickstart script to check your system:

```powershell
.\.venv\Scripts\python.exe quickstart.py
```

This will:
- ✓ Check all dependencies are installed
- ✓ Verify you have enough RAM and disk space
- ✓ Detect GPU (if available)
- ✓ Recommend optimal configuration
- ✓ Optionally test the model

### Step 3: Run Your First Agent

Create a simple script or run this in Python:

```python
from phi4_agent import Phi4Agent

# Create the agent (model downloads automatically on first run)
agent = Phi4Agent(verbose=True)

# Ask a question
response = agent.run("What is 15 multiplied by 23?")
print(response)
```

### Step 4: Explore Examples

Run the examples script:

```powershell
.\.venv\Scripts\python.exe examples.py
```

## Understanding the First Run

**Important**: The first time you run the agent:
1. The Phi-4 model (~15GB) will download automatically
2. This may take 10-30 minutes depending on your internet speed
3. The model is cached in `~/.cache/huggingface/`
4. Subsequent runs will be much faster (loading from cache)

Progress will look like this:
```
Loading Phi-4 model on cuda with dtype torch.float16...
This may take a few minutes on first run as the model downloads...
Downloading model files... [progress bars]
Model loaded successfully!
Model size: ~14.7B parameters
```

## Hardware-Specific Instructions

### If You Have a GPU (NVIDIA)

**Recommended settings for best performance:**

```python
import torch
from phi4_agent import Phi4Agent

agent = Phi4Agent(
    device="cuda",              # Use GPU
    torch_dtype=torch.float16,  # Half precision (faster, less VRAM)
    max_new_tokens=512,
    verbose=True
)
```

**Requirements:**
- NVIDIA GPU with 8GB+ VRAM
- CUDA toolkit installed
- Expected inference time: 1-3 seconds per response

### If You're Using CPU

**Recommended settings for CPU:**

```python
from phi4_agent import Phi4Agent

agent = Phi4Agent(
    device="cpu",              # Force CPU
    max_new_tokens=256,        # Reduce for faster generation
    verbose=True
)
```

**Requirements:**
- 16GB+ RAM recommended
- Expected inference time: 10-30 seconds per response

**Tip**: Use shorter `max_new_tokens` (128-256) for faster responses on CPU.

## Common Use Cases

### 1. Simple Question Answering

```python
from phi4_agent import Phi4Agent

agent = Phi4Agent()
response = agent.run("Explain what a neural network is")
print(response)
```

### 2. Mathematical Calculations

```python
agent = Phi4Agent()
response = agent.run("Calculate 1547 * 892")
print(response)
```

### 3. Multi-step Reasoning

```python
agent = Phi4Agent()
question = """
If a car travels 60 miles per hour for 2.5 hours,
how far does it travel? Then, if it continues at
75 mph for another hour, what's the total distance?
"""
response = agent.run(question)
print(response)
```

### 4. Using Custom Tools

```python
from phi4_agent import Phi4Agent
from langchain.tools import Tool

# Define a custom tool
def search_database(query: str) -> str:
    # Your database search logic here
    return f"Found results for: {query}"

# Create tools list
tools = [
    Tool(
        name="DatabaseSearch",
        func=search_database,
        description="Search the database for information"
    )
]

# Create agent with custom tools
agent = Phi4Agent(tools=tools)
response = agent.run("Search for customer records")
```

### 5. Using Configuration Presets

```python
from phi4_agent import Phi4Agent
from config import get_preset

# Use a preset for faster responses
fast_settings = get_preset("fast")
agent = Phi4Agent(**fast_settings)

# Or creative responses
creative_settings = get_preset("creative")
agent = Phi4Agent(**creative_settings)
```

## Optimizing Performance

### Reduce Memory Usage

```python
agent = Phi4Agent(
    max_new_tokens=256,        # Reduce from default 512
    torch_dtype=torch.float16, # Use half precision on GPU
)
```

### Faster Responses

```python
agent = Phi4Agent(
    max_new_tokens=128,        # Generate less text
    temperature=0.5,           # Less randomness
)
```

### Better Quality

```python
agent = Phi4Agent(
    max_new_tokens=1024,       # Allow longer responses
    temperature=0.7,           # Balanced creativity
)
```

## Troubleshooting First-Time Setup

### Model Download Fails

**Problem**: Download interrupted or fails

**Solution**:
- Check internet connection
- Ensure 20GB+ free disk space
- Try again - HuggingFace has automatic resume
- Clear cache: Delete `~/.cache/huggingface/hub/models--microsoft--phi-4/`

### Out of Memory on GPU

**Problem**: `CUDA out of memory` error

**Solution**:
```python
# Use CPU instead
agent = Phi4Agent(device="cpu")

# Or reduce max_new_tokens
agent = Phi4Agent(max_new_tokens=128)
```

### Out of Memory on CPU

**Problem**: System runs out of RAM

**Solution**:
- Close other applications
- Reduce `max_new_tokens` to 128 or 256
- Consider using a machine with more RAM

### Slow on CPU

**Problem**: Taking too long to respond

**Solution**:
- This is normal for CPU inference (10-30 seconds)
- Reduce `max_new_tokens` for faster responses
- Consider using a GPU-enabled machine

### Import Errors

**Problem**: `ModuleNotFoundError`

**Solution**:
```powershell
# Reinstall dependencies
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Make sure you're using the virtual environment
```

## Next Steps

1. **Read the main README.md** for full documentation
2. **Run examples.py** to see different use cases
3. **Check config.py** for performance presets
4. **Experiment with custom tools** for your specific needs

## Getting Help

If you encounter issues:
1. Run `quickstart.py` to verify your setup
2. Check the "Troubleshooting" section in README.md
3. Ensure you have sufficient hardware resources
4. Try the examples first to verify basic functionality

## Tips for Success

✓ **First run takes time**: Be patient during the initial model download
✓ **Use GPU if available**: 10-20x faster than CPU
✓ **Start simple**: Test with basic examples before complex tasks
✓ **Monitor resources**: Use Task Manager to check RAM/GPU usage
✓ **Adjust settings**: Tune `max_new_tokens` and `temperature` for your needs
✓ **Cache is your friend**: Model loads much faster after first download

Happy building! 🚀
