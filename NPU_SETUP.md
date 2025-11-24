# NPU Support for Phi-4 Agent

## Current Status: ⚠️ Network Required

NPU support has been **added to the code** but requires additional packages that need internet connectivity.

## What You Have Now:

✅ **Code is NPU-ready** - The `phi4_agent.py` includes NPU support via OpenVINO
✅ **OpenVINO installed** - Base package is installed
✅ **CPU mode working** - Agent runs on CPU without issues

## To Enable NPU (When Network is Available):

### Step 1: Install Required Packages
```powershell
.\.venv\Scripts\python.exe -m pip install optimum-intel nncf
```

### Step 2: Enable NPU in Code
Change line 322 in `phi4_agent.py`:
```python
# Change from:
agent = Phi4Agent(verbose=True, use_npu=False)

# To:
agent = Phi4Agent(verbose=True, use_npu=True)
```

## What is NPU?

**NPU (Neural Processing Unit)** is Intel's AI accelerator found in:
- Intel Core Ultra processors (Meteor Lake and newer)
- Some Intel Arc GPUs
- Designed specifically for AI workloads

## How NPU Support Works:

1. **OpenVINO** - Intel's toolkit for optimizing and deploying AI models
2. **Optimum Intel** - Integration between HuggingFace and OpenVINO
3. **Model Conversion** - Automatically converts Phi-4 to OpenVINO IR format
4. **NPU Execution** - Runs inference on Intel NPU hardware

## Checking if You Have NPU:

```powershell
# Check for Intel NPU
Get-WmiObject Win32_PnPEntity | Where-Object { $_.Name -like "*NPU*" }

# Check your processor
systeminfo | findstr /C:"Processor"
```

## Performance Expectations:

| Device | Speed | Power Efficiency |
|--------|-------|------------------|
| **CPU** | Baseline | Normal |
| **NPU** | 2-3x faster | 50% less power |
| **NVIDIA GPU** | 10-20x faster | High power |

## Current Code Features:

The `phi4_agent.py` already includes:

```python
class Phi4Agent:
    def __init__(
        self,
        use_npu: bool = False,  # ✅ NPU parameter available
        ...
    ):
```

When `use_npu=True`:
- Uses `OVModelForCausalLM` instead of `AutoModelForCausalLM`
- Exports model to OpenVINO IR format
- Runs inference on NPU device
- Falls back to CPU if NPU not available

## Troubleshooting:

### Error: "OpenVINO is not installed"
```powershell
.\.venv\Scripts\python.exe -m pip install openvino optimum optimum-intel
```

### Error: "NPU device not found"
- Your system may not have an NPU
- Check processor model (needs Intel Core Ultra or newer)
- OpenVINO will fall back to CPU

### Network Errors
- The packages `optimum-intel` and `nncf` require internet
- Download on a different network if corporate firewall blocks
- Or use CPU mode (current setup)

## Alternative: Manual NPU Setup

If pip install fails, download wheels manually:

1. Go to https://pypi.org/project/optimum-intel/#files
2. Download `optimum_intel-1.23.0-py3-none-any.whl`
3. Install locally:
```powershell
.\.venv\Scripts\python.exe -m pip install path\to\optimum_intel-1.23.0-py3-none-any.whl
```

## Current Recommendation:

✅ **Use CPU mode** (working now) until you have reliable internet access
🔄 **Enable NPU later** when you can install `optimum-intel` package

The agent works great on CPU - NPU is an optimization, not a requirement!
