# Phi-4 Agent with LangChain (Local Model)

A LangChain-based agent implementation using Microsoft's Phi-4 model running locally. This project demonstrates how to create an intelligent agent that can use tools and reasoning to solve complex tasks, all running on your own hardware without API calls.

## Features

- 🤖 **Local Phi-4 Model**: Downloads and runs Microsoft's Phi-4 model locally using transformers
- 🔧 **Tool Support**: Extensible agent with custom tool integration
- 💬 **Conversational**: Memory-enabled agent for context-aware interactions
- 🎯 **ReAct Pattern**: Implements the ReAct (Reasoning + Acting) framework
- 🚀 **Easy to Use**: Simple API for both agent and standalone LLM usage
- 🔒 **Privacy**: All inference happens locally - no data sent to external APIs
- ⚡ **GPU Accelerated**: Automatically uses CUDA if available, falls back to CPU

## Prerequisites

- Python 3.8+
- **Disk Space**: ~15GB for the Phi-4 model
- **RAM**: 16GB+ recommended for CPU inference
- **GPU** (Optional but recommended): 
  - NVIDIA GPU with 8GB+ VRAM for faster inference
  - CUDA toolkit installed
- Internet connection for initial model download

## Installation

1. **Clone or navigate to this directory**

2. **Install dependencies:**
   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

   **Note**: Installing PyTorch with CUDA support (for GPU acceleration):
   ```powershell
   # For CUDA 11.8
   .\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   .\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Verify your setup** (recommended):
   ```powershell
   .\.venv\Scripts\python.exe quickstart.py
   ```
   This will check your system, verify dependencies, and optionally run a test inference.

4. **First Run**: The model will automatically download (~15GB) on first use and be cached for future runs.

## System Requirements

### Minimum (CPU)
- 16GB RAM
- 20GB free disk space
- Inference time: ~10-30 seconds per response

### Recommended (GPU)
- NVIDIA GPU with 8GB+ VRAM
- 16GB system RAM
- 20GB free disk space
- Inference time: ~1-3 seconds per response

## Usage

### Basic Agent Usage

```python
from phi4_agent import Phi4Agent

# Create the agent (model downloads on first run)
# Uses GPU automatically if available, otherwise CPU
agent = Phi4Agent(verbose=True)

# Ask a question
response = agent.run("What is 25 multiplied by 17?")
print(response)
```

### Specify Device and Precision

```python
import torch
from phi4_agent import Phi4Agent

# Force CPU usage
agent = Phi4Agent(device="cpu", verbose=True)

# Use GPU with float16 for faster inference
agent = Phi4Agent(
    device="cuda",
    torch_dtype=torch.float16,
    verbose=True
)
```

### Using Custom Tools

```python
from phi4_agent import Phi4Agent
from langchain.tools import Tool

# Define a custom tool
def get_weather(location: str) -> str:
    # Your weather API logic here
    return f"Weather in {location}: Sunny, 72°F"

custom_tools = [
    Tool(
        name="Weather",
        func=get_weather,
        description="Get current weather for a location."
    )
]

# Create agent with custom tools
agent = Phi4Agent(tools=custom_tools, verbose=True)
response = agent.run("What's the weather in New York?")
```

### Standalone LLM (No Agent)

```python
from phi4_agent import create_phi4_llm

# Create standalone LLM
llm = create_phi4_llm(temperature=0.7, device="cuda")

# Generate text
response = llm.invoke("Explain machine learning in simple terms:")
print(response)
```

### Adding Tools Dynamically

```python
from phi4_agent import Phi4Agent
from langchain.tools import Tool

agent = Phi4Agent()

# Define and add a new tool
new_tool = Tool(
    name="CustomTool",
    func=lambda x: f"Processed: {x}",
    description="A custom tool that processes input."
)

agent.add_tool(new_tool)
```

## Running Examples

The project includes several examples demonstrating different use cases:

```powershell
# Verify setup first (recommended)
.\.venv\Scripts\python.exe quickstart.py

# Run all examples
.\.venv\Scripts\python.exe examples.py

# Or run the main agent script
.\.venv\Scripts\python.exe phi4_agent.py

# Check optimal configuration for your hardware
.\.venv\Scripts\python.exe config.py
```

## Project Structure

```
phi/
├── .env.example        # Environment variables template
├── .venv/              # Virtual environment
├── requirements.txt    # Python dependencies
├── phi4_agent.py       # Main agent implementation
├── examples.py         # Usage examples
├── config.py           # Configuration presets
├── quickstart.py       # Setup verification script
└── README.md           # This file
```

## Configuration Options

When creating a `Phi4Agent`, you can customize:

- `model_id`: HuggingFace model ID (default: "microsoft/phi-4")
- `temperature`: Sampling temperature 0.0-1.0 (default: 0.7)
- `max_new_tokens`: Maximum tokens to generate (default: 512)
- `device`: Device to run on - "cuda", "cpu", or None for auto-detect
- `torch_dtype`: Precision - torch.float16 (GPU), torch.float32 (CPU), or None for auto
- `tools`: List of LangChain tools for the agent
- `verbose`: Show agent reasoning steps (default: True)

## Example Use Cases

1. **Mathematical Calculations**: Use built-in calculator tool
2. **Information Retrieval**: Add custom API tools for weather, news, etc.
3. **Text Processing**: String manipulation and analysis
4. **Reasoning Tasks**: Multi-step problem solving
5. **Custom Workflows**: Chain multiple tools together

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce `max_new_tokens` (try 256 or 128)
   - Use CPU instead of GPU if VRAM is insufficient
   - Close other applications to free up RAM
   - Use `torch.float16` on GPU to reduce memory usage

2. **Slow Inference on CPU**
   - CPU inference is significantly slower than GPU
   - Consider using a GPU or reducing `max_new_tokens`
   - First response is slower due to model loading

3. **Model Download Fails**
   - Check your internet connection
   - Ensure you have sufficient disk space (~15GB)
   - Try downloading again - HuggingFace has automatic resume

4. **CUDA Out of Memory**
   - Your GPU doesn't have enough VRAM
   - Try `device="cpu"` or use a machine with more VRAM
   - Reduce batch size or max_new_tokens

5. **Import Errors**
   - Run `pip install -r requirements.txt`
   - Make sure you're using the virtual environment
   - For GPU support, install PyTorch with CUDA

## Advanced Usage

### Custom Prompt Templates

```python
from langchain.prompts import PromptTemplate

# Modify the agent's prompt template in phi4_agent.py
# to customize the agent's behavior
```

### Memory Management

```python
# The agent includes conversation memory by default
# Access it via agent.memory for custom operations
```

## Contributing

Feel free to extend this project with:
- Additional tools
- Custom prompt templates
- Error handling improvements
- Performance optimizations

## Resources

- [Phi-4 Model on HuggingFace](https://huggingface.co/microsoft/phi-4)
- [LangChain Documentation](https://python.langchain.com/)
- [HuggingFace Hub Documentation](https://huggingface.co/docs/hub/index)

## License

This project is open source and available for educational and commercial use.

## Notes

- Phi-4 is optimized for reasoning and problem-solving tasks
- The model works best with clear, specific instructions
- **First run**: Model downloads automatically (~15GB, may take 10-30 minutes)
- **Subsequent runs**: Model loads from cache (much faster)
- **GPU recommended**: Inference is 10-20x faster on GPU vs CPU
- **Memory usage**: ~8GB VRAM (GPU) or ~16GB RAM (CPU)
- All inference happens locally - your data never leaves your machine

## Performance Tips

1. **Use GPU if available**: Set `device="cuda"` and `torch_dtype=torch.float16`
2. **Reduce token generation**: Lower `max_new_tokens` for faster responses
3. **Batch processing**: Process multiple queries in sequence to amortize model loading time
4. **Model caching**: The model is cached after first download in `~/.cache/huggingface/`

---

**Happy Building! 🚀**
