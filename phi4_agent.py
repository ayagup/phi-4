"""
Phi-4 Agent using LangChain with Local Model
This module provides a LangChain agent powered by Microsoft's Phi-4 model running locally.
"""

import os
import sys
import warnings

# Prevent torchvision import issues by disabling vision features
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Block torchvision import to avoid compatibility issues
class BlockTorchvisionImport:
    def find_spec(self, fullname, path, target=None):
        if fullname == 'torchvision' or fullname.startswith('torchvision.'):
            return None
        return None

sys.meta_path.insert(0, BlockTorchvisionImport())

from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from collections import deque

# LangChain imports
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_core.language_models import BaseLLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import Generation, LLMResult
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, BitsAndBytesConfig
import torch

# OpenVINO imports for NPU support
try:
    from optimum.intel import OVModelForCausalLM
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    OVModelForCausalLM = None

# Load environment variables
load_dotenv()


class Phi4Agent:
    """
    A LangChain agent using the Phi-4 model running locally.
    
    The Phi-4 model is Microsoft's latest small language model optimized for
    reasoning and problem-solving tasks. This implementation downloads and runs
    the model locally using transformers.
    """
    
    def __init__(
        self,
        # model_id: str = "microsoft/phi-4",
        model_id: str = "C:\\Users\\mayangupta\\Documents\\models\\phi-4-int8-openvino",
        temperature: float = 0.3,
        max_new_tokens: int = 8192,
        max_context_length: int = 16384,  # Maximum context window for Phi-4
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        tools: Optional[List[Tool]] = None,
        verbose: bool = True,
        use_intel_gpu: bool = False,  # Intel integrated GPU via OpenVINO
        use_npu: bool = False,  # Intel NPU support
        load_in_8bit: bool = False,  # 8-bit quantization
        load_in_4bit: bool = False  # 4-bit quantization (NF4)
    ):
        """
        Initialize the Phi-4 Agent with local model.
        
        Args:
            model_id: The HuggingFace model ID for Phi-4 or local path
            temperature: Sampling temperature (0.0 to 1.0)
            max_new_tokens: Maximum number of tokens to generate
            max_context_length: Maximum context window size (default: 16384 for Phi-4)
            device: Device to run model on ('cuda', 'cpu', 'gpu', 'npu', or None for auto-detect)
            torch_dtype: Torch dtype for model (None for auto, torch.float16 recommended for GPU)
            tools: List of LangChain tools for the agent to use
            verbose: Whether to print agent reasoning steps
            use_intel_gpu: Whether to use Intel integrated GPU via OpenVINO (recommended)
            use_npu: Whether to use Intel NPU via OpenVINO
            load_in_8bit: Whether to load model with 8-bit quantization (CUDA only)
            load_in_4bit: Whether to load model with 4-bit quantization (NF4, CUDA only)
        """
        # Check if Intel GPU is requested
        if use_intel_gpu:
            if not OPENVINO_AVAILABLE:
                print("Warning: OpenVINO not fully installed. Falling back to CPU.")
                print("For Intel GPU acceleration, install: pip install optimum-intel")
                device = "cpu"
            else:
                device = "gpu"
                print("Intel GPU acceleration enabled via OpenVINO")
        
        # Check if NPU is requested
        elif use_npu:
            if not OPENVINO_AVAILABLE:
                print("Warning: OpenVINO not fully installed. Falling back to CPU.")
                print("For Intel NPU acceleration, install: pip install optimum-intel")
                device = "cpu"
            else:
                device = "npu"
                print("Intel NPU acceleration enabled via OpenVINO")
        
        # Detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Verify CUDA is actually available if requested
        if device == "cuda" and not torch.cuda.is_available():
            print(f"Warning: CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        
        # Quantization requires CUDA or GPU
        if (load_in_8bit or load_in_4bit) and device not in ["cuda", "gpu"]:
            print(f"Warning: Quantization requires CUDA or GPU. Disabling quantization.")
            load_in_8bit = False
            load_in_4bit = False
        
        # Can't use both 8-bit and 4-bit
        if load_in_8bit and load_in_4bit:
            print("Warning: Cannot use both 8-bit and 4-bit quantization. Using 4-bit.")
            load_in_8bit = False
        
        # Set dtype based on device if not specified
        if torch_dtype is None:
            if load_in_8bit or load_in_4bit:
                torch_dtype = None  # Let bitsandbytes handle dtype
            else:
                torch_dtype = torch.float16 if device in ["cuda", "gpu"] else torch.float32
        
        print(f"Loading Phi-4 model on {device.upper()}...")
        if load_in_4bit:
            print("Using 4-bit quantization (NF4, reduces memory usage by ~75%)...")
        elif load_in_8bit:
            print("Using 8-bit quantization (reduces memory usage by ~50%)...")
        if device in ["gpu", "npu"]:
            print(f"Using OpenVINO for Intel {device.upper()} acceleration...")
        else:
            print("This may take a few minutes on first run...")
        
        # Disable SSL verification if needed (for corporate proxies)
        import os
        os.environ.setdefault('CURL_CA_BUNDLE', '')
        os.environ.setdefault('REQUESTS_CA_BUNDLE', '')
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        # Load model based on device
        if device in ["gpu", "npu"]:
            # Use OpenVINO for Intel GPU/NPU acceleration
            device_name = "GPU" if device == "gpu" else "NPU"
            print(f"Loading model with OpenVINO backend for Intel {device_name}...")
            self.model = OVModelForCausalLM.from_pretrained(
                model_id,
                export=True if load_in_4bit or load_in_8bit else False,  # Export to OpenVINO IR format if needed
                device=device_name,  # Use GPU or NPU device
                trust_remote_code=True,
                ov_config={"PERFORMANCE_HINT": "LATENCY"}  # Optimize for responsiveness
            )
        elif load_in_4bit:
            # Use 4-bit quantization (NF4 - Normal Float 4)
            print("Loading model with 4-bit NF4 quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  # Use NormalFloat4 quantization
                bnb_4bit_use_double_quant=True,  # Double quantization for extra memory savings
                bnb_4bit_compute_dtype=torch.float16  # Compute in float16 for speed
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        elif load_in_8bit:
            # Use 8-bit quantization
            print("Loading model with 8-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # Use standard PyTorch model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
        
        # Only move to CPU for standard PyTorch models (not OpenVINO or quantized models)
        if device == "cpu" and not (load_in_8bit or load_in_4bit):
            self.model = self.model.to(device)
        
        # Store device for later use
        self.device = device
        self.use_openvino = device in ["gpu", "npu"]
        
        # For OpenVINO models, we'll use direct generation instead of pipeline
        if self.use_openvino:
            print("Using direct generation with OpenVINO model...")
            self.llm = None  # We'll implement custom generation
            self.max_new_tokens = max_new_tokens
            self.temperature = temperature
        else:
            # Create text generation pipeline for PyTorch models
            text_pipeline = TextGenerationPipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                repetition_penalty=1.1,
                do_sample=True if temperature > 0 else False,
            )
            
            # Wrap pipeline in LangChain HuggingFacePipeline
            self.llm = HuggingFacePipeline(pipeline=text_pipeline)
        
        print("Model loaded successfully!")
        if hasattr(self.model, 'num_parameters'):
            print(f"Model size: ~{self.model.num_parameters() / 1e9:.2f}B parameters")
        if load_in_4bit:
            print("4-bit NF4 quantization active - using ~75% less memory")
        elif load_in_8bit:
            print("8-bit quantization active - using ~50% less memory")
        
        # Set up default tools if none provided
        self.tools = tools or self._get_default_tools()
        
        # Store max context length and create chat memory
        self.max_context_length = max_context_length
        self.chat_memory = deque(maxlen=100)  # Store last 100 messages
        
        # Create the agent (LangGraph manages state internally)
        self.agent = self._create_agent(verbose)
    
    def _get_default_tools(self) -> List[Tool]:
        """Get default tools for the agent."""
        
        def calculator(expression: str) -> str:
            """Evaluate a mathematical expression."""
            try:
                result = eval(expression, {"__builtins__": {}}, {})
                return str(result)
            except Exception as e:
                return f"Error: {str(e)}"
        
        def string_length(text: str) -> str:
            """Get the length of a string."""
            return str(len(text))
        
        return [
            Tool(
                name="Calculator",
                func=calculator,
                description="Useful for performing mathematical calculations. Input should be a valid mathematical expression."
            ),
            Tool(
                name="StringLength",
                func=string_length,
                description="Returns the length of a given string. Input should be the string to measure."
            )
        ]
    
    def _create_agent(self, verbose: bool):
        """Create a simple agent using the Phi-4 model with tool access."""
        
        # Build tool descriptions
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}" 
            for tool in self.tools
        ])
        
        self.verbose = verbose
        self.tool_descriptions = tool_descriptions
        return None  # We'll use a custom run loop instead
    
    def _generate_text(self, prompt: str) -> str:
        """Generate text using either OpenVINO direct generation or HuggingFace pipeline."""
        if self.use_openvino:
            # Direct generation for OpenVINO models
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_length = inputs['input_ids'].shape[1]
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                top_p=0.95,
                repetition_penalty=1.1,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
            # Decode the full output
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Decode only the new tokens (after the prompt)
            new_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            return generated_text.strip()
        else:
            # Use HuggingFace pipeline
            llm_output = self.llm.invoke(prompt)
            
            # Handle different output formats
            if isinstance(llm_output, str):
                return llm_output
            elif isinstance(llm_output, dict):
                return llm_output.get('generated_text', llm_output.get('text', str(llm_output)))
            elif isinstance(llm_output, list) and len(llm_output) > 0:
                first_result = llm_output[0]
                if isinstance(first_result, dict):
                    return first_result.get('generated_text', first_result.get('text', str(first_result)))
                else:
                    return str(first_result)
            else:
                return str(llm_output)
    
    def run(self, query: str, use_memory: bool = False) -> str:
        """
        Run the agent with a given query.
        
        Args:
            query: The question or task for the agent
            use_memory: Whether to include chat history in the context
            
        Returns:
            The agent's response
        """
        try:
            # Simple ReAct loop with tool calling
            max_iterations = 5
            
            # Build the prompt with available tools
            tool_names = ", ".join([tool.name for tool in self.tools])
            
            # Build chat history context if requested
            history_context = ""
            if use_memory and len(self.chat_memory) > 0:
                history_context = "\n\nPrevious conversation:\n"
                for msg in self.chat_memory:
                    history_context += f"{msg}\n"
                history_context += "\n"
            
            prompt = f"""Answer the following question. You have access to these tools:

{self.tool_descriptions}

Use this format:
Question: the input question
Thought: think about what to do
Action: tool name (one of: {tool_names})
Action Input: input for the tool
Observation: result from the tool
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer
{history_context}
Question: {query}
Thought:"""

            if self.verbose:
                print(f"\nQuery: {query}\n")
            
            # Get LLM response using the unified generation method
            try:
                response = self._generate_text(prompt)
                
                if self.verbose:
                    print(f"LLM Response:\n{response}\n")
                
            except Exception as e:
                return f"Error invoking LLM: {str(e)}\nType: {type(e).__name__}"
            
            # Parse for tool usage (simple parsing)
            if "Final Answer:" in response:
                # Extract final answer
                answer = response.split("Final Answer:")[-1].strip()
                return answer
            elif "Action:" in response and "Action Input:" in response:
                # Extract tool name and input
                lines = response.split("\n")
                action = None
                action_input = None
                
                for line in lines:
                    if line.startswith("Action:"):
                        action = line.replace("Action:", "").strip()
                    elif line.startswith("Action Input:"):
                        action_input = line.replace("Action Input:", "").strip()
                
                # Execute tool if found
                if action and action_input:
                    for tool in self.tools:
                        if tool.name.lower() == action.lower():
                            observation = tool.func(action_input)
                            
                            if self.verbose:
                                print(f"Tool: {action}")
                                print(f"Input: {action_input}")
                                print(f"Observation: {observation}\n")
                            
                            # Get final answer with observation
                            final_prompt = f"{prompt}\n{response}\nObservation: {observation}\nThought:"
                            final_response = self._generate_text(final_prompt)
                            
                            if "Final Answer:" in final_response:
                                answer = final_response.split("Final Answer:")[-1].strip()
                                return answer
                            
                            return final_response
            
            # Return raw response if no special format detected
            return response
            
        except Exception as e:
            return f"Error running agent: {str(e)}"
    
    def add_tool(self, tool: Tool):
        """
        Add a new tool to the agent.
        
        Args:
            tool: A LangChain Tool object
        """
        self.tools.append(tool)
        # Recreate the agent with the new tool
        self.agent = self._create_agent(verbose=self.verbose)
    
    def chat(self, message: str) -> str:
        """
        Chat with the agent (alias for run method with memory enabled).
        
        Args:
            message: The user's message
            
        Returns:
            The agent's response
        """
        # Add user message to memory
        self.chat_memory.append(f"User: {message}")
        
        # Get response with memory
        response = self.run(message, use_memory=True)
        
        # Extract final answer if present
        if "Final Answer:" in response:
            answer = response.split("Final Answer:")[-1].strip()
            # Remove trailing instruction text
            if "\n\n" in answer:
                answer = answer.split("\n\n")[0].strip()
        else:
            answer = response
        
        # Add agent response to memory
        self.chat_memory.append(f"Agent: {answer}")
        
        return answer
    
    def clear_memory(self):
        """Clear the chat memory."""
        self.chat_memory.clear()
    
    def get_memory_size(self) -> int:
        """Get the current number of messages in memory."""
        return len(self.chat_memory)
    
    def interactive_chat(self):
        """
        Start an interactive chat session with the agent.
        Type 'exit', 'quit', or 'bye' to end the session.
        """
        print("\n" + "="*60)
        print("INTERACTIVE CHAT MODE")
        print("="*60)
        print("Chat with the Phi-4 Agent. Type 'exit', 'quit', or 'bye' to end.")
        print("Available tools:")
        for tool in self.tools:
            print(f"  - {tool.name}: {tool.description}")
        print("\nSpecial commands:")
        print("  - 'clear': Clear chat history")
        print("  - 'memory': Show memory usage")
        print(f"\nContext: Up to 100 messages, {self.max_context_length} tokens max")
        print("="*60 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'bye', 'q']:
                    print("\nGoodbye! Chat session ended.")
                    break
                
                # Check for special commands
                if user_input.lower() == 'clear':
                    self.clear_memory()
                    print("✓ Chat memory cleared.\n")
                    continue
                
                if user_input.lower() == 'memory':
                    mem_count = self.get_memory_size()
                    print(f"✓ Memory: {mem_count}/200 messages ({mem_count//2} exchanges)\n")
                    continue
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Get agent response with memory
                print("\nAgent: ", end="", flush=True)
                response = self.chat(user_input)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n\nChat session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try again.\n")


def create_phi4_llm(
    model_id: str = "microsoft/phi-4",
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None
) -> HuggingFacePipeline:
    """
    Create a standalone Phi-4 LLM instance (without agent capabilities).
    
    Args:
        model_id: The HuggingFace model ID for Phi-4
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate
        device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
        torch_dtype: Torch dtype for model
        
    Returns:
        HuggingFacePipeline instance configured for Phi-4
    """
    # Detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set dtype based on device if not specified
    if torch_dtype is None:
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Loading Phi-4 model on {device}...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch_dtype,  # Use 'dtype' instead of deprecated 'torch_dtype'
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    if device == "cpu":
        model = model.to(device)
    
    # Create text generation pipeline directly
    text_pipeline = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
        repetition_penalty=1.1,
        do_sample=True if temperature > 0 else False,
    )
    
    return HuggingFacePipeline(pipeline=text_pipeline)


if __name__ == "__main__":
    import sys
    
    # Example usage
    print("Initializing Phi-4 Agent with Intel GPU acceleration...")
    print("=" * 60)
    
    # Create the agent with Intel GPU via OpenVINO
    # This provides hardware acceleration on Intel integrated graphics
    agent = Phi4Agent(verbose=True, use_intel_gpu=True)
    
    # Check if chat mode is requested
    if len(sys.argv) > 1 and sys.argv[1] in ['chat', '--chat', '-c']:
        # Start interactive chat mode
        agent.interactive_chat()
    else:
        # Run single test query
        print("\n" + "="*50)
        print("Testing Phi-4 Agent")
        print("="*50 + "\n")
        
        response = agent.run("What is 25 multiplied by 17?")
        print(f"\nFinal Answer: {response}")
        
        print("\n" + "="*60)
        print("To start interactive chat mode, run:")
        print("  python phi4_agent.py chat")
        print("="*60)
