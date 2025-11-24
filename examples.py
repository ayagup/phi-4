"""
Example usage of the Phi-4 Agent with LangChain (Local Model)
This script demonstrates various ways to use the Phi-4 agent running locally.
"""

import os
from phi4_agent import Phi4Agent, create_phi4_llm
from langchain_core.tools import Tool
import torch


def example_1_basic_usage():
    """Example 1: Basic agent usage with default tools."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Agent Usage")
    print("="*60 + "\n")
    
    # Create the agent (this will download the model on first run)
    agent = Phi4Agent(verbose=True)
    
    # Ask a mathematical question
    response = agent.run("What is 15 multiplied by 23, plus 100?")
    print(f"\nFinal Answer: {response}\n")


def example_2_custom_tools():
    """Example 2: Using custom tools with the agent."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Agent with Custom Tools")
    print("="*60 + "\n")
    
    # Define custom tools
    def get_weather(location: str) -> str:
        """Mock weather tool - in production, this would call a real API."""
        weather_data = {
            "New York": "Sunny, 72°F",
            "London": "Cloudy, 15°C",
            "Tokyo": "Rainy, 20°C",
        }
        return weather_data.get(location, f"Weather data not available for {location}")
    
    def reverse_string(text: str) -> str:
        """Reverse a string."""
        return text[::-1]
    
    custom_tools = [
        Tool(
            name="Weather",
            func=get_weather,
            description="Get current weather for a location. Input should be a city name."
        ),
        Tool(
            name="ReverseString",
            func=reverse_string,
            description="Reverse a string. Input should be the string to reverse."
        )
    ]
    
    # Create agent with custom tools
    agent = Phi4Agent(tools=custom_tools, verbose=True)
    
    # Use the custom tools
    response = agent.run("What's the weather in London?")
    print(f"\nFinal Answer: {response}\n")


def example_3_standalone_llm():
    """Example 3: Using Phi-4 as a standalone LLM without agent capabilities."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Standalone LLM (No Agent)")
    print("="*60 + "\n")
    
    # Create a standalone Phi-4 LLM
    llm = create_phi4_llm(temperature=0.5, max_new_tokens=256)
    
    # Use it directly for text generation
    prompt = "Explain what machine learning is in simple terms:"
    response = llm.invoke(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}\n")


def example_4_reasoning_task():
    """Example 4: Using the agent for reasoning tasks."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Reasoning Task")
    print("="*60 + "\n")
    
    agent = Phi4Agent(verbose=True)
    
    # Complex reasoning question
    question = """
    If a train travels 120 miles in 2 hours, what is its average speed?
    Then, how far would it travel in 5 hours at the same speed?
    """
    
    response = agent.run(question)
    print(f"\nFinal Answer: {response}\n")


def example_5_adding_tools_dynamically():
    """Example 5: Dynamically adding tools to an existing agent."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Dynamically Adding Tools")
    print("="*60 + "\n")
    
    # Create agent with default tools
    agent = Phi4Agent(verbose=True)
    
    # Define a new tool
    def word_count(text: str) -> str:
        """Count words in a text."""
        return str(len(text.split()))
    
    new_tool = Tool(
        name="WordCount",
        func=word_count,
        description="Count the number of words in a text. Input should be the text to analyze."
    )
    
    # Add the tool dynamically
    print("Adding WordCount tool to the agent...")
    agent.add_tool(new_tool)
    
    # Use the new tool
    response = agent.run("How many words are in this sentence: 'The quick brown fox jumps over the lazy dog'?")
    print(f"\nFinal Answer: {response}\n")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("PHI-4 AGENT EXAMPLES (Local Model)")
    print("="*60)
    print("\nNote: The model will be downloaded on first run (~15GB)")
    print("Subsequent runs will use the cached model.\n")
    
    # Check for GPU
    import torch
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    else:
        print("No GPU detected. Running on CPU (will be slower)\n")
    
    try:
        # Run examples (comment out any you don't want to run)
        example_1_basic_usage()
        # example_2_custom_tools()
        # example_3_standalone_llm()
        # example_4_reasoning_task()
        # example_5_adding_tools_dynamically()
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        print("\nNote: Make sure you have:")
        print("1. Installed all dependencies (pip install -r requirements.txt)")
        print("2. Enough disk space for the model (~15GB)")
        print("3. Enough RAM (16GB+ recommended, or GPU with 8GB+ VRAM)")
        print("4. Internet connection for first-time model download")


if __name__ == "__main__":
    main()
