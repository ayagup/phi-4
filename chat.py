"""
Interactive Chat with Phi-4 Agent
Run this script to start chatting with the Phi-4 model with 100-message memory.
"""

from phi4_agent import Phi4Agent

def main():
    print("Loading Phi-4 Agent...")
    print("Please wait, this may take a moment...\n")
    
    # Create agent with Intel GPU acceleration and maximum context
    agent = Phi4Agent(
        verbose=False, 
        use_intel_gpu=True,
        max_context_length=16384,  # Maximum context window
        max_new_tokens=2048  # Allow longer responses
    )
    
    print("\n" + "="*60)
    print("INTERACTIVE CHAT WITH PHI-4")
    print("="*60)
    print("The agent has access to these tools:")
    for tool in agent.tools:
        print(f"  • {tool.name}: {tool.description}")
    print("\n📝 Memory: Up to 100 messages (50 exchanges)")
    print(f"🔧 Context: {agent.max_context_length} tokens maximum")
    print("\nCommands:")
    print("  • Type your question or message")
    print("  • 'clear' - Clear chat history")
    print("  • 'memory' - Show memory usage")
    print("  • 'exit', 'quit', or 'bye' - End chat")
    print("  • Ctrl+C - Interrupt")
    print("="*60 + "\n")
    
    message_count = 0
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'bye', 'q']:
                print(f"\n👋 Goodbye! Had {message_count} exchanges. Thanks for chatting!")
                break
            
            # Check for special commands
            if user_input.lower() == 'clear':
                agent.clear_memory()
                message_count = 0
                print("✓ Chat memory cleared.\n")
                continue
            
            if user_input.lower() == 'memory':
                mem_count = agent.get_memory_size()
                print(f"✓ Memory: {mem_count}/200 messages ({mem_count//2} exchanges)\n")
                continue
            
            # Skip empty input
            if not user_input:
                continue
            
            # Get agent response with memory
            message_count += 1
            print(f"\nPhi-4 [{message_count}]: ", end="", flush=True)
            
            # Use the chat method which automatically manages memory
            response = agent.chat(user_input)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Chat interrupted after {message_count} exchanges. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again.\n")

if __name__ == "__main__":
    main()
