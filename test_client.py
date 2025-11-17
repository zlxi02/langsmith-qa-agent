"""
Simple client to test the LangServe API

Make sure server.py is running first:
  python server.py
"""

from langserve import RemoteRunnable

# Connect to the remote agent
agent = RemoteRunnable("http://localhost:8000/agent")

# Test questions
questions = [
    "How does LangSmith tracing work?",
    "What is LangSmith evaluation?",
    "How do I deploy agents with LangSmith?"
]

print("="*60)
print("Testing LangServe API")
print("="*60)

for i, question in enumerate(questions, 1):
    print(f"\n{'='*60}")
    print(f"Query {i}: {question}")
    print('='*60)
    
    # Invoke the agent
    result = agent.invoke({"question": question})
    
    # Print the formatted output
    print(result["formatted_output"])

print("\n" + "="*60)
print("✅ All queries completed!")
print("="*60)

# Example: Streaming
print("\n\n" + "="*60)
print("Testing Streaming")
print("="*60)
print("\nQuestion: How does LangSmith help with debugging?")
print("-"*60)

for chunk in agent.stream({"question": "How does LangSmith help with debugging?"}):
    if "formatted_output" in chunk:
        print(chunk["formatted_output"])
        break

print("\n" + "="*60)
print("✅ Streaming test completed!")
print("="*60)

