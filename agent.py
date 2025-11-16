"""
LangGraph Agent for LangSmith Q&A

3-Node Architecture:
1. Retriever Node - Gets relevant docs from FAISS
2. Generator Node - LLM creates answer using docs
3. Formatter Node - Structures the final output
"""

import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAI, AsyncOpenAI
import httpx


# ==============================================================================
# STEP 1: Define State
# ==============================================================================

class GraphState(TypedDict):
    """
    State that flows through the agent graph.
    Each node reads from and writes to this state.
    """
    question: str           # User's query
    documents: str          # Retrieved documentation chunks
    answer: str            # Generated answer from LLM
    formatted_output: str  # Final formatted response


# ==============================================================================
# STEP 2: Node Functions
# ==============================================================================

def retriever_node(state: GraphState) -> dict:
    """
    Node 1: Retrieves relevant documentation using the FAISS retriever tool.
    
    Input: state["question"]
    Output: Updates state["documents"]
    """
    print("🔍 Node 1: Retrieving relevant documents...")
    
    question = state["question"]
    
    # Call the retriever tool (passed in via create_agent)
    # This tool was created in app.py and wraps the FAISS retriever
    documents = retriever_tool.invoke(question)
    
    print(f"   Retrieved {len(documents.split('\\n\\n'))} document chunks")
    
    return {"documents": documents}


def generator_node(state: GraphState) -> dict:
    """
    Node 2: Generates answer using LLM with retrieved documents as context.
    
    Input: state["question"] + state["documents"]
    Output: Updates state["answer"]
    """
    print("🤖 Node 2: Generating answer with LLM...")
    
    question = state["question"]
    documents = state["documents"]
    
    # Configure ChatOpenAI to use company gateway
    # Set dummy OPENAI_API_KEY if not already set
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "dummy"
    
    gateway_base_url = "https://gateway.salesforceresearch.ai/openai/process/v1"
    gateway_headers = {"X-Api-Key": os.getenv("X_API_KEY")}
    
    # Create custom HTTP clients with gateway configuration
    http_client = httpx.Client(
        base_url=gateway_base_url,
        headers=gateway_headers,
        timeout=60.0
    )
    
    async_http_client = httpx.AsyncClient(
        base_url=gateway_base_url,
        headers=gateway_headers,
        timeout=60.0
    )
    
    sync_client = OpenAI(
        base_url=gateway_base_url,
        api_key="dummy",
        http_client=http_client
    )
    
    async_client = AsyncOpenAI(
        base_url=gateway_base_url,
        api_key="dummy",
        http_client=async_http_client
    )
    
    # Create LLM with gateway clients
    # Pass the chat.completions subresource, not the full client
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        client=sync_client.chat.completions,
        async_client=async_client.chat.completions
    )
    
    # Create prompt with system message and user question
    system_prompt = """You are a helpful LangSmith documentation expert. 
Use the provided documentation to answer the user's question accurately and concisely.
If the documentation doesn't contain enough information, say so.

Documentation:
{documents}"""
    
    messages = [
        SystemMessage(content=system_prompt.format(documents=documents)),
        HumanMessage(content=question)
    ]
    
    # Generate answer
    response = llm.invoke(messages)
    answer = response.content
    
    print(f"   Generated answer ({len(answer)} characters)")
    
    return {"answer": answer}


def formatter_node(state: GraphState) -> dict:
    """
    Node 3: Formats the final output with question and answer.
    
    Input: state["question"] + state["answer"]
    Output: Updates state["formatted_output"]
    """
    print("✨ Node 3: Formatting output...")
    
    question = state["question"]
    answer = state["answer"]
    
    # Format the final output
    formatted_output = f"""
╔══════════════════════════════════════════════════════════════╗
║                    LANGSMITH Q&A AGENT                       ║
╚══════════════════════════════════════════════════════════════╝

📝 Question:
{question}

💡 Answer:
{answer}

══════════════════════════════════════════════════════════════
"""
    
    print("   Output formatted ✓")
    
    return {"formatted_output": formatted_output}


# ==============================================================================
# STEP 3: Create Agent Graph
# ==============================================================================

# Global variable to hold the retriever tool (will be set by create_agent)
retriever_tool = None


def create_agent(retriever_tool_instance):
    """
    Creates and compiles the LangGraph agent.
    
    Args:
        retriever_tool_instance: The Tool object from app.py that wraps FAISS retriever
        
    Returns:
        Compiled LangGraph agent ready to invoke
    """
    global retriever_tool
    retriever_tool = retriever_tool_instance
    
    print("\n🔧 Building LangGraph agent...")
    
    # Initialize the graph with our state schema
    workflow = StateGraph(GraphState)
    
    # Add the 3 nodes
    workflow.add_node("retrieve", retriever_node)
    workflow.add_node("generate", generator_node)
    workflow.add_node("format", formatter_node)
    
    # Define the edges (flow between nodes)
    workflow.set_entry_point("retrieve")  # Start with retrieval
    workflow.add_edge("retrieve", "generate")  # retrieve → generate
    workflow.add_edge("generate", "format")    # generate → format
    workflow.add_edge("format", END)           # format → end
    
    print("   ✓ Added 3 nodes: retrieve → generate → format")
    
    # Compile the graph
    agent = workflow.compile()
    
    print("   ✓ Agent compiled and ready!\n")
    
    return agent

