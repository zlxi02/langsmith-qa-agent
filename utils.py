"""
Shared utility functions for the LangSmith Q&A Agent

Contains reusable code for:
- Gateway embeddings configuration
- FAISS index operations
- Retriever tool creation
"""

import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import Tool
from openai import OpenAI, AsyncOpenAI
import httpx


def get_gateway_embeddings():
    """
    Create OpenAI embeddings configured for the company gateway.
    
    Returns:
        OpenAIEmbeddings: Configured embeddings instance
    """
    # Set dummy OPENAI_API_KEY if not already set (required by OpenAI SDK)
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "dummy"
    
    # Gateway configuration
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
    
    # Create embeddings with gateway clients
    embeddings = OpenAIEmbeddings(
        client=sync_client.embeddings,
        async_client=async_client.embeddings,
        model="text-embedding-3-small"
    )
    
    return embeddings


def create_retriever_tool(vectorstore):
    """
    Create a LangChain Tool from a FAISS vectorstore.
    
    Args:
        vectorstore: FAISS vectorstore instance
        
    Returns:
        Tool: Wrapped retriever tool for agent use
    """
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
    )
    
    # Wrap in Tool
    retriever_tool = Tool(
        name="langsmith_retriever_tool",
        description="Retrieves relevant information from LangSmith documentation. "
                    "Use this tool when you need to answer questions about LangSmith features, "
                    "such as tracing, evaluation, or other capabilities.",
        func=lambda query: "\n\n".join([doc.page_content for doc in retriever.invoke(query)])
    )
    
    return retriever_tool


def load_retriever_tool(index_path="faiss_langsmith_index"):
    """
    Load an existing FAISS index and create a retriever tool.
    
    Args:
        index_path: Path to the FAISS index directory
        
    Returns:
        Tool: Ready-to-use retriever tool
        
    Raises:
        Exception: If FAISS index cannot be loaded
    """
    # Get embeddings configuration
    embeddings = get_gateway_embeddings()
    
    # Load existing FAISS index
    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True  # Required for loading
    )
    
    # Create and return tool
    return create_retriever_tool(vectorstore)

