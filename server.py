"""
LangServe API Server for LangSmith Q&A Agent

Wraps the 3-node LangGraph agent in a FastAPI server with:
- REST API endpoints
- Streaming support
- Interactive playground
- Auto-generated documentation
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from agent import create_agent
from utils import load_retriever_tool


def load_agent():
    """
    Load the FAISS index and create the agent.
    This is faster than re-ingesting docs every time.
    """
    print("🔧 Loading FAISS index and creating agent...")
    
    # Load retriever tool from existing FAISS index (shared utility)
    try:
        retriever_tool = load_retriever_tool("faiss_langsmith_index")
        print("✓ Loaded existing FAISS index")
        print("✓ Created retriever tool")
    except Exception as e:
        print(f"❌ Error loading FAISS index: {e}")
        print("   Please run 'python app.py' first to create the index!")
        exit(1)
    
    # Create the agent
    agent = create_agent(retriever_tool)
    
    print("✓ Agent ready!\n")
    
    return agent


# Create FastAPI app
app = FastAPI(
    title="LangSmith Q&A Agent API",
    description="A 3-node LangGraph agent that answers questions about LangSmith documentation",
    version="1.0.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the agent
agent = load_agent()

# Add LangServe routes with proper tracing configuration
# This creates all the endpoints automatically:
# - POST /agent/invoke (single query)
# - POST /agent/batch (multiple queries)
# - POST /agent/stream (streaming response)
# - POST /agent/stream_log (detailed streaming)
# - GET /agent/playground (interactive UI)
add_routes(
    app,
    agent,
    path="/agent",
    enable_feedback_endpoint=True,  # Optional: allow feedback on responses
    # Per-request config for consistent tracing
    per_req_config_modifier=lambda config, request: {
        **config,
        "run_name": "LangSmith Q&A Agent",
        "tags": ["langsmith-qa", "3-node-agent", "rag", "api"],
    }
)

# Custom health check endpoint
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "LangSmith Q&A Agent API is running!",
        "endpoints": {
            "invoke": "POST /agent/invoke",
            "stream": "POST /agent/stream",
            "batch": "POST /agent/batch",
            "playground": "GET /agent/playground",
            "docs": "GET /docs"
        }
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "faiss_index": "loaded",
        "agent": "ready",
        "langsmith_tracing": os.getenv("LANGCHAIN_TRACING_V2") == "true"
    }


if __name__ == "__main__":
    import uvicorn
    
    # Check for required env vars
    if not os.getenv("X_API_KEY"):
        print("❌ Error: X_API_KEY environment variable not set!")
        print("   Set it with: export X_API_KEY='your-gateway-api-key'")
        exit(1)
    
    print("\n" + "="*60)
    print("🚀 Starting LangSmith Q&A Agent API Server")
    print("="*60)
    print("\n📍 Server will be available at:")
    print("   - API: http://localhost:8000")
    print("   - Docs: http://localhost:8000/docs")
    print("   - Playground: http://localhost:8000/agent/playground")
    print("\n" + "="*60 + "\n")
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

