# LangSmith Q&A Agent

A production-ready Retrieval-Augmented Generation (RAG) system built with LangGraph, featuring a 3-node agentic workflow that answers questions about LangSmith documentation with high accuracy and full observability.

## Overview

A production-ready Q&A agent demonstrating key LangChain/LangGraph patterns:

- **3-Node LangGraph Agent**: Sequential pipeline with Retrieve → Generate → Format nodes for clear separation of concerns and observability
- **RAG System**: Built on LangChain's document loaders, text splitters, embeddings, and FAISS vector store for semantic search
- **LangSmith & LangServe**: Full observability with tracing, plus REST API deployment with streaming support and interactive playground
- **Custom Gateway**: Salesforce OpenAI Gateway integration with custom authentication headers and base URL configuration

### Architecture Diagram

```
User Query → [API/Playground] → LangGraph Agent → LangSmith (Tracing)
                                        ↓
                                  Retrieve Node
                                        ↓
                                   FAISS Index
                                (cosine similarity)
                                        ↓
                               Doc Chunks (top 3)
                                        ↓
                                  Generate Node
                                        ↓
                            ChatOpenAI (GPT-4o-mini)
                             Question + Doc Context
                                        ↓
                                 Generated Answer
                                        ↓
                                   Format Node
                                        ↓
                               Formatted Output
```

**Flow:** Sequential execution (retrieve → generate → format)  
**Total latency:** ~3-4s per query

## Technical Architecture

### 1. **Data Ingestion Pipeline** (`app.py`)
The system scrapes and indexes LangSmith documentation through a multi-stage pipeline:

```python
Web URLs → Document Loader → Text Splitter → Embeddings → FAISS Index
```

**Key Components:**
- **WebBaseLoader**: Fetches documentation from configured URLs
- **RecursiveCharacterTextSplitter**: Chunks documents (512 chars, 100 overlap) to preserve semantic boundaries
- **OpenAI Embeddings** (text-embedding-3-small): Converts chunks to 1536-dim vectors
- **FAISS**: CPU-optimized vector store for semantic similarity search

**Why this matters:** The quality of retrieval directly impacts answer accuracy. Proper chunking ensures context is preserved while maintaining searchability.

### 2. **Agent Architecture** (`agent.py`)
A stateful LangGraph workflow with 3 specialized nodes:

#### Node 1: Retrieve
- **Input:** User question
- **Process:** Semantic search over FAISS index (k=3 top chunks)
- **Output:** Relevant documentation passages
- **Tracing:** Captures retrieval latency and matched documents

#### Node 2: Generate  
- **Input:** Question + retrieved documents
- **Process:** GPT-4o-mini generates answer from context
- **Output:** Natural language response
- **Tracing:** Captures LLM tokens, latency, and model metadata

#### Node 3: Format
- **Input:** Question + generated answer
- **Process:** Structures final output
- **Output:** Clean formatted response
- **Tracing:** Marks completion of pipeline

**State Management:**
```python
class GraphState(TypedDict, total=False):
    question: str           # Input (required)
    documents: str          # Output from Retrieve
    answer: str            # Output from Generate  
    formatted_output: str  # Final output from Format
```

The state flows through the graph: `retrieve → generate → format → END`

### 3. **API Layer** (`server.py`)
FastAPI server with LangServe integration providing:

**Auto-generated Endpoints:**
- `POST /agent/invoke` - Single query execution
- `POST /agent/stream` - Server-sent events streaming
- `POST /agent/batch` - Batch processing
- `GET /agent/playground` - Interactive web UI
- `GET /docs` - Auto-generated OpenAPI documentation

**Custom Endpoints:**
- `GET /` - Health check + endpoint directory
- `GET /health` - Detailed system status (FAISS, agent, tracing)

**CORS Configuration:** Enabled for frontend integration

### 4. **Gateway Integration** (`utils.py`)
All LLM/embedding calls route through Salesforce's OpenAI Gateway:

```python
Base URL: https://gateway.salesforceresearch.ai/openai/process/v1
Auth: X-Api-Key header
Models: GPT-4o-mini, text-embedding-3-small
```

**Benefits:**
- Centralized API management
- Usage tracking and rate limiting
- Cost optimization
- Compliance with enterprise policies

## Setup & Installation

### Prerequisites
- Python 3.9+
- Virtual environment (recommended)
- Salesforce OpenAI Gateway API key
- LangSmith API key (optional, for observability)

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv langsmith-qa-agent
source langsmith-qa-agent/bin/activate  # On Windows: langsmith-qa-agent\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Dependencies breakdown:**
- `langchain>=0.3.0` - Core LangChain framework
- `langgraph>=0.2.0` - Stateful agent workflows
- `langserve[all]>=0.3.0` - API deployment layer
- `faiss-cpu>=1.8.0` - Vector similarity search
- `fastapi>=0.109.0` + `uvicorn[standard]>=0.27.0` - API server
- `openai>=1.10.0` - OpenAI SDK for gateway
- `beautifulsoup4>=4.12.0` - Web scraping

### 2. Environment Variables

**Required (Custom Gateway):**
```bash
export X_API_KEY='your-salesforce-gateway-api-key'
```

> **Note:** This project uses Salesforce's OpenAI Gateway with custom authentication. To use your own enterprise gateway, modify the `base_url` and headers in `utils.py`. To use OpenAI directly instead, replace the custom client configuration with standard LangChain parameters (`openai_api_key` only).

**Recommended (LangSmith Observability):**
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY='your-langsmith-api-key'
export LANGCHAIN_PROJECT='langsmith-qa-agent'
```

Get your LangSmith API key at [smith.langchain.com](https://smith.langchain.com)

**Pro tip:** Add these to `~/.zshrc` or `~/.bashrc` for persistence:
```bash
echo 'export X_API_KEY="your-key"' >> ~/.zshrc
echo 'export LANGCHAIN_TRACING_V2=true' >> ~/.zshrc
echo 'export LANGCHAIN_API_KEY="your-langsmith-key"' >> ~/.zshrc
```

### 3. Documentation Source Configuration

Edit `docs_urls.py` to customize which documentation pages to index:

```python
LANGSMITH_DOC_URLS = [
    "https://docs.langchain.com/langsmith/home",
    "https://docs.langchain.com/langsmith/tracing",
    # Add more URLs...
]
```

**Considerations:**
- More URLs = better coverage but longer indexing time
- Choose foundational pages for core concepts
- Update and re-run `app.py` to refresh the index

## Usage

### Mode 1: Local Testing & Index Creation

```bash
python app.py
```

**What it does:**
1. ✅ Scrapes documentation from configured URLs
2. ✅ Creates embeddings (text-embedding-3-small)
3. ✅ Builds FAISS vector index → `faiss_langsmith_index/`
4. ✅ Tests agent with 3 sample queries
5. ✅ Outputs results to terminal

**When to use:** First-time setup, index refresh, or local testing

**Expected output:**
```
📥 Loading documentation from 10 URLs...
✓ Loaded 10 documentation pages
✓ Split into 247 chunks
✓ Created FAISS index at: /path/to/faiss_langsmith_index
✓ Created langsmith_retriever_tool
🔧 Building LangGraph agent...
✓ Added 3 nodes: retrieve → generate → format
```

### Mode 2: Production API Server

```bash
python server.py
```

**What it does:**
1. ✅ Loads existing FAISS index (from Mode 1)
2. ✅ Creates agent with retriever tool
3. ✅ Starts FastAPI server on port 8000
4. ✅ Exposes REST API + playground

**When to use:** Serving the agent to frontends, APIs, or clients

**Server endpoints:**
- **Interactive UI:** http://localhost:8000/agent/playground
- **API Docs:** http://localhost:8000/docs (Swagger UI)
- **Health Check:** http://localhost:8000/health

**Full startup command:**
```bash
X_API_KEY='your-key' \
LANGCHAIN_TRACING_V2=true \
LANGCHAIN_API_KEY='your-langsmith-key' \
LANGCHAIN_PROJECT='langsmith-qa-agent' \
python server.py
```

## API Reference

### LangServe Auto-Generated Endpoints

LangServe automatically creates these endpoints from your agent:

| Endpoint | Method | Purpose | Response Type |
|----------|--------|---------|---------------|
| `/agent/invoke` | POST | Single synchronous query | JSON |
| `/agent/stream` | POST | Streaming response (SSE) | Event Stream |
| `/agent/batch` | POST | Multiple queries at once | JSON Array |
| `/agent/stream_log` | POST | Detailed streaming with logs | Event Stream |
| `/agent/playground` | GET | Interactive web interface | HTML |

### Example Usage

#### 1. Single Query (curl)
```bash
curl -X POST "http://localhost:8000/agent/invoke" \
  -H "Content-Type: application/json" \
  -d '{"input": {"question": "How does LangSmith tracing work?"}}'
```

**Response:**
```json
{
  "output": {
    "output": "LangSmith tracing automatically captures all LLM calls, tool invocations, and agent steps..."
  }
}
```

#### 2. Streaming (curl)
```bash
curl -N -X POST "http://localhost:8000/agent/stream" \
  -H "Content-Type: application/json" \
  -d '{"input": {"question": "What is LangSmith evaluation?"}}'
```

Returns Server-Sent Events with real-time updates as each node executes.

#### 3. Python Client (RemoteRunnable)
```python
from langserve import RemoteRunnable

# Connect to the agent API
agent = RemoteRunnable("http://localhost:8000/agent")

# Single query
result = agent.invoke({"question": "How does LangSmith tracing work?"})
print(result["output"])

# Streaming
for chunk in agent.stream({"question": "What is LangSmith?"}):
    print(chunk)
```

#### 4. Python Client (Test Script)
```bash
python test_client.py
```

Runs pre-configured test queries with both invoke and streaming modes.

#### 5. Interactive Playground
Visit http://localhost:8000/agent/playground

- Visual interface for testing queries
- No code required
- See intermediate steps
- Test different inputs quickly

## LangSmith Observability

With `LANGCHAIN_TRACING_V2=true`, every agent execution is automatically traced to LangSmith.

### What Gets Traced

**Automatic instrumentation captures:**
- ✅ Each LangGraph node execution (retrieve, generate, format)
- ✅ Retriever tool invocations + matched documents
- ✅ LLM calls with full prompt, completion, tokens
- ✅ Timing and latency for each component
- ✅ Errors and stack traces
- ✅ Input/output at each step

### Trace Hierarchy

```
LangSmith Q&A Agent (root)
├─ retrieve (node 1) - 0.7s
│  └─ langsmith_retriever_tool
│     └─ VectorStoreRetriever (FAISS search)
├─ generate (node 2) - 3.5s, 241 tokens, $0.000072
│  └─ ChatOpenAI (GPT-4o-mini)
└─ format (node 3) - 0.0s
```

### Viewing Traces

1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Select your project (`langsmith-qa-agent`)
3. View list of all runs
4. Click into any run to see detailed trace tree
5. Analyze performance, costs, and outputs

**Tags for filtering:** `langsmith-qa`, `3-node-agent`, `rag`, `api`

## Project Structure

```
langsmith-qa-agent/
├── agent.py              # LangGraph agent (3 nodes)
├── app.py                # Local testing + FAISS index creation
├── server.py             # FastAPI + LangServe API server
├── utils.py              # Shared utilities (embeddings, retriever)
├── docs_urls.py          # Documentation URLs configuration
├── test_client.py        # Python client for testing API
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── faiss_langsmith_index/  # Vector store (created by app.py)
    ├── index.faiss       # FAISS index binary
    └── index.pkl         # Document metadata
```

## Design Decisions & Rationale

### Why LangGraph instead of LangChain Chains?
- **Statefulness:** Graph state flows through nodes, enabling complex logic
- **Observability:** Each node traced independently in LangSmith
- **Extensibility:** Easy to add conditional routing, loops, or additional nodes
- **Debugging:** Clear execution flow vs opaque chains

### Why FAISS over other vector stores?
- **Performance:** Optimized for CPU, no GPU required
- **Simplicity:** No external database or server needed
- **Portability:** Index saved to disk, easy to version control or deploy
- **Cost:** Free, open-source, runs locally

### Why 3 nodes (Retrieve → Generate → Format)?
- **Separation of concerns:** Each node has single responsibility
- **Traceability:** Can measure performance of each step independently
- **Flexibility:** Can swap out retrieval strategy or LLM without affecting others
- **Testing:** Can test each node in isolation

### Why GPT-4o-mini over GPT-4?
- **Cost efficiency:** ~50x cheaper than GPT-4
- **Speed:** Faster response times (avg 3.5s vs 10s+)
- **Sufficient quality:** For documentation Q&A, mini performs well
- **Can upgrade:** Model configurable in `agent.py`

### Why LangServe?
- **Auto-generated API:** Don't write boilerplate FastAPI routes
- **Streaming support:** Built-in SSE for real-time responses
- **Playground UI:** Free interactive testing interface
- **LangChain native:** Seamless integration with LangGraph agents

## Future Enhancements

- [ ] **Router Node:** Add 4th node to classify queries before retrieval
- [ ] **Multi-query retrieval:** Generate multiple search queries for better recall
- [ ] **Reranking:** Use cross-encoder to rerank retrieved chunks
- [ ] **Caching:** Cache embeddings and common queries
- [ ] **Feedback loop:** Collect user feedback via LangSmith
- [ ] **Evaluation:** Automated testing with LangSmith Datasets
- [ ] **Hybrid search:** Combine semantic + keyword (BM25) search

## Troubleshooting

### "ModuleNotFoundError: No module named 'faiss'"
```bash
pip install faiss-cpu
```

### "Error loading FAISS index"
Run `python app.py` first to create the index, then start the server.

### "LangSmith traces not appearing"
Check that all environment variables are set:
```bash
echo $LANGCHAIN_TRACING_V2  # should print "true"
echo $LANGCHAIN_API_KEY     # should print your key
```

### "Gateway authentication failed"
Verify `X_API_KEY` is correct and has access to the gateway.

## License

MIT License - Feel free to use this as a template for your own RAG agents!

