# LangSmith Q&A Agent

A 4-node LangGraph agent that acts as a General Q&A Expert for LangSmith Documentation.

## Features

- **Real Documentation:** Scrapes live LangSmith documentation from web
- **RAG System:** Vector-based retrieval for accurate answers
- **Custom Gateway:** Uses Salesforce OpenAI Gateway for API access
- **Configurable URLs:** Easily customize which docs to include in `docs_urls.py`

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   
   **Required:**
   ```bash
   export X_API_KEY='your-gateway-api-key-here'
   ```
   
   Note: This project uses the Salesforce OpenAI Gateway at `https://gateway.salesforceresearch.ai/openai/process/v1`
   
   **Optional (for LangSmith tracing):**
   ```bash
   export LANGCHAIN_TRACING_V2=true
   export LANGCHAIN_API_KEY='your-langsmith-api-key-here'
   export LANGCHAIN_PROJECT='langsmith-qa-agent'
   ```
   
   With LangSmith tracing enabled, you can monitor your agent's execution at [smith.langchain.com](https://smith.langchain.com)

3. **Customize documentation sources (optional):**
   Edit `docs_urls.py` to add/remove LangSmith documentation URLs

4. **Run the application:**
   
   **Option A: Test locally (creates FAISS index + tests agent):**
   ```bash
   X_API_KEY='your-key' python app.py
   ```
   
   **Option B: Start API server (requires FAISS index from Option A first):**
   ```bash
   X_API_KEY='your-key' python server.py
   ```
   
   **With LangSmith tracing:**
   ```bash
   X_API_KEY='your-key' \
   LANGCHAIN_TRACING_V2=true \
   LANGCHAIN_API_KEY='your-langsmith-key' \
   LANGCHAIN_PROJECT='langsmith-qa-agent' \
   python server.py
   ```

## API Endpoints

Once `server.py` is running, access:

- **API Documentation:** http://localhost:8000/docs
- **Interactive Playground:** http://localhost:8000/agent/playground
- **Health Check:** http://localhost:8000/health

### Example API Usage:

**Single Query:**
```bash
curl -X POST http://localhost:8000/agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"question": "How does LangSmith tracing work?"}}'
```

**Streaming:**
```bash
curl -N -X POST http://localhost:8000/agent/stream \
  -H "Content-Type: application/json" \
  -d '{"input": {"question": "How does LangSmith tracing work?"}}'
```

**From Python:**
```python
from langserve import RemoteRunnable

agent = RemoteRunnable("http://localhost:8000/agent")
result = agent.invoke({"question": "How does LangSmith tracing work?"})
print(result["formatted_output"])
```

## Project Status

- [x] Step 1: Initial Setup & RAG Ingestion
  - [x] Load LangSmith documentation from web
  - [x] Create FAISS vector store
  - [x] Build retriever tool
- [x] Step 2: 3-Node LangGraph Agent
  - [x] Retriever node
  - [x] Generator node (LLM)
  - [x] Formatter node
- [x] Step 3: LangSmith Integration
  - [x] Tracing support
- [x] Step 4: LangServe API
  - [x] FastAPI server
  - [x] Streaming endpoints
  - [x] Interactive playground
  - [x] Auto-generated docs
- [ ] Step 5: Add Router Node (4-node agent)

