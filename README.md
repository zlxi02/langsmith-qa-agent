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
   ```bash
   export X_API_KEY='your-gateway-api-key-here'
   ```
   
   Note: This project uses the Salesforce OpenAI Gateway at `https://gateway.salesforceresearch.ai/openai/process/v1`

3. **Customize documentation sources (optional):**
   Edit `docs_urls.py` to add/remove LangSmith documentation URLs

4. **Run the application:**
   ```bash
   python app.py
   ```

## Project Status

- [x] Step 1: Initial Setup & RAG Ingestion
- [ ] Step 2: TBD
- [ ] Step 3: TBD
- [ ] Step 4: TBD

