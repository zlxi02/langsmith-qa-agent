import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import Tool
import openai
from openai import OpenAI
from docs_urls import LANGSMITH_DOC_URLS
from agent import create_agent

def setup_vector_store():
    """
    Set up the RAG ingestion pipeline:
    1. Load LangSmith documentation from web URLs
    2. Split documents into chunks
    3. Create embeddings and store in FAISS vector database
    4. Create a retriever tool for the agent
    """
    
    # Set dummy OPENAI_API_KEY if not already set (required by OpenAI SDK)
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "dummy"
    
    # Step 1: Load Documents from Web
    print(f"📥 Loading documentation from {len(LANGSMITH_DOC_URLS)} URLs...")
    loader = WebBaseLoader(LANGSMITH_DOC_URLS)
    raw_docs = loader.load()
    print(f"✓ Loaded {len(raw_docs)} documentation pages")
    
    # Step 2: Text Splitting
    # RecursiveCharacterTextSplitter breaks text into chunks intelligently,
    # trying to preserve semantic boundaries (paragraphs, sentences)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,      # Maximum size of each chunk
        chunk_overlap=100    # Overlap between chunks to preserve context
    )
    
    # Split the documents into chunks
    docs = text_splitter.split_documents(raw_docs)
    print(f"✓ Split into {len(docs)} chunks")
    
    # Step 3: Create Embeddings and FAISS Index
    # OpenAIEmbeddings converts text into vector representations
    # FAISS (Facebook AI Similarity Search) stores these vectors for fast retrieval
    
    # Configure OpenAI clients to use company gateway
    from openai import AsyncOpenAI
    
    gateway_base_url = "https://gateway.salesforceresearch.ai/openai/process/v1"
    gateway_headers = {"X-Api-Key": os.getenv("X_API_KEY")}
    
    # Create both sync and async clients with gateway configuration
    # Important: We create custom HTTP clients with the headers baked in
    import httpx
    
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
    
    # Pass embeddings clients to OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(
        client=sync_client.embeddings,
        async_client=async_client.embeddings,
        model="text-embedding-3-small"
    )
    
    # Create FAISS vector store from documents
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Save the FAISS index to disk
    faiss_index_path = "faiss_langsmith_index"
    vectorstore.save_local(faiss_index_path)
    print(f"✓ Created FAISS index at: {os.path.abspath(faiss_index_path)}")
    
    # Step 4: Create Retriever Tool
    # The retriever will find the most relevant chunks for a given query
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
    )
    
    # Wrap the retriever in a Tool so it can be used by the agent
    langsmith_retriever_tool = Tool(
        name="langsmith_retriever_tool",
        description="Retrieves relevant information from LangSmith documentation. "
                    "Use this tool when you need to answer questions about LangSmith features, "
                    "such as tracing, evaluation, or other capabilities.",
        func=lambda query: "\n\n".join([doc.page_content for doc in retriever.invoke(query)])
    )
    
    print(f"✓ Created langsmith_retriever_tool")
    
    return langsmith_retriever_tool

if __name__ == "__main__":
    # Check for company gateway API key
    if not os.getenv("X_API_KEY"):
        print("⚠️  Warning: X_API_KEY environment variable not set!")
        print("   Please set it before running: export X_API_KEY='your-gateway-api-key'")
    else:
        print("✓ X_API_KEY found")
    
    print("\n" + "="*60)
    print("STEP 1: RAG INGESTION PROCESS")
    print("="*60 + "\n")
    
    # Execute the RAG ingestion
    tool = setup_vector_store()
    
    print("\n" + "="*60)
    print("INGESTION COMPLETE!")
    print("="*60)
    print(f"\nThe 'langsmith_retriever_tool' is ready for use in the agent.")
    
    # Create the 3-node agent
    print("\n" + "="*60)
    print("STEP 2: CREATING & TESTING 3-NODE AGENT")
    print("="*60 + "\n")
    
    agent = create_agent(tool)
    
    # Test the agent with sample queries
    test_queries = [
        "How does LangSmith tracing work?",
        "What is LangSmith evaluation?",
        "How do I deploy agents with LangSmith?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"🔄 Processing Query: {query}")
        print('='*60 + "\n")
        
        # Invoke the agent (goes through all 3 nodes)
        result = agent.invoke({"question": query})
        
        # Print the formatted output
        print(result["formatted_output"])
    
    print("\n" + "="*60)
    print("✅ 3-NODE AGENT WORKING! Step 2 Complete.")
    print("="*60)

