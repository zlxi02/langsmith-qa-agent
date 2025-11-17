import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from docs_urls import LANGSMITH_DOC_URLS
from agent import create_agent
from utils import get_gateway_embeddings, create_retriever_tool

def setup_vector_store():
    """
    Set up the RAG ingestion pipeline:
    1. Load LangSmith documentation from web URLs
    2. Split documents into chunks
    3. Create embeddings and store in FAISS vector database
    4. Create a retriever tool for the agent
    """
    
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
    # Get embeddings configured for company gateway (shared utility)
    embeddings = get_gateway_embeddings()
    
    # Create FAISS vector store from documents
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Save the FAISS index to disk
    faiss_index_path = "faiss_langsmith_index"
    vectorstore.save_local(faiss_index_path)
    print(f"✓ Created FAISS index at: {os.path.abspath(faiss_index_path)}")
    
    # Step 4: Create Retriever Tool (shared utility)
    langsmith_retriever_tool = create_retriever_tool(vectorstore)
    
    print(f"✓ Created langsmith_retriever_tool")
    
    return langsmith_retriever_tool

if __name__ == "__main__":
    # Check for company gateway API key
    if not os.getenv("X_API_KEY"):
        print("⚠️  Warning: X_API_KEY environment variable not set!")
        print("   Please set it before running: export X_API_KEY='your-gateway-api-key'")
        exit(1)
    else:
        print("✓ X_API_KEY found")
    
    # Check for LangSmith API key (optional but recommended for tracing)
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        if os.getenv("LANGCHAIN_API_KEY"):
            print("✓ LangSmith tracing enabled")
            print(f"  Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
        else:
            print("⚠️  Warning: LANGCHAIN_TRACING_V2 is true but LANGCHAIN_API_KEY not set!")
    else:
        print("ℹ️  LangSmith tracing disabled (set LANGCHAIN_TRACING_V2=true to enable)")
    
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

