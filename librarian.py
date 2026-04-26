import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank

# --- 1. CONFIGURATION ---
# We use a 10-minute timeout to allow your Mac to load the model without crashing
Settings.llm = Ollama(model="llama3.2:3b", request_timeout=3600.0)

# The "Translator" - Turns text into math (Vectors)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# --- 2. DATA LOADING ---
print("Step 1: Reading your documents from the /data folder...")
if not os.path.exists("./data"):
    os.makedirs("./data")
    print("Created /data folder. Please put a PDF in there and run again!")
    exit()

documents = SimpleDirectoryReader("./data").load_data()

# --- 3. INDEXING (THE FILING CABINET) ---
print("Step 2: Indexing documents. Your MacBook's GPU is working now...")
index = VectorStoreIndex.from_documents(documents)

# --- 4. THE SUPERVISOR (RE-RANKER) ---
# This looks at the top 10 results and picks the absolute best 3 
# to show the AI. This prevents the AI from getting confused.
print("Step 3: Initializing the Re-Ranker supervisor...")
rerank_postprocessor = SentenceTransformerRerank(
    model="BAAI/bge-reranker-base", 
    top_n=3
)

# --- 5. THE QUERY ENGINE ---
# We find 10 candidates first, then filter them down to 3
query_engine = index.as_query_engine(
    similarity_top_k=10, 
    node_postprocessors=[rerank_postprocessor]
)

# --- 6. EXECUTION ---
question = "What are the 3 most important points in this document?"
print(f"\nQuestion: {question}")
print("Thinking... (This may take a minute on the first run)")

try:
    response = query_engine.query(question)
    print("\n--- ADVANCED AI RESPONSE ---")
    print(response)
except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("TIP: Make sure the Ollama app is open and 'llama3.2:3b' is downloaded.")