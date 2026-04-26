import streamlit as st
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank

# --- Page Config ---
st.set_page_config(page_title="Agentic-RAG-on-Silicon", layout="wide")
st.title("🚀 Agentic RAG on Silicon")
st.markdown("### Privacy-First Document Intelligence")

# --- Setup Models (Cached to save memory) ---
@st.cache_resource
def load_models():
    # High timeout for MacBook Air first-run stability
    llm = Ollama(model="llama3.2:3b", request_timeout=3600.0)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=3)
    return llm, embed_model, reranker

llm, embed, reranker = load_models()
Settings.llm = llm
Settings.embed_model = embed

# --- Sidebar for Uploads ---
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file:
        # Save file to /data folder
        if not os.path.exists("./data"):
            os.makedirs("./data")
        with open(os.path.join("./data", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved {uploaded_file.name}")

# --- Main Logic ---
if "index" not in st.session_state:
    st.session_state.index = None

if st.button("Build/Update Knowledge Base"):
    with st.spinner("Indexing and applying MLX acceleration..."):
        documents = SimpleDirectoryReader("./data").load_data()
        st.session_state.index = VectorStoreIndex.from_documents(documents)
        st.success("Librarian is ready!")

# --- Chat Interface ---
if st.session_state.index:
    query_engine = st.session_state.index.as_query_engine(
        similarity_top_k=10, 
        node_postprocessors=[reranker]
    )
    
    user_query = st.text_input("Ask your document anything:")
    
    if user_query:
        with st.spinner("Reasoning over document..."):
            response = query_engine.query(user_query)
            st.markdown("#### AI Response:")
            st.write(str(response))
else:
    st.info("Please upload a PDF and click 'Build' to start.")