import streamlit as st
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core.postprocessor import SentenceTransformerRerank # Commented out to save RAM

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Agentic-RAG-on-Silicon", layout="wide", page_icon="🚀")
st.title("🚀 Agentic RAG on Silicon")
st.markdown("### Privacy-First Document Intelligence (Optimized for Speed)")

# --- 2. MODEL & SETTINGS SETUP ---
@st.cache_resource
def load_models():
    # Temperature 0.0 is the "Fastest" mode - no creative wandering
    llm = Ollama(
        model="llama3.2:3b", 
        request_timeout=120.0, # Reduced timeout so it fails fast instead of hanging
        additional_kwargs={"temperature": 0.0}
    )
    # Using a very lightweight embedding model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    return llm, embed_model

llm, embed = load_models()
Settings.llm = llm
Settings.embed_model = embed

# --- 3. CUSTOM PROMPT TEMPLATE ---
qa_prompt_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Using ONLY the context above, provide a detailed and structured answer. "
    "If you don't know the answer, say you don't know.\n\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_prompt = PromptTemplate(qa_prompt_str)

# --- 4. SIDEBAR & DATA LOADING ---
if "index" not in st.session_state:
    st.session_state.index = None

with st.sidebar:
    st.header("📁 Document Management")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file:
        if not os.path.exists("./data"):
            os.makedirs("./data")
        with open(os.path.join("./data", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Build Knowledge Base"):
        with st.spinner("Indexing..."):
            documents = SimpleDirectoryReader("./data").load_data()
            st.session_state.index = VectorStoreIndex.from_documents(documents)
            st.success("Complete!")

# --- 5. CHAT INTERFACE ---
if st.session_state.index:
    # Removed the Reranker here to ensure immediate responses
    query_engine = st.session_state.index.as_query_engine(
        similarity_top_k=3, # Reduced from 8 to 3 for instant processing
        text_qa_template=qa_prompt
    )
    
    user_query = st.text_input("💬 Ask about your document:")
    
    if user_query:
        with st.spinner("Generating..."):
            try:
                response = query_engine.query(user_query)
                st.markdown("---")
                st.markdown("#### 🤖 AI Analysis:")
                st.markdown(str(response))
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("👋 Upload a PDF and Build the base to start.")