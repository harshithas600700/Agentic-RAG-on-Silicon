import streamlit as st
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Agentic-RAG-on-Silicon", layout="wide", page_icon="🚀")
st.title("🚀 Agentic RAG on Silicon")
st.markdown("### Privacy-First Document Intelligence (Advanced Mode)")

# --- 2. MODEL & SETTINGS SETUP ---
@st.cache_resource
def load_models():
    # Temperature 0.7 makes the AI more descriptive/less robotic
    llm = Ollama(
        model="llama3.2:3b", 
        request_timeout=3600.0, 
        additional_kwargs={"temperature": 0.7}
    )
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Increased top_n to 7 to give the AI more context to talk about
    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-base", 
        top_n=7
    )
    return llm, embed_model, reranker

llm, embed, reranker = load_models()
Settings.llm = llm
Settings.embed_model = embed

# --- 3. CUSTOM PROMPT TEMPLATE (The "Anti-Short" Fix) ---
# This forces the AI to be detailed and structured
qa_prompt_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information above, act as a professional Research Assistant. "
    "Provide a comprehensive, detailed, and structured answer to the query. "
    "Break your answer into logical sections using bullet points or numbered lists. "
    "Explain the technical details and 'why' behind the facts. "
    "If the context doesn't contain the answer, say so, but summarize what IS available.\n\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_prompt = PromptTemplate(qa_prompt_str)

# --- 4. SIDEBAR & DATA LOADING ---
if "index" not in st.session_state:
    st.session_state.index = None

with st.sidebar:
    st.header("📁 Document Management")
    uploaded_file = st.file_uploader("Upload a PDF for your Knowledge Base", type="pdf")
    
    if uploaded_file:
        if not os.path.exists("./data"):
            os.makedirs("./data")
        with open(os.path.join("./data", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Build Knowledge Base"):
        with st.spinner("Processing documents with MLX acceleration..."):
            documents = SimpleDirectoryReader("./data").load_data()
            st.session_state.index = VectorStoreIndex.from_documents(documents)
            st.success("Indexing complete!")

# --- 5. CHAT INTERFACE ---
if st.session_state.index:
    # We retrieve 15 chunks, then rerank down to the best 7
    query_engine = st.session_state.index.as_query_engine(
        similarity_top_k=15, 
        node_postprocessors=[reranker],
        text_qa_template=qa_prompt
    )
    
    user_query = st.text_input("💬 Ask a detailed question about your document:")
    
    if user_query:
        with st.spinner("Reasoning and synthesizing response..."):
            try:
                response = query_engine.query(user_query)
                st.markdown("---")
                st.markdown("#### 🤖 Detailed AI Analysis:")
                st.markdown(str(response))
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("👋 Welcome! Please upload a PDF in the sidebar and click 'Build' to begin.")