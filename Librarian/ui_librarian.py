import streamlit as st
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from agents import run_consensus
# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Librarian Pro", layout="wide", page_icon="🕵️‍♀️")

st.markdown("""
    <style>
    .stChatFloatingInputContainer { background-color: rgba(0,0,0,0); }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MODELS & RECURSIVE SETTINGS ---
@st.cache_resource
def load_essentials():
    llm = Ollama(model="llama3.2:3b", request_timeout=3600.0, additional_kwargs={"temperature": 0.1})
    embed = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # This creates the 'Small' chunks (sentences) but remembers the 'Big' context (window)
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=2,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    return llm, embed, node_parser

llm, embed, node_parser = load_essentials()
Settings.llm = llm
Settings.embed_model = embed
Settings.node_parser = node_parser

# --- 3. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index" not in st.session_state:
    st.session_state.index = None

# --- 4. THE LAYOUT ---
col1, col2 = st.columns([1, 3])

with col1:
    st.header("📂 Data Hub")
    uploaded_file = st.file_uploader("Drop Research PDFs", type="pdf")
    
    if uploaded_file:
        if not os.path.exists("./data"): os.makedirs("./data")
        with open(os.path.join("./data", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File Saved: {uploaded_file.name}")

    if st.button("⚡ Build/Refresh Index"):
        with st.spinner("Recursive Indexing..."):
            documents = SimpleDirectoryReader("./data").load_data()
            # The index now uses the SentenceWindowNodeParser automatically
            st.session_state.index = VectorStoreIndex.from_documents(documents)
            st.toast("Contextual Index Built!", icon="🧠")

    if st.session_state.index:
        st.metric(label="Retrieval Mode", value="Sentence Window", delta="High Precision")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

with col2:
    st.header("💬 Librarian Intelligence")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if not st.session_state.index:
            st.error("Build the knowledge base first!")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Retrieving data and deploying Agents..."):
                    
                    # 1. RETRIEVE THE DATA
                    retriever = st.session_state.index.as_retriever(similarity_top_k=3)
                    nodes = retriever.retrieve(prompt)
                    
                    # --- NEW: EMPTY CONTEXT SAFEGUARD ---
                    # Check if the highest matching chunk has a terrible score (below 0.3)
                    if not nodes or nodes[0].score < 0.3:
                        score_val = nodes[0].score if nodes else 0
                        warning_msg = f"⚠️ **Low Confidence Score ({score_val:.2f}):** No relevant context found in your uploaded documents. I am aborting the AI agents to prevent hallucinations."
                        st.warning(warning_msg)
                        st.session_state.messages.append({"role": "assistant", "content": warning_msg})
                    
                    else:
                        # 2. APPLY THE SMALL-TO-BIG TRICK
                        postprocessor = MetadataReplacementPostProcessor(target_metadata_key="window")
                        processed_nodes = postprocessor.postprocess_nodes(nodes)
                        context_string = "\n\n".join([n.node.text for n in processed_nodes])
                        
                        # 3. HAND OFF TO CREWAI
                        consensus_data = run_consensus(context_data=context_string, user_query=prompt)
                        
                        final_text = consensus_data["final_answer"]
                        analyst_draft = consensus_data["analyst_draft"]
                        best_score = nodes[0].score
                        
                        # --- NEW: UI EXPANDER SEPARATION ---
                        # Display the clean, final answer
                        st.markdown(final_text)
                        
                        # Tuck the internal debate into a clickable expander
                        with st.expander(f"🔍 View System Audit Log (Confidence: {best_score:.2f})"):
                            st.markdown("### 📝 Analyst's Initial Draft")
                            st.info(analyst_draft)
                            
                            st.markdown("### 📄 Raw Context Extracted from PDF")
                            st.caption(context_string)
                        
                        st.session_state.messages.append({"role": "assistant", "content": final_text})