import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import importlib.util
from typing import Any
from PyPDF2 import PdfReader

# Core imports (always needed)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    USE_MODERN_LANGCHAIN = True
except ImportError:
    # Fallback to older version
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # pyright: ignore[reportMissingImports]
    from langchain.vectorstores import FAISS  # pyright: ignore[reportMissingImports]
    USE_MODERN_LANGCHAIN = False

# Try to import RetrievalQA without raising module-level errors on Streamlit Cloud
lc_chains_spec = importlib.util.find_spec("langchain.chains")
if lc_chains_spec:
    try:
        from langchain.chains import RetrievalQA  # pyright: ignore[reportMissingImports]
        HAS_RETRIEVALQA = True
    except ImportError:
        HAS_RETRIEVALQA = False
else:
    HAS_RETRIEVALQA = False

# Force manual QA path to ensure compatibility with local HuggingFace pipeline
HAS_RETRIEVALQA = False

# Legacy alias kept for compatibility with older references/lints
class HuggingFaceLLM:  # pragma: no cover - kept for tooling compatibility
    ...

# Cached local HuggingFace pipeline (avoids remote API limits)
@st.cache_resource(show_spinner=False)
def load_local_hf_pipeline(model_name: str = "google/flan-t5-small"):
    """Load a local HuggingFace text2text pipeline to stay within PDF context."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)


# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDF Q&A Bot",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">📄 PDF Question Answering Bot</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for information
with st.sidebar:
    st.header("ℹ️ About this app")
    st.markdown("---")
    st.markdown("### 📚 About")
    st.info("""
    **PDF Q&A Bot** is a Retrieval-Augmented Generation (RAG) application that allows you to:
    - Upload PDF documents
    - Ask questions about the content
    - Get accurate answers with source references
    
    Built with LangChain, Streamlit, FAISS, and HuggingFace.
    """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload PDF Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to start asking questions"
    )
    
    if uploaded_file:
        # Process PDF button
        if st.button("🔄 Process PDF", type="primary"):
            with st.spinner("📑 Processing PDF... This may take a moment."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    
                    # Extract text from PDF
                    pdf_reader = PdfReader(tmp_path)
                    text = ""
                    total_pages = len(pdf_reader.pages)
                    
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    
                    if not text.strip():
                        st.error("❌ Could not extract text from PDF. The file might be scanned or encrypted.")
                        st.session_state.pdf_processed = False
                    else:
                        # Split text into chunks
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200,
                            length_function=len
                        )
                        chunks = splitter.split_text(text)
                        
                        # Create embeddings and vector store (HuggingFace only)
                        st.info("🔄 Using HuggingFace embeddings + LLM")
                        # Lazy import to avoid loading if not needed
                        try:
                            from langchain_community.embeddings import HuggingFaceEmbeddings
                        except ImportError:
                            from langchain.embeddings import HuggingFaceEmbeddings
                        embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2"
                        )
                        
                        # Configure local HuggingFace pipeline (no external API calls)
                        llm = load_local_hf_pipeline("google/flan-t5-small")
                        
                        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                        
                        # Create QA chain (only if an LLM is configured)
                        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                        
                        if llm is not None:
                            if HAS_RETRIEVALQA:
                                # Use RetrievalQA if available
                                qa_chain = RetrievalQA.from_chain_type(
                                    llm=llm,
                                    retriever=retriever,
                                    return_source_documents=True
                                )
                            else:
                                # Simple approach: store retriever and llm separately
                                qa_chain = {
                                    "retriever": retriever,
                                    "llm": llm
                                }
                        else:
                            qa_chain = None
                        
                        # Store in session state
                        st.session_state.vector_store = vector_store
                        st.session_state.qa_chain = qa_chain
                        st.session_state.pdf_processed = True
                        
                        # Clean up temp file
                        os.unlink(tmp_path)
                        
                        st.success(f"✅ PDF processed successfully!")
                        st.info(f"📊 Pages: {total_pages} | Chunks: {len(chunks)}")
                        st.session_state.messages = []  # Clear previous chat
                        
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
                    st.session_state.pdf_processed = False
        
        # Display PDF processing status
        if st.session_state.pdf_processed:
            st.success("✅ PDF is ready for questions!")
        else:
            st.info("👆 Click 'Process PDF' after uploading to enable Q&A")

with col2:
    st.subheader("💬 Ask Questions")
    
    if st.session_state.pdf_processed:
        if st.session_state.qa_chain is None:
            st.warning("PDF was processed, but the language model is not ready yet. Please reprocess the PDF or restart the app.")
            st.stop()
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message:
                    with st.expander("📌 View Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Source {i}:**")
                            st.text(source[:500] + "..." if len(source) > 500 else source)
        
        # Question input
        query = st.chat_input("Ask a question about the PDF:")
        
        if query:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": query})
            
            with st.chat_message("user"):
                st.markdown(query)
            
            # Generate answer
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        # Simple approach: retrieve and generate using local pipeline
                        qa_data = st.session_state.qa_chain
                        retriever = qa_data["retriever"]
                        llm = qa_data["llm"]
                        
                        # Retrieve relevant documents (support modern retriever API)
                        if hasattr(retriever, "get_relevant_documents"):
                            docs = retriever.get_relevant_documents(query)
                        elif hasattr(retriever, "invoke"):
                            docs = retriever.invoke(query)
                        else:
                            raise AttributeError(
                                "Retriever object does not support document lookup."
                            )
                        
                        # Create context from documents
                        context = "\n\n".join([doc.page_content for doc in docs])
                        
                        # Generate answer using local HuggingFace pipeline
                        prompt = f"""Answer the following question using ONLY the context below. If the answer is not in the context, reply with "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer:"""
                        
                        llm_output = llm(
                            prompt,
                            max_new_tokens=256,
                            do_sample=False,
                        )
                        answer = llm_output[0]["generated_text"].strip() if llm_output else "I couldn't generate an answer."
                        sources_text = [doc.page_content for doc in docs] if docs else []
                        
                        st.markdown(answer)
                        
                        # Store assistant response
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources_text
                        })
                        
                        # Display sources in expander
                        if sources_text:
                            with st.expander("📌 View Source Documents"):
                                for i, source in enumerate(sources_text, 1):
                                    st.markdown(f"**Source {i}:**")
                                    source_str = source[:500] + "..." if len(source) > 500 else source
                                    st.text(source_str)
                                    st.markdown("---")
                                    
                    except Exception as e:
                        error_msg = f"❌ Error generating answer: {str(e)}"
                        st.error(error_msg)
                        import traceback
                        st.code(traceback.format_exc())
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
    else:
        st.info("📤 Please upload and process a PDF first to start asking questions.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>Built with ❤️ using Streamlit, LangChain, FAISS, and HuggingFace</p>
    </div>
    """,
    unsafe_allow_html=True
)

