import streamlit as st
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings, OpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.chains import RetrievalQA
except ImportError:
    # Fallback for older LangChain versions
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import FAISS
    from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
    from langchain.llms import OpenAI
    from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import tempfile

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

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Choose embedding model
    use_openai = st.checkbox("Use OpenAI (Requires API Key)", value=True)
    
    if use_openai:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            st.warning("⚠️ OpenAI API key not found in .env file")
            st.info("Please add OPENAI_API_KEY to your .env file")
            use_openai = False
        else:
            st.success("✅ OpenAI API key loaded")
    
    st.markdown("---")
    st.markdown("### 📚 About")
    st.info("""
    **PDF Q&A Bot** is a Retrieval-Augmented Generation (RAG) application that allows you to:
    - Upload PDF documents
    - Ask questions about the content
    - Get accurate answers with source references
    
    Built with LangChain, Streamlit, FAISS, and OpenAI/HuggingFace.
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
                        
                        # Create embeddings and vector store
                        if use_openai and openai_api_key:
                            # Set API key in environment for newer versions
                            os.environ["OPENAI_API_KEY"] = openai_api_key
                            embeddings = OpenAIEmbeddings()
                            llm = OpenAI(temperature=0.2)
                        else:
                            st.info("🔄 Using HuggingFace embeddings (free alternative)")
                            embeddings = HuggingFaceEmbeddings(
                                model_name="sentence-transformers/all-MiniLM-L6-v2"
                            )
                            # For HuggingFace, you'll need to set up HuggingFaceHub for LLM
                            # For now, we'll still use OpenAI for generation if available
                            if openai_api_key:
                                os.environ["OPENAI_API_KEY"] = openai_api_key
                                llm = OpenAI(temperature=0.2)
                            else:
                                st.error("⚠️ Please set OPENAI_API_KEY in .env file for answer generation")
                                st.stop()
                        
                        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                        
                        # Create QA chain
                        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            retriever=retriever,
                            return_source_documents=True
                        )
                        
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
                        result = st.session_state.qa_chain({"query": query})
                        
                        answer = result["result"]
                        source_docs = result.get("source_documents", [])
                        
                        st.markdown(answer)
                        
                        # Store assistant response
                        sources_text = [doc.page_content for doc in source_docs]
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources_text
                        })
                        
                        # Display sources in expander
                        if source_docs:
                            with st.expander("📌 View Source Documents"):
                                for i, doc in enumerate(source_docs, 1):
                                    st.markdown(f"**Source {i}:**")
                                    st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                    st.markdown("---")
                                    
                    except Exception as e:
                        error_msg = f"❌ Error generating answer: {str(e)}"
                        st.error(error_msg)
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
    <p>Built with ❤️ using Streamlit, LangChain, FAISS, and OpenAI</p>
    </div>
    """,
    unsafe_allow_html=True
)

