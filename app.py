import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import importlib.util
from typing import Any, List, Tuple
from PyPDF2 import PdfReader
import re
import html
import numpy as np

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

def highlight_relevant_sentence(source_text: str, answer_text: str) -> str:
    escaped_source = html.escape(source_text)
    sentences = re.split(r'(?<=[.!?])\s+', escaped_source)
    if not sentences:
        return escaped_source

    answer_lower = answer_text.lower()
    best_idx = -1
    best_score = 0

    for idx, sentence in enumerate(sentences):
        words = set(re.findall(r"\w+", sentence.lower()))
        score = sum(1 for w in words if w and w in answer_lower)
        if score > best_score:
            best_score = score
            best_idx = idx

    highlighted_sentences = []
    for idx, sentence in enumerate(sentences):
        if idx == best_idx and best_score > 0:
            highlighted_sentences.append(f'<span class="source-highlight">{sentence}</span>')
        else:
            highlighted_sentences.append(sentence)

    return " ".join(highlighted_sentences)


def is_followup_question(query: str) -> bool:
    """Detect if query is a vague follow-up that needs context."""
    vague_patterns = [
        r"^what\s+(are|is|were|was)\s+(they|it|those|these|that|this)\??$",
        r"^(tell|explain|describe|list)\s+(me\s+)?(more|them|it|those|these)\??$",
        r"^(and|also|what about)\s+(the\s+)?(others?|rest|more)\??$",
        r"^(can you|please)\s+(explain|list|tell)\s+(them|it|more)\??$",
        r"^(how|why|when|where)\??$",
        r"^(yes|no|okay|go on|continue)\??$",
    ]
    q = query.strip().lower()
    for pattern in vague_patterns:
        if re.match(pattern, q):
            return True
    words = q.split()
    if len(words) <= 4 and any(w in ["they", "it", "them", "those", "these", "that", "this"] for w in words):
        return True
    return False


def expand_with_context(query: str, messages: List[dict]) -> str:
    """Expand vague follow-up with previous Q&A context."""
    if not messages or not is_followup_question(query):
        return query
    
    last_qa = []
    for msg in reversed(messages[-4:]):
        if msg["role"] == "user":
            last_qa.insert(0, f"Q: {msg['content']}")
        elif msg["role"] == "assistant" and "sources" in msg:
            last_qa.insert(0, f"A: {msg['content']}")
    
    if last_qa:
        context_str = " ".join(last_qa)
        return f"{context_str} {query}"
    return query


def compute_semantic_similarity(text1: str, text2: str, embeddings_model) -> float:
    """Compute cosine similarity between two texts using embeddings."""
    try:
        emb1 = embeddings_model.embed_query(text1)
        emb2 = embeddings_model.embed_query(text2)
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    except Exception:
        return 0.0


def compute_max_chunk_similarity(query: str, docs: list, embeddings_model) -> float:
    """Compute max similarity between query and individual document chunks."""
    if not docs or not embeddings_model:
        return 0.0
    try:
        query_emb = np.array(embeddings_model.embed_query(query))
        max_sim = 0.0
        for doc in docs:
            chunk_emb = np.array(embeddings_model.embed_query(doc.page_content))
            sim = np.dot(query_emb, chunk_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb))
            max_sim = max(max_sim, float(sim))
        return max_sim
    except Exception:
        return 0.0


def validate_answer_grounding(answer: str, context: str, embeddings_model=None) -> Tuple[bool, float]:
    """Validate that the answer is grounded in the context.
    Uses PURE semantic similarity - no keyword matching.
    Returns (is_grounded, score).
    """
    if not answer or not context:
        return False, 0.0
    
    # Short answers are usually extracted directly - trust them
    if len(answer.split()) <= 15:
        return True, 1.0
    
    # Pure semantic check
    if embeddings_model:
        try:
            semantic_score = compute_semantic_similarity(answer, context, embeddings_model)
            is_grounded = semantic_score >= 0.4
            return is_grounded, semantic_score
        except Exception:
            pass
    
    # Fallback: if no embeddings, trust the answer
    return True, 1.0


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
if "embeddings_model" not in st.session_state:
    st.session_state.embeddings_model = None

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
    .source-panel {
        height: 260px;           /* fixed vertical size */
        width: 100%;             /* span the width of the expander */
        overflow-y: auto;        /* vertical scroll inside the box */
        overflow-x: auto;        /* horizontal scroll inside the box */
        padding: 0.75rem;
        border-radius: 6px;
        background-color: #ffffff;   /* solid white for maximum contrast */
        border: 1px solid #d0d0d0;
        font-family: "Source Code Pro", monospace;
        font-size: 0.9rem;
        color: #111111;              /* very dark text */
        white-space: pre-wrap;       /* allow wrapping but keep line breaks */
        box-sizing: border-box;
    }
    .source-highlight {
        background-color: #c8f3b4;   /* slightly stronger green but still soft */
        color: #000000;              /* ensure highlighted text is dark */
    }
    .upload-sticky {
        position: sticky;
        top: 0;
        z-index: 5;
        padding-bottom: 1rem;
        background-color: inherit;   /* blend with main background */
    }
    .chat-footer-note {
        position: fixed;
        bottom: 0.25rem;
        left: 0;
        right: 0;
        text-align: center;
        color: gray;
        font-size: 0.8rem;
        pointer-events: none;  /* don't block clicks on chat input */
    }
    /* Scrollable Ask Question section (vertical layout: section 1 = upload, section 2 = ask) */
    .ask-section-scroll {
        max-height: 500px;
        overflow-y: auto;
        padding-right: 0.5rem;
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

# Main two-column layout: left = upload, right = Q&A
col_upload, col_chat = st.columns(2)

with col_upload:
    # SECTION 1: Upload PDF document (left side)
    st.markdown("<div class='upload-sticky'>", unsafe_allow_html=True)

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
                        # Lazy import to avoid loading if not needed
                        try:
                            from langchain_community.embeddings import HuggingFaceEmbeddings
                        except ImportError:
                            from langchain.embeddings import HuggingFaceEmbeddings
                        embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2"
                        )
                        st.session_state.embeddings_model = embeddings

                        # Configure local HuggingFace pipeline (no external API calls)
                        llm = load_local_hf_pipeline("google/flan-t5-base")

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

    # When the PDF has been processed, tint the primary button (Process PDF) green
    if st.session_state.pdf_processed:
        st.markdown(
            """
            <style>
            .stButton > button[kind="primary"] {
                background-color: #16a34a !important;  /* green */
                border-color: #16a34a !important;
                color: #ffffff !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

with col_chat:
    # SECTION 2: Ask Questions (right side)
    st.subheader("💬 Ask Questions")
    st.markdown("<div class='ask-section-scroll'>", unsafe_allow_html=True)

    if st.session_state.pdf_processed:
        if st.session_state.qa_chain is None:
            st.warning("PDF was processed, but the language model is not ready yet. Please reprocess the PDF or restart the app.")
            st.stop()
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        # Display chat history (all past messages)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    # Show only the relevant PDF chunks used to answer the question,
                    # with the most relevant sentence highlighted in light green.
                    with st.expander("📌 Source Documents"):
                        answer_text = message["content"]
                        combined_html_parts = []
                        for source in message["sources"]:
                            highlighted_html = highlight_relevant_sentence(source, answer_text)
                            combined_html_parts.append(highlighted_html)
                        combined_html = "<br><br>".join(combined_html_parts)
                        st.markdown(
                            f"<div class='source-panel'>{combined_html}</div>",
                            unsafe_allow_html=True,
                        )

        # Question input at the bottom of the ask section
        query = st.chat_input("Ask a question about the PDF:")

        if query:
            # Basic validation: ignore extremely short/noisy queries such as
            # single letters or random characters, which lead to meaningless
            # retrieval results.
            cleaned = query.strip()
            # Count alphanumeric characters only
            alnum_only = "".join(ch for ch in cleaned if ch.isalnum())
            if len(alnum_only) < 3 or len(cleaned.split()) < 1:
                warning_msg = "Please enter a more specific question based on the PDF content."
                st.session_state.messages.append({
                    "role": "user",
                    "content": query,
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": warning_msg,
                })
                st.rerun()

            try:
                # Retrieve components
                qa_data = st.session_state.qa_chain
                retriever = qa_data["retriever"]
                llm = qa_data["llm"]

                # Expand vague follow-up questions with conversation context
                expanded_query = expand_with_context(query, st.session_state.messages)
                search_query = expanded_query if expanded_query != query else query

                # Retrieve relevant documents (support modern retriever API)
                if hasattr(retriever, "get_relevant_documents"):
                    docs = retriever.get_relevant_documents(search_query)
                elif hasattr(retriever, "invoke"):
                    docs = retriever.invoke(search_query)
                else:
                    raise AttributeError(
                        "Retriever object does not support document lookup."
                    )

                # Build full context from retrieved chunks
                context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""

                # Semantic Similarity Check - use max chunk similarity for better accuracy
                semantic_score = 0.0
                if docs and st.session_state.embeddings_model:
                    semantic_score = compute_max_chunk_similarity(
                        search_query, docs, st.session_state.embeddings_model
                    )
                
                # Threshold 0.35 for individual chunk matching (more reliable than full context)
                SEMANTIC_THRESHOLD = 0.35
                is_relevant = bool(context.strip()) and semantic_score >= SEMANTIC_THRESHOLD

                # Debug info in sidebar
                with st.sidebar:
                    if search_query != query:
                        st.write(f"🔗 Expanded: {search_query[:60]}...")
                    st.write(f"🔍 Max Chunk Score: {semantic_score:.3f}")
                    st.write(f"📊 Threshold: {SEMANTIC_THRESHOLD}")
                    st.write(f"✅ Relevant: {is_relevant}")

                if not is_relevant:
                    if not context.strip():
                        answer = "No relevant content found in the PDF. Please try rephrasing your question."
                    else:
                        answer = f"The question doesn't seem related to the PDF content (score: {semantic_score:.2f}). Try asking something more specific to the document."
                    sources_text = []
                else:
                    # Use expanded query for better context in follow-up questions
                    question_for_llm = search_query if search_query != query else query
                    
                    # Build conversation history for context
                    chat_history = ""
                    recent_msgs = st.session_state.messages[-6:]  # Last 3 Q&A pairs
                    for msg in recent_msgs:
                        if msg["role"] == "user":
                            chat_history += f"User: {msg['content']}\n"
                        elif msg["role"] == "assistant":
                            chat_history += f"Assistant: {msg['content']}\n"
                    
                    # System prompt for consistent behavior
                    system_prompt = """You are a precise document assistant. Your rules:
1. Answer ONLY using information from the provided context
2. If asked "what are they" or similar, list ALL items mentioned for that topic
3. Be specific and complete - list all relevant details
4. If information is not in context, say "I don't have that information"
5. Never make up facts or use external knowledge"""

                    prompt = f"""{system_prompt}

{f"Previous conversation:{chr(10)}{chat_history}" if chat_history else ""}

Context from document:
{context}

Current question: {question_for_llm}

Answer:"""

                    llm_output = llm(
                        prompt,
                        max_new_tokens=300,
                        do_sample=False,
                    )
                    raw_answer = llm_output[0]["generated_text"].strip() if llm_output else ""
                    
                    # Step 3: Post-generation validation - check answer is grounded
                    is_grounded, grounding_score = validate_answer_grounding(
                        raw_answer, context, st.session_state.embeddings_model
                    )
                    
                    # Debug grounding in sidebar
                    with st.sidebar:
                        st.write(f"📎 Grounding Score: {grounding_score:.3f}")
                        st.write(f"✅ Grounded: {is_grounded}")
                    
                    if not raw_answer or not is_grounded:
                        answer = "I couldn't find a reliable answer in the PDF for this question."
                    else:
                        answer = raw_answer
                    
                    sources_text = [doc.page_content for doc in docs] if docs else []

                # Update chat history (user question + assistant answer)
                st.session_state.messages.append({"role": "user", "content": query})
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources_text,
                })

            except Exception as e:
                error_msg = "❌ An error occurred while processing your question. Please try again."
                st.session_state.messages.append({
                    "role": "user",
                    "content": query,
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })

            # Rerun so that the updated history is rendered above and the input
            # bar remains visually fixed at the bottom, similar to ChatGPT.
            st.rerun()
    else:
        st.info("📤 Please upload and process a PDF first to start asking questions.")

    st.markdown("</div>", unsafe_allow_html=True)

# Fixed footer text rendered near the bottom of the page, visually under the
# chat input bar.
st.markdown(
    """
    <div class='chat-footer-note'>
      Built with ❤️ using Streamlit, LangChain, FAISS, and HuggingFace
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

