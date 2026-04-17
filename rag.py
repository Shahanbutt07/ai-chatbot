import os
import tempfile
import shutil
import logging
from pathlib import Path
from functools import lru_cache
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VECTOR_STORE_DIR = "vector_stores"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVER_K = 4
MAX_CHUNKS = 500  # prevent extremely large docs from overwhelming the system

SUPPORTED_TYPES = {"pdf", "docx", "doc", "txt"}

@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def load_document(file_path: str, file_type: str) -> list:
    """Load document and return pages. Raises clear errors."""
    loaders = {
        "pdf": PyPDFLoader,
        "docx": Docx2txtLoader,
        "doc": Docx2txtLoader,
        "txt": TextLoader,
    }
    if file_type not in loaders:
        raise ValueError(f"Unsupported file type: {file_type}. Supported: {SUPPORTED_TYPES}")
    
    try:
        loader = loaders[file_type](file_path)
        docs = loader.load()
        if not docs:
            raise ValueError("Document appears to be empty.")
        return docs
    except Exception as e:
        raise RuntimeError(f"Failed to load document: {str(e)}")

def ingest_document(uploaded_file, conv_id: str) -> dict:
    """Process uploaded file and store in FAISS vectorstore."""
    file_ext = uploaded_file.name.split(".")[-1].lower()

    if file_ext not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported file type: .{file_ext}")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        logger.info(f"Loading document: {uploaded_file.name}")
        docs = load_document(tmp_path, file_ext)
    finally:
        # Always clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("Could not extract any text from document.")

    # Warn if document is very large
    if len(chunks) > MAX_CHUNKS:
        logger.warning(f"Large document: {len(chunks)} chunks. Trimming to {MAX_CHUNKS}.")
        chunks = chunks[:MAX_CHUNKS]

    # Build and save vectorstore
    logger.info(f"Building vectorstore with {len(chunks)} chunks...")
    embeddings = get_embeddings()
    
    store_path = os.path.join(VECTOR_STORE_DIR, conv_id)
    os.makedirs(store_path, exist_ok=True)

    # If vectorstore exists already, merge with new docs
    existing_store_path = Path(store_path) / "index.faiss"
    if existing_store_path.exists():
        existing = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
        new_store = FAISS.from_documents(chunks, embeddings)
        existing.merge_from(new_store)
        existing.save_local(store_path)
        logger.info("Merged with existing vectorstore.")
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(store_path)
        logger.info("Created new vectorstore.")

    return {
        "filename": uploaded_file.name,
        "num_chunks": len(chunks),
        "num_pages": len(docs),
    }

def load_vectorstore(conv_id: str):
    """Load FAISS vectorstore for a conversation."""
    store_path = os.path.join(VECTOR_STORE_DIR, conv_id)
    if not os.path.exists(store_path):
        return None
    try:
        embeddings = get_embeddings()
        return FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        logger.error(f"Failed to load vectorstore for {conv_id}: {str(e)}")
        return None

def has_vectorstore(conv_id: str) -> bool:
    """Check if a vectorstore exists for this conversation."""
    return os.path.exists(os.path.join(VECTOR_STORE_DIR, conv_id, "index.faiss"))

def delete_vectorstore(conv_id: str):
    """Delete vectorstore for a conversation."""
    store_path = os.path.join(VECTOR_STORE_DIR, conv_id)
    if os.path.exists(store_path):
        shutil.rmtree(store_path)
        logger.info(f"Deleted vectorstore for {conv_id}")

def format_docs(docs) -> str:
    """Format retrieved docs with page numbers if available."""
    formatted = []
    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "")
        source = doc.metadata.get("source", "")
        header = f"[Chunk {i+1}"
        if page:
            header += f" | Page {int(page)+1}"
        if source:
            header += f" | {Path(source).name}"
        header += "]"
        formatted.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(formatted)

def build_rag_chain(llm, conv_id: str, system_prompt: str):
    """Build a conversation-aware RAG chain."""
    vectorstore = load_vectorstore(conv_id)
    if vectorstore is None:
        return None

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K}
    )

    # Step 1 — Rephrase question using chat history
    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Given the chat history and the latest user question, "
            "rephrase it into a standalone question that can be understood "
            "without the chat history. Do NOT answer it. "
            "If it's already standalone, return it as-is."
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])
    rephrase_chain = rephrase_prompt | llm | StrOutputParser()

    # Step 2 — Answer using retrieved context
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""{system_prompt}

Use the document context below to answer the question.
- If the answer is in the context, answer clearly and cite the chunk/page.
- If the context doesn't contain the answer, say: "I couldn't find this in the uploaded document."
- Do not make up information.

Context:
{{context}}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])

    # Full RAG chain
    rag_chain = (
        RunnablePassthrough.assign(
            standalone_question=rephrase_chain
        )
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["standalone_question"]))
        )
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain