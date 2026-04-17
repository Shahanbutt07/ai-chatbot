# 🤖 AI Assistant — LangChain + Groq + RAG

A production-grade conversational AI chatbot with persistent memory, multi-conversation management, document Q&A (RAG), streaming responses, and persona switching.

## Features

| Feature | Description |
|---|---|
| 💬 Multi-conversation | Create, rename, and delete chat threads (like ChatGPT sidebar) |
| 🗄️ Persistent memory | All history saved to SQLite — survives page refreshes and restarts |
| ⚡ Streaming | Word-by-word streaming responses via Groq |
| 📄 RAG | Upload PDF/DOCX/TXT and chat with your documents (FAISS + HuggingFace embeddings) |
| 🎭 Personas | Switch between presets: Developer, Researcher, Tutor, Writer, or custom |
| 📊 Token tracking | Live session token counter in the sidebar |

## Stack

- **LLM**: LLaMA 3.3 70B via [Groq](https://groq.com) (fastest inference available)
- **Framework**: [LangChain](https://langchain.com) (chains, RAG, retrieval)
- **Vector Store**: FAISS (local, no cloud needed)
- **Embeddings**: `all-MiniLM-L6-v2` via HuggingFace
- **Memory**: SQLite (zero-config persistent storage)
- **UI**: Streamlit

## Setup

```bash
# 1. Clone / copy files
# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file
GROQ_API_KEY=your_groq_key_here
LANGCHAIN_API_KEY=your_langsmith_key_here   # optional, for tracing

# 4. Run
streamlit run app.py
```

## Project Structure

```
├── app.py           # Main Streamlit application
├── memory.py        # SQLite conversation persistence layer
├── rag.py           # Document ingestion + FAISS RAG chain
├── requirements.txt
├── .env             # Your API keys (never commit this)
├── chat_history.db  # Auto-created SQLite database
└── vector_stores/   # Auto-created FAISS indexes per conversation
```

## How RAG Works

1. Upload a PDF/DOCX/TXT via the file uploader
2. Document is chunked (1000 chars, 150 overlap) and embedded using MiniLM
3. FAISS index is saved locally, keyed to that conversation
4. On each question, a **history-aware retriever** rephrases the query using chat context, fetches top-4 chunks, and passes them to the LLM with the full conversation history
5. The badge "📄 RAG Active" confirms document mode is on

## What to Build Next (Tier 3+)

- [ ] User authentication (Streamlit-Authenticator)
- [ ] Multi-model switcher (GPT-4o, Gemini, Claude)
- [ ] Export chat to PDF/Markdown
- [ ] LangChain Agent with web search tool
- [ ] Dockerize for deployment