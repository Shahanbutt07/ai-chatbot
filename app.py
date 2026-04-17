from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

import streamlit as st
import os
import uuid
from dotenv import load_dotenv

from memory import (
    init_db, create_conversation, get_all_conversations,
    delete_conversation, save_message, load_messages,
    get_conversation_info, auto_title_conversation, rename_conversation,
    update_system_prompt
)
from rag import ingest_document, has_vectorstore, build_rag_chain, delete_vectorstore

load_dotenv()
init_db()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
if key := os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = key

if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY is missing. Please add it to your .env file.")
    st.stop()

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0f0f0f; border-right: 1px solid #1e1e1e; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    .doc-badge { background: #1e3a2f; color: #4ade80 !important; padding: 2px 8px;
                 border-radius: 12px; font-size: 11px; font-weight: 600; }
    .token-info { font-size: 11px; color: #666; text-align: right; }
    .main .block-container { padding-top: 1rem; max-width: 820px; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

PRESETS = {
    "🤖 General Assistant":  "You are a helpful, concise, and friendly AI assistant.",
    "👨‍💻 Senior Developer":   "You are an expert software engineer. Give clean, production-ready code with brief explanations.",
    "📚 Research Analyst":   "You are a thorough research analyst. Provide structured responses and highlight uncertainties.",
    "✍️  Creative Writer":    "You are a creative writing collaborator. Be imaginative, vivid, and engaging.",
    "🎓 Socratic Tutor":      "You are a Socratic tutor. Guide the user to answers through questions rather than giving them directly.",
    "⚙️  Custom...":          "__custom__",
}

MAX_FILE_SIZE_MB = 10
MAX_HISTORY_MESSAGES = 50

@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        streaming=True,
    )

llm = get_llm()

def build_plain_chain(system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    return prompt | llm | StrOutputParser()

def validate_file(uploaded_file) -> tuple[bool, str]:
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large ({size_mb:.1f}MB). Maximum allowed is {MAX_FILE_SIZE_MB}MB."
    return True, ""

def trim_history(history: list, max_messages: int = MAX_HISTORY_MESSAGES) -> list:
    if len(history) > max_messages:
        return history[-max_messages:]
    return history

# ── Session state init ─────────────────────────────────────────────────────
if "active_conv_id" not in st.session_state:
    st.session_state.active_conv_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "token_count" not in st.session_state:
    st.session_state.token_count = 0
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💬 Conversations")

    if st.button("＋ New Chat", use_container_width=True, type="primary"):
        new_id = str(uuid.uuid4())
        create_conversation(new_id)
        st.session_state.active_conv_id = new_id
        st.session_state.chat_history = []
        st.session_state.token_count = 0
        st.session_state.last_uploaded_file = None
        st.rerun()

    st.divider()

    conversations = get_all_conversations()
    if not conversations:
        st.caption("No conversations yet. Create one above!")

    for conv in conversations:
        col1, col2 = st.columns([5, 1])
        is_active = conv["id"] == st.session_state.active_conv_id
        label = f"{'▶ ' if is_active else ''}{conv['title']}"
        if has_vectorstore(conv["id"]):
            label += " 📄"
        with col1:
            if st.button(label, key=f"conv_{conv['id']}", use_container_width=True):
                st.session_state.active_conv_id = conv["id"]
                st.session_state.chat_history = load_messages(conv["id"])
                st.session_state.token_count = 0
                st.session_state.last_uploaded_file = None
                st.rerun()
        with col2:
            if st.button("🗑", key=f"del_{conv['id']}"):
                delete_conversation(conv["id"])
                delete_vectorstore(conv["id"])
                if st.session_state.active_conv_id == conv["id"]:
                    st.session_state.active_conv_id = None
                    st.session_state.chat_history = []
                    st.session_state.last_uploaded_file = None
                st.rerun()

    st.divider()
    st.markdown("### ⚙️ Settings")

    preset_label = st.selectbox("Persona", list(PRESETS.keys()), key="preset_select")
    chosen_preset = PRESETS[preset_label]

    if chosen_preset == "__custom__":
        system_prompt = st.text_area(
            "Custom system prompt",
            value="You are a helpful assistant.",
            height=100,
            max_chars=1000
        )
    else:
        system_prompt = chosen_preset

    # ── Fixed: no more raw SQLite in app.py ───────────────────────────────
    if st.session_state.active_conv_id and st.button("Apply Persona", use_container_width=True):
        try:
            update_system_prompt(st.session_state.active_conv_id, system_prompt)
            st.success("Persona applied!")
        except Exception as e:
            st.error(f"Failed to apply persona: {str(e)}")

    st.divider()
    st.markdown(f"<div class='token-info'>~{st.session_state.token_count:,} tokens this session</div>",
                unsafe_allow_html=True)

# ── Main area ──────────────────────────────────────────────────────────────
if st.session_state.active_conv_id is None:
    st.markdown("""
    <div style="text-align:center; padding: 80px 20px;">
        <h1 style="font-size:3rem;">🤖 AI Assistant</h1>
        <p style="color:#888; font-size:1.1rem;">Powered by LLaMA 3.3 · LangChain · Groq</p>
        <p style="color:#555; margin-top:2rem;">← Create a new chat to get started</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

conv_info = get_conversation_info(st.session_state.active_conv_id)

if conv_info is None:
    st.session_state.active_conv_id = None
    st.session_state.chat_history = []
    st.warning("Conversation not found. Please create a new one.")
    st.rerun()

active_system_prompt = conv_info["system_prompt"]

col_title, col_upload = st.columns([3, 2])

with col_title:
    conv_title = conv_info["title"]
    new_title = st.text_input(
        "",
        value=conv_title,
        label_visibility="collapsed",
        placeholder="Conversation title...",
        max_chars=60
    )
    if new_title and new_title != conv_title:
        rename_conversation(st.session_state.active_conv_id, new_title)

with col_upload:
    uploaded_file = st.file_uploader(
        "Attach document",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed",
    )
    if uploaded_file and uploaded_file.name != st.session_state.last_uploaded_file:
        is_valid, error_msg = validate_file(uploaded_file)
        if not is_valid:
            st.error(error_msg)
        else:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    meta = ingest_document(uploaded_file, st.session_state.active_conv_id)
                    st.session_state.last_uploaded_file = uploaded_file.name
                    st.success(f"✅ {meta['filename']} — {meta['num_chunks']} chunks indexed")
                except Exception as e:
                    st.error(f"❌ Failed to process document: {str(e)}")

if has_vectorstore(st.session_state.active_conv_id):
    st.markdown("<span class='doc-badge'>📄 RAG Active — answering from your document</span>",
                unsafe_allow_html=True)

st.divider()

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        st.chat_message("user").write(message.content)
    else:
        st.chat_message("assistant").write(message.content)

input_text = st.chat_input("Ask anything...")

if input_text:
    if len(st.session_state.chat_history) == 0:
        auto_title_conversation(st.session_state.active_conv_id, input_text)

    trimmed_history = trim_history(st.session_state.chat_history)

    st.chat_message("user").write(input_text)

    if has_vectorstore(st.session_state.active_conv_id):
        try:
            rag_chain = build_rag_chain(llm, st.session_state.active_conv_id, active_system_prompt)
            if rag_chain is None:
                st.error("❌ Could not load document. Please re-upload.")
                st.stop()
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
                for chunk in rag_chain.stream({
                    "question": input_text,
                    "chat_history": trimmed_history
                }):
                    if isinstance(chunk, str):
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            response = full_response
        except Exception as e:
            st.error(f"❌ RAG error: {str(e)}")
            st.stop()
    else:
        try:
            chain = build_plain_chain(active_system_prompt)
            with st.chat_message("assistant"):
                response = st.write_stream(
                    chain.stream({
                        "question": input_text,
                        "chat_history": trimmed_history
                    })
                )
        except Exception as e:
            st.error(f"❌ Chat error: {str(e)}")
            st.stop()

    try:
        save_message(st.session_state.active_conv_id, "human", input_text)
        save_message(st.session_state.active_conv_id, "ai", response)
    except Exception as e:
        st.warning(f"⚠️ Message saved in session but failed to save to database: {str(e)}")

    st.session_state.chat_history.append(HumanMessage(content=input_text))
    st.session_state.chat_history.append(AIMessage(content=response))
    st.session_state.token_count += len(input_text + response) // 4
    st.rerun()