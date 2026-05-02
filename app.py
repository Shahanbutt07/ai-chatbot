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
    st.error("GROQ_API_KEY is missing. Please add it to your .env file.")
    st.stop()

st.set_page_config(page_title="ORION — AI Intelligence", page_icon="◎", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Inter:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --ink:       #0A0A0B;
    --ink-2:     #111114;
    --ink-3:     #18181C;
    --ink-4:     #222228;
    --line:      #2A2A32;
    --line-hi:   #3A3A45;
    --gold:      #C9A84C;
    --gold-dim:  rgba(201,168,76,0.10);
    --gold-glow: rgba(201,168,76,0.20);
    --gold-pale: #E8D5A3;
    --silver:    #8A8A9A;
    --ash:       #5A5A6A;
    --smoke:     #3A3A4A;
    --white:     #F0EEE8;
    --white-dim: rgba(240,238,232,0.70);
    --r-sm: 5px; --r-md: 9px; --r-lg: 14px;
}

html, body, .stApp { background: var(--ink) !important; font-family: 'Inter', sans-serif !important; color: var(--white) !important; }
.main .block-container { padding: 24px 40px 16px !important; max-width: 900px !important; }
*:focus { outline: none !important; }
[data-baseweb] *:focus { outline: none !important; box-shadow: none !important; }
::-webkit-scrollbar { width: 3px; } ::-webkit-scrollbar-track { background: transparent; } ::-webkit-scrollbar-thumb { background: var(--smoke); border-radius: 3px; }

[data-testid="stSidebar"] { background: var(--ink-2) !important; border-right: 1px solid var(--line) !important; }
[data-testid="stSidebar"] * { color: var(--white) !important; }

.orion-logo { padding: 22px 20px 18px; border-bottom: 1px solid var(--line); }
.orion-wordmark { font-family: 'Playfair Display', serif; font-weight: 900; font-size: 26px; letter-spacing: 8px; text-transform: uppercase; color: var(--gold) !important; line-height: 1; margin-bottom: 3px; }
.orion-tagline { font-family: 'JetBrains Mono', monospace; font-size: 9px; letter-spacing: 2px; text-transform: uppercase; color: var(--ash) !important; }
.orion-divline { height: 1px; background: linear-gradient(90deg, var(--gold) 0%, transparent 70%); margin-top: 14px; }

.sb-label { font-family: 'JetBrains Mono', monospace; font-size: 9px; letter-spacing: 2.5px; text-transform: uppercase; color: var(--ash) !important; padding: 14px 20px 6px; display: block; }

[data-testid="stSidebar"] .stButton button {
    background: transparent !important; color: var(--silver) !important; border: 1px solid transparent !important;
    border-radius: var(--r-sm) !important; font-size: 12px !important; font-family: 'Inter', sans-serif !important;
    font-weight: 300 !important; text-align: left !important; padding: 7px 10px !important;
    height: auto !important; min-height: 34px !important; transition: all 0.15s ease !important; letter-spacing: 0.2px !important;
}
[data-testid="stSidebar"] .stButton button:hover { background: var(--ink-3) !important; color: var(--white-dim) !important; border-color: var(--line) !important; }

[data-testid="stSidebar"] .stButton[data-testid*="primary"] button,
button[kind="primary"] {
    background: transparent !important; color: var(--gold) !important; border: 1px solid var(--gold) !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important; font-weight: 500 !important;
    letter-spacing: 2px !important; text-transform: uppercase !important; border-radius: var(--r-sm) !important; height: 40px !important;
}
button[kind="primary"]:hover { background: var(--gold-dim) !important; box-shadow: 0 0 20px var(--gold-glow) !important; }

[data-testid="stSidebar"] .stSelectbox > div > div { background: var(--ink-3) !important; border: 1px solid var(--line) !important; border-radius: var(--r-sm) !important; color: var(--white) !important; font-size: 12px !important; font-family: 'Inter', sans-serif !important; }
[data-testid="stSidebar"] .stSelectbox > div > div:hover { border-color: var(--line-hi) !important; }

[data-testid="stSidebar"] .stTextArea textarea { background: var(--ink-3) !important; border: 1px solid var(--line) !important; border-radius: var(--r-sm) !important; color: var(--white) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important; resize: none !important; }
[data-testid="stSidebar"] .stTextArea textarea:focus { border-color: var(--gold) !important; box-shadow: 0 0 0 2px var(--gold-dim) !important; }

[data-testid="stSidebar"] hr, hr { border: none !important; border-top: 1px solid var(--line) !important; margin: 8px 0 !important; }

[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { font-family: 'JetBrains Mono', monospace !important; font-size: 9px !important; font-weight: 500 !important; letter-spacing: 2.5px !important; text-transform: uppercase !important; color: var(--ash) !important; padding: 12px 20px 4px !important; margin: 0 !important; }

.welcome-wrap { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 82vh; text-align: center; padding: 40px 20px; }
.orion-hero { font-family: 'Playfair Display', serif; font-weight: 900; font-size: 72px; letter-spacing: 18px; text-transform: uppercase; color: var(--gold) !important; line-height: 1; margin-bottom: 6px; }
.hero-rule { width: 120px; height: 1px; background: linear-gradient(90deg, transparent, var(--gold), transparent); margin: 0 auto 16px; }
.hero-sub { font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 3px; text-transform: uppercase; color: var(--ash) !important; margin-bottom: 56px; }

.feat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; max-width: 580px; margin: 0 auto; background: var(--line); border: 1px solid var(--line); border-radius: var(--r-md); overflow: hidden; }
.feat-cell { background: var(--ink-2); padding: 20px 18px; text-align: left; }
.feat-num { font-family: 'Playfair Display', serif; font-size: 28px; font-weight: 700; color: var(--gold) !important; line-height: 1; margin-bottom: 8px; opacity: 0.45; }
.feat-title { font-family: 'Inter', sans-serif; font-size: 12px; font-weight: 500; color: var(--white-dim) !important; margin-bottom: 4px; }
.feat-desc { font-size: 11px; font-weight: 300; color: var(--ash) !important; line-height: 1.5; }
.welcome-cta { margin-top: 44px; font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: var(--smoke) !important; }
.welcome-cta::before { content: '← '; }

.stTextInput input {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid var(--line) !important;
    border-radius: 0 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 15px !important;
    letter-spacing: 0.2px !important;
    color: var(--white) !important;
    padding: 0 0 8px !important;
    box-shadow: none !important;
}
.stTextInput input:focus { border-bottom-color: var(--gold) !important; box-shadow: none !important; outline: none !important; }
.stTextInput input::placeholder { color: var(--smoke) !important; font-style: italic; }

[data-testid="stFileUploader"] { background: var(--ink-3) !important; border: 1px dashed var(--line-hi) !important; border-radius: var(--r-md) !important; padding: 6px 12px !important; }
[data-testid="stFileUploader"]:hover { border-color: var(--gold) !important; }
[data-testid="stFileUploader"] * { font-size: 11px !important; color: var(--silver) !important; }

.rag-badge { display: inline-flex; align-items: center; gap: 8px; background: rgba(201,168,76,0.08); border: 1px solid rgba(201,168,76,0.25); color: var(--gold-pale) !important; padding: 5px 14px; border-radius: 3px; font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase; animation: fade-in 0.4s ease; }
.rag-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--gold); animation: pulse-dot 2s ease-in-out infinite; display: inline-block; }
@keyframes pulse-dot { 0%,100%{opacity:1} 50%{opacity:0.3} }
@keyframes fade-in { from{opacity:0;transform:translateY(-3px)} to{opacity:1;transform:translateY(0)} }

/* ── HIDE default Streamlit avatar images, replace with initials via CSS ── */
[data-testid="chatAvatarIcon-user"] img,
[data-testid="chatAvatarIcon-assistant"] img { display: none !important; }

[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] {
    width: 28px !important; height: 28px !important;
    border-radius: 6px !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important; font-weight: 500 !important;
    flex-shrink: 0 !important;
}
[data-testid="chatAvatarIcon-user"] {
    background: var(--ink-4) !important;
    border: 1px solid var(--line-hi) !important;
    color: var(--silver) !important;
}
[data-testid="chatAvatarIcon-user"]::after {
    content: 'YOU' !important;
    color: var(--silver) !important;
    font-size: 8px !important;
    letter-spacing: 0.5px !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background: #1C1400 !important;
    border: 1px solid var(--gold) !important;
    color: var(--gold) !important;
}
[data-testid="chatAvatarIcon-assistant"]::after {
    content: '◎' !important;
    color: var(--gold) !important;
    font-size: 12px !important;
}

/* ── MESSAGES ── */
[data-testid="stChatMessage"] { background: transparent !important; border: none !important; padding: 10px 0 !important; }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { font-size: 14px !important; line-height: 1.75 !important; color: var(--white-dim) !important; font-weight: 300 !important; font-family: 'Inter', sans-serif !important; }
[data-testid="stChatMessage"] span { color: var(--white-dim) !important; font-family: 'Inter', sans-serif !important; }
[data-testid="stChatMessage"] code { background: var(--ink-4) !important; border: 1px solid var(--line) !important; border-radius: 3px !important; padding: 2px 6px !important; font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; color: var(--gold-pale) !important; }
[data-testid="stChatMessage"] pre { background: var(--ink-3) !important; border: 1px solid var(--line) !important; border-left: 2px solid var(--gold) !important; border-radius: 0 var(--r-sm) var(--r-sm) 0 !important; padding: 16px !important; }
[data-testid="stChatMessage"] pre code { background: transparent !important; border: none !important; padding: 0 !important; color: var(--white-dim) !important; }

/* ── CHAT INPUT — kill Streamlit's red outline completely ── */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div {
    background: var(--ink-3) !important;
    border: 1px solid var(--line-hi) !important;
    border-radius: var(--r-md) !important;
    outline: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"]:focus-within,
[data-testid="stChatInput"]:focus-within > div,
[data-testid="stChatInput"]:focus-within > div > div {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px var(--gold-dim) !important;
    outline: none !important;
}
/* Kill any red/orange/default focus rings from Streamlit internals */
[data-testid="stChatInput"] *:focus,
[data-testid="stChatInput"] *:focus-visible,
[data-testid="stChatInput"] *:focus-within { outline: none !important; box-shadow: none !important; border-color: transparent !important; }
[data-testid="stChatInput"] textarea { background: transparent !important; color: var(--white) !important; font-family: 'Inter', sans-serif !important; font-size: 14px !important; font-weight: 300 !important; border: none !important; outline: none !important; box-shadow: none !important; }
[data-testid="stChatInput"] textarea::placeholder { color: var(--ash) !important; font-style: italic; }
/* The send button inside chat input */
[data-testid="stChatInput"] button { background: var(--gold-dim) !important; border: 1px solid var(--gold) !important; border-radius: 6px !important; color: var(--gold) !important; }

.stAlert { background: var(--ink-3) !important; border-radius: var(--r-sm) !important; border: 1px solid var(--line) !important; font-size: 12px !important; }
.stSuccess { border-left: 2px solid var(--gold) !important; background: var(--gold-dim) !important; }
.stError { border-left: 2px solid #884444 !important; background: rgba(136,68,68,0.08) !important; }

.token-badge { display: inline-flex; align-items: center; gap: 6px; font-family: 'JetBrains Mono', monospace; font-size: 9px; letter-spacing: 1.5px; text-transform: uppercase; color: var(--smoke) !important; padding: 8px 20px 16px; }
.token-badge::before { content: '◎  '; color: var(--gold); opacity: 0.5; }

label, .stSelectbox label, .stTextArea label { font-family: 'JetBrains Mono', monospace !important; font-size: 9px !important; letter-spacing: 2px !important; text-transform: uppercase !important; color: var(--ash) !important; }
.stSpinner > div { border-color: var(--gold) transparent transparent transparent !important; }
.stCaption { color: var(--smoke) !important; font-size: 10px !important; font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stSidebar"] .stButton, [data-testid="stSidebar"] .stSelectbox, [data-testid="stSidebar"] .stTextArea { padding: 0 12px !important; }
* { transition: background 0.12s ease, border-color 0.12s ease, color 0.1s ease; }
</style>
""", unsafe_allow_html=True)

PRESETS = {
    "◎ General Assistant":   "You are a helpful, concise, and friendly AI assistant.",
    "▸ Senior Developer":    "You are an expert software engineer. Give clean, production-ready code with brief explanations.",
    "▸ Research Analyst":    "You are a thorough research analyst. Provide structured responses and highlight uncertainties.",
    "▸ Creative Writer":     "You are a creative writing collaborator. Be imaginative, vivid, and engaging.",
    "▸ Socratic Tutor":      "You are a Socratic tutor. Guide the user to answers through questions rather than giving them directly.",
    "▸ Custom...":           "__custom__",
}

MAX_FILE_SIZE_MB = 10
MAX_HISTORY_MESSAGES = 50

@st.cache_resource
def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"), streaming=True)

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
        return False, f"File too large ({size_mb:.1f} MB). Maximum is {MAX_FILE_SIZE_MB} MB."
    return True, ""

def trim_history(history: list, max_messages: int = MAX_HISTORY_MESSAGES) -> list:
    return history[-max_messages:] if len(history) > max_messages else history

if "active_conv_id"     not in st.session_state: st.session_state.active_conv_id = None
if "chat_history"       not in st.session_state: st.session_state.chat_history = []
if "token_count"        not in st.session_state: st.session_state.token_count = 0
if "last_uploaded_file" not in st.session_state: st.session_state.last_uploaded_file = None

with st.sidebar:
    st.markdown("""
    <div class="orion-logo">
        <div class="orion-wordmark">ORION</div>
        <div class="orion-tagline">Intelligence Platform · v2</div>
        <div class="orion-divline"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<span class='sb-label'>Sessions</span>", unsafe_allow_html=True)

    if st.button("＋  New Session", use_container_width=True, type="primary"):
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
        st.caption("No sessions yet.")

    for conv in conversations:
        col1, col2 = st.columns([5, 1])
        is_active = conv["id"] == st.session_state.active_conv_id
        label = f"{'▸ ' if is_active else ''}{conv['title']}"
        if has_vectorstore(conv["id"]): label += " ·"
        with col1:
            if st.button(label, key=f"conv_{conv['id']}", use_container_width=True):
                st.session_state.active_conv_id = conv["id"]
                st.session_state.chat_history = load_messages(conv["id"])
                st.session_state.token_count = 0
                st.session_state.last_uploaded_file = None
                st.rerun()
        with col2:
            if st.button("✕", key=f"del_{conv['id']}"):
                delete_conversation(conv["id"])
                delete_vectorstore(conv["id"])
                if st.session_state.active_conv_id == conv["id"]:
                    st.session_state.active_conv_id = None
                    st.session_state.chat_history = []
                    st.session_state.last_uploaded_file = None
                st.rerun()

    st.divider()
    st.markdown("### ◎ Persona")

    preset_label = st.selectbox("Select", list(PRESETS.keys()), key="preset_select", label_visibility="collapsed")
    chosen_preset = PRESETS[preset_label]

    if chosen_preset == "__custom__":
        system_prompt = st.text_area("Prompt", value="You are a helpful assistant.", height=90, max_chars=1000, label_visibility="collapsed")
    else:
        system_prompt = chosen_preset

    if st.session_state.active_conv_id and st.button("Apply Persona", use_container_width=True):
        try:
            update_system_prompt(st.session_state.active_conv_id, system_prompt)
            st.success("Persona applied.")
        except Exception as e:
            st.error(f"Failed: {e}")

    st.divider()
    st.markdown(f"<div class='token-badge'>{st.session_state.token_count:,} tokens this session</div>", unsafe_allow_html=True)

# ── MAIN ───────────────────────────────────────────────────────────────────
if st.session_state.active_conv_id is None:
    st.markdown("""
    <div class="welcome-wrap">
        <div class="orion-hero">ORION</div>
        <div class="hero-rule"></div>
        <div class="hero-sub">Intelligence Platform &nbsp;·&nbsp; LLaMA &nbsp;·&nbsp; LangChain &nbsp;·&nbsp; Groq</div>
        <div class="feat-grid">
            <div class="feat-cell">
                <div class="feat-num">01</div>
                <div class="feat-title">Persistent Memory</div>
                <div class="feat-desc">Every session stored with full context history</div>
            </div>
            <div class="feat-cell">
                <div class="feat-num">02</div>
                <div class="feat-title">Document Q&amp;A</div>
                <div class="feat-desc">Upload PDFs and docs for RAG-powered answers</div>
            </div>
            <div class="feat-cell">
                <div class="feat-num">03</div>
                <div class="feat-title">Persona Engine</div>
                <div class="feat-desc">Switch expert roles instantly per use case</div>
            </div>
        </div>
        <div class="welcome-cta">Open a new session to begin</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

conv_info = get_conversation_info(st.session_state.active_conv_id)
if conv_info is None:
    st.session_state.active_conv_id = None
    st.session_state.chat_history = []
    st.warning("Session not found.")
    st.rerun()

active_system_prompt = conv_info["system_prompt"]

col_title, col_upload = st.columns([3, 2])
with col_title:
    conv_title = conv_info["title"]
    new_title = st.text_input("", value=conv_title, label_visibility="collapsed", placeholder="Session title...", max_chars=60)
    if new_title and new_title != conv_title:
        rename_conversation(st.session_state.active_conv_id, new_title)

with col_upload:
    uploaded_file = st.file_uploader("Attach document", type=["pdf", "docx", "txt"], label_visibility="collapsed")
    if uploaded_file and uploaded_file.name != st.session_state.last_uploaded_file:
        is_valid, error_msg = validate_file(uploaded_file)
        if not is_valid:
            st.error(error_msg)
        else:
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                try:
                    meta = ingest_document(uploaded_file, st.session_state.active_conv_id)
                    st.session_state.last_uploaded_file = uploaded_file.name
                    st.success(f"{meta['filename']} — {meta['num_chunks']} chunks indexed")
                except Exception as e:
                    st.error(f"Failed to process document: {e}")

if has_vectorstore(st.session_state.active_conv_id):
    st.markdown(
        "<div style='padding:6px 0 10px'>"
        "<span class='rag-badge'><span class='rag-dot'></span>&nbsp; Document intelligence active</span>"
        "</div>",
        unsafe_allow_html=True
    )

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
                st.error("Could not load document. Please re-upload.")
                st.stop()
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
                for chunk in rag_chain.stream({"question": input_text, "chat_history": trimmed_history}):
                    if isinstance(chunk, str):
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            response = full_response
        except Exception as e:
            st.error(f"RAG error: {e}")
            st.stop()
    else:
        try:
            chain = build_plain_chain(active_system_prompt)
            with st.chat_message("assistant"):
                response = st.write_stream(chain.stream({"question": input_text, "chat_history": trimmed_history}))
        except Exception as e:
            st.error(f"Chat error: {e}")
            st.stop()

    try:
        save_message(st.session_state.active_conv_id, "human", input_text)
        save_message(st.session_state.active_conv_id, "ai", response)
    except Exception as e:
        st.warning(f"Saved in session but failed to persist: {e}")

    st.session_state.chat_history.append(HumanMessage(content=input_text))
    st.session_state.chat_history.append(AIMessage(content=response))
    st.session_state.token_count += len(input_text + response) // 4
    st.rerun()
