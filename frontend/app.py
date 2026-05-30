"""Streamlit UI for the Nexus AI Orchestrator.

Fully decoupled from local AI/LangGraph – communicates exclusively
via REST API calls to the FastAPI backend at http://localhost:8000.
"""

from __future__ import annotations

import os
import re
import uuid
import html
from pathlib import Path

import requests
import streamlit as st


def sanitize_workspace_filename(filename: str) -> str:
    """Filesystem-safe basename for workspace uploads (spaces, brackets, etc.)."""
    name = Path(filename).name
    stem, ext = os.path.splitext(name)
    ext = ext.lower()
    stem = re.sub(r"[\s()\[\]{}]+", "_", stem)
    stem = re.sub(r"[^\w.\-]+", "_", stem, flags=re.ASCII)
    stem = re.sub(r"_+", "_", stem).strip("._")
    if not stem:
        stem = "document"
    return f"{stem}{ext}"

# ── Backend URL ──
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Orchestrator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# CUSTOM CSS  (preserved from original)
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #1f2937; }
    :root {
        --primary: #4F46E5;
        --primary-light: #EEF2FF;
        --bg-neutral: #F9FAFB;
        --border-color: #E5E7EB;
        --text-secondary: #6B7280;
    }
    section[data-testid="stSidebar"] {
        background-color: #FAFAFA;
        border-right: 1px solid var(--border-color);
        padding-top: 1.5rem;
    }
    section[data-testid="stSidebar"] hr { margin: 2rem 0 1.5rem 0; border-color: #E5E7EB; opacity: 0.6; }
    .sidebar-section-header {
        font-size: 0.7rem; font-weight: 600; color: #9CA3AF;
        text-transform: uppercase; letter-spacing: 0.05em;
        margin-top: 0; margin-bottom: 0.75rem;
    }
    .user-profile-container {
        display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem;
        background-color: white; border-radius: 8px; border: 1px solid #E5E7EB; margin-bottom: 0.5rem;
    }
    .user-avatar {
        width: 36px; height: 36px; border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        display: flex; align-items: center; justify-content: center;
        color: white; font-weight: 600; font-size: 0.875rem; flex-shrink: 0;
    }
    .user-name { flex: 1; font-size: 0.875rem; color: #111827; font-weight: 500; }
    section[data-testid="stSidebar"] button[kind="primary"] {
        background-color: #F3F4F6 !important; color: #374151 !important;
        border: 1px solid #E5E7EB !important; font-weight: 500 !important;
        transition: all 0.15s ease !important;
    }
    section[data-testid="stSidebar"] button[kind="primary"]:hover {
        background-color: #E5E7EB !important; border-color: #D1D5DB !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"] {
        background-color: transparent !important; color: #6B7280 !important;
        border: 1px solid #E5E7EB !important; font-weight: 400 !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: #F9FAFB !important; color: #374151 !important;
    }
    .app-header { border-bottom: 1px solid var(--border-color); padding-bottom: 1rem; margin-bottom: 2rem; }
    .app-title {
        font-size: 1.5rem; font-weight: 700; color: #111827;
        display: flex; align-items: center; gap: 0.5rem;
    }
    .app-subtitle { font-size: 0.9rem; color: var(--text-secondary); margin-top: 0.25rem; }
    .workspace-badge {
        display: inline-flex; align-items: center; padding: 4px 12px;
        background-color: var(--primary-light); color: var(--primary);
        border-radius: 99px; font-size: 0.75rem; font-weight: 600;
        letter-spacing: 0.025em; margin-bottom: 0.5rem;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
    .stChatMessage { animation: fadeIn 0.3s ease-out both; background-color: transparent !important; padding: 0.5rem 0; border: none; }
    .user-bubble {
        background-color: var(--primary); color: white; padding: 1rem 1.25rem;
        border-radius: 12px 12px 0 12px; margin-left: auto; width: fit-content;
        max-width: 80%; box-shadow: 0 1px 2px rgba(0,0,0,0.05); line-height: 1.5;
    }
    .assistant-bubble {
        background-color: white; border: 1px solid var(--border-color); color: #1f2937;
        padding: 1rem 1.25rem; border-radius: 12px 12px 12px 0; width: fit-content;
        max-width: 90%; box-shadow: 0 1px 2px rgba(0,0,0,0.02); line-height: 1.6;
    }
    div[data-testid="stChatMessageContent"] {
        background-color: transparent !important; box-shadow: none !important;
        border: none !important; padding: 0 !important;
    }
    .stChatInput { max-width: 900px; margin: 0 auto; padding-bottom: 3rem; }
    .stChatInput [data-testid="stChatInputContainer"] {
        border-radius: 12px; border: 1px solid var(--border-color);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.01);
    }
    .stChatInput [data-testid="stChatInputContainer"]:focus-within {
        border-color: var(--primary); box-shadow: 0 0 0 2px rgba(79,70,229,0.2);
    }
    footer { visibility: hidden; }
    .footer-text {
        position: fixed; left: 0; bottom: 0; width: 100%; background-color: #fff;
        color: #9CA3AF; text-align: center; padding: 0.75rem; font-size: 0.75rem;
        border-top: 1px solid #f3f4f6; z-index: 1000;
    }
    /* Connection status badge */
    .conn-ok { color: #10B981; font-weight: 600; font-size: 0.8rem; }
    .conn-fail { color: #EF4444; font-weight: 600; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="footer-text">Built by Harsh Sharma • Autonomous Enterprise AI Orchestrator</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════

if "login_complete" not in st.session_state:
    st.session_state.login_complete = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None
if "workspace_id" not in st.session_state:
    st.session_state.workspace_id = "default"
if "workspace_name" not in st.session_state:
    st.session_state.workspace_name = "Default Workspace"
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "uploaded_doc_list" not in st.session_state:
    st.session_state.uploaded_doc_list = []
if "show_debug" not in st.session_state:
    st.session_state.show_debug = False
if "all_sessions" not in st.session_state:
    st.session_state.all_sessions = {}  # {session_id: {"name": ..., "history": [...]}}

# ══════════════════════════════════════════════════════════════════
# HELPER: API CALL
# ══════════════════════════════════════════════════════════════════

def call_chat_api(message: str, uploaded_docs: list[str] | None = None) -> dict:
    """Send a message to the FastAPI backend and return the response."""
    payload = {
        "user_id": st.session_state.user_id,
        "message": message,
        "workspace_id": st.session_state.workspace_id,
        "session_id": st.session_state.session_id,
        "tenant_id": "default",
        "uploaded_docs": uploaded_docs or [],
    }
    resp = requests.post(f"{API_BASE}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def check_backend_health() -> bool:
    """Return True if the FastAPI backend is reachable."""
    try:
        r = requests.get(f"{API_BASE}/api/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ── Session management (client-side) ──

def _save_current_session():
    """Persist current chat into session registry."""
    sid = st.session_state.session_id
    name = "New Chat"
    if st.session_state.chat_history:
        first_user = next((m for m in st.session_state.chat_history if m["role"] == "user"), None)
        if first_user:
            name = first_user["content"][:30] + "..."
    st.session_state.all_sessions[sid] = {
        "name": name,
        "history": list(st.session_state.chat_history),
    }


def _switch_session(sid: str):
    """Switch to an existing session."""
    _save_current_session()
    entry = st.session_state.all_sessions.get(sid, {"name": "New Chat", "history": []})
    st.session_state.session_id = sid
    st.session_state.chat_history = list(entry["history"])


def _new_chat():
    """Create a brand new chat session."""
    _save_current_session()
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.chat_history = []


def login(username: str):
    if not username:
        return
    st.session_state.user_id = f"user_{username.lower().replace(' ', '_')}"
    st.session_state.username = username
    st.session_state.login_complete = True
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    st.rerun()


def logout():
    st.session_state.clear()
    st.rerun()


# ══════════════════════════════════════════════════════════════════
# AUTH SCREEN
# ══════════════════════════════════════════════════════════════════

if not st.session_state.login_complete:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
        st.markdown("""
            <h1 style='text-align: center; color: #111827;'>AI Orchestrator</h1>
            <p style='text-align: center; color: #6B7280; margin-bottom: 2rem;'>
                Autonomous Enterprise Control Plane
            </p>
        """, unsafe_allow_html=True)
        with st.container(border=True):
            username_input = st.text_input("Username", placeholder="e.g. Harsh Sharma")
            if st.button("Initialize Session", use_container_width=True, type="primary"):
                login(username_input)
    st.stop()

# ══════════════════════════════════════════════════════════════════
# SIDEBAR (logged in)
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    # User profile
    st.markdown(f"""
        <div class="user-profile-container">
            <div class="user-avatar">{st.session_state.username[0].upper()}</div>
            <div class="user-name">{st.session_state.username}</div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Sign out", use_container_width=True, key="signout_btn"):
            logout()

    st.divider()

    # Backend status
    st.markdown('<p class="sidebar-section-header">Backend</p>', unsafe_allow_html=True)
    if check_backend_health():
        st.markdown('<span class="conn-ok">● Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="conn-fail">● Disconnected</span>', unsafe_allow_html=True)
        st.caption("Start backend: `python -m uvicorn backend.app.main:app --reload`")

    st.divider()

    # Workspace
    st.markdown('<p class="sidebar-section-header">Workspace</p>', unsafe_allow_html=True)
    st.markdown(f"**{st.session_state.workspace_name}**")

    st.divider()

    # Chat sessions
    st.markdown('<p class="sidebar-section-header">Chats</p>', unsafe_allow_html=True)

    if st.button("+ New chat", use_container_width=True, type="secondary", key="new_chat_btn"):
        _new_chat()
        st.rerun()

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # List saved sessions
    all_sess = st.session_state.all_sessions
    # Also ensure the current session appears
    _save_current_session()

    for sid, entry in sorted(all_sess.items(), key=lambda x: x[0], reverse=True):
        is_active = sid == st.session_state.session_id
        lbl = entry["name"] if len(entry["name"]) < 22 else entry["name"][:20] + "..."

        col_btn, col_del = st.columns([6, 1])
        with col_btn:
            if st.button(lbl, key=f"chat_{sid}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                _switch_session(sid)
                st.rerun()
        with col_del:
            if st.button("🗑", key=f"del_{sid}"):
                if sid in all_sess:
                    del all_sess[sid]
                if is_active:
                    _new_chat()
                st.rerun()

    st.divider()

    # Debug toggle
    st.session_state.show_debug = st.toggle("Debug", value=st.session_state.show_debug, key="debug_toggle")

# ══════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════════

st.markdown(f"""
    <div class="app-header">
        <div class="workspace-badge">WORKSPACE: {st.session_state.workspace_name.upper()}</div>
        <div class="app-title">AI Orchestrator</div>
        <div class="app-subtitle">Orchestrate LLMs, tools, and knowledge with deterministic control.</div>
    </div>
""", unsafe_allow_html=True)

# Document uploader
with st.expander("📂 Workspace Documents (RAG Context)", expanded=False):
    st.info("Documents uploaded here are persistent and available to all chats in this workspace.")
    uploaded_files = st.file_uploader(
        "Upload", type=["txt", "md", "pdf"], accept_multiple_files=True, label_visibility="collapsed"
    )
    if st.session_state.uploaded_doc_list:
        st.markdown("**Active Knowledge Base:**")
        for doc in st.session_state.uploaded_doc_list:
            st.markdown(f"- 📄 `{doc}`")
    else:
        st.caption("No documents indexed yet.")

# Chat feed
chat_container = st.container()

with chat_container:
    if not st.session_state.chat_history:
        st.markdown(f"""
            <div style='text-align: center; color: #9CA3AF; margin-top: 4rem;'>
                <h3>Ready to assist in {st.session_state.workspace_name}</h3>
                <p>Ask a question, request research, or analyze documents.</p>
            </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        with st.chat_message(role):
            if role == "user":
                st.markdown(f'<div class="user-bubble">{content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-bubble">{content}</div>', unsafe_allow_html=True)
                if "details" in msg and st.session_state.show_debug:
                    st.json(msg["details"])

# Voice mic (iframe)
import streamlit.components.v1 as components

input_container = st.container()
with input_container:
    col_input, col_mic = st.columns([15, 1])
    with col_input:
        prompt = st.chat_input("Command the orchestrator...", disabled=st.session_state.processing)
    with col_mic:
        mic_html = """
        <!DOCTYPE html><html><head><style>
            body{margin:0;padding:0;overflow:hidden;background:transparent}
            #voice-mic-wrapper{display:flex;align-items:flex-end;justify-content:center;height:100vh}
            #voice-mic-btn{width:42px;height:42px;border-radius:8px;background:transparent;border:none;color:#6B7280;cursor:pointer;transition:all .2s ease;margin-bottom:18px}
            #voice-mic-btn:hover{background-color:#F3F4F6;color:#4F46E5}
            #voice-mic-btn.listening{background-color:#FEE2E2;color:#DC2626;animation:pulse 1.5s infinite}
            @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(220,38,38,.4)}70%{box-shadow:0 0 0 6px rgba(220,38,38,0)}100%{box-shadow:0 0 0 0 rgba(220,38,38,0)}}
        </style></head><body>
        <div id="voice-mic-wrapper"><button id="voice-mic-btn" type="button" title="Click to speak">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/>
            </svg>
        </button></div>
        <script>(function(){const btn=document.getElementById("voice-mic-btn");if(!("webkitSpeechRecognition" in window)&&!("SpeechRecognition" in window)){btn.style.display="none";return}const SR=window.SpeechRecognition||window.webkitSpeechRecognition;const r=new SR();r.interimResults=true;r.lang="en-US";let l=false;r.onstart=()=>{l=true;btn.classList.add("listening")};r.onend=()=>{l=false;btn.classList.remove("listening")};r.onresult=(e)=>{let t="";for(let i=e.resultIndex;i<e.results.length;i++){if(e.results[i].isFinal)t+=e.results[i][0].transcript}if(t){const ta=window.parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');if(ta){const s=Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,"value").set;const c=ta.value;const p=c&&!c.endsWith(" ")?c+" ":c;s.call(ta,p+t);ta.dispatchEvent(new Event("input",{bubbles:true}));ta.focus()}}};btn.onclick=()=>{l?r.stop():r.start()}})();</script>
        </body></html>
        """
        st.markdown(
            f'<iframe srcdoc="{html.escape(mic_html)}" style="border:none;width:100%;height:70px;overflow:hidden;" allow="microphone" sandbox="allow-scripts allow-same-origin"></iframe>',
            unsafe_allow_html=True,
        )

st.markdown("<style>.stChatInput{width:100% !important;}</style>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PROCESSING LOGIC – REST API call to FastAPI backend
# ══════════════════════════════════════════════════════════════════

if prompt:
    # Optimistic update
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(f'<div class="user-bubble">{prompt}</div>', unsafe_allow_html=True)

    st.session_state.processing = True

    # Handle file uploads (save to disk, pass paths to backend)
    doc_paths = []
    if uploaded_files:
        base_dir = os.path.join("workspaces", st.session_state.user_id, st.session_state.workspace_id, "files")
        os.makedirs(base_dir, exist_ok=True)
        for f in uploaded_files:
            safe_name = sanitize_workspace_filename(f.name)
            path = os.path.join(base_dir, safe_name)
            with open(path, "wb") as w:
                w.write(f.getbuffer())
            doc_paths.append(path)
            if safe_name not in st.session_state.uploaded_doc_list:
                st.session_state.uploaded_doc_list.append(safe_name)

    try:
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Processing..."):
                    result = call_chat_api(prompt, uploaded_docs=doc_paths)

        final_answer = result.get("answer", "")

        # Prefix with intent badge
        intent = result.get("intent")
        if intent == "rag":
            final_answer = f"📚 **Document Context**\n\n{final_answer}"
        elif intent == "research":
            final_answer = f"🌐 **Web Research**\n\n{final_answer}"

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": final_answer,
            "details": {
                "intent": intent,
                "confidence": result.get("confidence"),
                "model_used": result.get("model_used"),
                "status": result.get("execution_status"),
                "errors": result.get("errors", []),
            },
        })

        st.session_state.processing = False
        _save_current_session()
        st.rerun()

    except requests.exceptions.ConnectionError:
        st.session_state.processing = False
        st.error("🚨 Cannot reach the backend. Is the FastAPI server running on port 8000?")
    except Exception as e:
        st.session_state.processing = False
        st.error(f"Error: {str(e)}")
