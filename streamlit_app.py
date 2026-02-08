
"""Streamlit UI for the Autonomous Enterprise AI Orchestrator.

Professional, enterprise-ready interface for the LangGraph control plane.
UI-only: collects user input and invokes orchestrator - no business logic here.
"""

from __future__ import annotations

import os
from typing import List
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# LAZY LOADING: AIOrchestrator imported only when needed (first user message)
# This prevents Render startup timeout by avoiding heavy model loading at import time
_orchestrator_instance = None

def get_orchestrator():
    """Lazily initialize the orchestrator on first use."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        from orchestrator import AIOrchestrator
        _orchestrator_instance = AIOrchestrator(enable_checkpointing=True)
    return _orchestrator_instance

from storage.sqlite_store import (
    create_workspace, list_workspaces, rename_workspace, list_workspace_documents,
    create_chat_session, list_chat_sessions, update_chat_session_name, load_chat_messages, delete_chat_session
)

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Orchestrator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    /* ─── Global Reset & Typography ─── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1f2937; /* Gray-800 */
    }

    /* ─── Color System ─── */
    :root {
        --primary: #4F46E5; /* Indigo-600 */
        --primary-light: #EEF2FF; /* Indigo-50 */
        --bg-neutral: #F9FAFB; /* Gray-50 */
        --border-color: #E5E7EB; /* Gray-200 */
        --text-secondary: #6B7280; /* Gray-500 */
    }

    /* ─── Sidebar Polish ─── */
    section[data-testid="stSidebar"] {
        background-color: #FAFAFA;
        border-right: 1px solid var(--border-color);
        padding-top: 1.5rem;
    }
    
    section[data-testid="stSidebar"] hr {
        margin: 2rem 0 1.5rem 0;
        border-color: #E5E7EB;
        opacity: 0.6;
    }
    
    .sidebar-section-header {
        font-size: 0.7rem;
        font-weight: 600;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0;
        margin-bottom: 0.75rem;
    }
    
    /* User Profile Styling */
    .user-profile-container {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem;
        background-color: white;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        margin-bottom: 0.5rem;
    }
    
    .user-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 0.875rem;
        flex-shrink: 0;
    }
    
    .user-name {
        flex: 1;
        font-size: 0.875rem;
        color: #111827;
        font-weight: 500;
    }
    
    /* Sign Out Link */
    .sign-out-link {
        font-size: 0.75rem;
        color: #6B7280;
        text-decoration: none;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        transition: all 0.15s ease;
    }
    
    .sign-out-link:hover {
        background-color: #F3F4F6;
        color: #374151;
    }
    
    /* Chat List Styling */
    .chat-item {
        display: flex;
        align-items: center;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        margin-bottom: 0.25rem;
        cursor: pointer;
        transition: all 0.15s ease;
        background-color: transparent;
    }
    
    .chat-item:hover {
        background-color: #F3F4F6;
    }
    
    .chat-item.active {
        background-color: #EEF2FF;
        color: #4F46E5;
        font-weight: 500;
    }
    
    /* Button overrides for sidebar */
    section[data-testid="stSidebar"] button[kind="primary"] {
        background-color: #F3F4F6 !important;
        color: #374151 !important;
        border: 1px solid #E5E7EB !important;
        font-weight: 500 !important;
        transition: all 0.15s ease !important;
    }
    
    section[data-testid="stSidebar"] button[kind="primary"]:hover {
        background-color: #E5E7EB !important;
        border-color: #D1D5DB !important;
    }
    
    section[data-testid="stSidebar"] button[kind="secondary"] {
        background-color: transparent !important;
        color: #6B7280 !important;
        border: 1px solid #E5E7EB !important;
        font-weight: 400 !important;
    }
    
    section[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: #F9FAFB !important;
        color: #374151 !important;
    }
    
    /* Delete button - red for destructive action */
    section[data-testid="stSidebar"] button[key="delete_chat"] {
        background-color: #FEE2E2 !important;
        color: #DC2626 !important;
        border-color: #FECACA !important;
    }
    
    section[data-testid="stSidebar"] button[key="delete_chat"]:hover {
        background-color: #DC2626 !important;
        color: white !important;
        border-color: #DC2626 !important;
    }


    /* ─── Header & Branding ─── */
    .app-header {
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 1rem;
        margin-bottom: 2rem;
    }
    
    .app-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827; /* Gray-900 */
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .app-subtitle {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
    }

    /* ─── Status Badge ─── */
    .workspace-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        background-color: var(--primary-light);
        color: var(--primary);
        border-radius: 99px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.025em;
        margin-bottom: 0.5rem;
    }

    /* ─── Chat Interface & Bubbles ─── */
    div[data-testid="stVerticalBlock"] > div.element-container {
        width: 100%;
    }
    
    .stChatMessage {
        background-color: transparent !important;
        padding: 0.5rem 0;
        border: none;
    }
    
    /* Animation for messages */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(4px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stChatMessage { animation: fadeIn 0.3s ease-out both; }
    
    /* User Bubble (Right Aligned) */
    .user-bubble {
        background-color: var(--primary);
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 12px 12px 0 12px;
        margin-left: auto;
        width: fit-content;
        max-width: 80%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        line-height: 1.5;
    }
    
    /* Assistant Bubble (Left Aligned) */
    .assistant-bubble {
        background-color: white;
        border: 1px solid var(--border-color);
        color: #1f2937;
        padding: 1rem 1.25rem;
        border-radius: 12px 12px 12px 0;
        width: fit-content;
        max-width: 90%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        line-height: 1.6;
    }
    
    /* Remove default streamlit bubble styling if possible, or override */
    div[data-testid="stChatMessageContent"] {
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
        padding: 0 !important;
    }
    
    /* Avatar sizes */
    .stChatMessage .stImage {
        width: 32px;
        height: 32px;
    }

    /* ─── Input Area ─── */
    .stChatInput {
        max-width: 900px;
        margin: 0 auto;
        padding-bottom: 3rem;
    }
    
    .stChatInput [data-testid="stChatInputContainer"] {
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.01);
    }
    
    .stChatInput [data-testid="stChatInputContainer"]:focus-within {
        border-color: var(--primary);
        box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.2);
    }
    
    /* ─── Footer ─── */
    footer { visibility: hidden; }
    .footer-text {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background-color: #fff;
        color: #9CA3AF; /* Gray-400 */
        text-align: center; 
        padding: 0.75rem; 
        font-size: 0.75rem;
        border-top: 1px solid #f3f4f6;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="footer-text">Built by Harsh Sharma • Autonomous Enterprise AI Orchestrator</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════

if "login_complete" not in st.session_state: st.session_state.login_complete = False
if "user_id" not in st.session_state: st.session_state.user_id = None
if "username" not in st.session_state: st.session_state.username = None
if "workspace_id" not in st.session_state: st.session_state.workspace_id = None
if "workspace_name" not in st.session_state: st.session_state.workspace_name = None
if "session_id" not in st.session_state: st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "processing" not in st.session_state: st.session_state.processing = False
if "uploaded_doc_list" not in st.session_state: st.session_state.uploaded_doc_list = []
if "show_debug" not in st.session_state: st.session_state.show_debug = False
if "renaming_session" not in st.session_state: st.session_state.renaming_session = False
if "voice_buffer" not in st.session_state: st.session_state.voice_buffer = ""
if "final_query" not in st.session_state: st.session_state.final_query = None

# ══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def _load_session_history(session_id: str):
    """Load history for a specific session."""
    st.session_state.session_id = session_id
    msgs = load_chat_messages(st.session_state.user_id, st.session_state.workspace_id, session_id)
    st.session_state.chat_history = msgs
    st.session_state.renaming_session = False

def _reset_session_for_workspace():
    """Reset state when switching workspaces (creates new chat)."""
    new_sid = str(uuid.uuid4())
    create_chat_session(st.session_state.user_id, st.session_state.workspace_id, new_sid, "New Chat")
    st.session_state.session_id = new_sid
    st.session_state.chat_history = []
    st.session_state.uploaded_doc_list = []
    
    if st.session_state.user_id and st.session_state.workspace_id:
        docs, _ = list_workspace_documents(st.session_state.user_id, st.session_state.workspace_id)
        st.session_state.uploaded_doc_list = [os.path.basename(d["file_path"]) for d in docs]

def login(username: str):
    if not username: return
    st.session_state.user_id = f"user_{username.lower().replace(' ', '_')}"
    st.session_state.username = username
    st.session_state.login_complete = True
    
    workspaces = list_workspaces(st.session_state.user_id)
    if workspaces:
        st.session_state.workspace_id = workspaces[0]["workspace_id"]
        st.session_state.workspace_name = workspaces[0]["name"]
    else:
        ws_id = create_workspace(st.session_state.user_id, "My First Workspace")
        st.session_state.workspace_id = ws_id
        st.session_state.workspace_name = "My First Workspace"
    
    _reset_session_for_workspace()
    st.rerun()

def logout():
    st.session_state.clear()
    st.rerun()

# ══════════════════════════════════════════════════════════════════
# AUTHENTICATION SCREEN
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
# APP LAYOUT (LOGGED IN)
# ══════════════════════════════════════════════════════════════════

# ── Sidebar: Chats & Navigation ──
with st.sidebar:
    # User Profile Section
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
    
    # Workspace Section
    st.markdown('<p class="sidebar-section-header">Workspace</p>', unsafe_allow_html=True)
    
    workspaces = list_workspaces(st.session_state.user_id)
    ws_map = {ws["name"]: ws["workspace_id"] for ws in workspaces}
    
    current_idx = 0
    if st.session_state.workspace_id in ws_map.values():
        current_idx = list(ws_map.values()).index(st.session_state.workspace_id)
    
    selected_ws_name = st.selectbox(
        "Current Workspace", 
        list(ws_map.keys()), 
        index=current_idx, 
        label_visibility="collapsed",
        key="workspace_selector"
    )
    
    if ws_map[selected_ws_name] != st.session_state.workspace_id:
        st.session_state.workspace_id = ws_map[selected_ws_name]
        st.session_state.workspace_name = selected_ws_name
        _reset_session_for_workspace()
        st.rerun()

    with st.expander("Create Workspace", expanded=False):
        new_ws = st.text_input("Workspace name", placeholder="My Workspace", label_visibility="collapsed")
        if st.button("Create", use_container_width=True, key="create_ws") and new_ws:
            create_workspace(st.session_state.user_id, new_ws)
            st.rerun()
            
    st.divider()
    
    # Chat History Section
    st.markdown('<p class="sidebar-section-header">Chats</p>', unsafe_allow_html=True)
    
    if st.button("+ New chat", use_container_width=True, type="secondary", key="new_chat_btn"):
        new_sid = str(uuid.uuid4())
        create_chat_session(st.session_state.user_id, st.session_state.workspace_id, new_sid, "New Chat")
        st.session_state.session_id = new_sid
        st.session_state.chat_history = []
        st.session_state.renaming_session = False
        st.rerun()
    
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        
    sessions = list_chat_sessions(st.session_state.user_id, st.session_state.workspace_id)
    
    for sess in sessions:
        sid = sess["session_id"]
        is_active = (sid == st.session_state.session_id)
        
        col_btn, col_menu = st.columns([6, 1])
        
        with col_btn:
            lbl = sess["name"] if len(sess["name"]) < 22 else sess["name"][:20] + "..."
            if st.button(
                lbl, 
                key=f"chat_{sid}", 
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                _load_session_history(sid)
                st.rerun()
        
        with col_menu:
            with st.popover("⋮", use_container_width=True):
                st.caption("Manage Chat")
                
                # Rename
                new_name_val = st.text_input("Name", value=sess["name"], key=f"input_{sid}")
                if st.button("Rename", key=f"ren_{sid}", use_container_width=True):
                    update_chat_session_name(sid, new_name_val)
                    st.rerun()
                
                st.divider()
                
                if "final_query" not in st.session_state:
                    st.session_state.final_query = None

                # Delete
                if st.button("Delete", key=f"del_{sid}", type="primary", use_container_width=True):
                    delete_chat_session(sid)
                    
                    # If we deleted the active chat, switch to another or create new
                    if is_active:
                        remaining = [s for s in sessions if s["session_id"] != sid]
                        if remaining:
                            # Switch to the first available (usually most recent)
                            _load_session_history(remaining[0]["session_id"])
                        else:
                            # No chats left, ensure a fresh one exists
                            new_sid = str(uuid.uuid4())
                            create_chat_session(st.session_state.user_id, st.session_state.workspace_id, new_sid, "New Chat")
                            st.session_state.session_id = new_sid
                            st.session_state.chat_history = []
                            st.session_state.renaming_session = False
                    
                    st.rerun()

    st.divider()
    
    # Debug Section
    st.session_state.show_debug = st.toggle("Debug", value=st.session_state.show_debug, key="debug_toggle")


# ── Main Chat Area ──

# Custom Header
st.markdown(f"""
    <div class="app-header">
        <div class="workspace-badge">WORKSPACE: {st.session_state.workspace_name.upper()}</div>
        <div class="app-title">AI Orchestrator</div>
        <div class="app-subtitle">Orchestrate LLMs, tools, and knowledge with deterministic control.</div>
    </div>
""", unsafe_allow_html=True)

if not os.getenv("GOOGLE_API_KEY"): st.error("🚨 GOOGLE_API_KEY missing.")

# Documents EXPANDER
with st.expander("📂 Workspace Documents (RAG Context)", expanded=False):
    st.info("Documents uploaded here are persistent and available to all chats in this workspace.")
    uploaded_files = st.file_uploader(
        "Upload",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    if st.session_state.uploaded_doc_list:
        st.markdown("**Active Knowledge Base:**")
        for doc in st.session_state.uploaded_doc_list:
             st.markdown(f"- 📄 `{doc}`")
    else:
        st.caption("No documents indexed yet.")

# Chat Feed
chat_container = st.container()

with chat_container:
    if not st.session_state.chat_history:
        st.markdown(
            f"""
            <div style='text-align: center; color: #9CA3AF; margin-top: 4rem;'>
                <h3>Ready to assist in {st.session_state.workspace_name}</h3>
                <p>Ask a question, request research, or analyze documents.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
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

# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
# INPUT AREA WITH VOICE - COLUMN LAYOUT
# ══════════════════════════════════════════════════════════════════
import streamlit.components.v1 as components

# ─────────────────────────────────────────────
# INPUT AREA (SINGLE SOURCE OF TRUTH)
# ─ Uses st.chat_input + mic injection
# ─────────────────────────────────────────────

input_container = st.container()

with input_container:
    col_input, col_mic = st.columns([15, 1])

    with col_input:
        prompt = st.chat_input(
            "Command the orchestrator...",
            disabled=st.session_state.processing
        )

    with col_mic:
        components.html(
            """
            <div id="voice-mic-wrapper">
                <button id="voice-mic-btn" type="button" title="Click to speak">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"
                        viewBox="0 0 24 24" fill="none" stroke="currentColor"
                        stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                        <line x1="12" y1="19" x2="12" y2="23"/>
                        <line x1="8" y1="23" x2="16" y2="23"/>
                    </svg>
                </button>
            </div>

            <style>
                #voice-mic-wrapper {
                    display: flex;
                    align-items: flex-end;
                    justify-content: center;
                    height: 100%;
                }

                #voice-mic-btn {
                    width: 42px;
                    height: 42px;
                    border-radius: 8px;
                    background: transparent;
                    border: none;
                    color: #6B7280;
                    cursor: pointer;
                    transition: all 0.2s ease;
                    margin-bottom: 18px; /* aligns with chat input */
                }

                #voice-mic-btn:hover {
                    background-color: #F3F4F6;
                    color: #4F46E5;
                }

                #voice-mic-btn.listening {
                    background-color: #FEE2E2;
                    color: #DC2626;
                    animation: pulse 1.5s infinite;
                }

                @keyframes pulse {
                    0% { box-shadow: 0 0 0 0 rgba(220,38,38,.4); }
                    70% { box-shadow: 0 0 0 6px rgba(220,38,38,0); }
                    100% { box-shadow: 0 0 0 0 rgba(220,38,38,0); }
                }
            </style>

            <script>
            (function () {
                const btn = document.getElementById("voice-mic-btn");

                if (!("webkitSpeechRecognition" in window) && !("SpeechRecognition" in window)) {
                    btn.style.display = "none";
                    return;
                }

                const SpeechRecognition =
                    window.SpeechRecognition || window.webkitSpeechRecognition;
                const recognition = new SpeechRecognition();
                recognition.interimResults = true;
                recognition.lang = "en-US";

                let listening = false;

                recognition.onstart = () => {
                    listening = true;
                    btn.classList.add("listening");
                };

                recognition.onend = () => {
                    listening = false;
                    btn.classList.remove("listening");
                };

                recognition.onresult = (event) => {
                    let transcript = "";
                    for (let i = event.resultIndex; i < event.results.length; i++) {
                        if (event.results[i].isFinal) {
                            transcript += event.results[i][0].transcript;
                        }
                    }

                    if (transcript) {
                        const textarea = window.parent.document.querySelector(
                            'textarea[data-testid="stChatInputTextArea"]'
                        );
                        if (textarea) {
                            const setter = Object.getOwnPropertyDescriptor(
                                HTMLTextAreaElement.prototype, "value"
                            ).set;

                            const current = textarea.value;
                            const prefix =
                                current && !current.endsWith(" ")
                                    ? current + " "
                                    : current;

                            setter.call(textarea, prefix + transcript);
                            textarea.dispatchEvent(new Event("input", { bubbles: true }));
                            textarea.focus();
                        }
                    }
                };

                btn.onclick = () => {
                    listening ? recognition.stop() : recognition.start();
                };
            })();
            </script>
            """,
            height=70,
        )

# Optional alignment polish
st.markdown(
    """
    <style>
        .stChatInput {
            width: 100% !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# import streamlit.components.v1 as components

# # Layout: [ Chat Input (90%) | Mic (10%) ]
# # We use columns to visually reserve space for the mic button next to the input
# input_container = st.container()
# with input_container:
#     # Reserved space columns
#     # Note: Streamlit chat_input is fixed-bottom, so it will overlay the page.
#     # By placing it in a column, we constrain its width.
#     col_input, col_mic = st.columns([15, 1])

#     with col_input:
#         prompt = st.chat_input("Command the orchestrator...", disabled=st.session_state.processing)

#     with col_mic:
#         # Voice Component - Resides in the right column
#         # Because chat_input is fixed to bottom, we mimic that placement
#         voice_component_html = """
#         <div id="voice-mic-wrapper">
#             <button id="voice-mic-btn" type="button" title="Click to speak">
#                 <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
#                     <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
#                     <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
#                     <line x1="12" y1="19" x2="12" y2="23"></line>
#                     <line x1="8" y1="23" x2="16" y2="23"></line>
#                 </svg>
#             </button>
#             <div id="mic-tooltip">Click to speak</div>
#         </div>

#         <style>
#             /* Wrapper matches the column flow but ensures bottom placement */
#             #voice-mic-wrapper {
#                 display: flex;
#                 flex-direction: column;
#                 align-items: center;
#                 justify-content: center;
#                 height: 100%;
#                 width: 100%;
#             }

#             /* The button itself - matching Streamlit input style */
#             #voice-mic-btn {
#                 width: 42px;
#                 height: 42px;
#                 border-radius: 8px;
#                 background-color: transparent;
#                 border: 1px solid transparent;
#                 color: #6B7280;
#                 cursor: pointer;
#                 transition: all 0.2s ease;
#                 display: flex;
#                 align-items: center;
#                 justify-content: center;
#                 margin-bottom: 24px; /* Align with chat input baseline */
#             }

#             #voice-mic-btn:hover {
#                 background-color: #F3F4F6;
#                 color: #4F46E5;
#             }

#             #voice-mic-btn.listening {
#                 background-color: #FEE2E2;
#                 color: #DC2626;
#                 animation: pulse 1.5s infinite;
#             }

#             #voice-mic-btn.success {
#                 color: #10B981;
#             }

#             #voice-mic-btn.error {
#                 color: #F59E0B;
#             }

#             @keyframes pulse {
#                 0% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.4); }
#                 70% { box-shadow: 0 0 0 6px rgba(220, 38, 38, 0); }
#                 100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }
#             }
            
#             #mic-tooltip {
#                 position: absolute;
#                 bottom: 60px;
#                 background: #1F2937;
#                 color: white;
#                 padding: 4px 8px;
#                 font-family: sans-serif;
#                 font-size: 11px;
#                 border-radius: 4px;
#                 opacity: 0;
#                 pointer-events: none;
#                 transition: opacity 0.2s;
#                 white-space: nowrap;
#                 right: 0;
#             }
            
#             #voice-mic-btn:hover + #mic-tooltip,
#             #voice-mic-btn.listening + #mic-tooltip {
#                 opacity: 1;
#             }
#         </style>

#         <script>
#         (function() {
#             const btn = document.getElementById('voice-mic-btn');
#             const tooltip = document.getElementById('mic-tooltip');
            
#             if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
#                 btn.style.display = 'none';
#                 return;
#             }

#             const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
#             const recognition = new SpeechRecognition();
#             recognition.continuous = false;
#             recognition.interimResults = true;
#             recognition.lang = 'en-US';

#             let isListening = false;

#             recognition.onstart = function() {
#                 isListening = true;
#                 btn.className = 'listening';
#                 tooltip.textContent = 'Listening...';
#             };

#             recognition.onend = function() {
#                 isListening = false;
#                 btn.className = '';
#                 tooltip.textContent = 'Click to speak';
#             };

#             recognition.onerror = function(event) {
#                 isListening = false;
#                 btn.className = 'error';
#                 tooltip.textContent = 'Error: ' + event.error;
#             };

#             recognition.onresult = function(event) {
#                 let finalTranscript = '';
#                 for (let i = event.resultIndex; i < event.results.length; ++i) {
#                     if (event.results[i].isFinal) {
#                         finalTranscript += event.results[i][0].transcript;
#                     }
#                 }

#                 if (finalTranscript) {
#                     const parentDoc = window.parent.document;
#                     const textarea = parentDoc.querySelector('textarea[data-testid="stChatInputTextArea"]');
#                     if (textarea) {
#                         const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                        
#                         const current = textarea.value;
#                         const prefix = current && !current.endsWith(' ') ? current + ' ' : current;
#                         nativeInputValueSetter.call(textarea, prefix + finalTranscript);
                        
#                         textarea.dispatchEvent(new Event('input', { bubbles: true }));
#                         textarea.focus();
                        
#                         btn.className = 'success';
#                         setTimeout(() => btn.className = '', 1000);
#                     }
#                 }
#             };

#             btn.onclick = function(e) {
#                 e.preventDefault();
#                 if (isListening) recognition.stop();
#                 else recognition.start();
#             };
#         })();
#         </script>
#         """
        
#         # Inject the component
#         components.html(voice_component_html, height=70)

# # CSS to ensure alignment
# st.markdown("""
# <style>
#     /* target the column containing the mic to align it to bottom */
#     div[data-testid="column"]:nth-of-type(2) {
#         display: flex;
#         align-items: flex-end;
#         padding-bottom: 2rem; /* Setup alignment with chat input */
#     }
    
#     /* Force chat input to respect column width */
#     .stChatInput {
#         width: 100% !important;
#     }
# </style>
# """, unsafe_allow_html=True)


# Processing Logic
if prompt:
    # Optimistic Update
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(f'<div class="user-bubble">{prompt}</div>', unsafe_allow_html=True)
    
    st.session_state.processing = True
    
    inputs = {
        "query": prompt,
        "tenant_id": "demo-tenant",
        "user_id": st.session_state.user_id,
        "session_id": st.session_state.session_id,
        "workspace_id": st.session_state.workspace_id,
        "approved": True
    }
    
    # Handle Uploads
    doc_paths = []
    if uploaded_files:
        base_dir = os.path.join("workspaces", st.session_state.user_id, st.session_state.workspace_id, "files")
        os.makedirs(base_dir, exist_ok=True)
        for f in uploaded_files:
             safe_name = os.path.basename(f.name)
             path = os.path.join(base_dir, safe_name)
             with open(path, "wb") as w: w.write(f.getbuffer())
             doc_paths.append(path)
             if safe_name not in st.session_state.uploaded_doc_list:
                 st.session_state.uploaded_doc_list.append(safe_name)
        inputs["uploaded_docs"] = doc_paths
    
    try:
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Processing..."):
                    orchestrator = get_orchestrator()
                    result = orchestrator.invoke(**inputs)
        
        final_answer = result.get("final_answer", "")
        
        if result.get("intent") == "rag":
            final_answer = f"📚 **Document Context**\n\n{final_answer}"
        elif result.get("intent") == "research":
             final_answer = f"🌐 **Web Research**\n\n{final_answer}"

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": final_answer,
            "details": {
                "intent": result.get("intent"),
                "confidence": result.get("intent_confidence"),
                "model_used": result.get("model_used"),
                "status": result.get("execution_status"),
                "errors": result.get("errors", [])
            }
        })
        
        st.session_state.processing = False
        st.rerun()

    except Exception as e:
        st.session_state.processing = False
        st.error(f"Error: {str(e)}")

# import streamlit.components.v1 as components

# components.html(
#     """
#     <style>
#         .chat-input-wrapper {
#             position: fixed;
#             bottom: 24px;
#             left: 50%;
#             transform: translateX(-50%);
#             width: min(900px, 92%);
#             background: #F9FAFB;
#             border: 1px solid #E5E7EB;
#             border-radius: 999px;
#             display: flex;
#             align-items: center;
#             padding: 8px 10px;
#             gap: 8px;
#             z-index: 1000;
#         }

#         .chat-input-wrapper input {
#             flex: 1;
#             border: none;
#             background: transparent;
#             font-size: 15px;
#             padding: 10px 12px;
#             outline: none;
#             color: #111827;
#         }

#         .icon-btn {
#             width: 40px;
#             height: 40px;
#             border-radius: 999px;
#             border: none;
#             background: transparent;
#             cursor: pointer;
#             display: flex;
#             align-items: center;
#             justify-content: center;
#             color: #6B7280;
#             transition: all 0.15s ease;
#         }

#         .icon-btn:hover {
#             background: #EEF2FF;
#             color: #4F46E5;
#         }

#         .icon-btn.listening {
#             background: #FEE2E2;
#             color: #DC2626;
#             animation: pulse 1.4s infinite;
#         }

#         @keyframes pulse {
#             0% { box-shadow: 0 0 0 0 rgba(220,38,38,.4); }
#             70% { box-shadow: 0 0 0 6px rgba(220,38,38,0); }
#             100% { box-shadow: 0 0 0 0 rgba(220,38,38,0); }
#         }
#     </style>

#     <div class="chat-input-wrapper">
#         <button id="mic-btn" class="icon-btn" title="Click to speak">🎤</button>
#         <input id="chat-input" placeholder="Command the orchestrator..." />
#         <button id="send-btn" class="icon-btn" title="Send">➤</button>
#     </div>

#     <script>
#         const input = document.getElementById("chat-input");
#         const sendBtn = document.getElementById("send-btn");
#         const micBtn = document.getElementById("mic-btn");

#         function submit() {
#             const text = input.value.trim();
#             if (!text) return;

#             const textarea = window.parent.document.querySelector(
#                 'textarea[data-testid="stChatInputTextArea"]'
#             );

#             if (textarea) {
#                 const setter = Object.getOwnPropertyDescriptor(
#                     HTMLTextAreaElement.prototype, "value"
#                 ).set;

#                 setter.call(textarea, text);
#                 textarea.dispatchEvent(new Event("input", { bubbles: true }));
#                 textarea.focus();

#                 textarea.dispatchEvent(
#                     new KeyboardEvent("keydown", {
#                         bubbles: true,
#                         cancelable: true,
#                         key: "Enter",
#                         code: "Enter"
#                     })
#                 );
#             }

#             input.value = "";
#         }

#         sendBtn.onclick = submit;
#         input.addEventListener("keydown", (e) => {
#             if (e.key === "Enter") submit();
#         });

#         if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
#             const SpeechRecognition =
#                 window.SpeechRecognition || window.webkitSpeechRecognition;
#             const recognition = new SpeechRecognition();
#             recognition.interimResults = true;
#             recognition.lang = "en-US";

#             let listening = false;

#             recognition.onstart = () => {
#                 listening = true;
#                 micBtn.classList.add("listening");
#             };

#             recognition.onend = () => {
#                 listening = false;
#                 micBtn.classList.remove("listening");
#             };

#             recognition.onresult = (event) => {
#                 let transcript = "";
#                 for (let i = event.resultIndex; i < event.results.length; i++) {
#                     if (event.results[i].isFinal) {
#                         transcript += event.results[i][0].transcript;
#                     }
#                 }
#                 if (transcript) {
#                     input.value += (input.value ? " " : "") + transcript;
#                     input.focus();
#                 }
#             };

#             micBtn.onclick = () => {
#                 listening ? recognition.stop() : recognition.start();
#             };
#         } else {
#             micBtn.style.display = "none";
#         }
#     </script>
#     """,
#     height=90,
# )

# # Processing Logic
# if query:
#     prompt = query
#     # Optimistic Update
#     st.session_state.chat_history.append({"role": "user", "content": prompt})
#     with chat_container:
#         with st.chat_message("user"):
#             st.markdown(f'<div class="user-bubble">{prompt}</div>', unsafe_allow_html=True)
    
#     st.session_state.processing = True
    
#     inputs = {
#         "query": prompt,
#         "tenant_id": "demo-tenant",
#         "user_id": st.session_state.user_id,
#         "session_id": st.session_state.session_id,
#         "workspace_id": st.session_state.workspace_id,
#         "approved": True
#     }
    