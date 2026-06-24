import streamlit as st
import httpx
import json
import asyncio
import requests
import time
from uuid import uuid4

st.set_page_config(page_title="RAG Agent", page_icon="🤖", layout="wide")

st.title("🤖 RAG Agent")

# --- Persistent Error Display ---
if "error_message" not in st.session_state:
    st.session_state["error_message"] = None

if st.session_state["error_message"]:
    st.error(st.session_state["error_message"])
    if st.button("Dismiss Error"):
        st.session_state["error_message"] = None
        st.rerun()


def _query_param(name: str) -> str | None:
    value = st.query_params.get(name)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def split_context(context_str: str) -> tuple[str, str]:
    if not context_str:
        return "", ""
    milvus_part = ""
    exa_part = ""
    tag_milvus = "[Source: Milvus (Local Knowledge Base)]"
    tag_exa = "[Source: Exa Web Search (Internet)]"
    if tag_milvus in context_str or tag_exa in context_str:
        parts = context_str.split(tag_exa)
        milvus_raw = parts[0]
        if tag_milvus in milvus_raw:
            milvus_part = milvus_raw.replace(tag_milvus, "").strip()
        else:
            milvus_part = milvus_raw.strip()
        if len(parts) > 1:
            exa_part = parts[1].strip()
    else:
        milvus_part = context_str
    return milvus_part, exa_part


def _history_to_messages(history: list[dict]) -> list[dict[str, str]]:
    messages = []
    last_tool_content = None
    for entry in history:
        role = entry.get("role") or entry.get("type", "message")
        content = entry.get("content", "")
        if role in ("tool", "function"):
            last_tool_content = content
        elif role == "human" or role == "user":
            messages.append({"role": "user", "content": content})
        elif role in ("ai", "assistant") and content:
            msg = {"role": "assistant", "content": content}
            if last_tool_content:
                msg["retrieved_context"] = last_tool_content
                last_tool_content = None
            messages.append(msg)
    return messages


def load_history(base_url: str, conversation_id: str, auth_token: str) -> list[dict[str, str]]:
    resp = requests.get(
        f"{base_url}/api/agent/conversation/history",
        params={"conversation_id": conversation_id},
        headers={"Authorization": f"Bearer {auth_token}"},
        timeout=10,
    )
    resp.raise_for_status()
    return _history_to_messages(resp.json().get("history", []))


def create_conversation(base_url: str, auth_token: str) -> str:
    response = requests.get(
        f"{base_url}/api/agent/conversation/create",
        headers={"Authorization": f"Bearer {auth_token}"},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()["conversation_id"]


# --- State management ---
if "base_url" not in st.session_state or not st.session_state["base_url"]:
    st.session_state["base_url"] = "http://localhost:9229"

if "auth_token" not in st.session_state:
    st.session_state["auth_token"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

if "conversation_id" not in st.session_state:
    st.session_state["conversation_id"] = _query_param("conversation_id")
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "loaded_history_for" not in st.session_state:
    st.session_state["loaded_history_for"] = None
if "is_streaming" not in st.session_state:
    st.session_state["is_streaming"] = False
if "pending_query" not in st.session_state:
    st.session_state["pending_query"] = None
if "interrupted" not in st.session_state:
    st.session_state["interrupted"] = False
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = "pdf_uploader_1"
if "enable_exa" not in st.session_state:
    st.session_state["enable_exa"] = False

for sensitive_parameter in ("auth_token", "username", "user_id"):
    if sensitive_parameter in st.query_params:
        del st.query_params[sensitive_parameter]

if st.session_state["conversation_id"]:
    st.query_params["conversation_id"] = st.session_state["conversation_id"]

# --- Authentication Wall ---
if st.session_state["auth_token"] is None:
    st.subheader("Please sign in or register to access the agent.")

    tab_login, tab_register = st.tabs(["🔑 Sign In", "📝 Register"])

    with tab_login:
        login_user = st.text_input("Username", key="login_username")
        login_pass = st.text_input("Password", type="password", key="login_password")
        if st.button("Sign In", type="primary"):
            if not login_user or not login_pass:
                st.error("Please enter both username and password.")
            else:
                try:
                    resp = requests.post(
                        f"{st.session_state['base_url']}/api/auth/login",
                        json={"username": login_user, "password": login_pass},
                        timeout=10
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state["auth_token"] = data["token"]
                        st.session_state["username"] = data["username"]
                        st.session_state["user_id"] = data["user_id"]
                        st.session_state["conversation_id"] = None
                        st.session_state["messages"] = []
                        st.session_state["loaded_history_for"] = None

                        st.toast(f"Welcome back, {data['username']}!")
                        st.rerun()
                    else:
                        st.error(f"Login failed: {resp.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error connecting to backend at {st.session_state['base_url']}: {e}")

    with tab_register:
        reg_user = st.text_input("Username", key="reg_username")
        reg_pass = st.text_input("Password", type="password", key="reg_password")
        reg_pass_confirm = st.text_input("Confirm Password", type="password", key="reg_password_confirm")
        if st.button("Register"):
            if not reg_user or not reg_pass:
                st.error("Please enter both username and password.")
            elif reg_pass != reg_pass_confirm:
                st.error("Passwords do not match.")
            else:
                try:
                    resp = requests.post(
                        f"{st.session_state['base_url']}/api/auth/register",
                        json={"username": reg_user, "password": reg_pass},
                        timeout=10
                    )
                    if resp.status_code == 200:
                        st.success("Registration successful! You can now log in using the 'Sign In' tab.")
                    else:
                        st.error(f"Registration failed: {resp.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error connecting to backend at {st.session_state['base_url']}: {e}")
    st.stop()

# --- Fetch Recent Conversations ---
recent_chats = []
conversations_loaded = False
if st.session_state["auth_token"]:
    try:
        resp = requests.get(
            f"{st.session_state['base_url']}/api/agent/conversation/list",
            headers={"Authorization": f"Bearer {st.session_state['auth_token']}"},
            timeout=5,
        )
        if resp.status_code == 200:
            recent_chats = resp.json()
            conversations_loaded = True
    except Exception as e:
        print(f"Error fetching conversation list: {e}")

if conversations_loaded:
    conversation_ids = {chat["conversation_id"] for chat in recent_chats}
    if st.session_state["conversation_id"] not in conversation_ids:
        try:
            conversation_id = create_conversation(
                st.session_state["base_url"],
                st.session_state["auth_token"],
            )
            st.session_state["conversation_id"] = conversation_id
            st.session_state["messages"] = []
            st.session_state["loaded_history_for"] = conversation_id
            st.query_params["conversation_id"] = conversation_id
            st.rerun()
        except Exception as error:
            st.error(f"Could not create a conversation: {error}")
            st.stop()

# --- Sidebar config ---
with st.sidebar:
    st.header("🤖 EC Master Agent")
    st.write(f"Logged in as: **{st.session_state['username']}**")

    # ➕ New Chat Button
    new_chat = st.button("➕ New Chat", type="primary", use_container_width=True, disabled=st.session_state["is_streaming"])

    # Toggle for Exa AI Search
    enable_exa = st.toggle("Enable Exa AI Search", value=st.session_state.get("enable_exa", False))
    st.session_state["enable_exa"] = enable_exa

    st.markdown("---")
    st.markdown("### 💬 Recent Chats")

    # Scrollable / list of conversations
    if not recent_chats:
        st.caption("No recent conversations.")
    else:
        for chat in recent_chats:
            chat_id = chat["conversation_id"]
            title = chat["title"] or "New Chat"

            # Format/truncate title
            display_title = title if len(title) <= 25 else title[:22] + "..."

            # Highlight current active session
            is_active = (chat_id == st.session_state["conversation_id"])

            if st.button(
                f"💬 {display_title}",
                key=f"chat_select_{chat_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state["conversation_id"] = chat_id
                st.query_params["conversation_id"] = chat_id
                try:
                    st.session_state["messages"] = load_history(st.session_state["base_url"], chat_id, st.session_state["auth_token"])
                    st.session_state["loaded_history_for"] = chat_id
                except Exception as e:
                    st.session_state["messages"] = []
                    st.session_state["loaded_history_for"] = chat_id
                    st.sidebar.error(f"Could not load history: {e}")
                st.rerun()

    st.markdown("---")

    # Advanced settings collapsed in an expander
    with st.expander("⚙️ Advanced Settings"):
        base_url_input = st.text_input("API Base URL", value=st.session_state["base_url"])
        st.session_state["base_url"] = base_url_input

        clear_session = st.button("Clear Current Session", type="secondary", use_container_width=True, disabled=st.session_state["is_streaming"])
        logout_button = st.button("Log Out", type="primary", use_container_width=True)

st.query_params["conversation_id"] = st.session_state["conversation_id"]

conv_id = st.session_state["conversation_id"]

# --- Log Out ---
if logout_button:
    st.session_state["auth_token"] = None
    st.session_state["username"] = None
    st.session_state["user_id"] = None
    st.session_state["conversation_id"] = None
    st.session_state["messages"] = []
    st.session_state["loaded_history_for"] = None
    st.query_params.clear()
    st.rerun()

# --- Create new session ---
if new_chat:
    try:
        conversation_id = create_conversation(
            st.session_state["base_url"],
            st.session_state["auth_token"],
        )
        st.session_state["conversation_id"] = conversation_id
        st.session_state["messages"] = []
        st.session_state["pending_query"] = None
        st.session_state["loaded_history_for"] = conversation_id
        st.query_params["conversation_id"] = conversation_id
        st.toast("New conversation started!")
        st.rerun()
    except Exception as error:
        st.error(f"Could not create a conversation: {error}")

# --- Clear session ---
if clear_session:
    try:
        async def do_clear():
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.delete(
                    f"{st.session_state['base_url']}/api/agent/conversation/clear",
                    json={"conversation_id": conv_id},
                    headers={"Authorization": f"Bearer {st.session_state['auth_token']}"}
                )
                resp.raise_for_status()
                return resp.json()

        result = asyncio.new_event_loop().run_until_complete(do_clear())
        st.session_state["messages"] = []
        st.session_state["pending_query"] = None
        st.session_state["loaded_history_for"] = conv_id
        st.toast(result.get("message", "Session cleared"))
        st.rerun()
    except Exception as e:
        st.error(f"Failed to clear session: {e}")

# --- Load persisted server history once per conversation ---
if (
    not st.session_state["is_streaming"]
    and st.session_state["pending_query"] is None
    and st.session_state["loaded_history_for"] != conv_id
):
    try:
        st.session_state["messages"] = load_history(st.session_state["base_url"], conv_id, st.session_state["auth_token"])
        st.session_state["loaded_history_for"] = conv_id
    except Exception as e:
        st.warning(f"Could not load persisted history: {e}")
        st.session_state["loaded_history_for"] = conv_id

# --- Display chat history (capped to last 100 to prevent frontend lag in long threads) ---
display_messages = st.session_state["messages"][-100:] if len(st.session_state["messages"]) > 100 else st.session_state["messages"]
if len(st.session_state["messages"]) > 100:
    st.info(f"💬 Showing the most recent 100 messages out of {len(st.session_state['messages'])} total. The older conversation context remains fully preserved in the database checkpointer.")

for msg in display_messages:
    with st.chat_message(msg["role"]):
        if msg.get("retrieved_context"):
            milvus_ctx, exa_ctx = split_context(msg["retrieved_context"])
            if milvus_ctx and exa_ctx:
                with st.expander("🔍 Retrieved Context (Milvus & Exa)", expanded=False):
                    tab1, tab2 = st.tabs(["📚 Milvus (Local DB)", "🌐 Exa AI (Web Search)"])
                    with tab1:
                        st.markdown(milvus_ctx)
                    with tab2:
                        st.markdown(exa_ctx)
            elif milvus_ctx:
                with st.expander("🔍 Retrieved Context", expanded=False):
                    st.markdown(milvus_ctx)
            elif exa_ctx:
                with st.expander("🌐 Exa AI Web Search Context", expanded=False):
                    st.markdown(exa_ctx)
        if msg.get("reasoning"):
            with st.expander("💭 Thinking Process", expanded=False):
                st.markdown(msg["reasoning"])
        st.markdown(msg["content"])

# --- Document Workspace Section (Unified on the same visual layer using st.bottom) ---
with st.bottom:
    # CSS injection for unified Gemini/ChatGPT style look
    st.markdown("""
<style>
/* Make the bottom container block look like a single unified card */
div[data-testid="stBottomBlockContainer"] > div[data-testid="stVerticalBlock"],
div[data-testid="stBottomBlockContainer"] > div {
    position: relative !important;
    background-color: var(--background-color);
    border: 1px solid var(--secondary-background-color);
    border-radius: 16px;
    padding: 16px 20px 14px 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    display: flex !important;
    flex-direction: column !important;
    gap: 12px !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}

/* Highlight border and shadow on focus */
div[data-testid="stBottomBlockContainer"] > div[data-testid="stVerticalBlock"]:focus-within,
div[data-testid="stBottomBlockContainer"] > div:focus-within {
    border-color: var(--primary-color) !important;
    box-shadow: 0 4px 24px rgba(255, 75, 75, 0.1) !important;
}

/* Swap visual order of elements inside the vertical block: Chat input at top, Columns at bottom */
div[data-testid="stBottomBlockContainer"] > div > div:has(div[data-testid="stChatInput"]),
div[data-testid="stBottomBlockContainer"] > div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stChatInput"]) {
    order: 1 !important;
}
div[data-testid="stBottomBlockContainer"] > div > div:has(div[data-testid="column"]),
div[data-testid="stBottomBlockContainer"] > div[data-testid="stVerticalBlock"] > div:has(div[data-testid="column"]) {
    order: 2 !important;
}

/* Prevent widget labels from wrapping inside the bottom block */
div[data-testid="stBottomBlockContainer"] label[data-testid="stWidgetLabel"] {
    white-space: nowrap !important;
}

/* Adjust the bottom block background container to be transparent */
div[data-testid="stBottom"] {
    background-color: transparent !important;
}

/* Align columns vertically inside the bottom container */
div[data-testid="stBottomBlockContainer"] div[data-testid="column"] {
    display: flex;
    align-items: center;
}

/* Push ingest checkbox to the right but leave space for the absolute send button */
div[data-testid="stBottomBlockContainer"] div[data-testid="column"]:last-child {
    padding-right: 48px !important;
    justify-content: flex-end;
}

/* Make stChatInput and its wrappers fully transparent and borderless */
div[data-testid="stChatInput"],
div[data-testid="stChatInput"] div {
    border: none !important;
    background-color: transparent !important;
    box-shadow: none !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Style the textarea to take full width and wrap beautifully at the top of the card */
div[data-testid="stChatInput"] textarea {
    background-color: transparent !important;
    border: none !important;
    color: var(--text-color) !important;
    padding: 2px 0 10px 0 !important;
    box-shadow: none !important;
    width: 100% !important;
}

/* Absolutely position the send button to the bottom-right corner of the card */
div[data-testid="stChatInput"] button {
    position: absolute !important;
    bottom: 12px !important;
    right: 20px !important;
    z-index: 100 !important;
    background-color: rgb(0, 122, 255) !important;
    color: white !important;
    border: none !important;
    border-radius: 50% !important;
    width: 32px !important;
    height: 32px !important;
    min-width: 32px !important;
    min-height: 32px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 2px 8px rgba(0, 122, 255, 0.3) !important;
    transition: background-color 0.2s, transform 0.1s !important;
}
div[data-testid="stChatInput"] button:hover {
    background-color: rgb(0, 102, 204) !important;
    transform: scale(1.05) !important;
}
div[data-testid="stChatInput"] button svg {
    fill: white !important;
    color: white !important;
}

/* Style the file uploader dropzone to look compact and premium */
div[data-testid="stFileUploader"] {
    padding: 0 !important;
}
section[data-testid="stFileUploaderDropzone"] {
    background-color: var(--secondary-background-color) !important;
    border: 1px dashed rgba(49, 51, 63, 0.15) !important;
    border-radius: 8px !important;
    padding: 6px 12px !important;
}
</style>
""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Attach a document",
        type=["pdf", "docx", "md", "txt"],
        key=st.session_state["uploader_key"],
        label_visibility="collapsed",
    )

    # --- User input ---
    query = st.chat_input("Type your message...", disabled=st.session_state["is_streaming"])

if query and query.strip() and not st.session_state["is_streaming"]:
    upload_success = True
    if uploaded_file is not None:
        with st.spinner("Uploading and processing document..."):
            try:
                # Prepare file payload
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
                }
                headers = {
                    "Authorization": f"Bearer {st.session_state['auth_token']}"
                }

                response = requests.post(
                    f"{st.session_state['base_url']}/api/ingestion/documents",
                    files=files,
                    params={"conversation_id": st.session_state["conversation_id"]},
                    headers=headers,
                    timeout=60
                )

                if response.status_code == 202:
                    res_json = response.json()
                    job_id = res_json["job_id"]
                    status_placeholder = st.empty()
                    deadline = time.monotonic() + 300
                    while time.monotonic() < deadline:
                        job_response = requests.get(
                            f"{st.session_state['base_url']}/api/ingestion/jobs/{job_id}",
                            headers=headers,
                            timeout=15,
                        )
                        job_response.raise_for_status()
                        job = job_response.json()
                        job_status = job["status"]
                        status_placeholder.info(
                            f"Ingestion {job_status}: {uploaded_file.name}"
                        )
                        if job_status in {"completed", "duplicate"}:
                            status_placeholder.success(
                                f"Document ready: {job['chunks_written']} chunks indexed."
                            )
                            break
                        if job_status == "failed":
                            raise RuntimeError(job.get("error") or "Document ingestion failed")
                        time.sleep(2)
                    else:
                        raise TimeoutError("Document ingestion did not finish within five minutes")
                    st.session_state["uploader_key"] = f"pdf_uploader_{uuid4()}"
                else:
                    detail = response.json().get("detail", "Unknown error")
                    st.error(f"❌ Upload failed: {detail}")
                    upload_success = False
            except Exception as ex:
                st.error(f"⚠️ Error uploading document: {ex}")
                upload_success = False

    if upload_success:
        st.session_state["pending_query"] = query.strip()
        st.session_state["messages"].append({"role": "user", "content": query.strip()})
        st.session_state["is_streaming"] = True
        st.session_state["interrupted"] = False
        st.rerun()

if st.session_state["pending_query"] and st.session_state["is_streaming"]:
    pending_query = st.session_state["pending_query"]

    response_text = ""
    reasoning_text = ""
    tool_call_content = ""
    milvus_content = ""
    exa_content = ""
    final_status = None
    final_metadata = {}

    with st.chat_message("assistant"):
        tool_expander_placeholder = st.empty()
        reasoning_expander_placeholder = st.empty()
        placeholder = st.empty()
        with st.spinner("Generating response..."):
            try:
                payload = {
                    "query": pending_query,
                    "conversation_id": conv_id,
                    "enable_exa": st.session_state.get("enable_exa", False)
                }
                with requests.post(
                    f"{st.session_state['base_url']}/api/agent/conversation/chat",
                    json=payload,
                    headers={"Authorization": f"Bearer {st.session_state['auth_token']}"},
                    timeout=120,
                    stream=True,
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if st.session_state.get("interrupted", False):
                            break
                        if not line or line == b"data: [DONE]":
                            continue
                        if line.startswith(b"data: "):
                            line = line[6:]
                        try:
                            chunk = json.loads(line)
                            status = chunk.get("status", "")
                            content = chunk.get("content", "")
                            metadata = chunk.get("metadata") or {}

                            if status == "response":
                                if chunk.get("type") == "response.output_text.delta":
                                    response_text += content
                                    placeholder.markdown(response_text + "▌")
                                elif chunk.get("type") == "response.reasoning.delta":
                                    reasoning_text += content
                                    with reasoning_expander_placeholder:
                                        with st.expander("💭 Thinking Process", expanded=False):
                                            st.markdown(reasoning_text)
                            elif status == "function":
                                tool_name = chunk.get("type", "tool")
                                if "Exa" in tool_name or tool_name == "ExaSearchResults":
                                    exa_content = content
                                else:
                                    milvus_content = content

                                # Re-create the tagged tool_call_content to save to message history
                                tool_call_content = ""
                                if milvus_content:
                                    tool_call_content += f"[Source: Milvus (Local Knowledge Base)]\n{milvus_content}\n\n"
                                if exa_content:
                                    tool_call_content += f"[Source: Exa Web Search (Internet)]\n{exa_content}"

                                with tool_expander_placeholder:
                                    if milvus_content and exa_content:
                                        with st.expander("🔍 Retrieved Context (Milvus & Exa)", expanded=False):
                                            tab1, tab2 = st.tabs(["📚 Milvus (Local DB)", "🌐 Exa AI (Web Search)"])
                                            with tab1:
                                                st.markdown(milvus_content)
                                            with tab2:
                                                st.markdown(exa_content)
                                    elif milvus_content:
                                        with st.expander("🔍 Retrieved Context", expanded=False):
                                            st.markdown(milvus_content)
                                    elif exa_content:
                                        with st.expander("🌐 Exa AI Web Search Context", expanded=False):
                                            st.markdown(exa_content)
                            elif status == "usage":
                                tokens = f"Input: {metadata.get('input_tokens_used', 0)} | Output: {metadata.get('output_tokens_used', 0)} tokens"
                                response_text += f"\n\n*{tokens}*\n"
                                placeholder.markdown(response_text)
                            elif status == "complete":
                                final_status = "complete"
                                final_metadata = metadata
                                rating = metadata.get("rating", "N/A")
                                title = metadata.get("title", "N/A")
                                completion = f"\n\n---\n*Rating: {rating} | Title: {title}*"
                                response_text += completion
                                placeholder.markdown(response_text)
                        except Exception:
                            try:
                                response_text += line.decode("utf-8", errors="replace")
                                placeholder.markdown(response_text + "▌")
                            except Exception:
                                pass
            except requests.exceptions.ConnectionError:
                st.session_state["error_message"] = f"Cannot connect to API at `{base_url}`. Make sure the backend server is running."
            except requests.exceptions.Timeout:
                st.session_state["error_message"] = "Request timed out. The backend may be slow or unresponsive."
            except requests.exceptions.HTTPError as e:
                st.session_state["error_message"] = f"HTTP error: {e.response.status_code} - {e.response.text[:200]}"
            except Exception as e:
                st.session_state["error_message"] = f"Error: {e}"
            finally:
                st.session_state["is_streaming"] = False
                st.session_state["pending_query"] = None
                if response_text or reasoning_text:
                    # If we only have reasoning text, still display it
                    msg = {"role": "assistant", "content": response_text}
                    if tool_call_content:
                        msg["retrieved_context"] = tool_call_content
                    if reasoning_text:
                        msg["reasoning"] = reasoning_text
                    st.session_state["messages"].append(msg)
                    st.session_state["loaded_history_for"] = conv_id
                if st.session_state.get("interrupted"):
                    st.session_state["interrupted"] = False
                    try:
                        st.toast("Chat interrupted")
                    except Exception:
                        pass
                st.rerun()

elif query and query.strip():
    st.toast("Please wait for the current response to finish.")

# --- Interrupt button ---
if st.session_state.get("is_streaming", False):
    st.text("Streaming...")
