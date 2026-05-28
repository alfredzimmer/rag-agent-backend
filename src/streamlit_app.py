import streamlit as st
import httpx
import json
import asyncio
import requests
from uuid import uuid4

st.set_page_config(page_title="RAG Agent Test", page_icon="🤖", layout="wide")

st.title("🤖 RAG Agent Tester")

def _query_param(name: str) -> str | None:
    value = st.query_params.get(name)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _history_to_messages(history: list[dict]) -> list[dict[str, str]]:
    messages = []
    for entry in history:
        role = entry.get("type", "message")
        content = entry.get("content", "")
        if role == "human":
            messages.append({"role": "user", "content": content})
        elif role == "ai" and content:
            messages.append({"role": "assistant", "content": content})
    return messages


def load_history(base_url: str, conversation_id: str) -> list[dict[str, str]]:
    resp = requests.get(
        f"{base_url}/api/agent/conversation/history",
        params={"conversation_id": conversation_id},
        timeout=10,
    )
    resp.raise_for_status()
    return _history_to_messages(resp.json().get("history", []))


# --- State management ---
if "base_url" not in st.session_state:
    st.session_state["base_url"] = "http://localhost:9229"
if "user_id" not in st.session_state:
    st.session_state["user_id"] = _query_param("user_id") or str(uuid4())
if "conversation_id" not in st.session_state:
    st.session_state["conversation_id"] = _query_param("conversation_id") or str(uuid4())
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

st.query_params["conversation_id"] = st.session_state["conversation_id"]
st.query_params["user_id"] = st.session_state["user_id"]

# --- Sidebar config ---
with st.sidebar:
    st.header("Settings")
    base_url = st.text_input("API Base URL", key="base_url")
    user_id = st.text_input("User ID", key="user_id", help="Identifier for the current user session")
    new_session = st.button("New Session", type="primary", disabled=st.session_state["is_streaming"])
    clear_session = st.button("Clear Session", type="secondary", disabled=st.session_state["is_streaming"])

st.query_params["conversation_id"] = st.session_state["conversation_id"]
st.query_params["user_id"] = st.session_state["user_id"]

conv_id = st.session_state["conversation_id"]
st.info(f"Conversation ID: `{conv_id}`")

# --- Create new session ---
if new_session:
    st.session_state["conversation_id"] = str(uuid4())
    st.session_state["messages"] = []
    st.session_state["pending_query"] = None
    st.session_state["loaded_history_for"] = st.session_state["conversation_id"]
    st.query_params["conversation_id"] = st.session_state["conversation_id"]
    st.toast("New session created")
    st.rerun()

# --- Clear session ---
if clear_session:
    try:
        async def do_clear():
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.delete(
                    f"{base_url}/api/agent/conversation/clear",
                    json={"conversation_id": conv_id},
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
        st.session_state["messages"] = load_history(base_url, conv_id)
        st.session_state["loaded_history_for"] = conv_id
    except Exception as e:
        st.warning(f"Could not load persisted history: {e}")
        st.session_state["loaded_history_for"] = conv_id

# --- Display chat history ---
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- User input ---
query = st.chat_input("Type your message...", disabled=st.session_state["is_streaming"])

if query and query.strip() and not st.session_state["is_streaming"]:
    st.session_state["pending_query"] = query.strip()
    st.session_state["messages"].append({"role": "user", "content": query.strip()})
    st.session_state["is_streaming"] = True
    st.session_state["interrupted"] = False
    st.rerun()

if st.session_state["pending_query"] and st.session_state["is_streaming"]:
    pending_query = st.session_state["pending_query"]

    response_text = ""
    final_status = None
    final_metadata = {}

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Generating response..."):
            try:
                payload = {
                    "query": pending_query,
                    "conversation_id": conv_id,
                    "user_id": user_id,
                }
                with requests.post(
                    f"{base_url}/api/agent/conversation/chat",
                    json=payload,
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
                            metadata = chunk.get("metadata", {})

                            if status == "response":
                                response_text += content
                                placeholder.markdown(response_text + "▌")
                            elif status == "function":
                                tool_name = chunk.get("type", "tool")
                                response_text += f"\n\n`🔧 {tool_name}`\n"
                                if content:
                                    response_text += f"`{content}`\n"
                                placeholder.markdown(response_text + "▌")
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
                        except json.JSONDecodeError:
                            response_text += line.decode("utf-8", errors="replace")
                            placeholder.markdown(response_text + "▌")
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot connect to API at `{base_url}`. Make sure the backend server is running.")
            except requests.exceptions.Timeout:
                st.error("Request timed out. The backend may be slow or unresponsive.")
            except requests.exceptions.HTTPError as e:
                st.error(f"HTTP error: {e.response.status_code} - {e.response.text[:200]}")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                st.session_state["is_streaming"] = False
                st.session_state["pending_query"] = None
                if response_text:
                    st.session_state["messages"].append({"role": "assistant", "content": response_text})
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

# --- Refresh history button ---
if st.button("🔄 Load History", help="Reload the conversation history from the server", disabled=st.session_state["is_streaming"]):
    try:
        async def do_load():
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{base_url}/api/agent/conversation/history",
                    params={"conversation_id": conv_id},
                )
                resp.raise_for_status()
                return resp.json()

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(do_load())
        loop.close()
        
        st.session_state["messages"] = _history_to_messages(result.get("history", []))
        st.session_state["loaded_history_for"] = conv_id
        st.rerun()
    except Exception as e:
        st.error(f"Failed to load history: {e}")
