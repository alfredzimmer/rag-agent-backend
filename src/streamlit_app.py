"""Minimal Streamlit chat UI for the RAG agent API."""
import json
import os

import requests
import streamlit as st

API_URL = os.getenv("RAG_AGENT_API_URL", "http://127.0.0.1:9230")
CONVERSATION_API = f"{API_URL}/api/agent/conversation"

st.set_page_config(page_title="RAG Agent", page_icon="🤖", layout="wide")


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def api_get(path: str, **params) -> dict | list:
    resp = requests.get(f"{CONVERSATION_API}/{path}", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def load_messages(conversation_id: str) -> list[dict]:
    history = api_get("history", conversation_id=conversation_id)["history"]
    return [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m["role"] in ("user", "assistant")
    ]


def stream_chat(conversation_id: str, query: str, reasoning: bool):
    with requests.post(
        f"{CONVERSATION_API}/chat",
        json={"query": query, "conversation_id": conversation_id, "reasoning": reasoning},
        stream=True,
        timeout=600,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                yield json.loads(line)


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message.get("context"):
            with st.expander("📚 Retrieved context"):
                st.text(message["context"][:6000])
        if message.get("reasoning"):
            with st.expander("💭 Reasoning"):
                st.markdown(message["reasoning"])
        st.markdown(message["content"])
        if message.get("tokens"):
            st.caption(message["tokens"])


if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
    st.session_state.messages = []
if "reasoning_enabled" not in st.session_state:
    st.session_state.reasoning_enabled = env_bool("RAG_LLM_REASONING", False)

with st.sidebar:
    st.title("🤖 RAG Agent")

    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        deps = health.get("dependencies", {})
        if health.get("status") == "ok":
            st.success("API online", icon="✅")
        else:
            offline = [k for k, v in deps.items() if v != "online"]
            st.warning(f"API degraded: {', '.join(offline)} offline", icon="⚠️")
    except requests.RequestException:
        st.error(f"API unreachable at {API_URL}", icon="❌")
        st.stop()

    st.toggle("Reasoning", key="reasoning_enabled")

    if st.button("➕ New chat", use_container_width=True):
        st.session_state.conversation_id = api_get("create")["conversation_id"]
        st.session_state.messages = []
        st.rerun()

    sessions = api_get("list")
    if sessions:
        st.caption("Conversations")
    for session in sessions:
        current = session["conversation_id"] == st.session_state.conversation_id
        if st.button(
            ("🟢 " if current else "") + (session["title"] or "New Chat")[:40],
            key=session["conversation_id"],
            use_container_width=True,
        ):
            st.session_state.conversation_id = session["conversation_id"]
            st.session_state.messages = load_messages(session["conversation_id"])
            st.rerun()

    if st.session_state.conversation_id:
        st.divider()
        if st.button("🗑️ Delete current chat", use_container_width=True):
            requests.delete(
                f"{CONVERSATION_API}/clear",
                json={"conversation_id": st.session_state.conversation_id},
                timeout=15,
            )
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.rerun()

for message in st.session_state.messages:
    render_message(message)

if prompt := st.chat_input("Ask about your documents…"):
    if not st.session_state.conversation_id:
        st.session_state.conversation_id = api_get("create")["conversation_id"]

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Retrieving context…")
        context_box = status.empty()
        reasoning_box = None
        if st.session_state.reasoning_enabled:
            reasoning_box = st.expander("💭 Reasoning").empty()
        answer_box = st.empty()
        context = reasoning = answer = ""
        metadata = {}

        try:
            for event in stream_chat(
                st.session_state.conversation_id,
                prompt,
                st.session_state.reasoning_enabled,
            ):
                kind = event["type"]
                if kind == "retrieve":
                    context = event["content"]
                    context_box.text(context[:6000])
                    status.update(label="📚 Context retrieved", state="complete")
                elif kind == "response.reasoning.delta":
                    reasoning += event["content"]
                    if reasoning_box is not None:
                        reasoning_box.markdown(reasoning)
                elif kind == "response.output_text.delta":
                    answer += event["content"]
                    answer_box.markdown(answer + "▌")
                elif event["status"] in ("complete", "cancel"):
                    metadata = event["metadata"]
        except requests.RequestException as error:
            st.error(f"Request failed: {error}")

        answer_box.markdown(answer or "*(no answer)*")

    tokens = ""
    if metadata:
        tokens = (
            f"tokens: {metadata.get('input_tokens_used', 0)} in / "
            f"{metadata.get('output_tokens_used', 0)} out"
        )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "context": context,
            "reasoning": reasoning,
            "tokens": tokens,
        }
    )
    st.rerun()
