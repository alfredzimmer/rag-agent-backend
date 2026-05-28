import argparse
import asyncio
import sys
import uuid

sys.path.insert(0, "src")

from langchain_core.messages import HumanMessage

from rag.agent import RAGAgent
from rag.config import RAGConfig


async def run_chat_wrapper():
    agent = await RAGAgent.create(
        RAGConfig(llm_model="qwen3.6", collection_name="HeaderInContentTrial")
    )
    conversation_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    try:
        for query in ("What is Ziyutec?", "Tell me more about it"):
            print(f"QUERY {query}", flush=True)
            async for response in agent.chat(query, conversation_id, user_id):
                print(
                    response.status.value,
                    response.type,
                    repr(response.content[:200]),
                    flush=True,
                )
            print(f"END_QUERY {query}", flush=True)
    finally:
        await agent.close()


async def run_raw_events():
    agent = await RAGAgent.create(
        RAGConfig(llm_model="qwen3.6", collection_name="HeaderInContentTrial")
    )
    conversation_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": conversation_id, "user_id": user_id}}
    state = {"messages": [HumanMessage(content="What is Ziyutec?")]}

    try:
        async with asyncio.timeout(30):
            async for event in agent.agent.astream_events(state, config=config, version="v2"):
                node = event.get("metadata", {}).get("langgraph_node")
                name = event.get("name")
                print(f"EVENT {event['event']} node={node} name={name}", flush=True)
    finally:
        await agent.close()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-events", action="store_true")
    args = parser.parse_args()

    if args.raw_events:
        await run_raw_events()
    else:
        await run_chat_wrapper()


if __name__ == "__main__":
    asyncio.run(main())
