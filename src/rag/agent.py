from langchain_core.tools import tool
from vectordb import vector_store
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

model = ChatOpenAI(model="gpt-5-mini")

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve top-2 chunks from the embedded Wiki article matching the query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


tools = [retrieve_context]
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You have access to a tool that retrieves context from a book. "
            "Use the tool to help answer user queries.",
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

query = input('Ask you question: ')

for event in agent_executor.stream(
    {"input": query}
):
    event["messages"][-1].pretty_print()
