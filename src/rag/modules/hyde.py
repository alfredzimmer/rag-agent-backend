from langchain_ollama import ChatOllama

class HyDEGenerator:
    def __init__(self):
        self.llm = ChatOllama(model="qwen3:8b")
    
    def generate(self, query: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful expert. Write a passage that answers the following question. Do not include any conversational filler, just the answer content."},
            {"role": "user", "content": query}
        ]
        response = self.llm.invoke(messages)
        return response.content