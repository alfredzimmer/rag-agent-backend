from deepeval.models.base_model import DeepEvalBaseLLM
import ollama

class OLLAMA_DEEPEVAL_WRAPPER(DeepEvalBaseLLM):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def load_model(self):
        return ollama.Client()

    def generate(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']

    async def a_generate(self, prompt: str) -> str:
        client = ollama.AsyncClient()
        response = await client.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']

    def get_model_name(self):
        return self.model_name