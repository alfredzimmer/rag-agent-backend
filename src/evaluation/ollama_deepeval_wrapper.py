from deepeval.models.base_model import DeepEvalBaseLLM, DeepEvalBaseEmbeddingModel
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from typing import List

class LLMModelWrapper(DeepEvalBaseLLM):
    def __init__(self, model_name: str, num_ctx: int = 8192):
        """
        Args:
            model_name: The Ollama model tag (e.g., 'qwen3-30b-instruct')
            timeout: Request timeout in seconds. Increase this for complex metrics (e.g., 300s).
            num_ctx: Context window size in tokens. Defaults to 8192 for RAG tasks.
        """
        self.model_name = model_name
        self.num_ctx = num_ctx
        # We initialize the model config once here
        self.model = ChatOllama(
            model=model_name,
            num_ctx=num_ctx,     # Ensures model can read your entire RAG context
            temperature=0,       # Set to 0 for deterministic evaluations
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        res = self.model.invoke(prompt).content
        return res.content

    async def a_generate(self, prompt: str) -> str:
        res = await self.model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return self.model_name

class EmbeddingModelWrapper(DeepEvalBaseEmbeddingModel):
    def __init__(self, model_name: str = "qwen3-embedding:8b", num_ctx: int = 8192):
        """
        Initializes the Ollama embedding model via LangChain.
        
        Args:
            model_name: The tag of the model you ran 'ollama pull' with.
            num_ctx: Context window. IMPORTANT: Set this >= your chunk size.
                     Default Ollama context is often 2048, which truncates RAG docs.
        """
        self.model_name = model_name
        
        # We initialize the model ONCE here to persist the connection pool.
        # client_kwargs={"timeout": 300} prevents 'ReadTimeout' on large batches.
        self.embedding_model = OllamaEmbeddings(
            model=model_name,
            num_ctx=num_ctx,
            client_kwargs={"timeout": 300.0} 
        )

    def load_model(self):
        """Returns the cached LangChain embedding object."""
        return self.embedding_model

    def embed_text(self, text: str) -> List[float]:
        # 'embed_query' is for single strings
        return self.embedding_model.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # 'embed_documents' is for batches (optimized)
        return self.embedding_model.embed_documents(texts)

    async def a_embed_text(self, text: str) -> List[float]:
        # Async version for concurrent evaluations
        return await self.embedding_model.aembed_query(text)

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        return await self.embedding_model.aembed_documents(texts)

    def get_model_name(self):
        return self.model_name