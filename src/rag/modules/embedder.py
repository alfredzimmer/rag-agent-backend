from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import numpy as np
from FlagEmbedding import FlagModel


"""
  Uses Qwen/Embedding-0.6B
  returns a numpy array
"""
def compute_dense_vec(text):
  dense_model = FlagModel("Qwen/Qwen3-Embedding-0.6B", use_fp16=True)
  emb = dense_model.encode([text], batch_size=1, normalize_embeddins=True)
  return emb[0]

"""
Uses SPLADE naver/splade-v3 model
  Implemented from template on QDRANT 
  https://qdrant.tech/articles/sparse-vectors/#computing-the-sparse-vector
  returns indices, values that will be used by QDRANT sparse search
"""
def compute_sparse_vec(text):
  # needs to decouple the model_id here in the future
  model_id = "naver/splade-v3"
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForMaskedLM.from_pretrained(model_id)

  """
    Computes a vector from logits and attention mask using ReLU, log, and max operations.
  """
  tokens = tokenizer(text, return_tensors="pt")
  output = model(**tokens)
  logits, attention_mask = output.logits, tokens.attention_mask
  relu_log = torch.log(1 + torch.relu(logits))
  weighted_log = relu_log * attention_mask.unsqueeze(-1)
  max_val, _ = torch.max(weighted_log, dim=1)
  vec = max_val.squeeze()

  # process vec into a Qdrant readable pair
  indices = vec.nonzero().numpy().flatten()
  values = vec.detach().numpy()[indices]

  return indices, values 

