import json
import logging
import random
import requests
from pymilvus import connections, Collection

logging.basicConfig(level=logging.INFO)

MILVUS_URI = "http://127.0.0.1:19530"
OLLAMA_HOST = "http://127.0.0.1:11434"
API_URL = "http://127.0.0.1:9229"
MODEL = "qwen3.6"
COLLECTION_NAME = "rag_documents_v2"

def generate_question(text: str) -> str:
    prompt = f"You are an expert evaluator. Generate a short, specific question that can be answered SOLELY by the following text. Do not provide the answer, just the question.\n\nText: {text}\n\nQuestion:"
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()

def main():
    logging.info("Connecting to Milvus...")
    connections.connect(host="127.0.0.1", port="19530", db_name="rag2")
    from pymilvus import utility
    collections = utility.list_collections()
    logging.info(f"Available collections: {collections}")
    
    if COLLECTION_NAME not in collections:
        if collections:
            COLLECTION_NAME_ACTUAL = collections[0]
            logging.info(f"Using {COLLECTION_NAME_ACTUAL} instead")
        else:
            logging.error("No collections found!")
            return
    else:
        COLLECTION_NAME_ACTUAL = COLLECTION_NAME
        
    coll = Collection(COLLECTION_NAME_ACTUAL)
    coll.load()

    # Get some documents
    logging.info("Fetching documents...")
    # Using pk >= 0 or something similar to get some rows. auto_id creates int64 pk.
    res = coll.query(expr="pk >= 0", output_fields=["text", "source"], limit=100)
    
    # Filter out very short texts
    res = [r for r in res if len(r["text"]) > 200]
    
    # Sample 10
    if len(res) > 10:
        res = random.sample(res, 10)
    
    logging.info(f"Selected {len(res)} chunks for testing.")
    
    successful = 0
    total = len(res)
    
    results = []
    
    for i, r in enumerate(res):
        text = r["text"]
        source = r.get("source", "unknown")
        
        logging.info(f"[{i+1}/{total}] Generating question for source: {source}")
        try:
            question = generate_question(text)
        except Exception as e:
            logging.error(f"Failed to generate question: {e}")
            total -= 1
            continue
            
        logging.info(f"Question: {question}")
        
        # Create conversation
        c_resp = requests.get(f"{API_URL}/api/agent/conversation/create")
        c_resp.raise_for_status()
        conversation_id = c_resp.json()["conversation_id"]
        
        # Chat
        logging.info("Querying API...")
        chat_resp = requests.post(
            f"{API_URL}/api/agent/conversation/chat",
            json={"query": question, "conversation_id": conversation_id},
            stream=True
        )
        
        retrieved_context = ""
        for line in chat_resp.iter_lines():
            if line:
                event = json.loads(line)
                if event["type"] == "retrieve":
                    retrieved_context = event["content"]
        
        # Check recall
        # The retrieved context combines multiple docs. We check if our original text is in it.
        # However, slight formatting might differ, so we can check a large substring or just exact match.
        # Let's check a 100-char substring of the text to be safe.
        check_str = text[50:150] if len(text) > 150 else text[:50]
        
        is_hit = check_str in retrieved_context
        if is_hit:
            successful += 1
            logging.info("Result: HIT")
        else:
            logging.info("Result: MISS")
            
        results.append({
            "source": source,
            "question": question,
            "expected_text_snippet": check_str,
            "hit": is_hit
        })
        
    recall_rate = (successful / total) * 100 if total > 0 else 0
    logging.info(f"--- Evaluation Complete ---")
    logging.info(f"Recall Rate: {recall_rate:.2f}% ({successful}/{total})")
    
    with open("recall_results.json", "w") as f:
        json.dump({"recall_rate": recall_rate, "successful": successful, "total": total, "details": results}, f, indent=2)

if __name__ == "__main__":
    main()
