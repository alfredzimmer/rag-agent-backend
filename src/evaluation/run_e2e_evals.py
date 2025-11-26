from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.dataset import Golden
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRecallMetric, ContextualRelevancyMetric
from deepeval.dataset import EvaluationDataset
from ollama_deepeval_wrapper import OLLAMA_DEEPEVAL_WRAPPER
from pathlib import Path

qwen_judge = OLLAMA_DEEPEVAL_WRAPPER(model_name="qwen3:30b-instruct")

faithfulness = FaithfulnessMetric(
    threshold=0.7,
    model=qwen_judge,
    include_reason=True
)

relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model=qwen_judge,
    include_reason=True
)

contextual_recall = ContextualRecallMetric(
    threshold=0.7,
    model=qwen_judge,
    include_reason=True
)

contextual_relevance = ContextualRelevancyMetric(
    threshold=0.7,
    model=qwen_judge,
    include_reason=True
)

def run_e2e_evals():
    # Get the absolute path to the dataset file
    script_dir = Path(__file__).parent
    dataset_path = script_dir / 'rag-eval-dataset' / 'AU-2025NAElectrical1.json'
    
    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file(
        file_path=str(dataset_path),
        input_key_name="input",
        actual_output_key_name="output"
    )
    
    test_cases = []
    for golden in dataset.goldens:
        res, text_chunks = 

if __name__ == "__main__":
    run_e2e_evals()
    