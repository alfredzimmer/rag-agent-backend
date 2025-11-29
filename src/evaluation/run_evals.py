import os
import sys
import json
from pathlib import Path

# Add parent directory to path to allow imports from rag module
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRecallMetric, ContextualRelevancyMetric, ContextualPrecisionMetric
from ollama_deepeval_wrapper import LLMModelWrapper, EmbeddingModelWrapper
from deepeval.evaluate import AsyncConfig
from deepeval.evaluate import DisplayConfig

os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = "300"

llm_judge = LLMModelWrapper(model_name="qwen3:30b-instruct", num_ctx=20480)

faithfulness = FaithfulnessMetric(
    threshold=0.7,
    model=llm_judge,
    include_reason=True
)

relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model=llm_judge,
    include_reason=True
)

contextual_precision = ContextualPrecisionMetric(
    threshold=0.7,
    model=llm_judge,
    include_reason=True,
)

contextual_recall = ContextualRecallMetric(
    threshold=0.7,
    model=llm_judge,
    include_reason=True,
)

contextual_relevance = ContextualRelevancyMetric(
    threshold=0.7,
    model=llm_judge,
    include_reason=True,
)

def load_test_cases_from_json(file_path):
    """Load test cases from a JSON file and convert to LLMTestCase objects."""
    with open(file_path, 'r', encoding='utf-8') as f:
        test_cases_data = json.load(f)

    test_cases = []
    for tc_data in test_cases_data:
        test_case = LLMTestCase(
            input=tc_data['input'],
            actual_output=tc_data['actual_output'],
            retrieval_context=tc_data['retrieval_context'],
            expected_output=tc_data['expected_output']
        )
        test_cases.append(test_case)

    return test_cases


def run_evals():
    """Run end-to-end evaluations on pre-generated test cases."""
    script_dir = Path(__file__).parent
    testcases_dir = script_dir / 'rag-testcases'

    # Get all test case files
    testcase_files = list(testcases_dir.glob('*.test.json'))

    if not testcase_files:
        print("No test case files found in rag-testcases directory.")
        print("Please run generate_test_cases.py first to generate test cases.")
        return

    print(f"Found {len(testcase_files)} test case file(s)")

    # Load all test cases
    all_test_cases = []
    for testcase_file in testcase_files:
        print(f"Loading test cases from {testcase_file.name}...")
        test_cases = load_test_cases_from_json(testcase_file)
        all_test_cases.extend(test_cases)
        print(f"  ✓ Loaded {len(test_cases)} test cases")

    print(f"\nTotal test cases loaded: {len(all_test_cases)}")
    print("Running evaluation metrics...")

    evaluate(
        test_cases=all_test_cases,
        metrics=[faithfulness, relevancy, contextual_precision, contextual_recall, contextual_relevance],
        async_config=AsyncConfig(run_async=True, max_concurrent=3),
        display_config=DisplayConfig(verbose_mode=False)
    )

if __name__ == "__main__":
    run_evals()
    