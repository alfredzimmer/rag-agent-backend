import pytest
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRecallMetric, ContextualRelevancyMetric
from ollama_deepeval_wrapper import OLLAMA_DEEPEVAL_WRAPPER

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

def run_automated_evals(test_dataset: list[dict], agent):
    print(f"Running evaluation on {len(test_dataset)} test cases...")

    metrics = [faithfulness, relevancy, contextual_recall, contextual_relevance]

    deepeval_test_cases = []

    for data in test_dataset:
        print(f"   Testing Input: {data['input'][:30]}...")
        
        agent_response = agent(data['input'])
        
        test_case = LLMTestCase(
            input=data['input'],
            actual_output=agent_response['output'],
            retrieval_context=agent_response['retrieved_context'],
            expected_output=data['expected_output']
        )
        deepeval_test_cases.append(test_case)

    results = evaluate(deepeval_test_cases, metrics)

    score_data = []
    for res in results:
        row = {
            "Input": res.input,
            "Actual Output": res.actual_output,
            "Passed": res.success,
        }
        # Extract individual metric scores
        for metric_data in res.metrics_data:
            row[f"{metric_data.name} Score"] = metric_data.score
            row[f"{metric_data.name} Reason"] = metric_data.reason[:50] + "..." # Truncate for display
        
        score_data.append(row)

    # Step E: Display Table
    df = pd.DataFrame(score_data)
    print("\n\n📊 EVALUATION RESULTS TABLE 📊")
    print(df.to_markdown(index=False))
    
    return df
