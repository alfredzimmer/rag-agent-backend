# Synthesis Dataset Refactoring Summary

## Overview
The `synthesize_dataset.py` module has been refactored to improve performance, scalability, and maintainability through the introduction of async operations and multiprocessing.

## Key Changes

### 1. New Module: `llm_worker.py`
- **Purpose**: Extracted all LLM-related operations into a separate, reusable worker class
- **Features**:
  - `LLMWorker` class handles all 4 stages of synthetic data generation
  - Async/await support for all LLM operations
  - Uses `AsyncOpenAI` client for concurrent API calls
  - Configurable instruction evaluation (can be disabled via parameter)

### 2. Async Operations
The following operations now run asynchronously within each worker:

- **Instruction Evaluation (Stage 2)**: All questions are evaluated in parallel
- **Response Generation (Stage 3)**: All valid questions generate responses concurrently
- **Response Evaluation (Stage 4)**: Response evaluations happen in parallel with generation

This significantly reduces wall-clock time when processing multiple questions from a single context.

### 3. Multiprocessing Architecture

#### Previous Architecture:
```
DatasetSynthesizer
└── Sequential processing of contexts
    └── For each context:
        - Generate instructions (sync)
        - Evaluate each instruction (sync, sequential)
        - Generate each response (sync, sequential)
        - Evaluate each response (sync, sequential)
```

#### New Architecture:
```
DatasetSynthesizer
└── Splits contexts into chunks
    └── Spawns N worker processes (via multiprocessing.Pool)
        └── Each worker process:
            - Creates an LLMWorker instance
            - Processes its context chunk
                └── For each context (async):
                    - Generate instructions (async)
                    - Evaluate all instructions (async, parallel)
                    - Generate + evaluate all responses (async, parallel)
```

### 4. New Parameters

#### DatasetSynthesizer.__init__()
- `evaluate_instructions: bool = True` - Toggle instruction evaluation (Stage 2)
- `num_workers: int = 4` - Number of parallel worker processes

#### LLMWorker.__init__()
- `evaluate_instructions: bool = True` - Control whether to run Stage 2 evaluation
- All model parameters (instruction_model, response_model, etc.)
- `api_key: Optional[str] = None` - Optional explicit API key

### 5. Behavior Changes

#### Instruction Generation (Stage 1)
- **Unchanged**: Still generates once per context (synchronous within worker)
- Each context generates N instructions based on `questions_per_context` parameter

#### Instruction Evaluation (Stage 2)
- **New**: Can be disabled entirely via `evaluate_instructions=False`
- **Changed**: Now runs in parallel for all generated questions
- Questions are gathered and evaluated concurrently using `asyncio.gather()`

#### Response Generation & Evaluation (Stages 3 & 4)
- **Changed**: Now fully async and parallel
- For N valid questions, creates N concurrent tasks
- Each task generates a response AND evaluates it
- All tasks run in parallel using `asyncio.gather()`

### 6. Performance Benefits

1. **Async Within Workers**: Multiple API calls per context happen concurrently
2. **Multiprocessing Across Contexts**: Multiple contexts processed simultaneously
3. **Combined Effect**:
   - If you have 100 contexts, 5 questions/context, 4 workers:
   - Each worker handles ~25 contexts
   - Within each context, 5 questions are processed concurrently
   - Total speedup can be 4-20x depending on API latency

### 7. Usage Examples

#### Basic Usage (Default Behavior)
```python
synthesizer = DatasetSynthesizer(
    documents_dir="src/extraction/documents",
    outputs_dir="src/extraction/outputs"
)
# Uses 4 workers, evaluates instructions by default
```

#### Disable Instruction Evaluation
```python
synthesizer = DatasetSynthesizer(
    documents_dir="src/extraction/documents",
    outputs_dir="src/extraction/outputs",
    evaluate_instructions=False  # Skip Stage 2
)
```

#### Adjust Worker Count
```python
synthesizer = DatasetSynthesizer(
    documents_dir="src/extraction/documents",
    outputs_dir="src/extraction/outputs",
    num_workers=8  # Use 8 parallel processes
)
```

#### Combine Options
```python
synthesizer = DatasetSynthesizer(
    documents_dir="src/extraction/documents",
    outputs_dir="src/extraction/outputs",
    evaluate_instructions=False,  # Skip evaluation
    num_workers=8,                # 8 workers
    instruction_model="gpt-4o",   # Better model for instructions
    response_model="gpt-4o-mini"  # Faster model for responses
)
```

## Files Modified/Created

### Created:
- `llm_worker.py` - New worker module with LLMWorker class
- `test_refactor.py` - Test script to verify refactoring
- `REFACTORING_SUMMARY.md` - This document

### Modified:
- `synthesize_dataset.py`:
  - Removed: `_get_llm_client()`, `generate_instructions()`, `evaluate_instruction()`, `generate_response()`, `evaluate_instruction_response_pair()`
  - Modified: `__init__()` - Added new parameters
  - Modified: `synthesize_dataset()` - Complete rewrite with multiprocessing
  - Updated: Module docstring and comments

## Backward Compatibility

The refactoring maintains backward compatibility with existing code:
- All previous parameters still work with default values
- New parameters have sensible defaults
- Existing method signatures preserved where applicable
- Output format unchanged

## Testing

Run the test script to verify the refactoring:
```bash
python src/extraction/test_refactor.py
```

Expected output:
```
✓ All tests passed! The refactoring appears successful.
```

## Future Improvements

Potential enhancements to consider:
1. Rate limiting to avoid API throttling
2. Retry logic with exponential backoff
3. Progress tracking and resumption for long jobs
4. Configurable batch sizes for better memory management
5. Support for different LLM providers (Anthropic, local models)
