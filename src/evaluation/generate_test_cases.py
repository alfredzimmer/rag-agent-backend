import sys
import json
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to allow imports from rag module
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.agent import agent_call


def generate_test_cases_from_goldens():
    """Generate test cases from all golden files and save them to rag-testcases."""
    script_dir = Path(__file__).parent
    goldens_dir = script_dir / 'rag-eval-goldens'
    testcases_dir = script_dir / 'rag-testcases'

    # Ensure testcases directory exists
    testcases_dir.mkdir(exist_ok=True)

    # Get all JSON files from rag-eval-goldens
    golden_files = list(goldens_dir.glob('*.test.json'))

    if not golden_files:
        print("No golden files found in rag-eval-goldens directory")
        return

    print(f"Found {len(golden_files)} golden file(s)")

    # Process each golden file
    for golden_file in golden_files:
        print(f"\nProcessing: {golden_file.name}")

        # Load goldens from file
        with open(golden_file, 'r', encoding='utf-8') as f:
            goldens = json.load(f)

        print(f"  Generating test cases for {len(goldens)} goldens...")

        # Generate test cases with progress bar
        test_cases = []
        for golden in tqdm(goldens, desc=f"  {golden_file.name}"):
            try:
                # Call agent to get actual output and retrieval context
                res, text_chunks = agent_call(golden['input'])

                test_case = {
                    'input': golden['input'],
                    'actual_output': str(res),
                    'retrieval_context': text_chunks,
                    'expected_output': golden['output']
                }
                test_cases.append(test_case)
            except Exception as e:
                print(f"    Error processing golden '{golden['input'][:50]}...': {e}")
                continue

        # Save test cases to rag-testcases with same filename
        output_file = testcases_dir / golden_file.name
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(test_cases, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Saved {len(test_cases)} test cases to {output_file.name}")

    print(f"\n✓ Test case generation complete!")


if __name__ == "__main__":
    generate_test_cases_from_goldens()
