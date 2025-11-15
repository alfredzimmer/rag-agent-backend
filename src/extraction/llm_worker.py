"""
LLM Worker module for synthetic dataset generation.

This module provides a worker class that handles all LLM-related operations
for generating synthetic instruction-response pairs. Each instance represents
a worker that can process contexts independently, with support for async operations.
"""

import os
import asyncio
from typing import Optional
from pydantic import BaseModel
from openai import AsyncOpenAI


class InstructionList(BaseModel):
    """Pydantic model for instruction generation response."""
    instructions: list[str]


class ResponseList(BaseModel):
    """Pydantic model for response generation."""
    response: str


class InstructionEvaluation(BaseModel):
    """Pydantic model for instruction evaluation response."""
    passed: bool
    explanation: str


class ResponseEvaluation(BaseModel):
    """Pydantic model for response evaluation."""
    score: int
    explanation: str


class LLMWorker:
    """
    Worker class for handling LLM operations in synthetic dataset generation.

    Each instance represents an independent worker that can process contexts
    and generate instruction-response pairs using async operations for efficiency.
    """

    def __init__(
        self,
        instruction_model: str = "gpt-4o-mini",
        response_model: str = "gpt-4o-mini",
        instruction_eval_model: str = "gpt-4o-mini",
        response_eval_model: str = "gpt-4o-mini",
        evaluate_instructions: bool = True,
        api_key: Optional[str] = None
    ):
        """
        Initialize the LLM worker.

        Args:
            instruction_model: Model for generating instructions
            response_model: Model for generating responses
            instruction_eval_model: Model for evaluating instructions
            response_eval_model: Model for evaluating responses
            evaluate_instructions: Whether to evaluate instructions (Stage 2)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.instruction_model = instruction_model
        self.response_model = response_model
        self.instruction_eval_model = instruction_eval_model
        self.response_eval_model = response_eval_model
        self.evaluate_instructions = evaluate_instructions

        # Initialize async OpenAI client
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.client = AsyncOpenAI(api_key=api_key)

    async def generate_instructions(
        self,
        context: str,
        num_samples: int = 5,
        icl_question: Optional[str] = None
    ) -> Optional[list[str]]:
        """
        Stage 1: Generate diverse questions from a context.

        Args:
            context: The context to generate questions from
            num_samples: Number of questions to generate
            icl_question: In-context learning example question

        Returns:
            List of generated questions
        """
        if icl_question is None:
            icl_question = "What are the key design considerations for a low-noise amplifier in RF circuits?"

        prompt = f"""You are asked to come up with a set of {num_samples} diverse questions on Electrical Engineering based on the provided context.
Please follow these guiding principles when generating responses:
* Use proper grammar and punctuation.
* Always generate safe and respectful content. Do not generate content that is harmful, abusive, or offensive.
* Always generate content that is factually accurate and relevant to the prompt.
* The questions should be clear and human-like.
* The questions should be diverse and cover a wide range of topics.
* The questions should not be template-based or generic, it should be very diverse.
* Simply return the questions, do not return any answers or explanations.
* Strictly adhere to the prompt and generate responses in the same style and format as the example.

To better assist you with this task, here is an example:
### Question:
1. {icl_question}

Context:
{context}

Now generate {num_samples} such questions, remember to follow the principles mentioned above
and use the same format as the examples. Remember to use the same style and format as the example
above."""

        try:
            parsed_response = await self.client.responses.parse(
                model=self.instruction_model,
                input=[{"role": "user", "content": prompt}],
                temperature=0.8,
                text_format=InstructionList
            )
            return parsed_response.output_parsed.instructions if parsed_response.output_parsed else None
        except Exception as e:
            print(f"Error generating instructions: {e}")
            return None

    async def evaluate_instruction(self, question: str) -> Optional[dict]:
        """
        Stage 2: Evaluate if a question meets quality criteria.

        Args:
            question: The question to evaluate

        Returns:
            Dictionary with 'passed' (bool) and 'explanation' (str)
        """
        prompt = f"""Please evaluate whether the following question is suitable for an AI training dataset.

Question: {question}

Evaluation criteria:
1. Is it relevant to Electrical Engineering domain?
2. Is it safe and respectful (not harmful, abusive, or offensive)?
3. Can it be answered by a language model with domain knowledge?
4. Is it clear and well-formed?

Respond with ONLY "PASS" or "FAIL" followed by a brief reason (one sentence).
Format: PASS: reason OR FAIL: reason"""

        try:
            parsed_response = await self.client.responses.parse(
                model=self.instruction_eval_model,
                input=[{"role": "user", "content": prompt}],
                temperature=0.3,
                text_format=InstructionEvaluation
            )

            if parsed_response.output_parsed:
                return {
                    'passed': parsed_response.output_parsed.passed,
                    'explanation': parsed_response.output_parsed.explanation
                }
            return None
        except Exception as e:
            print(f"Error evaluating instruction: {e}")
            return None

    async def generate_response(
        self,
        question: str,
        context: str,
        creative: bool = False
    ) -> Optional[str]:
        """
        Stage 3: Generate a response to a question.

        Args:
            question: The question to answer
            context: Context to use for answering
            creative: If True, use creative persona; if False, use precise persona

        Returns:
            Generated response
        """
        if creative:
            persona = "You are a creative and engaging expert in Electrical Engineering. Provide detailed, insightful answers that include examples and analogies to help understanding."
        else:
            persona = "You are a precise and technical expert in Electrical Engineering. Provide accurate, detailed answers with technical depth and clarity."

        prompt = f"""{persona}

Context:
{context}

Question: {question}

Please provide a comprehensive answer to the question based on the context provided. Your answer should be informative, well-structured, and demonstrate expert knowledge."""

        temperature = 0.8 if creative else 0.5

        try:
            parsed_response = await self.client.responses.parse(
                model=self.response_model,
                input=[{"role": "user", "content": prompt}],
                temperature=temperature,
                text_format=ResponseList
            )
            return parsed_response.output_parsed.response if parsed_response.output_parsed else None
        except Exception as e:
            print(f"Error generating response: {e}")
            return None

    async def evaluate_instruction_response_pair(
        self,
        question: str,
        answer: str
    ) -> Optional[dict]:
        """
        Stage 4: Evaluate the quality of a question-answer pair.

        Args:
            question: The question
            answer: The generated answer

        Returns:
            Dictionary with 'score' (1-3) and 'explanation' (str)
        """
        prompt = f"""Please act as an impartial judge and evaluate the quality of the answer provided by an AI assistant
to the questions displayed below. Evaluate whether or not the answer is a good example of how AI
Assistant should respond to the user's instruction. Please assign a score using the following 3-point
scale:

1: It means the answer is incorrect, irrelevant, unsafe or provides incomplete and garbage information.
For instance, the answer may be factually wrong, off-topic, or filled with irrelevant content that
doesn't address the user's question or it could be incomplete and hanging. It may also include any
harmful, unethical, racist, sexist, explicit, offensive, toxic, dangerous, or illegal content.

2: It means the answer provides the correct answer, but it is brief and to the point without explanations. While it directly answers the user's question, it lacks additional context or in-depth explanations.

3: It means the answer is a perfect answer from an AI Assistant. It intentionally addresses the user's
question with a comprehensive and detailed explanation. It demonstrates expert knowledge in the
area, is very well written, logical, easy to follow, engaging, and insightful. And the answer is safe and
does not include any harmful content.

Question: {question}

Answer: {answer}

Begin your evaluation by providing an 1 sentence explanation. Be as objective as possible. After providing
your explanation, you must rate the answer on a scale of 1 to 3 as mentioned above.

Format your response as:
Explanation: [your explanation]
Score: [1, 2, or 3]"""

        try:
            parsed_response = await self.client.responses.parse(
                model=self.response_eval_model,
                input=[{"role": "user", "content": prompt}],
                temperature=0.3,
                text_format=ResponseEvaluation
            )

            if parsed_response.output_parsed:
                return {
                    'score': parsed_response.output_parsed.score,
                    'explanation': parsed_response.output_parsed.explanation
                }
            return None
        except Exception as e:
            print(f"Error evaluating response: {e}")
            return None

    async def process_context(
        self,
        context_dict: dict,
        questions_per_context: int = 5,
        creative_responses: bool = False,
        min_quality_score: int = 2
    ) -> list[dict]:
        """
        Process a single context through the complete pipeline.

        Args:
            context_dict: Context dictionary containing cluster info and chunks
            questions_per_context: Number of questions to generate per context
            creative_responses: Use creative persona for responses
            min_quality_score: Minimum quality score (1-3) to include in dataset

        Returns:
            List of high-quality instruction-response pairs for this context
        """
        dataset = []

        # Combine chunks in context
        context_text = "\n\n".join([chunk['content'] for chunk in context_dict['chunks']])

        # Stage 1: Generate instructions
        questions = await self.generate_instructions(context_text, questions_per_context)
        if not questions:
            print(f"  - Failed to generate questions for cluster {context_dict['cluster_id']}")
            return dataset

        # Stage 2: Evaluate instructions (if enabled)
        valid_questions = []
        if self.evaluate_instructions:
            # Run evaluations in parallel
            eval_tasks = [self.evaluate_instruction(q) for q in questions]
            eval_results = await asyncio.gather(*eval_tasks)

            for q, eval_result in zip(questions, eval_results):
                if eval_result is None:
                    valid_questions.append(q)
                elif eval_result['passed']:
                    valid_questions.append(q)
        else:
            # Skip evaluation, use all generated questions
            valid_questions = questions

        if not valid_questions:
            print(f"  - No valid questions for cluster {context_dict['cluster_id']}")
            return dataset

        # Stage 3 & 4: Generate and evaluate responses (in parallel)
        async def process_question(question: str) -> Optional[dict]:
            """Process a single question: generate response and evaluate."""
            # Generate response
            answer = await self.generate_response(question, context_text, creative_responses)
            if answer is None:
                return None

            # Evaluate pair
            evaluation = await self.evaluate_instruction_response_pair(question, answer)

            if evaluation is None:
                # Failed to evaluate, include anyway with default score
                return {
                    'instruction': question,
                    'response': answer,
                    'context': context_text[:500] + "..." if len(context_text) > 500 else context_text,
                    'cluster_id': context_dict['cluster_id'],
                    'quality_score': 1,
                    'quality_explanation': "Failed to evaluate response"
                }
            elif evaluation['score'] >= min_quality_score:
                return {
                    'instruction': question,
                    'response': answer,
                    'context': context_text[:500] + "..." if len(context_text) > 500 else context_text,
                    'cluster_id': context_dict['cluster_id'],
                    'quality_score': evaluation['score'],
                    'quality_explanation': evaluation['explanation']
                }

            return None

        # Process all valid questions in parallel
        question_tasks = [process_question(q) for q in valid_questions]
        results = await asyncio.gather(*question_tasks)

        # Filter out None results
        dataset = [r for r in results if r is not None]

        return dataset
