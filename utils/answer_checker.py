# utils/answer_checker.py
import os
import re
import json
import logging
import asyncio
from typing import Optional, Dict
from datasets import load_dataset
from dotenv import load_dotenv
import litellm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RETRY_DELAY = 1

class AnswerChecker:
    def __init__(self, model_name: str = 'gemini/gemini-1.5-flash', 
                 semaphore: Optional[asyncio.Semaphore] = None,
                 no_think_tags: bool = False):
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError('GEMINI_API_KEY is not found in environment variables')
        os.environ['GEMINI_API_KEY'] = api_key
        
        self.model_name = model_name
        self.semaphore = semaphore or asyncio.Semaphore(10)
        self.no_think_tags = no_think_tags
        
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "retry_calls": 0,
            "failed_calls": 0,
            "rate_limit_retries": 0,
            "other_retries": 0,
            "retry_counts": {1: 0, 2: 0, 3: 0},
            "exact_match_failures_but_correct": 0,
            "interesting_cases": []
        }

    def _strip_think_tags(self, text: str) -> str:
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    def _extract_final_number(self, text: str) -> Optional[float]:
        try:
            # 1. First check DeepSeek's "Final Answer" format
            if "Final Answer" in text:
                final_part = text.split("Final Answer")[-1]
                numbers = re.findall(r'-?\d+\.?\d*', final_part)
                if numbers:
                    return float(numbers[-1])
            # Check for #### format
            if '####' in text:
                final_part = text.split('####')[-1].strip()
                numbers = re.findall(r'-?\d+\.?\d*', final_part)
                return float(numbers[-1]) if numbers else None
            
            # Check for LaTeX boxed format
            if r'\boxed{' in text:
                boxed_content = text.split(r'\boxed{')[-1].split('}')[0]
                numbers = re.findall(r'-?\d+\.?\d*', boxed_content)
                return float(numbers[-1]) if numbers else None
            
            # General number extraction
            numbers = re.findall(r'-?\d+\.?\d*', text)
            return float(numbers[-1]) if numbers else None
        except (ValueError, IndexError, TypeError):
            return None
        

    def _parse_gemini_response(self, response: str) -> Dict:
        verdict_match = re.search(r'VERDICT:\s*(CORRECT|INCORRECT)', response)
        explanation_match = re.search(r'ASSESSMENT:(.*?)VERDICT::', response, re.DOTALL)
        
        return {
            "is_correct": verdict_match.group(1) == "CORRECT" if verdict_match else False,
            "explanation": explanation_match.group(1).strip() if explanation_match else "No explanation found"
        }

    def _has_valid_think_tags(self, text: str) -> bool:
        if self.no_think_tags:
            return True
            
        open_tags = len(re.findall(r'<think>', text))
        close_tags = len(re.findall(r'</think>', text))
        return (open_tags == close_tags and 
                open_tags > 0 and 
                text.strip().startswith('<think>') and 
                bool(re.search(r'<think>.*?</think>', text, re.DOTALL)))

    async def _call_gemini_with_retry(self, prompt: str, max_retries: int = 3) -> Optional[Dict]:
        async with self.semaphore:
            for attempt in range(max_retries):
                try:
                    self.stats["total_calls"] += 1
                    response = await litellm.acompletion(
                        model=self.model_name,
                        messages=[{"content": prompt, "role": "user"}],
                        temperature=0.1
                    )
                    self.stats["successful_calls"] += 1
                    return self._parse_gemini_response(response.choices[0].message.content)
                except Exception as e:
                    if 'rate limit' in str(e).lower():
                        self.stats["rate_limit_retries"] += 1
                    else:
                        self.stats["other_retries"] += 1
                    
                    if attempt < max_retries - 1:
                        await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
                        self.stats["retry_counts"][attempt + 1] += 1
                    else:
                        self.stats["failed_calls"] += 1
                        logger.error(f"Failed after {max_retries} retries: {str(e)}")
                        return None

    async def check_answer_async(self, question: str, model_answer: str, 
                               ground_truth: str, max_retries: int = 3) -> Dict:
        if not self._has_valid_think_tags(model_answer):
            return {
                "is_correct": False,
                "explanation": "Invalid think tags format",
                "extracted_answer": None,
                "extracted_ground_truth": None,
                "check_method": "format_check"
            }

        cleaned_answer = self._strip_think_tags(model_answer)
        extracted_answer = self._extract_final_number(cleaned_answer)
        extracted_truth = self._extract_final_number(ground_truth)

        if extracted_answer and extracted_truth:
            if abs(extracted_answer - extracted_truth) < 1e-6:
                return {
                    "is_correct": True,
                    "explanation": f"Exact match: {extracted_answer} = {extracted_truth}",
                    "extracted_answer": extracted_answer,
                    "extracted_ground_truth": extracted_truth,
                    "check_method": "exact_match"
                }

        gemini_response = await self._call_gemini_with_retry(
            f"""Compare these answers for the question: {question}
            Model Answer: {cleaned_answer}
            Ground Truth: {ground_truth}
            Focus only on numerical equivalence.""",
            max_retries
        )

        if gemini_response:
            if gemini_response['is_correct'] and extracted_answer is not None:
                self.stats["exact_match_failures_but_correct"] += 1
                if len(self.stats["interesting_cases"]) < 10:
                    self.stats["interesting_cases"].append({
                        "question": question,
                        "model_answer": model_answer,
                        "ground_truth": ground_truth,
                        "gemini_explanation": gemini_response['explanation']
                    })
            return {**gemini_response, 
                    "extracted_answer": extracted_answer,
                    "extracted_ground_truth": extracted_truth,
                    "check_method": "gemini"}

        return {
            "is_correct": False,
            "explanation": "Verification failed",
            "extracted_answer": extracted_answer,
            "extracted_ground_truth": extracted_truth,
            "check_method": "failed"
        }