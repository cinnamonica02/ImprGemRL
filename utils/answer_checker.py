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
        
        # Format model name properly for litellm
        self.model_name = self._format_model_name(model_name)
        logger.info(f"Using model: {self.model_name}")
        
        # Set up API keys based on model provider
        self._setup_api_keys()
        
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

    def _format_model_name(self, model_name: str) -> str:
        """Format the model name for LiteLLM by adding provider prefix if needed."""
        model_lower = model_name.lower()
        
        # Already has provider prefix
        if any(model_lower.startswith(p + '/') for p in ['openai', 'anthropic', 'gemini', 'huggingface']):
            return model_name
            
        # Add appropriate provider based on model name
        if any(name in model_lower for name in ['gpt-3', 'gpt-4', 'davinci']):
            return f"openai/{model_name}"
        elif any(name in model_lower for name in ['claude']):
            return f"anthropic/{model_name}"
        elif any(name in model_lower for name in ['gemini', 'palm']):
            return f"gemini/{model_name}"
        elif any(name in model_lower for name in ['llama', 'mistral', 'phi', 'unsloth']):
            return f"huggingface/{model_name}"
        else:
            # Default to Hugging Face for unknown models
            return f"huggingface/{model_name}"
    
    def _setup_api_keys(self):
        """Set up API keys based on the model provider."""
        if 'gemini/' in self.model_name:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError('GEMINI_API_KEY is not found in environment variables')
            os.environ['GEMINI_API_KEY'] = api_key
        elif 'openai/' in self.model_name:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError('OPENAI_API_KEY is not found in environment variables')
            os.environ['OPENAI_API_KEY'] = api_key
        elif 'anthropic/' in self.model_name:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError('ANTHROPIC_API_KEY is not found in environment variables')
            os.environ['ANTHROPIC_API_KEY'] = api_key
        elif 'huggingface/' in self.model_name:
            api_key = os.getenv('HUGGINGFACE_API_KEY')
            if api_key:
                os.environ['HUGGINGFACE_API_KEY'] = api_key
                # No error if missing as some HF models can be used without API key
        
    def _strip_think_tags(self, text: str) -> str:
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
    def get_stats(self) -> Dict:
        """ Get current retry statistics """
        return {
            **self.stats,
            "success_rate": (self.stats["successful_calls"] / self.stats["total_calls"] * 100) if self.stats["total_calls"] > 0 else 0,
            "exact_match_disagreement_rate": (self.stats["exact_match_failures_but_correct"] / self.stats["successful_calls"] * 100) if self.stats["successful_calls"] > 0 else 0
            
        }

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
                    logger.info(f"Calling LLM with model: {self.model_name}")
                    
                    response = await litellm.acompletion(
                        model=self.model_name,
                        messages=[{"content": prompt, "role": "user"}],
                        temperature=0.1
                    )
                    self.stats["successful_calls"] += 1
                    return self._parse_gemini_response(response.choices[0].message.content)
                except Exception as e:
                    logger.warning(f"LLM call attempt {attempt+1} failed: {str(e)}")
                    if 'rate limit' in str(e).lower():
                        self.stats["rate_limit_retries"] += 1
                    else:
                        self.stats["other_retries"] += 1
                    
                    if attempt < max_retries - 1:
                        retry_delay = RETRY_DELAY * (2 ** attempt)
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
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
                "explanation": "Answer must contain <think> tags with content",
                "extracted_answer": None,
                "extracted_ground_truth": None
            }

        cleaned_answer = self._strip_think_tags(model_answer)
        extracted_answer = self._extract_final_number(cleaned_answer)
        extracted_truth = self._extract_final_number(ground_truth)

        if extracted_answer is not None and extracted_truth is not None:
            if abs(extracted_answer - extracted_truth) < 1e-6:
                return {
                    "is_correct": True,
                    "explanation": f"Exact match: {extracted_answer} = {extracted_truth}",
                    "extracted_answer": extracted_answer,
                    "extracted_ground_truth": extracted_truth,
                    "check_method": "exact_match"
                }
            
        # Define the prompt for all cases where we need LLM verification
        prompt = f""" You are an expert math answer checker. Focus Only on final numerical answer, ignoring any reasoning
        steps or mistakes in the working. 

    Question: {question}

    Model's Answer:
    {cleaned_answer}

    Ground Truth:
    {ground_truth}

    Extracted numerical answers for references:
    - Model's final number: {extracted_answer if extracted_answer is not None else "Not found"}
    - Ground truth final number: {extracted_truth if extracted_truth is not None else "Not found"}


    Provide your assessment in the following format:

    ASSESSMENT: 
    Explain whether the final numerical answers match, focusing only on the numbers.

    VERDICT::

    CORRECT or INCORRECT


    """

        # Call LLM with retry logic
        result = await self._call_gemini_with_retry(prompt, max_retries)
        if result:
            # track cases where exact match failed but gemini marks as correct
            if result["is_correct"] and extracted_answer is not None and extracted_truth is not None:
                self.stats["exact_match_failures_but_correct"] += 1
                if len(self.stats["interesting_cases"]) < 10:
                    self.stats["interesting_cases"].append({
                        "question": question,
                        "model_answer": model_answer,
                        "ground_truth": ground_truth,
                        "gemini_explanation": result['explanation']
                    })

            # Add extracted answers and check method to the result
            result["extracted_answer"] = extracted_answer
            result["extracted_ground_truth"] = extracted_truth
            result["check_method"] = "gemini"
            return result

        return {
            "is_correct": False,
            "explanation": "Verification failed",
            "extracted_answer": extracted_answer,
            "extracted_ground_truth": extracted_truth,
            "check_method": "failed"
        }