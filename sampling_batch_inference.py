
import json 
import subprocess 
import time
import asyncio
import aiohttp
import re 
from typing import Optional, List, Dict, Any, Tuple
from utils.answer_checker import AnswerChecker
from tqdm import tqdm
import logging
import argparse
import json
import os
import sys
from datasets import load_dataset
from pathlib import Path
from config import (
    GSM8K_SYSTEM_PROMPT,
    ARC_SYSTEM_PROMPT,
    SGLANG_CONFIG,
    MAX_CONCURRENT_SGLANG, 
    MAX_CONCURRENT_GEMINI,
    RETRY_DELAY
)

def format_arc_question(question: str, output: Optional[str] = None) -> str:
    """Format an ARC question with optional output grid."""
    pass

def calculate_metrics(samples: List[Dict]) -> Dict:
    """Calculate metrics for a set of samples"""
    # Count valid samples (those with is_correct field)
    valid_samples = [s for s in samples if "is_correct" in s]
    total_samples = len(samples) # include invalid samples in denom

    # Calc basic metrics 
    correct_samples = [s for s in valid_samples if s["is_correct"]]
    num_correct = len(correct-samples)

    # calculate average characters for correct and incorrect answers
    correct_chars = [len(s["output"]) for s in correct_samples]
    incorrect_samples = [s for s in valid_samples if not s["is_correct"]]
    incorrect_chars = [len(s["output"]) if not s["is_correct"]]
def has_single_think_tag_pair(text: str) -> bool:
    """Check if the text has a single pair of <think> tags."""
    open_tags = len(re.findall(r'<think>', text))
    close_tags = len(re.findall(r'</think>', text))
    return open_tags == 1 and close_tags == 1
    
def extract_partial_response(text: str, suppress_logs: bool = False) -> str:
    """Extract the partial response from the text."""
    if not has_single_think_tag_pair(text):
        return text

    result = partial + "Hold on a second, thats wrong. Let me think through this again."
    

    # Dbug logging if not suppressed
    if not suppress_logs:
        print("\nProcessing incorrect answer with wait")
        print("Original:", text)
        print("Modified:", result)
        print("-" * 50)

    return result

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str,
                        help='Model name/path to use for inference')
    
    parser.add_argument('--dataset', type=str, choices=['gsm8k', 'arc'], default='gsm8k',
                        help='Dataset to use (gsm8k or arc)')
    
    parser.add_argument('--arc-file', type=str,
                        help='For ARC dataset, specify which json file to use (e.g., arc-agi_evaluation.json)')
    
    parser.add_argument('--gsm8k-file', type=str, default='gsm8k_train_64.json',
                        help='For GSM8K dataset, specify which json file to use (e.g., gsm8k_train_64.json)')
    
    parser.add_argument('--subset-file', type=str,
                        help='JSON file containing list of problem IDs to run (optional)')
    
    parser.add_argument('--num-samples', type=int, default=8,
                        help='Number of samples to generate per question (default: 8)')
    
    parser.add_argument('--wait', action='store_true',
                        help='Enable wait functionality for incorrect answers')
    
    parser.add_argument('--think-tags', action='store_true',
                        help='Require <think> tags in answers')
    
    parser.add_argument('--suppress-logs', action='store_true',
                        help='Suppress debug logging output')
    
    return parser.parse_args()


def format_grid_display(grid: List[List[int]], indent: int =2) -> str:
    """Format a grid in a readable way for display purposes only"""
    max_width = max(len(str(num)) for row in grid for num in row)
    rows = []
    for row in grid:
        numbers = [str(num).rjust(max_width) for num in row]
        rows.append(" ".join(numbers))

        # join rows with newlines and indent
        indent_str = " " * indent
        return "[\n" + f"{indent_str}{',\n'.join(rows)}\n" + "]"
def start_server(model_name: str) -> subprocess.Popen:
    # start server with SGLang
    pass
    
def load_arc_dataset(file_path: str) -> List[Dict[str, str]]:
    pass 

async def process_questions(dataset: List[Dict[str, str]], 
                            model_name: str, 
                            num_samples: int = 8, 
                            wait: bool = False, 
                            think_tags: bool = False, 
                            suppress_logs: bool = False) -> List[Dict[str, Any]]:
    async def process_single_question(dataset: List[Dict], model_name: str, results_dir: str, num_samples:int, 
                                      is_arc:bool = False, wait_enabled: bool = False, suppress_logs: bool = False,
                                        no_think_tags: bool = False) -> Tuple[List[Dict], int, Optional[Dict]]:
        """Process all questions concurrently with rate limiting"""
        # Initialize rate limiters for sglang and gemini using semaphores
        sglang_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SGLANG) 
        gemini_semaphore = asyncio.Semaphore(MAX_CONCURRENT_GEMINI) 

        # initialize answer checker with Gemini semaphore - and check whether or not to check for thinking tags 
        checker = AnswerChecker(semaphore=gemini_semaphore, no_think_tags = no_think_tags) if not is_arc else None
        # Track succesful wait retries
        wait_successes = 0
        async def process_single_question(session: aiohttp.ClientSession, example: Dict[str, str]) -> Dict:
            nonlocal wait_successes
            if is_arc:
                #Format ARC question with training examples
                question = format_arc_question(example["input"], example.get["output"])
                model_answers = await get_model_answer(session, question, model_name, num_samples,
                sglang_semaphore, is_arc=True, wait_enabled=wait_enabled)

                # Process each sample

                samples = []
                for answer in model_answers:
                    check_result = check_arc_answer(
                        question=example,
                        model_answer=answer,
                        ground_thuth=example["output"],
                        no_think_tags=no_think_tags
                    ) 

                    # If wait in enabled and answer is incorrect, but has valid think tags, try again
                    if wait_enabled and not check_result["is_correct"] and has_single_think_tag_pair(answer):
                        wait_successes += 1
                        answer = extract_partial_response(answer, suppress_logs=suppress_logs)
                        check_result = check_arc_answer(
                            question=example,
                            model_answer=answer,
                            ground_truth=example["output"],
                            no_think_tags=no_think_tags
                        )

                    sample = {
                        "is_correct": check_results["is_correct"],
                    }
                    #Only include extracted answer if grid was succesfully extracted
                    if check_result["extracted_answer"] : 
                        sample["extracted_answer"] = check_result["extracted_answer"]

                    samples.append(sample)

                    result = {
                        "task_id": example["task_id"], 
                        "prompt": question,
                        "input": example["input"],
                        "ground_truth": example["output"], 
                        "samples": samples,

                    }

                    pass

            else:
                # Process GSM8K question
                model_answers = await get_model_answer(session, example["question"], model_name,
                                                       sglang_semaphore, num_samples, wait_enabled=wait_enabled)
                # Process each samples
                samples = [] 
                for answer in model_answers:
                    check_result = await checker.check_answer_async(
                        question=example["question"],
                        model_answer=answer,
                        ground_truth=example["answer"]
                    )

                    # If wait is enabled and answer is incorrect but has valid think tags try again
                    if wait_enabled and not check_result["is_correct"] and has_single_think_tag_pair(answer):
                        partial_response = extract_partial_response(answer, suppress_logs)
                        retry_answers = await get_model_answer(session, example["question"], model_name,
                                                                sglang_semaphore, num_samples=1, wait_enabled=True,
                                                                partial_response=partial_response)
                        if retry_answers and retry_answers[0]:
                            #Combine partial response with new completion
                            combined_answer = partial_response + retry_answers[0]
                            #Check combined answer:
                            retry_result = await checker.check_answer_async(
                                question=example["question"],
                                model_answer=combined_answer,
                                ground_truth=example["answer"]
                            )
                            ## To Do  LATER 

                    samples.append({
                        "output": answer,
                        "is_correct": check_result["is_correct"]
                    })

                result = {
                    "question": example["question"],
                    "answer": example["answer"],
                    "samples": samples
                }

            #Calculate metrics for given question
            result["metrics"] = calculate_metrics(result["samples"])

            # Only print metrics if not suppressing logs
            if not supress_logs:
                print(f"\nQuestion: {len{results}} metrics:")
                for metric, value in result["metrics"].items():
                    if isinstance(value, float):  # Check if value is a float, otherwise it's an nstance



                        
                    

## Main function 

async def main():
    args = parse_args()
    model_name = args.model 

    # Create model specific results directory with dataset details 
    model_short_name = model_name.split('/')[-1]
    if args.dataset == 'gsm8k':
        dataset_name = os.path.splittext(args.gsm8k_file)[0] # remove json extention
    else:
        dataset_name = os.path.splitext(args.gsm8k_file)[0] if args.arc_file else 'arc'
        if args.subset_file:
            subset_name = os.path.splitext(os.path.basename(args.subset_file))[0]
            dataset_name += f"_{subset_name}"

    # Add suffixes for wait and think tags
    if args.wait:
        dataset_name += "_w"
    if args.think_tags:
        dataset_name += "_tt"
    # setting up results dir where we want to save things 
    results_dir = Path(f"results/{model_short_name}/{dataset_name}")
    os.makedirs(results_dir, exist_ok=True)

    # Load subset IDs if specified 
    subset_ids = None
    if args.subset_file:
        with open(args.subset_file, 'r') as f:
            subset_ids = json.load(f)

    if args.dataset == 'gsm8k':
        try:
            with open(args.gsm8k_file, 'r') as f:
                dataset = json.load(f)
                if subset_ids:
                    dataset = [ex for i, ex in enumerate(dataset) if str(i) in subset_ids]
        except FileNotFoundError:
            print(f"Could not find GSM8K dataset file:  {args.gsm8k_file}")
            print("Please run downloaded_gsm8k.py first to create the dataset file.")
            sys.exit(1)
    else: # ARC dataset
        if not args.arc_file:
            print("Please specify the ARC dataset file using the --arc-file argument.")
            sys.exit(1)

        try:
            dataset, _  = load_arc_dataset(args.arc_file. subset_ids)
        except FileNotFoundError:
            print(f"Could not find ARC dataset file: {args.arc_file}")
            sys.exit(1)

    # Start  Server -
    server_process = start_server(model_name)
    if not server_process:
        sys.exit(1)

    try:
        # Process all questions 
        results, wait_succesess, gemini_stats = await process_questions(dataset, model_name, results_dir,
                                                                        is_ard =(args.dataset == 'arc'),
                                                                        wait_enabled=args.wait,
                                                                        suppress_logs=args.suppress_logs,
                                                                        no_think_tags=not args.think_tags)
        # calculate stats 
        pass_at_k_count = sum(1 for r in results if r["metrics"]["pass_at_k"])
        majority_count = sum(1 for r in results if r["metrics"]["majority_at_k"]) 
        total_count = len(results)

        # Calculate overall average lengths 
        all_correct_lengths = []
        all_incorrect_lengths = []
        for result in results:
            correct_samples = [s for s in result["samples"] if s.get("is_correct", False)]
            incorrect_samples = [s for s in result["samples"] if not s.get("is_correct", False)]
            all_correct_lengths.extend(len(s["output"]) for s in correct_samples)
            all_incorrect_lengths.extend(len(s["output"]) for s in incorrect_samples)
        avg_correct_length = sum(all_correct_lengths) / len(all_correct_lengths) if all_correct_lengths else 0
        all_correct_length = sum(all_incorrect_lengths) / len(all_incorrect_lengths) if all_incorrect_lengths else 0

        # Prepare final results
        final_results = {
            "model": model_name,
            "dataset": args.dataset, 
            "subset_info": f"Using  subset of {len(subset_ids)} problems" if subset_ids else None,
            "samples_per_question": args.num_samples,
            "wait_enabled": args.wait,
            "wait_succeses": wait_successes if args.wait else None,
            "total_questions": total_count,
            "pass_at_k": {
                "count": pass_at_k_count,
                "total": total_count,
                "percentage": (pass_at_k_count / total_count) * 100 if total_count > 0 else 0
            },
            "majority_at_k": {
                "count": majority_count,
                "total": total_count,
                "percentage": (majority_count / total_count) * 100 if total_count > 0 else 0
            },
            "average_length": {
                "correct_responses": avg_correct_length,
                "incorrect_responses": avg_incorrect_length
            },
            "gemini_stats": gemini_stats

        }

        # Save final results
        with open(f"{results_dir}/final_results.json", "w") as f:
            json.dump(final_results, f, indent=2)

        print(f"\nFinal Statistics:")
        print(f"Model: {model_name}")
        print(f"Dataset: {args.dataset}")
        if subset_ids:
            print(f"Subset: {len(subset_ids)} problems")
        print(f"Number of samples per question: {args.num_samples}")
        print(f"Wait functionality: {'enabled' if args.wait else 'disabled'}")
        if args.wait:
            print(f"Answers corrected through wait: {wait_successes}")
        print(f"\nMetrics:")
        print(f"Total questions attempted: {total_count}")
        print(f"Pass at{args.num_samples} (at least 1 correct): ({(pass_at_k_count / total_count) * 100:.2f}%)")
        print(f"Majority at {args.num_samples} (majority correct): ({(majority_count / total_count) * 100:.2f}%)")
        print(f"\nResponse Lengths:")
        print(f"\nAverage Lengths: {avg_correct_length:.2f} (correct), {avg_incorrect_length:.2f} (incorrect)")





























































        ##############################################################################################################




            
    

    
    
    

























































































































































# # Import the configuration
# from config import InferenceConfig

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Install missing packages if needed
# missing_packages = []

# try:
#     from dotenv import load_dotenv
#     load_dotenv()  # Load environment variables from .env file
# except ImportError:
#     missing_packages.append("python-dotenv")

# try:
#     import litellm
# except ImportError:
#     missing_packages.append("litellm")

# if missing_packages:
#     logger.info(f"Installing missing packages: {', '.join(missing_packages)}")
#     import subprocess
#     for package in missing_packages:
#         subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
#     # Now try importing again
#     if "python-dotenv" in missing_packages:
#         from dotenv import load_dotenv
#         load_dotenv()
#     if "litellm" in missing_packages:
#         import litellm

# # Now import AnswerChecker
# from utils.answer_checker import AnswerChecker

# def check_api_keys():
#     """Check which API keys are available"""
#     available_models = []
    
#     if os.getenv("GEMINI_API_KEY"):
#         available_models.append("gemini/gemini-1.5-flash")
    
#     if os.getenv("OPENAI_API_KEY"):
#         available_models.append("openai/gpt-3.5-turbo")
    
#     if os.getenv("ANTHROPIC_API_KEY"):
#         available_models.append("anthropic/claude-instant-1")
    
#     if os.getenv("HUGGINGFACE_API_KEY"):
#         available_models.append("huggingface/microsoft/phi-2")  # A smaller model that should work with HF API
    
#     return available_models

# async def process_example(example, answer_checker: AnswerChecker):
#     """
#     Process a single example:
#     - Extracts the question and ground truth answer.
#     - Simulates a model answer by wrapping the ground truth in <think> tags.
#     - Uses the AnswerChecker to evaluate the answer.
#     """
#     question = example["question"]
#     ground_truth = example["answer"]

#     # For demonstration purposes, we simulate a model answer.
#     # In a real scenario, this would be generated by your model.
#     model_answer = f"<think>{ground_truth}</think>"

#     result = await answer_checker.check_answer_async(
#         question=question,
#         model_answer=model_answer,
#         ground_truth=ground_truth,
#         max_retries=3
#     )
#     return result

# def load_gsm8k_dataset(subset='main', split='test', file_path=None):
#     """
#     Load GSM8K dataset with proper error handling
#     """
#     try:
#         if file_path and os.path.exists(file_path):
#             logger.info(f"Loading GSM8K dataset from file: {file_path}")
#             with open(file_path, 'r') as f:
#                 data = json.load(f)
#             # Convert to dataset-like format
#             return [{"question": item.get("question", ""), 
#                     "answer": item.get("answer", "")} 
#                    for item in data]
#         else:
#             # Ensure the subset is explicitly provided
#             logger.info(f"Loading GSM8K dataset from HuggingFace - subset: '{subset}', split: '{split}'")
#             try:
#                 # Try to load with explicit subset and split parameters
#                 dataset = load_dataset("openai/gsm8k", name=subset, split=split)
#                 logger.info(f"Successfully loaded GSM8K dataset with {len(dataset)} examples")
#                 return dataset
#             except Exception as e:
#                 logger.warning(f"Error with name parameter: {e}")
#                 # Try alternate parameter format
#                 dataset = load_dataset("openai/gsm8k", subset, split=split)
#                 logger.info(f"Successfully loaded GSM8K dataset with {len(dataset)} examples")
#                 return dataset
#     except Exception as e:
#         logger.error(f"Failed to load dataset: {e}")
#         raise

# async def main():
#     # Load default configuration
#     config = InferenceConfig()
    
#     # Check which API keys are available
#     available_models = check_api_keys()
#     if not available_models:
#         logger.error("No API keys found. Please set at least one API key in your environment.")
#         logger.info("You can set one of the following environment variables:")
#         logger.info("GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, HUGGINGFACE_API_KEY")
#         return
    
#     # Parse command line arguments
#     parser = argparse.ArgumentParser(description="Run batch inference on datasets")
#     parser.add_argument("--dataset", type=str, default="gsm8k", help="Dataset to use (default: gsm8k)")
#     parser.add_argument("--model", type=str, help=f"Model name to use. Available models: {', '.join(available_models)}")
#     parser.add_argument("--num-samples", type=int, default=config.sample_size or 5, help="Number of samples to process")
#     parser.add_argument("--gsm8k-file", type=str, help="Path to GSM8K dataset file (optional)")
#     parser.add_argument("--dataset-subset", type=str, default=config.dataset_subset, help=f"Dataset subset (default: {config.dataset_subset})")
#     parser.add_argument("--dataset-split", type=str, default=config.dataset_split, help=f"Dataset split (default: {config.dataset_split})")
#     parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    
#     args = parser.parse_args()
    
#     # List available models if requested
#     if args.list_models:
#         logger.info("Available models:")
#         for model in available_models:
#             logger.info(f"  - {model}")
#         return
    
#     # Use the first available model if none specified
#     model_name = args.model or available_models[0]
#     logger.info(f"Using model: {model_name}")
    
#     # Load the dataset based on the arguments
#     logger.info(f"Loading {args.dataset} dataset...")
    
#     try:
#         if args.dataset.lower() == "gsm8k":
#             dataset = load_gsm8k_dataset(
#                 subset=args.dataset_subset,
#                 split=args.dataset_split,
#                 file_path=args.gsm8k_file if args.gsm8k_file else None
#             )
#         else:
#             raise ValueError(f"Unsupported dataset: {args.dataset}")
#     except Exception as e:
#         logger.error(f"Error loading dataset: {str(e)}")
#         return
    
#     if not dataset:
#         logger.error("Failed to load dataset. Exiting.")
#         return

#     # Initialize the AnswerChecker with the specified model
#     logger.info(f"Initializing AnswerChecker with model: {model_name}")
#     answer_checker = AnswerChecker(model_name=model_name)

#     # Process the specified number of samples
#     num_samples = min(args.num_samples, len(dataset))
#     subset = dataset[:num_samples] if isinstance(dataset, list) else dataset.select(range(num_samples))
    
#     logger.info(f"Processing {num_samples} examples...")
#     tasks = [process_example(example, answer_checker) for example in subset]

#     # Run the tasks concurrently
#     results = await asyncio.gather(*tasks)

#     # Log results for each example
#     for idx, res in enumerate(results):
#         logger.info(f"Result for example {idx}:\n{res}\n")

#     # Log aggregated statistics
#     stats = answer_checker.get_stats()
#     logger.info(f"Aggregated Stats: {stats}")
    
#     # Save results to file
#     results_file = config.results_file
#     with open(results_file, 'w') as f:
#         json.dump({
#             "results": [{"example_index": idx, "result": res} for idx, res in enumerate(results)],
#             "stats": stats,
#             "config": {
#                 "model_name": model_name,
#                 "dataset": args.dataset,
#                 "dataset_subset": args.dataset_subset,
#                 "num_samples": num_samples,
#             }
#         }, f, indent=2)
    
#     logger.info(f"Results saved to {results_file}")

if __name__ == "__main__":
    asyncio.run(main())
