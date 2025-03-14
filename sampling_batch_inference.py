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

def has_single_think_tag_pair(text: str) -> bool:
    """Check if text has exactly one pair of think tags."""
    open_tags = len(re.findall(r'<think>', text))
    close_tags = len(re.findall(r'</think>', text))
    return open_tags == 1 and close_tags == 1

def extract_partial_response(text: str, suppress_logs: bool = False) -> str:
    """Extract the partial response from the text."""
    if not has_single_think_tag_pair(text):
        return text

    # Extract content between <think> tags
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if not match:
        return text

    partial = match.group(0)  # Get the entire <think>...</think> part
    result = partial + " Hold on a second, that's wrong. Let me think through this again."

    # Debug logging if not suppressed
    if not suppress_logs:
        print("\nProcessing incorrect answer with wait")
        print("Original:", text)
        print("Modified:", result)
        print("-" * 50)

    return result

def calculate_metrics(samples: List[Dict]) -> Dict:
    """Calculate metrics for a set of samples"""
    # Count valid samples (those with is_correct field)
    valid_samples = [s for s in samples if "is_correct" in s]
    total_samples = len(samples) # include invalid samples in denom

    # Calc basic metrics
    correct_samples = [s for s in valid_samples if s["is_correct"]]
    num_correct = len(correct_samples)  # Fix: was len(correct-samples)

    # calculate average characters for correct and incorrect answers
    correct_chars = [len(s["output"]) for s in correct_samples]
    incorrect_samples = [s for s in valid_samples if not s["is_correct"]]
    incorrect_chars = [len(s["output"]) for s in incorrect_samples]  # Fix: missing list comprehension

    metrics = {
        "percent_correct": (num_correct / total_samples) * 100 if total_samples > 0 else 0,
        "pass_at_k": num_correct > 0,
        "majority_at_k": num_correct > (total_samples / 2),
        "avg_chars_correct": sum(correct_chars) / len(correct_chars) if correct_chars else 0,
        "avg_chars_incorrect": sum(incorrect_chars) / len(incorrect_chars) if incorrect_chars else 0
    }

    return metrics

async def get_model_answer(session: aiohttp.ClientSession, question: str, model_name: str,
                          sglang_semaphore: asyncio.Semaphore, num_samples: int,
                          is_arc: bool = False, wait_enabled: bool = False,
                          partial_response: Optional[str] = None,
                          server_process_ref: List[subprocess.Popen] = None) -> List[str]:
    """Get multiple model answers for a question asynchronously with server recovery"""
    url = f"http://{SGLANG_CONFIG['host']}:{SGLANG_CONFIG['port']}/v1/chat/completions"
    
    system_prompt = ARC_SYSTEM_PROMPT if is_arc else GSM8K_SYSTEM_PROMPT

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': question}
    ]

    # If this is a retry with wait functionality, add the partial response
    if wait_enabled and partial_response:
        messages.append({'role': 'assistant', 'content': partial_response})

    data = {
        'model': model_name,
        'messages': messages,
        'temperature': 1.0,
        'top_p': 100,
        'top_k': 50,
        'max_tokens': 8192,
        'n': num_samples if not partial_response else 1
    }

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            async with sglang_semaphore:
                async with session.post(url, json=data, timeout=120) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"Server error (status {response.status}): {error_text}")
                        
                        # If this is a server error and we have a server process reference, try to restart
                        if 500 <= response.status < 600 and server_process_ref and server_process_ref[0]:
                            if attempt < max_attempts - 1:
                                print(f"Server error detected, attempting to restart (attempt {attempt+1})")
                                # Kill old process
                                server_process_ref[0].terminate()
                                try:
                                    server_process_ref[0].wait(timeout=5)
                                except subprocess.TimeoutExpired:
                                    server_process_ref[0].kill()
                                
                                # Start new process
                                new_process = start_server(model_name)
                                if new_process:
                                    server_process_ref[0] = new_process
                                    time.sleep(5)  # Wait for server to be ready
                                    continue  # Try again
                        
                        return [""] * (num_samples if not partial_response else 1)

                    result = await response.json()

                    if "choices" not in result or not result["choices"]:
                        print("No choices in response", result)
                        return [""] * (num_samples if not partial_response else 1)

                    # Extract and return the actual model outputs
                    return [choice["message"]["content"] for choice in result["choices"]]

        except Exception as e:
            print(f"Error getting model answer (attempt {attempt+1}): {str(e)}")
            
            # If server connection failed and we have a server process reference, try to restart
            if "Cannot connect to host" in str(e) and server_process_ref and server_process_ref[0]:
                if attempt < max_attempts - 1:
                    print(f"Connection failed, attempting to restart server (attempt {attempt+1})")
                    # Kill old process if it's still running
                    if server_process_ref[0].poll() is None:
                        server_process_ref[0].terminate()
                        try:
                            server_process_ref[0].wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            server_process_ref[0].kill()
                    
                    # Start new process
                    new_process = start_server(model_name)
                    if new_process:
                        server_process_ref[0] = new_process
                        time.sleep(5)  # Wait for server to be ready
                        continue  # Try again
            
            if attempt == max_attempts - 1:
                return [""] * (num_samples if not partial_response else 1)
            
            # Wait before retrying
            await asyncio.sleep(2)

def check_arc_answer(question: Dict, model_answer: str, ground_truth: str, no_think_tags: bool = False) -> Dict:
    """Check if an ARC answer is correct."""
    # Basic implementation - this would need to be expanded for proper ARC evaluation
    result = {
        "is_correct": False,
        "extracted_answer": None,
    }

    # Extract answer grid if possible
    try:
        # This is a simplified placeholder - actual implementation would be more complex
        # and would compare the extracted grid with the ground truth
        if "<think>" in model_answer and "</think>" in model_answer:
            # Extract content between think tags
            match = re.search(r'<think>(.*?)</think>', model_answer, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                # Simple check if the extracted answer contains the ground truth
                result["is_correct"] = ground_truth in extracted
                result["extracted_answer"] = extracted
        elif no_think_tags:
            # If think tags aren't required, check if answer contains ground truth
            result["is_correct"] = ground_truth in model_answer
            result["extracted_answer"] = model_answer
    except Exception as e:
        print(f"Error checking ARC answer: {str(e)}")

    return result

def format_arc_question(question: str, output: Optional[str] = None) -> str:
    """Format an ARC question with optional output grid."""
    formatted_question = f"Question: {question}\n\n"

    if output:
        formatted_question += f"Please solve this question and provide the output as a grid format similar to:\n{output}"
    else:
        formatted_question += "Please solve this question and provide your answer as a grid."

    return formatted_question

def load_arc_dataset(file_path: str, subset_ids: Optional[List[str]] = None) -> Tuple[List[Dict], int]:
    """Load ARC dataset from file with optional subset filtering."""
    try:
        with open(file_path, 'r') as f:
            dataset = json.load(f)

        # Filter by subset_ids if provided
        if subset_ids:
            dataset = [item for item in dataset if str(item.get("task_id", "")) in subset_ids]

        return dataset, len(dataset)
    except Exception as e:
        print(f"Error loading ARC dataset: {e}")
        return [], 0

def start_server(model_name: str, max_retries: int = 3) -> subprocess.Popen:
    """Start SGLang server for the specified model with retries and verbose logging."""
    for attempt in range(max_retries):
        try:
            cmd = [
                "python", "-m", "sglang.launch_server",
                "--model-path", model_name,  # Changed from --model to --model-path
                "--port", str(SGLANG_CONFIG["port"]),
                "--host", SGLANG_CONFIG["host"]
            ]

            print(f"Starting SGLang server (attempt {attempt+1}/{max_retries}) with command: {' '.join(cmd)}")
            # Capture output but also send to console for debugging
            server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Create threads to monitor output in real-time
            def print_output(stream, prefix):
                for line in iter(stream.readline, ''):
                    print(f"{prefix}: {line.strip()}")
                    
            import threading
            stdout_thread = threading.Thread(target=print_output, args=(server_process.stdout, "SERVER OUT"))
            stderr_thread = threading.Thread(target=print_output, args=(server_process.stderr, "SERVER ERR"))
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()

            # Wait for server to start - increased wait time for larger models
            start_wait = 30  # Increased from 10 to 30 seconds
            print(f"Waiting {start_wait} seconds for server to initialize...")
            time.sleep(start_wait)

            # Check if server process is still running
            if server_process.poll() is not None:
                print(f"Server process exited with code: {server_process.returncode}")
                if attempt < max_retries - 1:
                    print("Retrying...")
                    continue
                return None

            # Test connection to server before returning
            try:
                import requests
                health_url = f"http://{SGLANG_CONFIG['host']}:{SGLANG_CONFIG['port']}/v1/models"
                print(f"Testing server with request to: {health_url}")
                response = requests.get(health_url, timeout=10)
                if response.status_code == 200:
                    print("Server started successfully and responding to requests")
                    print(f"Server response: {response.text}")
                    return server_process
                else:
                    print(f"Server started but health check failed with status {response.status_code}: {response.text}")
                    if attempt < max_retries - 1:
                        print("Killing server and retrying...")
                        server_process.terminate()
                        time.sleep(5)  # Increased cool-down time
                        continue
            except requests.exceptions.RequestException as e:
                print(f"Server health check failed: {e}")
                print("Checking if server is still running...")
                if server_process.poll() is None:
                    print("Server process is still running despite connection failure")
                else:
                    print(f"Server process has exited with code: {server_process.returncode}")
                
                if attempt < max_retries - 1:
                    print("Killing server and retrying...")
                    try:
                        server_process.terminate()
                        server_process.wait(timeout=5)
                    except Exception as kill_error:
                        print(f"Error killing server: {kill_error}")
                    time.sleep(5)  # Increased cool-down time
                    continue
            
            # If we got here but couldn't confirm health, return process anyway
            print("Returning server process without confirming health")
            return server_process
            
        except Exception as e:
            print(f"Error starting server: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(5)  # Increased cool-down time
            else:
                return None

    return None

# Fix the process_questions function by removing the nested function and fixing indentation
async def process_questions(dataset: List[Dict[str, str]],
                            model_name: str,
                            num_samples: int = 8,
                            wait: bool = False,
                            think_tags: bool = False,
                            suppress_logs: bool = False,
                            server_process_ref: List[subprocess.Popen] = None) -> Tuple[List[Dict[str, Any]], int, Optional[Dict]]:
    """Process all questions asynchronously and return results."""
    # Initialize rate limiters using semaphores
    sglang_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SGLANG)
    gemini_semaphore = asyncio.Semaphore(MAX_CONCURRENT_GEMINI)

    # Initialize answer checker with Gemini semaphore - no_think_tags is the opposite of think_tags
    checker = AnswerChecker(semaphore=gemini_semaphore, no_think_tags=not think_tags)

    # Track successful wait retries
    wait_successes = 0
    results = []

    # Create HTTP session for all requests
    async with aiohttp.ClientSession() as session:
        for example in tqdm(dataset, desc="Processing questions"):
            is_arc = "output" in example and "input" in example  # Check if it's ARC format

            if is_arc:
                # Format ARC question with training examples
                question = format_arc_question(example["input"], example.get("output"))
                model_answers = await get_model_answer(
                    session, question, model_name, sglang_semaphore,
                    num_samples, is_arc=True, wait_enabled=wait,
                    server_process_ref=server_process_ref
                )

                # Process each sample
                samples = []
                for answer in model_answers:
                    check_result = check_arc_answer(
                        question=example,
                        model_answer=answer,
                        ground_truth=example["output"],
                        no_think_tags=not think_tags
                    )

                    # If wait is enabled and answer is incorrect, but has valid think tags, try again
                    if wait and not check_result["is_correct"] and has_single_think_tag_pair(answer):
                        wait_successes += 1
                        partial_response = extract_partial_response(answer, suppress_logs=suppress_logs)
                        retry_answers = await get_model_answer(
                            session, question, model_name, sglang_semaphore,
                            num_samples=1, is_arc=True, wait_enabled=True,
                            partial_response=partial_response,
                            server_process_ref=server_process_ref
                        )

                        if retry_answers and retry_answers[0]:
                            combined_answer = partial_response + retry_answers[0]
                            retry_check = check_arc_answer(
                                question=example,
                                model_answer=combined_answer,
                                ground_truth=example["output"],
                                no_think_tags=not think_tags
                            )

                            if retry_check["is_correct"]:
                                check_result = retry_check
                                answer = combined_answer

                    sample = {
                        "output": answer,
                        "is_correct": check_result["is_correct"]
                    }

                    # Only include extracted answer if grid was successfully extracted
                    if check_result["extracted_answer"]:
                        sample["extracted_answer"] = check_result["extracted_answer"]

                    samples.append(sample)

                result = {
                    "task_id": example.get("task_id", "unknown"),
                    "prompt": question,
                    "input": example["input"],
                    "ground_truth": example["output"],
                    "samples": samples
                }

            else:
                # Process GSM8K question
                model_answers = await get_model_answer(
                    session, example["question"], model_name, sglang_semaphore,
                    num_samples, wait_enabled=wait,
                    server_process_ref=server_process_ref
                )

                # Process each sample
                samples = []
                for answer in model_answers:
                    check_result = await checker.check_answer_async(
                        question=example["question"],
                        model_answer=answer,
                        ground_truth=example["answer"]
                    )

                    # If wait is enabled and answer is incorrect but has valid think tags, try again
                    if wait and not check_result["is_correct"] and has_single_think_tag_pair(answer):
                        partial_response = extract_partial_response(answer, suppress_logs=suppress_logs)
                        retry_answers = await get_model_answer(
                            session, example["question"], model_name, sglang_semaphore,
                            num_samples=1, wait_enabled=True, partial_response=partial_response,
                            server_process_ref=server_process_ref
                        )

                        if retry_answers and retry_answers[0]:
                            # Combine partial response with new completion
                            combined_answer = partial_response + retry_answers[0]
                            # Check combined answer
                            retry_result = await checker.check_answer_async(
                                question=example["question"],
                                model_answer=combined_answer,
                                ground_truth=example["answer"]
                            )

                            if retry_result["is_correct"]:
                                wait_successes += 1
                                check_result = retry_result
                                answer = combined_answer

                    samples.append({
                        "output": answer,
                        "is_correct": check_result["is_correct"]
                    })

                result = {
                    "question": example["question"],
                    "answer": example["answer"],
                    "samples": samples
                }

            # Calculate metrics for given question
            result["metrics"] = calculate_metrics(result["samples"])

            # Only print metrics if not suppressing logs
            if not suppress_logs:
                print(f"\nQuestion {len(results)} metrics:")
                for metric, value in result["metrics"].items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.2f}")
                    else:
                        print(f"  {metric}: {value}")

            results.append(result)

    # Return the results along with wait success count and any Gemini stats
    gemini_stats = checker.stats if hasattr(checker, "stats") else None
    return results, wait_successes, gemini_stats

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run batch inference on datasets.")
    parser.add_argument("--model", type=str, required=True, help="Model name to use")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "arc"],
                        help="Dataset to use (gsm8k or arc)")
    parser.add_argument("--gsm8k-file", type=str, default="data/gsm8k.json",
                        help="Path to GSM8K dataset file")
    parser.add_argument("--arc-file", type=str, help="Path to ARC dataset file")
    parser.add_argument("--subset-file", type=str, help="Path to subset IDs file")
    parser.add_argument("--num-samples", type=int, default=8,
                        help="Number of samples per question")
    parser.add_argument("--wait", action="store_true", help="Enable wait functionality")
    parser.add_argument("--think-tags", action="store_true", default=True, 
                        help="Require think tags in answers")
    parser.add_argument("--suppress-logs", action="store_true", 
                        help="Suppress detailed logging")
    return parser.parse_args()

async def main():
    args = parse_args()
    model_name = args.model

    # Create model specific results directory with dataset details
    model_short_name = model_name.split('/')[-1]
    if args.dataset == 'gsm8k':
        dataset_name = os.path.splitext(args.gsm8k_file)[0]  # Fixed: was splittext
    else:
        dataset_name = os.path.splitext(args.arc_file)[0] if args.arc_file else 'arc'  # Fixed incorrect variable
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
            print(f"Could not find GSM8K dataset file: {args.gsm8k_file}")
            print("Please run download_gsm8k.py first to create the dataset file.")
            sys.exit(1)
    else:  # ARC dataset
        if not args.arc_file:
            print("Please specify the ARC dataset file using the --arc-file argument.")
            sys.exit(1)

        try:
            dataset, _  = load_arc_dataset(args.arc_file, subset_ids)
        except FileNotFoundError:
            print(f"Could not find ARC dataset file: {args.arc_file}")
            sys.exit(1)

    # Start Server
    server_process = start_server(model_name)
    if not server_process:
        sys.exit(1)
    
    # Create a mutable reference to the server process
    server_process_ref = [server_process]

    try:
        # Process all questions
        results, wait_successes, gemini_stats = await process_questions(
            dataset,
            model_name,
            num_samples=args.num_samples,
            wait=args.wait,
            think_tags=args.think_tags,
            suppress_logs=args.suppress_logs,
            server_process_ref=server_process_ref  # Pass server process reference
        )

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
        avg_incorrect_length = sum(all_incorrect_lengths) / len(all_incorrect_lengths) if all_incorrect_lengths else 0

        # Prepare final results
        final_results = {
            "model": model_name,
            "dataset": args.dataset,
            "subset_info": f"Using subset of {len(subset_ids)} problems" if subset_ids else None,
            "samples_per_question": args.num_samples,
            "wait_enabled": args.wait,
            "wait_successes": wait_successes if args.wait else None,
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

        # Print final statistics
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
        print(f"Pass@{args.num_samples} (at least 1 correct): {pass_at_k_count}/{total_count} ({(pass_at_k_count / total_count) * 100:.2f}%)")
        print(f"Majority@{args.num_samples} (majority correct): {majority_count}/{total_count} ({(majority_count / total_count) * 100:.2f}%)")
        print(f"\nResponse Lengths:")
        print(f"Average Lengths: {avg_correct_length:.2f} (correct), {avg_incorrect_length:.2f} (incorrect)")

    finally:
        # Make sure to terminate the server process when done
        if server_process_ref and server_process_ref[0]:
            print("Shutting down server...")
            server_process_ref[0].terminate()
            try:
                server_process_ref[0].wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process_ref[0].kill()

if __name__ == "__main__":
    asyncio.run(main())