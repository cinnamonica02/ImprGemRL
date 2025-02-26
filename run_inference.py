# utils/run_inference.py
import os
import json
import asyncio
from tqdm import tqdm
from sglang import RuntimeEndpoint
from datasets import load_dataset
from .config import InferenceConfig
from utils.answer_checker import AnswerChecker

class SGInferenceRunner:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.checker = AnswerChecker()
        self.client = RuntimeEndpoint(f"http://localhost:{self.config.sglang_port}")
        
        # Prompt template from DeepSeek's documentation
        self.prompt_template = (
            "Solve this math problem step by step. Put your final answer within \boxed{{}}.\n"
            "Question: {question}\n"
            "Answer: Let's think step by step."
        )

    async def start_server(self):
        """Start the SGLang server in background"""
        os.system(
            f"sglang launch --model-path {self.config.model_name} "
            f"--port {self.config.sglang_port} "
            f"--tokenizer-mode auto &"
        )
        await asyncio.sleep(20)  # Wait for server initialization

    async def generate_batch(self, questions: list[str]) -> list[str]:
        """Generate answers for a batch of questions"""
        prompts = [self.prompt_template.format(question=q) for q in questions]
        
        responses = await self.client.run_batch(
            prompts=prompts,
            temperature=self.config.temperature,
            max_length=self.config.max_length,
            batch_size=self.config.batch_size
        )
        
        return [r.strip() for r in responses]

    async def process_dataset(self):
        """Main processing pipeline"""
        # Load dataset
        dataset = load_dataset(
            self.config.dataset_path,
            self.config.dataset_subset,
            split=self.config.dataset_split
        )
        
        if self.config.sample_size:
            dataset = dataset.select(range(self.config.sample_size))
            
        # Process in batches
        results = []
        questions = [ex["question"] for ex in dataset]
        
        with tqdm(total=len(questions), desc="Processing batches") as pbar:
            for i in range(0, len(questions), self.config.batch_size):
                batch_questions = questions[i:i+self.config.batch_size]
                batch_answers = await self.generate_batch(batch_questions)
                
                # Validate answers
                for j, (question, answer) in enumerate(zip(batch_questions, batch_answers)):
                    gt = dataset[i+j]["answer"]
                    check_result = await self.checker.check_answer_async(
                        question=question,
                        model_answer=answer,
                        ground_truth=gt
                    )
                    
                    results.append({
                        "question": question,
                        "model_answer": answer,
                        "ground_truth": gt,
                        "is_correct": check_result["is_correct"],
                        "check_method": check_result["check_method"]
                    })
                
                pbar.update(len(batch_questions))
        
        # Save results
        with open(self.config.results_file, "w") as f:
            json.dump(results, f, indent=2)
            
        # Print summary
        accuracy = sum(r["is_correct"] for r in results) / len(results)
        print(f"\nFinal Accuracy: {accuracy:.2%}")
        return results

async def main():
    config = InferenceConfig(
        sample_size=100  # Remove for full dataset
    )
    
    runner = SGInferenceRunner(config)
    await runner.start_server()
    try:
        await runner.process_dataset()
    finally:
        # Clean up server
        os.system("pkill -f sglang")

if __name__ == "__main__":
    asyncio.run(main())