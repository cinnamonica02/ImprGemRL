from dataclasses import dataclass

@dataclass
class InferenceConfig:
    model_name: str = "deepseek-ai/deepseek-r1-distill-quen-1.5b"
    batch_size: int = 8
    sglang_port: int = 30000
    max_length: int = 2048
    temperature: float = 0.1
    dataset_path: str = "openai/gsm8k"
    dataset_split: str = "test"
    dataset_subset: str = "main"
    sample_size: int = None  # Set to number for testing
    results_file: str = "inference_results.json"