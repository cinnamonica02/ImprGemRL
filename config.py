"""Configuration settings for the RL agent."""



 # System prompts for different tasks 

GSM8K_SYSTEM_PROMPT = """You are a helpdul math assistant. Solve problems steps by step using the following commands:
 
    1. Put your step-by-step reasoning inside <think> tags 
    2. After your reasoning, provide the final numerical answer
    3. End with final answer in a clear format


    Example:
    <think>
    1. First, I need to add 3 and 4 
    2. 3 + 4 = 7
    3. Then multiply by 2
    4. 7 * 2 = 14 
    </think> 
    The answer is 14. 

    #### 14""" 


ARC_SYSTEM_PROMPT = """ You are a helpful assistant that solves ARC (Abstraction and Reasoning Challenge) problems.

    Given input grids and their corresponding output grids, indentify the patterns and solve the problem.


    Example:
    <think>
    1. The output grid is the same as the input grid
    2. The output grid is rotated 90 degrees clockwise
    3. The output grid is rotated 180 degrees clockwise
    4. The output grid is rotated 270 degrees clockwise
    </think>
    The answer is 90 degrees clockwise rotation.

    #### 90 degrees clockwise rotation"""


# SGLang server configuration
SGLANG_CONFIG = {
    "host": "localhost",
    "port": 8000,  # Default port for SGLang server
    "timeout": 120,  # Timeout in seconds for API requests
    "max_retries": 3  # Maximum number of retries for failed API requests
}

# Rate limiting settings
MAX_CONCURRENT_SGLANG = 5  # Maximum no. of concurrent requests to SGLang
MAX_CONCURRENT_GEMINI = 10  # Maximum no. of concurrent requests to Gemini

# Retry settings
RETRY_DELAY = 1  # Delay in seconds between retries

# Model configuration defaults
DEFAULT_MODEL_CONFIG = {
    "temperature": 1.0,
    "top_p": 100,
    "top_k": 50,
    "max_tokens": 8192
}

# Paths for data and results
DEFAULT_PATHS = {
    "gsm8k_data": "data/gsm8k.json",
    "arc_data": "data/arc",
    "results": "results"
}

# Think tag detection regex patterns
THINK_TAG_OPEN = r"<think>"
THINK_TAG_CLOSE = r"</think>"
THINK_TAG_PATTERN = r"<think>(.*?)</think>"

# Answer extraction patterns
FINAL_ANSWER_PATTERN = r"Final Answer:\s*(.*?)(?:\n|$)"
BOXED_ANSWER_PATTERN = r"\\boxed{([^}]*)}"
HASH_ANSWER_PATTERN = r"####\s*(.*?)(?:\n|$)"

# Success metrics thresholds
SIMILARITY_THRESHOLD = 0.85  # Threshold for answer similarity matching
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
