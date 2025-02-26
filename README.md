### Setup 

> Were gonna go for vast for now but considering other affordable GPU Cloud instances like glow


1. Install dependencies


```bash

pip install --upgrade pip
pip install uv

uv pip install -r requirements.txt --system

# install SGLang components

uv pip install  sgl-kernel --force-reinstall --no-deps--system
uv-pip install "sglang[all] --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/ --system
```


2. Setup environment variables
```


### downloading the hf dataset


download_gsm8k.py


### downloading data samples

python3 download_gsm8k.py --split test --num_examples 100

python3 download_gsm8k.py --split test --num_examples 16

python3 download_gsm8k.py --split train --num_examples 7473

#python3 download_gsm8k.py --split train --num_examples -1


#python3 download_gsm8k.py --split test --num_examples -1


```


## Answer Checkers

The nature of your answer checker depends on your dataset. Some datasets are conducive to exact match checks (like ARC) while some may require LLM as a judge.

As templates in this repo, you have:

`utils/answer_checker.py` - A utility that uses Gemini 1.5 Flash to assess math answers. Features:
- Extracts answers from model output and ground truth
- Checks for exact match and falls back to Gemini for a more detailed check if there's no match
- Retries on API failures

To test the answer checker, run:

```bash
python3 test-scripts/test_gemini_flash.py

This will run several test cases:

- Simple addition problem
- Complex multi-step problem
- Incorrect answer scenario
- ARC relies on exact match and is done using arc_answer_checker.py
```

> Now we will run `run_inference.py` - Runs an SGLang server (faster than vLLM for >1 batch size) with DeepSeek R1 Distill Quen 1.5B and performs inference on the downloaded examples. 

```bash


# install dependencies 

pip install sglang torch transformers deepseek-ai

# run inference on the GSM8K dataset

python3 run_inference.py --datasetgsm8k --gsm8k-file gsm8k_test_100.json



