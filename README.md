# Ollama-MMLU-Pro

This is a modified version of [TIGER-AI-Lab/MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro), and it lets you run [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) benchmark with any openai compatible api.

<a href="https://colab.research.google.com/github/chigkim/Ollama-MMLU-Pro/blob/main/Ollama_MMLU_Pro.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Usage

For example, in order to run benchmark against Phi3 on Ollama, use:

```shell
pip install -r requirements.txt
python run_openai.py --url http://localhost:11434/v1 --model phi3
```

As default, it tests against all subjects, but you can use --category option to test only specific subject.

Subjects include: 'business', 'law', 'psychology', 'biology', 'chemistry', 'history', 'other', 'health', 'economics', 'math', 'physics', 'computer science', 'philosophy', 'engineering'

### Parallelism

You can optionally run multiple tests in parallel by using --parallel option. For example, to run 2 tests in parallel:

```shell
python run_openai.py --url http://localhost:11434/v1 --model llama3 --parallel 2
```
