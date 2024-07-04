# Ollama-MMLU-Pro

This is a modified version of [TIGER-AI-Lab/MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro), and it lets you run [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) benchmark via the OpenAI Chat Completion API. It's tested on Ollama and Llama.cpp, but it should also work with LMStudio, Koboldcpp, Oobabooga with openai extension, etc.

<a href="https://colab.research.google.com/github/chigkim/Ollama-MMLU-Pro/blob/main/Ollama_MMLU_Pro.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Usage

Change the config.toml according to your setup.

```shell
pip install -r requirements.txt
python run_openai.py
```
