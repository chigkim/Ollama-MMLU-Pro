# Ollama-MMLU-Pro

This is a modified version of run_gpt4o.py from [TIGER-AI-Lab/MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro), and it lets you run [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) benchmark via the OpenAI Chat Completion API. It's tested on Ollama and Llama.cpp, but it should also work with LMStudio, Koboldcpp, Oobabooga with openai extension, etc.

<a href="https://colab.research.google.com/github/chigkim/Ollama-MMLU-Pro/blob/main/Ollama_MMLU_Pro.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Usage

Change the config.toml according to your setup.

```shell
pip install -r requirements.txt
python run_openai.py
```

## Notes

* If an answer cannot be extracted from the model's response, the script will randomly assign an answer.
* The total score represents the number of correct answers out of the total number of attempts. This is the score from the original script.
* "Random Guess Attempts" indicates the total number of random guesses out of the total attempts.
* "Correct Random Guesses" shows the number of random guesses that were correct out of all the random guesses.
* "Adjusted Score Without Random Guesses" subtracts all random guesses from the correct answers and the total answers.
* The combined total score in the last column of the table is calculated as: the total number of correct answers across all categories / the total number of all attempts across all categories * 100.
* All the   scores in percentage are rounded numbers.