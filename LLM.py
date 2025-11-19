#!/usr/bin/env python3
import os
import subprocess
import sys

VENV_DIR = "venv"
REQ_FILE = "requirements.txt"

def create_venv():
    if not os.path.isdir(VENV_DIR):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
    else:
        print("Virtual environment already exists.")

def install_packages():
    """
    Installs packages using the current Python executable (venv or system).
    """
    python_bin = sys.executable  # use current Python

    
    # Pip is upgraded here to make sure the instalation is stable.
    print(f"Upgrading pip using {python_bin}...")
    subprocess.check_call([python_bin, "-m", "pip", "install", "--upgrade", "pip"])

    # Install requirements if they exist
    if os.path.isfile(REQ_FILE):
        print(f"Installing packages from {REQ_FILE} using {python_bin}...")
        subprocess.check_call([python_bin, "-m", "pip", "install", "-r", REQ_FILE])
    else:
        print(f"{REQ_FILE} not found. Make sure it exists in the project root.")

# This function prepares the model used
def load_model(model_name="gpt2"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

# This function translates the given prompt, into values the model can understand
def tokenize_prompt(prompt, tokenizer):
    return tokenizer(prompt, return_tensors="pt")

# This function extracts the resulting text
def generate_text(model, inputs, max_new_tokens=50, temperature=1.0, top_k=50):
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=1.2,  # penalize repeated phrases
        pad_token_id=model.config.eos_token_id,  # stop at EOS
        do_sample=True  # sampling prevents deterministic looping
    )
    return outputs

def decode_tokens(outputs, tokenizer, prompt_length=None):
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if prompt_length:
        # Remove the prompt from the output
        text = text[prompt_length:]
    return text.strip()

def run_llm_example():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    model_name = "google/flan-t5-small"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("CPU Q&A LLM ready! Type 'exit' to quit.")

    while True:
        prompt = input("Enter your prompt: ")
        if prompt.lower() in ["exit", "quit"]:
            print("Exiting LLM...")
            break

        # Instruction-tuned prompt
        input_text = f"Answer the question : {prompt}"

        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"LLM: {response}\n")

# ---------------- Main Script ---------------- #
def main():
    inside_venv = sys.prefix != sys.base_prefix
    if not inside_venv:
        create_venv()
        # Re-run inside the venv
        python_bin = os.path.join(VENV_DIR, "bin", "python") if os.name == "posix" else os.path.join(VENV_DIR, "Scripts", "python.exe")
        print("Re-running script inside virtual environment...")
        subprocess.check_call([python_bin, __file__])
        sys.exit(0)

    # Already inside venv
    install_packages()
    run_llm_example()


if __name__ == "__main__":
    main()
