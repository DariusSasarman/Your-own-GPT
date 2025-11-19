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
    Always upgrades pip.
    """
    python_bin = sys.executable  # use current Python

    # Upgrade pip
    print(f"Upgrading pip using {python_bin}...")
    subprocess.check_call([python_bin, "-m", "pip", "install", "--upgrade", "pip"])

    # Install requirements if they exist
    if os.path.isfile(REQ_FILE):
        print(f"Installing packages from {REQ_FILE} using {python_bin}...")
        subprocess.check_call([python_bin, "-m", "pip", "install", "-r", REQ_FILE])
    else:
        print(f"{REQ_FILE} not found. Make sure it exists in the project root.")

# ---------------- LLM Helper Functions ---------------- #
def load_model(model_name="gpt2"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def tokenize_prompt(prompt, tokenizer):
    return tokenizer(prompt, return_tensors="pt")

def generate_text(model, inputs, max_new_tokens=50, temperature=1.0, top_k=50):
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k
    )
    return outputs

def decode_tokens(outputs, tokenizer):
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_llm_example():
    import torch
    try:
        model, tokenizer = load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("CPU LLM ready! Type 'exit' to quit.")
    conversation_history = ""  # Optional: keep context for multi-turn

    while True:
        prompt = input("Enter your prompt (or write \"exit\" to exit): ")
        if prompt.lower() in ["exit", "quit"]:
            print("Exiting LLM...")
            break

        # Optionally append previous conversation for context
        full_prompt = conversation_history + prompt

        inputs = tokenize_prompt(full_prompt, tokenizer)
        outputs = generate_text(model, inputs, max_new_tokens=100)
        response = decode_tokens(outputs, tokenizer)

        print(f"LLM: {response}\n")

        # Update conversation history for multi-turn (optional)
        conversation_history += prompt + " " + response + " "

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
