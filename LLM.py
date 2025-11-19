#!/usr/bin/env python3
import os
import subprocess
import sys

VENV_DIR = "venv"
REQ_FILE = "requirements.txt"
SETUP_FLAG = os.path.join(VENV_DIR, ".setup_done")

def create_venv():
    if not os.path.isdir(VENV_DIR):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
    else:
        print("Virtual environment already exists.")


def install_packages():
    """Install packages once inside the venv."""
    python_bin = sys.executable

    print(f"Upgrading pip using {python_bin}...")
    subprocess.check_call([python_bin, "-m", "pip", "install", "--upgrade", "pip"])

    if os.path.isfile(REQ_FILE):
        print(f"Installing packages from {REQ_FILE} using {python_bin}...")
        subprocess.check_call([python_bin, "-m", "pip", "install", "-r", REQ_FILE])
    else:
        print(f"{REQ_FILE} not found.")


def load_model(model_name="google/flan-t5-small"):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer


def classify_prompt(prompt):
    prompt_lower = prompt.lower()

    succinct_indicators = ['what is', 'who is', 'when did', 'where is', 'how many', 'translate']
    if any(s in prompt_lower for s in succinct_indicators):
        if '?' in prompt and len(prompt.split()) < 10:
            return "succinct"

    verbose_indicators = ['explain', 'describe', 'walk me through', 'how do i', 'steps to']
    if any(s in prompt_lower for s in verbose_indicators):
        return "verbose"

    ambiguous_indicators = ['meaning of', 'best', 'should i', 'is', 'good or bad']
    if any(s in prompt_lower for s in ambiguous_indicators):
        return "ambiguous"

    return "ambiguous"


def RAG_prompt(prompt):
    import requests
    from bs4 import BeautifulSoup
    # from sklearn.feature_extraction.text import TfidfVectorizer
    # from sklearn.metrics.pairwise import cosine_similarity
    # later to improve this naive rag.
    
    q = prompt
    url = f"https://html.duckduckgo.com/html/?q={q}"

    prompt += "Context:"

    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
    soup = BeautifulSoup(html, "html.parser")

    for r in soup.select(".result__a"):
        if(len(prompt) <1800) :
            prompt += "\n" + r.get_text()
        else:
            break
    
    return prompt



def tokenize_prompt(prompt, tokenizer):
    return tokenizer(prompt, return_tensors="pt")


def generate_text(model, inputs, max_new_tokens=50, temperature=1.0, top_k=50):
    """Correct Flan-T5 generation (encoder-decoder)."""
    return model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        do_sample=True,
    )


def decode_tokens(outputs, tokenizer):
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def run_llm_example():
    model_name = "google/flan-t5-small"

    try:
        model, tokenizer = load_model(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("CPU Q&A LLM ready! Type 'exit' to quit.\n")

    while True:
        prompt = input("Enter your prompt: ")
        if prompt.lower() in ["exit", "quit"]:
            print("Exiting LLM...")
            break
        

        category = classify_prompt(prompt)

        prompt = RAG_prompt(prompt)

        if category == "succinct":
            input_text = f"Answer briefly: {prompt}"
            max_tokens = 50
        elif category == "verbose":
            input_text = f"Explain in detail: {prompt}"
            max_tokens = 200
        elif category == "ambiguous":
            input_text = f"Provide a thoughtful perspective on: {prompt}"
            max_tokens = 512

        inputs = tokenize_prompt(input_text, tokenizer)
        outputs = generate_text(model, inputs, max_new_tokens=max_tokens)
        response = decode_tokens(outputs, tokenizer)

        print(f"\nLLM: {response}\n")


# ---------------- Main Script ---------------- #
def main():
    inside_venv = sys.prefix != sys.base_prefix

    if not inside_venv:
        create_venv()
        python_bin = os.path.join(
            VENV_DIR, "bin", "python"
        ) if os.name == "posix" else os.path.join(VENV_DIR, "Scripts", "python.exe")

        print("Re-running script inside virtual environment...")
        subprocess.check_call([python_bin, __file__])
        sys.exit(0)

    # Now inside the venv
    if not os.path.exists(SETUP_FLAG):
        install_packages()
        open(SETUP_FLAG, "w").close()

    run_llm_example()


if __name__ == "__main__":
    main()
