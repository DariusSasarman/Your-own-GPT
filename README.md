# Your-own-GPT
---

Here's a demo of it working:

[Screencast from 2025-11-19 08-22-24.webm](https://github.com/user-attachments/assets/2094e700-9b0b-4c04-bcfa-acd388bc84b4)

---

## CPU-Based LLM Python Script 

I made this to be used as a 1-hour presentation support, where students would learn to use a Large Language model, and a little about how it all works in the background.

It's meant to be "drag, drop and run" by taking the two files mentioned lower in this README.md page.

---

## How to Run

1. Make sure Python 3.8+ is installed.
2. Install LLM.py and requirements.txt in the same folder, in your computer
3. Run the following command, in the comand-prompt:

```bash
python LLM.py
```

The explanaitons onward are mostly A.I.-generated based on the given ".py" file.

---

## Overview

The script automatically:

1. Creates a **virtual environment** (`venv`) if it doesn't exist.
2. Upgrades `pip` and installs required Python packages from `requirements.txt`.
3. Loads a **pre-trained CPU-friendly LLM** using Hugging Face Transformers.
4. Lets you interact with the model via the command line.

## Function List and Explanations

### `create_venv()`
- **Purpose:** Creates a virtual environment (`venv`) if it doesn’t already exist.
- **How it works:** Checks for the `venv` folder. If missing, it uses Python's built-in `venv` module to create an isolated Python environment, keeping project dependencies separate from system Python.

---

### `install_packages()`
- **Purpose:** Installs all required Python packages in the current environment.
- **How it works:**  
  1. Upgrades `pip` to the latest version.  
  2. Installs packages listed in `requirements.txt` if the file exists.  
  - Works with both the system Python and the virtual environment Python.

---

### `load_model(model_name="gpt2")`
- **Purpose:** Loads a pre-trained language model and tokenizer.
- **How it works:**  
  - Loads a tokenizer to convert text into tokens the model understands.  
  - Loads a causal language model (or another model) from Hugging Face.  
  - Returns both the tokenizer and model for later use.

---

### `tokenize_prompt(prompt, tokenizer)`
- **Purpose:** Converts a string prompt into input tokens for the model.
- **How it works:** Uses the tokenizer to transform user text into numerical tensors suitable for model processing.

---

### `generate_text(model, inputs, max_new_tokens=50)`
- **Purpose:** Generates text from the model based on input tokens.
- **How it works:**  
  - Uses parameters like `temperature`, `top_k`, and `repetition_penalty` to control output randomness and prevent repeated phrases.  
  - Returns the model's generated token IDs.

---

### `decode_tokens(outputs, tokenizer, prompt_length=None)`
- **Purpose:** Converts model output tokens back into readable text.
- **How it works:**  
  - Decodes the generated token IDs.  
  - Optionally removes the original prompt from the output for cleaner results.

---

### `run_llm_example()`
- **Purpose:** Provides an interactive command-line interface for the user to communicate with the LLM.
- **How it works:**  
  - Loads a small CPU-friendly language model (e.g., Flan-T5 small).  
  - Continuously prompts the user for input.  
  - Generates and prints responses.  
  - User can type `exit` or `quit` to end the session.

---

### `main()`
- **Purpose:** Coordinates the script execution and ensures it runs in a virtual environment.
- **How it works:**  
  1. Checks if the script is already running inside a virtual environment.  
  2. If not, creates the virtual environment and restarts the script inside it.  
  3. If yes, installs dependencies and runs the interactive LLM example.

---

## Execution Flow Summary

1. `main()` is called when the script runs.  
2. Virtual environment is checked and created if necessary.  
3. Dependencies are installed.  
4. Language model and tokenizer are loaded.  
5. User enters prompts → LLM generates answers.  
6. Loop continues until the user exits.

---

## Notes

- The first run may take time to download model weights.  
- Script is CPU-friendly; no GPU required.  
- The virtual environment keeps project dependencies separate and can be reset by deleting `venv/`.  
- `requirements.txt` can be updated to include additional Python packages.
