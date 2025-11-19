# Your-own-GPT
---

Here's a demo of it working:

[Screencast from 2025-11-19 08-22-24.webm](https://github.com/user-attachments/assets/2094e700-9b0b-4c04-bcfa-acd388bc84b4)

---

## CPU-Based LLM Python Script 

I made this to be used as a 1-hour presentation support, where students would learn to use a Large Language model, and a little about how it all works in the background.

---

## How to Run

1. Make sure Python 3.8+ is installed.
2. Install LLM.py and requirements.txt in the same folder, in your computer
3. Run the following command, in the comand-prompt:

```bash
python LLM.py
```

---

## Overview

The script automatically:

1. Creates a **virtual environment** (`venv`) if it doesn't exist.
2. Upgrades `pip` and installs required Python packages from `requirements.txt`.
3. Loads a **pre-trained CPU-friendly LLM** using Hugging Face Transformers.
4. Lets you interact with the model via the command line.

## Notes

- The first run may take time to download model weights.  
- Script is CPU-friendly; no GPU required.  
- The virtual environment keeps project dependencies separate and can be reset by deleting `venv/`.  
- `requirements.txt` can be updated to include additional Python packages.
