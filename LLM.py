#!/usr/bin/env python3
import os
import subprocess
import sys
import logging

VENV_DIR = "venv"
REQ_FILE = "requirements.txt"
SETUP_FLAG = os.path.join(VENV_DIR, ".setup_done")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_venv():
    """Create a virtual environment if it doesn't exist."""
    if not os.path.isdir(VENV_DIR):
        logger.info("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
    else:
        logger.info("Virtual environment already exists.")


def install_packages():
    """Install packages once inside the venv."""
    python_bin = sys.executable

    logger.info(f"Upgrading pip using {python_bin}...")
    subprocess.check_call([python_bin, "-m", "pip", "install", "--upgrade", "pip"])

    if os.path.isfile(REQ_FILE):
        logger.info(f"Installing packages from {REQ_FILE} using {python_bin}...")
        subprocess.check_call([python_bin, "-m", "pip", "install", "-r", REQ_FILE])
    else:
        logger.warning(f"{REQ_FILE} not found.")


def load_model(model_name="google/flan-t5-small"):
    """Load the Flan-T5 model and tokenizer."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    logger.info(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    logger.info("Model loaded successfully!")
    return model, tokenizer


def search_wikipedia(query):
    """Search Wikipedia API for relevant information."""
    import requests
    
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 3,
        "srprop": "snippet"
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("query", {}).get("search", []):
            # Remove HTML tags from snippet
            import re
            snippet = re.sub(r'<[^>]+>', '', item.get("snippet", ""))
            results.append({
                "title": item.get("title", ""),
                "snippet": snippet
            })
        
        return results
    except Exception as e:
        logger.error(f"Wikipedia search failed: {e}")
        return []


def search_duckduckgo(query):
    """Fetch search results from DuckDuckGo API."""
    import requests
    
    url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1
    }
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        results = []
        
        # Try Abstract first
        if data.get("AbstractText"):
            results.append(data["AbstractText"])
        
        # Extract from RelatedTopics
        related = data.get("RelatedTopics", [])
        for r in related:
            if "Text" in r:
                results.append(r["Text"])
            elif "Topics" in r:
                for t in r["Topics"]:
                    if "Text" in t:
                        results.append(t["Text"])
        
        return results
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")
        return []


def build_prompt_with_context(query, context_info):
    """
    Build an optimized prompt for Flan-T5.
    Flan-T5 works best with clear instructions and structured input.
    """
    if context_info:
        prompt = f"""Answer the following question using the provided context. Be concise and factual.
        Context:
        {context_info}

        Question: {query}
        Answer:"""
    else:
        # No context available - use direct question format
        prompt = f"""Answer this question concisely and factually:

        Question: {query}
        Answer:"""
    
    return prompt


def get_search_context(query, max_chars=800):
    """
    Get search context from multiple sources.
    Try Wikipedia first (more reliable), then DuckDuckGo.
    """
    context_parts = []
    
    # Try Wikipedia first
    wiki_results = search_wikipedia(query)
    if wiki_results:
        logger.info(f"Found {len(wiki_results)} Wikipedia results")
        for result in wiki_results[:2]:  # Use top 2 results
            context_parts.append(f"{result['title']}: {result['snippet']}")
    
    # If Wikipedia didn't return enough, try DuckDuckGo
    if len(context_parts) < 2:
        ddg_results = search_duckduckgo(query)
        if ddg_results:
            logger.info(f"Found {len(ddg_results)} DuckDuckGo results")
            context_parts.extend(ddg_results[:3])
    
    if not context_parts:
        logger.warning("No search results found from any source")
        return None
    
    # Join and truncate to max_chars
    context = "\n".join(context_parts)
    if len(context) > max_chars:
        context = context[:max_chars] + "..."
    
    return context


def is_math_query(query):
    """Detect if query is a simple math problem."""
    import re
    # Look for patterns like "2+2", "5*3", "what is 10-5"
    math_pattern = r'\d+\s*[\+\-\*\/]\s*\d+'
    return bool(re.search(math_pattern, query))


def solve_math(query):
    """Solve simple math expressions."""
    import re
    
    # Extract the math expression
    match = re.search(r'(\d+)\s*([\+\-\*\/])\s*(\d+)', query)
    if not match:
        return None
    
    try:
        num1 = float(match.group(1))
        op = match.group(2)
        num2 = float(match.group(3))
        
        if op == '+':
            result = num1 + num2
        elif op == '-':
            result = num1 - num2
        elif op == '*':
            result = num1 * num2
        elif op == '/':
            if num2 == 0:
                return "Cannot divide by zero"
            result = num1 / num2
        else:
            return None
        
        # Format result nicely
        if result == int(result):
            return str(int(result))
        else:
            return f"{result:.2f}"
    except:
        return None


def tokenize_prompt(prompt, tokenizer, max_length=512):
    """Tokenize the prompt with truncation."""
    return tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )


def generate_text(model, inputs, max_new_tokens=128):
    """
    Generate text from the model.
    Using greedy decoding for factual questions.
    """
    return model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Greedy decoding for factual answers
        num_beams=4,      # Beam search for better quality
        early_stopping=True
    )


def decode_tokens(outputs, tokenizer):
    """Decode model output tokens to text."""
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def run_llm_example():
    """Main LLM interaction loop."""
    model_name = "google/flan-t5-small"

    try:
        model, tokenizer = load_model(model_name)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)

    print("\n" + "="*60)
    print("🤖 CPU Q&A LLM with RAG Ready!")
    print("="*60)
    print("Commands:")
    print("  • Type your question to get an answer")
    print("  • Type 'exit' or 'quit' to stop")
    print("  • Type 'help' for more information")
    print("="*60 + "\n")

    while True:
        try:
            query = input("You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ["exit", "quit"]:
                print("\n👋 Goodbye!")
                break
            
            if query.lower() == "help":
                print("\nThis LLM uses RAG (Retrieval-Augmented Generation):")
                print("• Searches Wikipedia and DuckDuckGo for information")
                print("• Uses Flan-T5-small to generate answers")
                print("• Can handle math questions directly")
                print("• Best for factual questions\n")
                continue
            
            logger.info(f"Processing query: {query}")
            
            # Check if it's a math question
            if is_math_query(query):
                answer = solve_math(query)
                if answer:
                    print(f"\n🤖 LLM: {answer}\n")
                    continue
            
            # Get search context
            context = get_search_context(query)
            
            # Build prompt
            prompt = build_prompt_with_context(query, context)
            
            # Tokenize and generate
            inputs = tokenize_prompt(prompt, tokenizer)
            outputs = generate_text(model, inputs, max_new_tokens=128)
            response = decode_tokens(outputs, tokenizer)

            # Post-process response
            if not response or len(response) < 3:
                response = "I don't have enough information to answer that question confidently."
            
            print(f"\n🤖 LLM: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            print(f"\n❌ Error: Unable to process your question. Please try again.\n")


def main():
    """Main entry point with venv setup."""
    inside_venv = sys.prefix != sys.base_prefix

    if not inside_venv:
        create_venv()
        python_bin = os.path.join(
            VENV_DIR, "bin", "python"
        ) if os.name == "posix" else os.path.join(VENV_DIR, "Scripts", "python.exe")

        logger.info("Re-running script inside virtual environment...")
        subprocess.check_call([python_bin, __file__])
        sys.exit(0)

    # Now inside the venv
    if not os.path.exists(SETUP_FLAG):
        install_packages()
        with open(SETUP_FLAG, "w") as f:
            f.write("")  # Create flag file

    run_llm_example()


if __name__ == "__main__":
    main()