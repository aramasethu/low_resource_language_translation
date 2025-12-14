#!/usr/bin/env python3
"""
Simple script to translate prompts to any language using Tower model with MPS optimization.
Model stays loaded in memory for fast repeated queries.

Usage:
    python scripts/simple_french_translation.py "Hello, how are you?" "French"
    python scripts/simple_french_translation.py "Bonjour" "English"
    python scripts/simple_french_translation.py  # Interactive mode
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys

# Model
TOWER_MODEL = "Unbabel/TowerInstruct-7B-v0.1"

# Determine device - use MPS on Apple Silicon for faster inference
if torch.backends.mps.is_available():
    device = "mps"
    print("Using MPS (Metal Performance Shaders) for Apple Silicon acceleration")
elif torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA")
else:
    device = "cpu"
    print("Using CPU")

# Load model once (keep in memory)
print(f"\nLoading Tower model ({TOWER_MODEL})...")
print("This may take a minute - model will stay loaded for fast queries...")

tower_tokenizer = AutoTokenizer.from_pretrained(TOWER_MODEL)
tower_model = AutoModelForCausalLM.from_pretrained(
    TOWER_MODEL,
    device_map=device if device != "mps" else None,
    torch_dtype=torch.float16 if device != "cpu" else torch.float32
)

# Move to device if not using device_map
if device == "mps":
    tower_model = tower_model.to(device)

# Set pad token if not set
if tower_tokenizer.pad_token is None:
    tower_tokenizer.pad_token = tower_tokenizer.eos_token

print("Model loaded! Ready for translations.\n")

def translate(text, target_language="French", model=tower_model, tokenizer=tower_tokenizer):
    """Translate a single text to the target language using Tower's chat template."""
    # Create chat messages using Tower's format
    messages = [
        {
            "role": "user",
            "content": f"Translate the following text to {target_language}: {text}"
        }
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate with better stopping criteria
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # Reduced from 100
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,  # Reduce repetition
        )
    
    # Decode only the new tokens (not the prompt)
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    translation = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Clean up translation - stop at common stop phrases
    stop_phrases = [
        "\n\n",
        "\nTranslate",
        "\nEnglish:",
        "\nFrench:",
        "\nSpanish:",
        "\nGerman:",
        "\nItalian:",
        "\nPortuguese:",
        "\nRussian:",
        "\nChinese:",
        "\nJapanese:",
        "\nKorean:",
        "\nHindi:",
        "\nMarathi:",
        "\nKonkani:",
        "\nArabic:",
        "again!",
        "again",
    ]
    
    for phrase in stop_phrases:
        if phrase in translation:
            translation = translation.split(phrase)[0]
    
    # Strip whitespace and return
    translation = translation.strip()
    
    return translation

# Example usage
if __name__ == "__main__":
    # Check if text provided as command line argument
    if len(sys.argv) > 1:
        # Command line mode: translate the provided text
        if len(sys.argv) >= 3:
            # Both text and language provided
            text = sys.argv[1]
            target_language = sys.argv[2]
        else:
            # Only text provided, default to French
            text = sys.argv[1]
            target_language = "French"
        
        print("="*60)
        print("TRANSLATION")
        print("="*60)
        print(f"Original: {text}")
        print(f"Target:   {target_language}")
        
        translation = translate(text, target_language)
        print(f"Result:   {translation}")
        print("="*60)
    else:
        # Interactive mode: keep asking for translations
        print("="*60)
        print("Interactive Translation")
        print("Model is loaded and ready!")
        print("Type 'quit' or 'exit' to stop")
        print("="*60)
        
        while True:
            try:
                text = input("\nEnter text to translate: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not text:
                    continue
                
                target_language = input("Enter target language (default: French): ").strip()
                if not target_language:
                    target_language = "French"
                
                print(f"Translating to {target_language}...")
                translation = translate(text, target_language)
                print(f"Result: {translation}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
