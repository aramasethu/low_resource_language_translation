#!/usr/bin/env python3
"""
Interactive French translation using Tower model with MPS optimization.
Model stays loaded in memory - just call translate() repeatedly!
"""

from transformers import AutoTokenizer, pipeline
import torch

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
tower_pipeline = pipeline(
    "text-generation",
    model=TOWER_MODEL,
    tokenizer=tower_tokenizer,
    device=device,
    torch_dtype=torch.float16 if device != "cpu" else torch.float32
)

print("Model loaded! Ready for translations.\n")

# Simple prompt template
def create_prompt(text):
    return f"Translate the following text to French: {text}"

def translate(text, model_pipeline=tower_pipeline):
    """Translate a single text to French."""
    prompt = create_prompt(text)
    
    output = model_pipeline(
        prompt,
        max_new_tokens=100,
        do_sample=False,
        return_full_text=False
    )
    
    translation = output[0]['generated_text'].strip()
    return translation

# Interactive mode
if __name__ == "__main__":
    print("="*60)
    print("Interactive French Translation")
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
            
            print("Translating...")
            translation = translate(text)
            print(f"French: {translation}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

