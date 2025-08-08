# main.py
import time
import contextlib
from tinygrad.tensor import Tensor
from tinygrad.nn.state import get_parameters, load_state_dict, safe_load, safe_save
from tinygrad.nn.optim import Adam

from model import GPT
import lora

# --- 1. CONTEXT MANAGERS FOR STATE ---
@contextlib.contextmanager
def no_grad():
    """A context manager to disable gradient tracking for inference."""
    prev = Tensor.no_grad
    Tensor.no_grad = True
    try:
        yield
    finally:
        Tensor.no_grad = prev

@contextlib.contextmanager
def train_mode():
    """A context manager to enable training mode for optimizers and layers."""
    prev = Tensor.training
    Tensor.training = True
    try:
        yield
    finally:
        Tensor.training = prev

# --- 2. CONFIGURATION ---
MODEL_WEIGHTS_PATH = "weights/model.safetensors"
LORA_ADAPTER_PATH = "weights/pirate_adapter.safetensors"
PRIVATE_DATA_PATH = "data/private_data.txt"

PYTHIA_CONFIG = {"dim": 128, "n_layers": 6, "n_heads": 4}
LEARNING_RATE = 1e-3
EPOCHS = 25

# --- 3. THE FRAMEWORK CORE ---

def load_base_model_and_tokenizer(config):
    with open(PRIVATE_DATA_PATH, 'r') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda l: ''.join([itos.get(i, '?') for i in l])
    
    model = GPT(**config, vocab_size=vocab_size)
    
    weights = safe_load(MODEL_WEIGHTS_PATH)
    weights = {k:v for k,v in weights.items() if 'wte' not in k and 'lm_head' not in k and 'embed_in' not in k and 'embed_out' not in k}
    load_state_dict(model, weights, strict=False)
    print(f"Loaded base model weights from {MODEL_WEIGHTS_PATH}")
    
    return model, encode, decode

def generate_text(model, encode, decode, start_string, max_len=30):
    with no_grad():
        tokens = encode(start_string)
        for _ in range(max_len):
            input_tensor = Tensor([tokens])
            logits = model(input_tensor)
            next_token_logit = logits[0, -1, :]
            next_token = next_token_logit.argmax().item()
            tokens.append(next_token)
        return decode(tokens)

def main():
    print("--- On-Device Fine-Tuning Simulation using tinygrad ---")
    
    model, encode, decode = load_base_model_and_tokenizer(PYTHIA_CONFIG)
    
    print("\nFreezing all base model parameters...")
    for p in get_parameters(model):
        p.requires_grad = False

    print("Injecting LoRA adapters into the model...")
    trainable_params = lora.inject_lora_into_model(model)
    total_params = sum(p.numel() for p in trainable_params)
    print(f"Total trainable parameters (LoRA only): {total_params} ({total_params/1e6:.4f}M)")
    
    print("\n--- Running inference BEFORE fine-tuning ---")
    prompt = "Ahoy, "
    generated = generate_text(model, encode, decode, prompt)
    print(f"Prompt: '{prompt}'")
    print(f"Generated: '{generated}'") # The repetitive "lll" is expected before training

    print("\n--- Starting LoRA fine-tuning on private data ---")
    with open(PRIVATE_DATA_PATH, 'r') as f:
        train_data = f.read()
    input_tensor = Tensor([encode(train_data[:-1])])
    target_tensor = Tensor([encode(train_data[1:])])
    
    optimizer = Adam(trainable_params, lr=LEARNING_RATE)
    
    with train_mode():
        for i in range(EPOCHS):
            optimizer.zero_grad()
            loss = model(input_tensor).sparse_categorical_crossentropy(target_tensor)
            loss.backward()
            optimizer.step()
            print(f"Epoch {i+1}/{EPOCHS} | Loss: {loss.item():.4f}")

    print("\n--- Saving the personalized LoRA adapter ---")
    lora_state_dict = lora.get_lora_state_dict(model)
    safe_save(lora_state_dict, LORA_ADAPTER_PATH)
    print(f"Saved {len(lora_state_dict)} LoRA tensors to {LORA_ADAPTER_PATH}")
    
    print("\n--- Verifying: Loading adapter into a new, clean model ---")
    fresh_model, _, _ = load_base_model_and_tokenizer(PYTHIA_CONFIG)
    _ = lora.inject_lora_into_model(fresh_model)
    saved_adapter_weights = safe_load(LORA_ADAPTER_PATH)
    lora.load_lora_state_dict(fresh_model, saved_adapter_weights)
    print("Adapter weights loaded successfully.")
    
    print("\n--- Running inference with the re-loaded adapter ---")
    generated = generate_text(fresh_model, encode, decode, prompt)
    print(f"Prompt: '{prompt}'")
    print(f"Generated: '{generated}'")

if __name__ == "__main__":
    main()