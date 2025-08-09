# main.py
import time
import contextlib
from tinygrad.tensor import Tensor
from tinygrad.nn.state import get_parameters, load_state_dict, safe_load, safe_save, get_state_dict
from tinygrad.nn.optim import Adam

from model import GPT
import lora

def find_lora_params(model_or_module):
    """
    Recursively traverses the model to find all LoRA parameters (A and B matrices).
    """
    found_params = []
    for name, layer in model_or_module.__dict__.items():
        if isinstance(layer, lora.LoRAWrapper):
            found_params.extend(layer.trainable_params)
        elif isinstance(layer, list):
            for sub_layer in layer:
                if hasattr(sub_layer, '__dict__'):
                    found_params.extend(find_lora_params(sub_layer))
        elif hasattr(layer, '__dict__') and not isinstance(layer, lora.LoRALayer):
            found_params.extend(find_lora_params(layer))
    return found_params


@contextlib.contextmanager
def train_mode():
    """A context manager to enable training mode. This is required by the optimizer."""
    prev = Tensor.training
    Tensor.training = True
    try:
        yield
    finally:
        Tensor.training = prev

MODEL_WEIGHTS_PATH = "weights/model.safetensors"
LORA_ADAPTER_PATH = "weights/pirate_adapter.safetensors"
PRIVATE_DATA_PATH = "data/private_data.txt"

PYTHIA_CONFIG = {"dim": 128, "n_layers": 6, "n_heads": 4}
LEARNING_RATE = 1e-3
EPOCHS = 200


TUNE_NORM_LAYERS = False


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
    weights = {k:v for k,v in weights.items() if 'wte' not in k and 'lm_head' not in k}
    load_state_dict(model, weights, strict=False)
    print(f"Loaded base model weights from {MODEL_WEIGHTS_PATH}")
    return model, encode, decode

def generate_text(model, encode, decode, start_string, max_len=100, temperature=0.5):
    tokens = encode(start_string)
    for _ in range(max_len):
        input_tensor = Tensor([tokens])
        logits = model(input_tensor)
        logits = logits[:, -1, :] / temperature
        probs = logits.softmax()
        next_token = probs.multinomial().item()
        tokens.append(next_token)
    return decode(tokens)

def main():
    print("--- On-Device Fine-Tuning Simulation using tinygrad ---")
    model, encode, decode = load_base_model_and_tokenizer(PYTHIA_CONFIG)
    
    print("\n1. Freezing all model parameters by default...")
    for p in get_state_dict(model).values():
        p.requires_grad = False

    print("2. Injecting LoRA adapters (these are born trainable)...")
    lora.inject_lora_into_model(model)

    print("3. Unfreezing specific layers for fine-tuning...")
    
    # ALWAYS unfreeze wte and lm_head for a new vocabulary. This is mandatory.
    print(" -> Unfreezing wte and lm_head (mandatory for new vocab)")
    for p in get_state_dict(model.wte).values(): p.requires_grad = True
    for p in get_state_dict(model.lm_head).values(): p.requires_grad = True
    
    # CONDITIONALLY unfreeze normalization layers based on the flag.
    if TUNE_NORM_LAYERS:
        print(" -> Unfreezing normalization layers (optional)")
        for p in get_state_dict(model.ln_f).values(): p.requires_grad = True
        for block in model.h:
            for p in get_state_dict(block.ln_1).values(): p.requires_grad = True
            for p in get_state_dict(block.ln_2).values(): p.requires_grad = True
    else:
        print(" -> Keeping normalization layers frozen.")

    print("4. Collecting all trainable parameters...")
    trainable_params = get_parameters(model)
    
    total_params = sum(p.numel() for p in trainable_params)
    lora_params_list = find_lora_params(model)
    lora_only_params = sum(p.numel() for p in lora_params_list)

    print(f"\nTotal trainable parameters: {total_params} ({total_params/1e6:.4f}M)")
    print(f" -> LoRA parameters: {lora_only_params}")
    print(f" -> Other fine-tuned parameters: {total_params - lora_only_params}")
    
    print("\n--- Running inference BEFORE fine-tuning (expect gibberish) ---")
    prompt = "Ahoy, "
    generated = generate_text(model, encode, decode, prompt)
    print(f"Prompt: '{prompt}'")
    print(f"Generated: '{generated}'")

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
            if (i+1) % 10 == 0:
                print(f"Epoch {i+1}/{EPOCHS} | Loss: {loss.item():.4f}")

    print("\n--- Saving the personalized LoRA adapter ---")
    lora_state_dict = lora.get_lora_state_dict(model)
    safe_save(lora_state_dict, LORA_ADAPTER_PATH)
    print(f"Saved {len(lora_state_dict)} LoRA tensors to {LORA_ADAPTER_PATH}")
    
    print("\n--- Running inference AFTER fine-tuning ---")
    generated = generate_text(model, encode, decode, prompt, temperature=0.5)
    print(f"Prompt: '{prompt}'")
    print(f"Generated: '{generated}'")

if __name__ == "__main__":
    main()