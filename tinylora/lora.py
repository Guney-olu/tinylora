# lora.py
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.nn.state import safe_save, safe_load, get_state_dict

class LoRALayer:
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        self.A = Tensor.uniform(in_features, rank, requires_grad=True)
        self.B = Tensor.zeros(rank, out_features, requires_grad=True)
        self.scaling = alpha / rank

    def __call__(self, x: Tensor) -> Tensor:
        return x.dot(self.A).dot(self.B) * self.scaling

class LoRAWrapper:
    def __init__(self, original_layer: Linear, rank=4, alpha=1.0):
        assert isinstance(original_layer, Linear), "Can only wrap Linear layers"
        self.original_layer = original_layer
        # Linear.weight is (out_features, in_features).
        # LoRALayer expects (in_features, out_features).
        in_features = original_layer.weight.shape[1]
        out_features = original_layer.weight.shape[0]
        self.lora_layer = LoRALayer(in_features, out_features, rank, alpha)
        
    def __call__(self, x: Tensor) -> Tensor:
        return self.original_layer(x) + self.lora_layer(x)

    @property
    def trainable_params(self):
        return [self.lora_layer.A, self.lora_layer.B]

def inject_lora_into_model(model):
    trainable_lora_params = []
    for name, layer in model.__dict__.items():
        if isinstance(layer, Linear):
            print(f"  -> Wrapping layer: {name}")
            wrapper = LoRAWrapper(layer, rank=8)
            setattr(model, name, wrapper)
            trainable_lora_params.extend(wrapper.trainable_params)
        elif isinstance(layer, list):
            for i, sub_layer in enumerate(layer):
                if hasattr(sub_layer, '__dict__'):
                     trainable_lora_params.extend(inject_lora_into_model(sub_layer))
        elif hasattr(layer, '__dict__'):
            trainable_lora_params.extend(inject_lora_into_model(layer))
    return trainable_lora_params

def get_lora_state_dict(model, prefix=""):
    lora_sd = {}
    for name, layer in model.__dict__.items():
        current_prefix = f"{prefix}.{name}" if prefix else name
        if isinstance(layer, LoRAWrapper):
            lora_sd[f"{current_prefix}.lora_layer.A"] = layer.lora_layer.A
            lora_sd[f"{current_prefix}.lora_layer.B"] = layer.lora_layer.B
        elif isinstance(layer, list):
            for i, sub_layer in enumerate(layer):
                if hasattr(sub_layer, '__dict__'):
                    lora_sd.update(get_lora_state_dict(sub_layer, prefix=f"{current_prefix}.{i}"))
        elif hasattr(layer, '__dict__'):
            lora_sd.update(get_lora_state_dict(layer, prefix=current_prefix))
    return lora_sd

def load_lora_state_dict(model, state_dict, prefix=""):
    for name, layer in model.__dict__.items():
        current_prefix = f"{prefix}.{name}" if prefix else name
        if isinstance(layer, LoRAWrapper):
            key_A = f"{current_prefix}.lora_layer.A"
            key_B = f"{current_prefix}.lora_layer.B"
            if key_A in state_dict and key_B in state_dict:
                layer.lora_layer.A.assign(state_dict[key_A].to(layer.lora_layer.A.device))
                layer.lora_layer.B.assign(state_dict[key_B].to(layer.lora_layer.B.device))
        elif isinstance(layer, list):
            for i, sub_layer in enumerate(layer):
                if hasattr(sub_layer, '__dict__'):
                    load_lora_state_dict(sub_layer, state_dict, prefix=f"{current_prefix}.{i}")
        elif hasattr(layer, '__dict__'):
            load_lora_state_dict(layer, state_dict, prefix=current_prefix)