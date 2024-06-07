from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def initialize_model(model_name: str, cache_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    return tokenizer, model


def apply_lora(model):
    lora_config = LoraConfig(
        r=8,  # Rank of the low-rank matrices
        lora_alpha=16,  # Scaling factor
        lora_dropout=0.1,  # Dropout rate
        target_modules=["q_proj", "v_proj"],  # Target modules for LoRA
    )
    model = get_peft_model(model, lora_config)
    return model
