from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.utils import load_config


def load_tokenizer(cfg: dict):
    return AutoTokenizer.from_pretrained(cfg["model_id"])


def load_base_model(cfg: dict, dtype, device):
    return AutoModelForSeq2SeqLM.from_pretrained(
        cfg["model_id"],
        dtype=dtype,
        device_map=device if device.type == "cuda" else None,
    )


def build_lora_model(base_model, cfg: dict):
    lc = cfg["lora"]
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lc["r"],
        lora_alpha=lc["lora_alpha"],
        lora_dropout=lc["lora_dropout"],
        target_modules=lc["target_modules"],
    )
    return get_peft_model(base_model, lora_config), lora_config


def load_finetuned_model(cfg: dict, adapter_dir, dtype, device):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg["model_id"],
        dtype=dtype,
        device_map=device if device.type == "cuda" else None,
    )
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer
