from datasets import load_dataset

from src.utils import load_config


def load_opus100(cfg: dict):
    data = load_dataset(cfg["dataset_id"], cfg["dataset_config"])
    return data


def expand_translation(ds):
    if "translation" in ds.column_names:
        ds = ds.map(
            lambda ex: {
                "fr": ex["translation"]["fr"],
                "en": ex["translation"]["en"],
            },
            remove_columns=["translation"],
        )
    return ds


def subset(ds, n: int):
    if n is not None and len(ds) > n:
        return ds.select(range(n))
    return ds


def build_tokenize_fn(tokenizer, max_length: int, padding: bool):
    def fn(batch):
        model_inputs = tokenizer(
            batch["fr"],
            max_length=max_length,
            padding=padding,
            truncation=True,
        )
        labels = tokenizer(
            batch["en"],
            max_length=max_length,
            padding=padding,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return fn


def prepare_datasets(cfg: dict, tokenizer, cache_dir):
    data = load_opus100(cfg)
    d = cfg["data"]

    splits = {
        "train": subset(data["train"], d["train_subset"]),
        "valid": subset(data["validation"], d["valid_subset"]),
        "test": subset(data["test"], d["test_subset"]),
    }
    splits = {name: expand_translation(ds) for name, ds in splits.items()}

    fn = build_tokenize_fn(tokenizer, d["max_length"], d["padding"])

    cols = [c for c in splits["train"].column_names if c not in ("fr", "en")]
    processed = {}
    for name, ds in splits.items():
        p = ds.map(
            fn,
            batched=True,
            remove_columns=cols,
            load_from_cache_file=True,
            cache_file_name=None,
        )
        processed[name] = p

    return processed, splits
