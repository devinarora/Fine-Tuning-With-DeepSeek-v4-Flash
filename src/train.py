import json
import logging
import os
import time
from pathlib import Path

import torch
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from src.data import prepare_datasets
from src.model import build_lora_model, load_base_model, load_tokenizer
from src.utils import (
    ensure_output_dirs,
    get_dtype,
    load_config,
    resolve_device,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


def estimate_time(cfg: dict, n_train: int) -> dict:
    t = cfg["training"]
    steps_per_epoch = n_train // (t["per_device_train_batch_size"] * t.get("gradient_accumulation_steps", 1))
    total_steps = steps_per_epoch * t["num_train_epochs"]
    low = total_steps * 0.3
    high = total_steps * 0.6
    return {
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "est_seconds_min": low,
        "est_seconds_max": high,
        "est_minutes_range": (low / 60.0, high / 60.0),
    }


def main():
    cfg = load_config()
    set_seed(cfg["training"]["seed"])
    device = resolve_device()
    dtype = get_dtype(cfg)
    dirs = ensure_output_dirs(cfg)

    logger.info("device=%s dtype=%s", device, dtype)
    logger.info("loading tokenizer/model: %s", cfg["model_id"])
    tokenizer = load_tokenizer(cfg)
    base_model = load_base_model(cfg, dtype, device)
    peft_model, lora_config = build_lora_model(base_model, cfg)
    peft_model.print_trainable_parameters()

    processed, _ = prepare_datasets(cfg, tokenizer, None)
    n_train = len(processed["train"])
    logger.info("train=%d valid=%d test=%d", n_train, len(processed["valid"]), len(processed["test"]))

    eval_subset = cfg["data"].get("eval_subset", len(processed["valid"]))
    eval_dataset = processed["valid"].select(range(min(eval_subset, len(processed["valid"]))))
    logger.info("training-time eval on %d samples (of %d available)", len(eval_dataset), len(processed["valid"]))

    est = estimate_time(cfg, n_train)
    logger.info(
        "TIME ESTIMATE: ~%.0f-%.0f min (%.0f-%.0f s) over %d steps",
        est["est_minutes_range"][0],
        est["est_minutes_range"][1],
        est["est_seconds_min"],
        est["est_seconds_max"],
        est["total_steps"],
    )

    t = cfg["training"]
    args = Seq2SeqTrainingArguments(
        output_dir=dirs["checkpoints"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        warmup_steps=t["warmup_steps"],
        weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"],
    eval_strategy=t["eval_strategy"],
    eval_steps=t.get("eval_steps", 2000),
    save_strategy=t["save_strategy"],
    save_steps=t.get("save_steps", 1000),
    save_total_limit=t["save_total_limit"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        greater_is_better=False,
        fp16=t["fp16"],
        bf16=t["bf16"],
        predict_with_generate=t["predict_with_generate"],
        generation_max_length=t["generation_max_length"],
        seed=t["seed"],
        dataloader_num_workers=t["dataloader_num_workers"],
        optim=t["optim"],
        lr_scheduler_type=t["lr_scheduler_type"],
        report_to=[],
    )

    def compute_metrics(eval_pred):
        try:
            import numpy as np

            predictions, labels = eval_pred
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            preds = np.asarray(predictions)
            if preds.ndim == 3:
                preds = preds.argmax(-1)
            preds = preds.astype(np.int64)
            preds = np.clip(preds, 0, tokenizer.vocab_size - 1)
            pred_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
            return {
                "avg_pred_len": float(
                    sum(len(p.split()) for p in pred_text) / max(1, len(pred_text))
                )
            }
        except Exception:
            return {}

    trainer = Seq2SeqTrainer(
        model=peft_model,
        args=args,
        train_dataset=processed["train"],
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
        compute_metrics=compute_metrics,
    )

    start = time.time()
    trainer.train(resume_from_checkpoint=cfg["training"].get("resume_from_checkpoint", False))
    elapsed = time.time() - start
    logger.info("TRAINING DONE in %.1f min", elapsed / 60.0)

    metrics = trainer.evaluate()
    logger.info("val metrics: %s", metrics)

    peft_model.save_pretrained(dirs["adapter"])
    tokenizer.save_pretrained(dirs["adapter"])
    logger.info("LoRA adapter saved to %s", dirs["adapter"])

    with open(dirs["metrics"] / "val_metrics.json", "w", encoding="utf-8") as fh:
        json.dump({**metrics, "elapsed_min": elapsed / 60.0, "time_estimate": est}, fh, indent=2)

    with open(dirs["metrics"] / "trainer_state.json", "w", encoding="utf-8") as fh:
        json.dump(trainer.state.log_history, fh, indent=2)

    logger.info("best checkpoint: %s", trainer.state.best_model_checkpoint)


if __name__ == "__main__":
    main()
