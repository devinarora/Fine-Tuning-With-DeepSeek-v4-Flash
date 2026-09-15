import copy
import json
import logging
import shutil
from pathlib import Path

import torch
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from src.data import prepare_datasets
from src.model import load_base_model, load_tokenizer
from src.utils import get_dtype, load_config, resolve_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("toy_test")

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond)))
    status = "PASS" if cond else "FAIL"
    logger.info("[%s] %s %s", status, name, detail)


def toy_config():
    cfg = load_config()
    cfg["model_id"] = "google/flan-t5-small"
    cfg["data"]["train_subset"] = 200
    cfg["data"]["valid_subset"] = 200
    cfg["data"]["test_subset"] = 200
    cfg["data"]["max_length"] = 64
    cfg["training"]["num_train_epochs"] = 2
    cfg["training"]["save_strategy"] = "steps"
    cfg["training"]["save_steps"] = 20
    cfg["training"]["eval_strategy"] = "steps"
    cfg["training"]["eval_steps"] = 20
    cfg["training"]["logging_steps"] = 10
    cfg["training"]["output_dir"] = "outputs/toy"
    cfg["training"]["load_best_model_at_end"] = True
    cfg["training"]["metric_for_best_model"] = "eval_loss"
    cfg["training"]["generation_max_length"] = 32
    return cfg


def build_trainer(cfg, ckpt_dir):
    tokenizer = load_tokenizer(cfg)
    dtype = get_dtype(cfg)
    device = resolve_device()
    base = load_base_model(cfg, dtype, device)
    from src.model import build_lora_model

    peft, _ = build_lora_model(base, cfg)
    processed, _ = prepare_datasets(cfg, tokenizer, None)
    t = cfg["training"]
    args = Seq2SeqTrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=t["learning_rate"],
        warmup_steps=t["warmup_steps"],
        weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"],
        eval_strategy=t["eval_strategy"],
        eval_steps=t["eval_steps"],
        save_strategy=t["save_strategy"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        greater_is_better=False,
        fp16=False,
        bf16=True,
        predict_with_generate=True,
        generation_max_length=t["generation_max_length"],
        seed=t["seed"],
        dataloader_num_workers=0,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        report_to=[],
    )
    trainer = Seq2SeqTrainer(
        model=peft,
        args=args,
        train_dataset=processed["train"],
        eval_dataset=processed["valid"],
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
        compute_metrics=None,
    )
    return trainer, tokenizer


def checkpoint_dirs(root):
    return sorted(Path(root).glob("checkpoint-*"))


def phase_save_and_best(cfg):
    logger.info("=== PHASE A: save + best-model selection ===")
    root = Path("outputs/toy")
    shutil.rmtree(root, ignore_errors=True)
    ckpt_dir = root / "checkpoints"
    trainer, _ = build_trainer(cfg, str(ckpt_dir))
    set_seed(cfg["training"]["seed"])
    trainer.train()
    total_steps = trainer.state.global_step
    expected = (cfg["data"]["train_subset"] // 8) * cfg["training"]["num_train_epochs"]
    check("global_step == expected total steps", total_steps == expected, f"{total_steps} vs {expected}")

    ckpts = checkpoint_dirs(ckpt_dir)
    check("multiple checkpoints written", len(ckpts) >= 2, f"found {len(ckpts)}")
    for d in ckpts:
        files = {p.name for p in d.iterdir()}
        need = {"adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json"}
        missing = need - files
        check(f"checkpoint {d.name} has full contents", not missing, f"missing={sorted(missing)}")

    best = trainer.state.best_model_checkpoint
    check("best_model_checkpoint set", best is not None, str(best))
    if best:
        best_step = int(str(best).rsplit("-", 1)[-1])
        losses = []
        for d in ckpts:
            st = json.loads((d / "trainer_state.json").read_text(encoding="utf-8"))
            if "eval_loss" in st:
                losses.append((d.name, st["eval_loss"], int(str(d).rsplit("-", 1)[-1])))
        if losses:
            min_step = min(losses, key=lambda x: x[1])[2]
            check("best_model_checkpoint == min eval_loss step", best_step == min_step,
                  f"best={best_step}, min_loss_step={min_step}, losses={losses}")
    return ckpt_dir


def phase_resume(cfg, ckpt_dir):
    logger.info("=== PHASE B: resume from checkpoint ===")
    root = Path("outputs/toy")
    resume_dir = root / "resume"
    shutil.rmtree(resume_dir, ignore_errors=True)
    target_steps = 25
    trainer, _ = build_trainer(cfg, str(resume_dir))
    set_seed(cfg["training"]["seed"])
    trainer.args.max_steps = target_steps
    trainer.train()
    first_global = trainer.state.global_step
    check("phase B first run stopped at target", first_global == target_steps, str(first_global))
    ckpts = checkpoint_dirs(resume_dir)
    check("phase B produced a checkpoint to resume from", len(ckpts) >= 1, f"{len(ckpts)}")

    resume_from = str(ckpts[-1])
    resumed_step = int(resume_from.rsplit("-", 1)[-1])
    trainer2, _ = build_trainer(cfg, str(resume_dir))
    set_seed(cfg["training"]["seed"])
    trainer2.args.max_steps = target_steps + 15
    trainer2.train(resume_from_checkpoint=resume_from)
    final = trainer2.state.global_step
    check("resumed global_step continues", final == target_steps + 15,
          f"final={final}, expected={target_steps + 15}")


def phase_adapter_reload(cfg, ckpt_dir):
    logger.info("=== PHASE C: reload best adapter + translate ===")
    best_ckpts = sorted(ckpt_dir.glob("checkpoint-*"))
    if not best_ckpts:
        check("best adapter available", False, "no checkpoints")
        return
    best = str(best_ckpts[-1])
    tokenizer = load_tokenizer(cfg)
    dtype = get_dtype(cfg)
    device = resolve_device()
    from peft import PeftModel

    base = load_base_model(cfg, dtype, device)
    adapter = PeftModel.from_pretrained(base, best)
    adapter = adapter.to(device)
    adapter.eval()
    src = ["Qu'est-ce que c'est que ça ?", "Ceci est un test."]
    enc = tokenizer(src, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    with torch.no_grad():
        gen = adapter.generate(**enc, max_new_tokens=32, do_sample=False)
    preds = tokenizer.batch_decode(gen, skip_special_tokens=True)
    logger.info("reloaded adapter translations: %s", preds)
    check("adapter reload + generation works", all(len(p.strip()) > 0 for p in preds), str(preds))


def main():
    cfg = toy_config()
    set_seed(cfg["training"]["seed"])
    ckpt_dir = phase_save_and_best(cfg)
    phase_resume(cfg, ckpt_dir)
    phase_adapter_reload(cfg, ckpt_dir)

    failed = [name for name, ok in CHECKS if not ok]
    logger.info("===== TOY TEST SUMMARY: %d passed, %d failed =====",
                len(CHECKS) - len(failed), len(failed))
    if failed:
        logger.error("FAILED CHECKS: %s", failed)
        raise SystemExit(1)


if __name__ == "__main__":
    main()