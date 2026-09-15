import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from src.data import prepare_datasets
from src.model import load_finetuned_model
from src.utils import ensure_output_dirs, get_dtype, load_config, resolve_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate")

METRIC_ALIASES = {
    "bleu": "bleu",
    "chrf": "chrF",
    "bertscore": "f1",
    "rouge": "rougeL",
}


def translate(model, tokenizer, fr_texts, max_length: int, device, dtype, batch_size: int):
    tokenizer.model_max_length = max_length
    outputs = []
    for i in tqdm(range(0, len(fr_texts), batch_size), desc="translating"):
        batch = fr_texts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_length,
                num_beams=4,
                do_sample=False,
            )
        outputs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outputs


def main():
    cfg = load_config()
    set_seed(cfg["training"]["seed"])
    device = resolve_device()
    dtype = get_dtype(cfg)
    dirs = ensure_output_dirs(cfg)

    adapter_dir = dirs["adapter"]
    if not (adapter_dir / "adapter_model.safetensors").exists() and not (adapter_dir / "adapter_model.bin").exists():
        logger.error("No adapter found in %s. Run src/train.py first.", adapter_dir)
        raise SystemExit(1)

    logger.info("loading finetuned model from %s", adapter_dir)
    model, tokenizer = load_finetuned_model(cfg, str(adapter_dir), dtype, device)
    model = model.to(device)

    processed, raw = prepare_datasets(cfg, tokenizer, None)
    ev = cfg["evaluation"]
    max_samples = ev["max_samples"]
    max_length = cfg["data"]["max_length"]

    test = raw["test"].select(range(min(max_samples, len(raw["test"]))))
    refs = list(test["en"])
    srcs = list(test["fr"])

    logger.info("translating %d test samples", len(srcs))
    preds = translate(model, tokenizer, srcs, max_length, device, dtype, ev["batch_size"])

    results = {}
    for name in ev["metrics"]:
        results[name] = {}
    try:
        import sacrebleu

        refs_flat = [[r] for r in refs]
        bleu = sacrebleu.corpus_bleu(preds, refs_flat)
        results["bleu"]["bleu"] = bleu.score
        results["bleu"]["precisions"] = list(bleu.precisions)
        chrf = sacrebleu.corpus_chrf(preds, refs_flat)
        results["chrf"]["chrF"] = chrf.score
        results["chrf"]["chrF_plus"] = chrf.score / 100.0
    except Exception as exc:
        logger.warning("sacrebleu failed: %s", exc)

    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rl_scores = [scorer.score(ref, pred)["rougeL"].fmeasure for ref, pred in zip(refs, preds)]
        results["rouge"]["rougeL"] = sum(rl_scores) / len(rl_scores)
    except Exception as exc:
        logger.warning("ROUGE failed: %s", exc)

    try:
        from bert_score import score as bert_score

        _, _, f1 = bert_score(preds, refs, lang="en", verbose=False, device=device)
        results["bertscore"]["f1"] = float(f1.mean().item())
        results["bertscore"]["precision"] = float(_.mean().item())
        results["bertscore"]["recall"] = float(_.mean().item())
    except Exception as exc:
        logger.warning("BERTScore failed: %s", exc)

    summary = {}
    for name, vals in results.items():
        for k, v in vals.items():
            if isinstance(v, (int, float)):
                summary[f"{name}.{k}"] = round(float(v), 4)
    logger.info("METRICS: %s", summary)

    df = pd.DataFrame({"source_fr": srcs, "prediction_en": preds, "reference_en": refs})
    out_csv = dirs["metrics"] / "predictions.csv"
    df.to_csv(out_csv, index=False)

    with open(dirs["metrics"] / "metrics_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("wrote %s and metrics_summary.json", out_csv)


def plot_training(metrics_dir, fig_dir):
    log_path = Path(metrics_dir) / "trainer_state.json"
    if not log_path.exists():
        logger.warning("trainer_state.json missing; skipping plots")
        return
    logs = json.loads(log_path.read_text(encoding="utf-8"))
    rows = [x for x in logs if "loss" in x or "eval_loss" in x]
    if not rows:
        logger.warning("no loss history to plot")
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    if "loss" in df:
        ax.plot(df["step"], df["loss"], label="train_loss", marker="o", markersize=3)
    if "eval_loss" in df:
        ax.plot(df["step"], df["eval_loss"], label="eval_loss", marker="s", markersize=4)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Training / Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(fig_dir) / "loss_curves.png", dpi=150)
    logger.info("saved loss_curves.png")


if __name__ == "__main__":
    main()