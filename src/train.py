"""
Train (Colab-style hyperparams and preprocessing)
"""

# --- disable TF/Flax ---
import os
os.environ["USE_TF"] = "0"; os.environ["USE_FLAX"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"; os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["WANDB_DISABLED"] = "true"   # match your notebook

from pathlib import Path
import argparse
import torch

from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
)

from src import config
from src import data as data_mod
from src import preprocess as pp

def _ensure_parent_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def build_argparser():
    ap = argparse.ArgumentParser("Fine-tune T5 on TAL summaries (Colab-style)")
    ap.add_argument("--model-checkpoint", type=str, default=config.MODEL_CHECKPOINT)
    ap.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    ap.add_argument("--train-batch-size", type=int, default=config.TRAIN_BATCH_SIZE)
    ap.add_argument("--eval-batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    ap.add_argument("--save-total-limit", type=int, default=config.SAVE_TOTAL_LIMIT)
    ap.add_argument("--logging-steps", type=int, default=config.LOGGING_STEPS)
    ap.add_argument("--eval-strategy", type=str, default=config.EVAL_STRATEGY, choices=["epoch","steps","no"])
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--output-dir", type=Path, default=config.MODEL_OUTPUT_DIR)
    ap.add_argument("--num-proc", type=int, default=None)
    ap.add_argument("--fp16", action="store_true", help="Force fp16 (GPU only). If omitted, auto-enable on CUDA.")
    return ap

def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)
    _ensure_parent_dir(Path(args.output_dir))

    # ---- Data ----
    df = data_mod.load_dataframe()  # requires MERGED_CSV
    ds = data_mod.make_hf_splits(df, test_size=0.2, seed=args.seed)
    tokenized = pp.tokenize_dataset_dict(ds, num_proc=args.num_proc)
    print(ds)

    # ---- Model ----
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)
    # GPU + fp16 (Colab-style)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.to("cuda")
    fp16_flag = args.fp16 or use_cuda  # enable if on CUDA unless explicitly disabled

    # ---- Training args (mirror your notebook) ----
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.epochs,
        fp16=bool(fp16_flag),
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,  # Transformers 4.55 name
        report_to=[],                      # keep external loggers off
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"] if args.eval_strategy != "no" else None,
        # No custom metrics here—same as your notebook
    )

    trainer.train()

    # ---- Save model + tokenizer like your notebook ----
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    pp.get_tokenizer().save_pretrained(str(args.output_dir))
    print(f"Saved fine-tuned model to: {args.output_dir}")

if __name__ == "__main__":
    main()
