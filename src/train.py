"""
Train a seq2seq summarization model on the merged TAL dataset.

Usage (from project root):
    python -m src.train
    # or override a few things:
    python -m src.train --epochs 3 --lr 3e-5 --train-batch-size 4 --eval-batch-size 4
"""

import os
# Tell Transformers to NOT use TF/Flax at all
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
# (Keep these too; they help on newer versions)
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
assert os.environ["USE_TF"] == "0" and os.environ["USE_FLAX"] == "0"


from pathlib import Path
import argparse
import numpy as np
import evaluate
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
    ap = argparse.ArgumentParser(description="Fine-tune a seq2seq model for TAL teaser summarization.")
    ap.add_argument("--model-checkpoint", type=str, default=config.MODEL_CHECKPOINT)
    ap.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    ap.add_argument("--warmup-ratio", type=float, default=config.WARMUP_RATIO)
    ap.add_argument("--train-batch-size", type=int, default=config.TRAIN_BATCH_SIZE)
    ap.add_argument("--eval-batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    ap.add_argument("--grad-accum-steps", type=int, default=config.GRADIENT_ACCUMULATION_STEPS)
    ap.add_argument("--logging-steps", type=int, default=config.LOGGING_STEPS)
    ap.add_argument("--save-total-limit", type=int, default=config.SAVE_TOTAL_LIMIT)
    ap.add_argument("--eval-strategy", type=str, default=config.EVAL_STRATEGY, choices=["epoch", "steps", "no"])
    ap.add_argument("--save-strategy", type=str, default=config.SAVE_STRATEGY, choices=["epoch", "steps", "no"])
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--output_dir", type=Path, default=config.MODEL_OUTPUT_DIR)
    ap.add_argument("--num-proc", type=int, default=None, help="Tokenization processes (None = single process).")
    ap.add_argument("--resume-from-checkpoint", type=str, default=None)
    ap.add_argument("--fp16", action="store_true", help="Use FP16 (if CUDA is available).")
    ap.add_argument("--bf16", action="store_true", help="Use BF16 (Ampere+).")
    return ap


def main():
    args = build_argparser().parse_args()

    # Reproducibility
    set_seed(args.seed)

    # Ensure output dir exists
    _ensure_parent_dir(Path(args.output_dir) if isinstance(args.output_dir, str) else args.output_dir)

    # -------- Load & tokenize data --------
    df = data_mod.load_dataframe()  # requires MERGED_CSV to exist
    print(f"Loaded {len(df)} rows from merged dataset at {config.MERGED_CSV}")

    ds = data_mod.make_hf_splits(df, test_size=0.2, seed=args.seed)
    print(ds)

    tokenized = pp.tokenize_dataset_dict(ds, num_proc=args.num_proc, desc="Tokenizing dataset")
    collator = pp.make_data_collator()

    # -------- Model --------
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)

    gen_cfg = model.generation_config
    gen_cfg.max_new_tokens = config.MAX_SUMMARY_LEN
    gen_cfg.num_beams = config.NUM_BEAMS
    gen_cfg.no_repeat_ngram_size = config.NO_REPEAT_NGRAM_SIZE
    gen_cfg.repetition_penalty = config.REPETITION_PENALTY
    gen_cfg.length_penalty = config.LENGTH_PENALTY
    gen_cfg.early_stopping = config.EARLY_STOPPING

    # -------- Metrics (ROUGE) --------
    rouge = evaluate.load("rouge")
    tok = pp.get_tokenizer()

    def postprocess_text(preds, labels):
        preds = [p.strip() for p in preds]
        labels = [l.strip() for l in labels]
        return preds, labels

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tok.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tok.pad_token_id)
        decoded_labels = tok.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        result = {k: round(v * 100, 2) for k, v in result.items()}

        # Track average generated length
        gen_lens = [np.count_nonzero(p != tok.pad_token_id) for p in preds]
        result["gen_len"] = round(float(np.mean(gen_lens)), 2)
        return result

    # FP16/BF16 options
    use_cuda = torch.cuda.is_available()
    fp16 = bool(args.fp16 and use_cuda)
    bf16 = bool(args.bf16 and use_cuda)

    # -------- Training args --------
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        fp16=fp16,
        bf16=bf16,
        predict_with_generate=True,
        generation_max_length=config.MAX_SUMMARY_LEN,
        generation_num_beams=config.NUM_BEAMS,
        report_to=[],  # disable wandb/etc by default
        load_best_model_at_end=(args.eval_strategy != "no"),
        metric_for_best_model="rougeLsum",
        greater_is_better=True,
    )

    # -------- Trainer --------
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"] if args.eval_strategy != "no" else None,
        data_collator=collator,
        processing_class=tok,
        compute_metrics=compute_metrics if args.eval_strategy != "no" else None,
    )

    # -------- Train --------
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # -------- Save final model/tokenizer --------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tok.save_pretrained(str(args.output_dir))
    print(f"Saved fine-tuned model & tokenizer to: {args.output_dir}")

    # -------- Final eval --------
    if args.eval_strategy != "no":
        metrics = trainer.evaluate()
        print("Final eval metrics:", metrics)


if __name__ == "__main__":
    main()
