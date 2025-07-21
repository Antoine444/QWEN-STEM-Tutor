#!/usr/bin/env python3
import os
import argparse
import logging

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig


def setup_environment():
    """
    Configure environment variables and CUDA settings for training.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a causal LM on MCQA dataset using TRL SFT"
    )
    parser.add_argument(
        "--model_name", type=str, default="najabba/sft-lr2e-5",
        help="Model identifier to load from Hugging Face"
    )
    parser.add_argument(
        "--dataset_repo", type=str,
        default="antoine-444/MNLP_M3_mcqa_dataset",
        help="Hugging Face dataset repository"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output_model",
        help="Directory to save the fine-tuned model"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-6,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01,
        help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4,
        help="Batch size per GPU/CPU"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=2,
        help="Number of steps to accumulate gradients"
    )
    parser.add_argument(
        "--save_steps", type=int, default=600,
        help="Steps between saving checkpoints"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=600,
        help="Steps between evaluations"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=300,
        help="Steps between logging metrics"
    )

    return parser.parse_args()


LETTERS = ["A", "B", "C", "D"]


def mcqa_to_prompt_completion(example, tokenizer):
    """
    Convert an MCQA example to a prompt-completion pair.
    """
    prompt = (
        "The following are multiple choice questions (with answers) "
        "about knowledge and skills in advanced master-level STEM courses.\n\n"
    )
    prompt += f"{example['question'].strip()}\n"
    prompt += "".join(
        f"{key}. {choice}\n" for key, choice in zip(LETTERS, example["choices"])
    )
    prompt += "Answer:"

    completion = f" {example['answer']}{tokenizer.eos_token}"
    return {"prompt": prompt, "completion": completion}


def load_data(dataset_repo: str, tokenizer):
    """
    Load and preprocess the train and validation splits.
    """
    train = load_dataset(dataset_repo, split="train").shuffle(seed=42)
    val = load_dataset(dataset_repo, split="validation").shuffle(seed=42)

    train = train.map(
        lambda x: mcqa_to_prompt_completion(x, tokenizer),
        remove_columns=["question", "choices", "answer", "dataset", "id", "rationale"]
    )
    val = val.map(
        lambda x: mcqa_to_prompt_completion(x, tokenizer),
        remove_columns=["question", "choices", "answer", "dataset", "id", "rationale"]
    )
    
    return train, val


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    setup_environment()

    logging.info("Loading tokenizer and model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    train_dataset, eval_dataset = load_data(args.dataset_repo, tokenizer)

    config = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        gradient_checkpointing=True,
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logging.info("Starting training")
    trainer.train()
    trainer.save_model(args.output_dir)
    logging.info("Training completed. Model saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
