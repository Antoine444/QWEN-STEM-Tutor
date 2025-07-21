import logging
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

# ─────────────────────────── Hyper-parameters ────────────────────────────
MODEL_NAME      = "Qwen/Qwen3-0.6B-Base"
OUTPUT_DIR      = "./dpo_full_bf16_5e-7_0.1_linear"

LEARNING_RATE   = 5e-7
BATCH_SIZE      = 1
GRAD_ACC_STEPS  = 16
EPOCHS          = 3
BETA            = 0.1
LR_SCHEDULER    = "linear"
WARMUP_RATIO    = 0.10
WEIGHT_DECAY    = 0.01
LOGGING_STEPS   = 50
SAVE_STEPS      = 400
SEED            = 42

# ─────────────────────────── Dataset Configuration ───────────────────────
RAW_DATASET     = "derko83/MNLP_M3_dpo_dataset"
TRAIN_SPLIT     = "train"
VALID_SPLIT     = "validation"
FILTERED_PATH   = "train_filtered_full_dataset_max1024"
MAX_LENGTH      = 1024


def filter_and_save_dataset():
    """
    Load the dataset, filter examples based on max length, and save the filtered dataset.
    This function uses the AutoTokenizer to encode prompts and responses, ensuring they do not exceed the specified `MAX_LENGTH`.
    It filters out examples where the combined length of the prompt and chosen/rejected responses exceeds `MAX_LENGTH`.
    The filtered dataset is then saved to disk at the specified `FILTERED_PATH`.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    ds = load_dataset(RAW_DATASET, split=TRAIN_SPLIT)

    def is_within_limit(example):
        prompt   = example["prompt"]
        chosen   = example["chosen"]
        rejected = example["rejected"]
        len_pc = len(tokenizer.encode(prompt + chosen, truncation=False))
        len_pr = len(tokenizer.encode(prompt + rejected, truncation=False))
        return len_pc <= MAX_LENGTH and len_pr <= MAX_LENGTH

    print("🔍 Filtrage des exemples dépassant la longueur max...")
    filtered = ds.filter(is_within_limit)

    print(f"💾 Sauvegarde du dataset filtré ({len(filtered)} exemples) dans `{FILTERED_PATH}`")
    filtered.save_to_disk(FILTERED_PATH)


def train_dpo():
    """
    Train a DPO model using the filtered dataset.
    This function initializes the DPOTrainer with the model, tokenizer, and training configuration.
    It sets up the training parameters such as batch size, learning rate, number of epochs,
    and other hyperparameters defined in the `DPOConfig`.
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    train_ds = load_from_disk(FILTERED_PATH)
    print(f"✅ Training set: {len(train_ds):,} examples")

    val_ds = load_dataset(RAW_DATASET, split=VALID_SPLIT)
    print(f"✅ Validation set: {len(val_ds):,} examples")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()

    dpo_cfg = DPOConfig(
        output_dir                   = OUTPUT_DIR,
        per_device_train_batch_size  = BATCH_SIZE,
        per_device_eval_batch_size   = BATCH_SIZE,
        gradient_accumulation_steps  = GRAD_ACC_STEPS,
        num_train_epochs             = EPOCHS,
        learning_rate                = LEARNING_RATE,
        beta                         = BETA,
        lr_scheduler_type            = LR_SCHEDULER,
        warmup_ratio                 = WARMUP_RATIO,
        weight_decay                 = WEIGHT_DECAY,
        max_length                   = MAX_LENGTH,
        eval_strategy                = "steps",
        eval_steps                   = SAVE_STEPS,
        logging_steps                = LOGGING_STEPS,
        save_steps                   = SAVE_STEPS,
        save_strategy                = "steps",
        logging_strategy             = "steps",
        bf16                         = True,
        fp16                         = False,
        seed                         = SEED,
        report_to                    = None,
        load_best_model_at_end       = True,
        metric_for_best_model        = "eval_loss",
    )

    trainer = DPOTrainer(
        model           = model,
        args            = dpo_cfg,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        processing_class= tokenizer,
    )

    print("🚀 Training started...")
    trainer.train()

    print("✅ Training completed!")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Model and tokenizer saved in `{OUTPUT_DIR}`")


if __name__ == "__main__":
    filter_and_save_dataset()
    train_dpo()
