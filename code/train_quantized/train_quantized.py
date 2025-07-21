#!/usr/bin/env python3
"""
QLoRA Fine-tuning Script for MCQA Model
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
from datasets import load_dataset
from trl import SFTTrainer

def parse_arguments():
    """"
    Parse command line arguments for QLoRA fine-tuning.
    Returns:
        argparse.Namespace: Parsed arguments including learning rate, batch size, epochs, and max length.
    """
    parser = argparse.ArgumentParser(description='QLoRA Fine-tuning for MCQA')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per device')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')

    return parser.parse_args()


def load_base_model():
    """"
    Load the quantized base model and tokenizer.
    Returns:
        tuple: Loaded model and tokenizer.
    """
    # Load model 4-bit + tokenizer
    model_id = "antoine-444/MNLP_M3_mcqa_model"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_storage_dtype="uint8",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def prepare_QLoRA_training(base_model):
    """"
    Prepare the model for QLoRA training.
    Args:
        base_model (AutoModelForCausalLM): The base model to prepare.
    Returns:
        AutoModelForCausalLM: The prepared model for QLoRA training.
    """
    print("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(base_model)

    # LoRA Configuration
    print("Setting up LoRA configuration...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"]  # Adjust according to your model
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


def prepare_input(example):
    """"
    Prepare the input for the MCQA dataset.
    Args:
        example (dict): A single example from the dataset.
    Returns:
        dict: A dictionary with the formatted input text.
    """
    question = example["question"]
    choices = example["choices"]
    input_text = f"{question} " + " ".join([f"({chr(65+i)}) {c}" for i, c in enumerate(choices)])

    return {"text": input_text}


def prepare_dataset(args, tokenizer):
    """"
    Load and preprocess the MCQA dataset.
    Args:
        args (argparse.Namespace): Parsed command line arguments.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding.
    Returns:
        Dataset: The preprocessed dataset ready for training.
    """
    print("Loading dataset...")
    dataset = load_dataset("najabba/MNLP_M3_quantized_dataset", split="train")

    print("Preprocessing dataset...")
    dataset = dataset.map(
        prepare_input,
        remove_columns=[col for col in dataset.column_names if col not in ("text", "id")]
    )

    # Tokenization function
    def tokenize(examples):
        """"
        Tokenize the input text.
        Args:            examples (dict): A batch of examples from the dataset.
        Returns:
            dict: Tokenized inputs ready for model training.
        """
        return tokenizer(examples["text"], padding="max_length", max_length=args.max_length, truncation=True)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize, batched=True)

    return tokenized_dataset


def train(args, model, train_dataset):
    """"
    Train the QLoRA model using the provided dataset.
    Args:
        args (argparse.Namespace): Parsed command line arguments.
        model (AutoModelForCausalLM): The model to train.
        train_dataset (Dataset): The preprocessed training dataset.
    Returns:
        tuple: The trained trainer and the directory where the model is saved.
    """
    # Set learning rate
    lr = args.learning_rate
    
    # Create save directory
    save_dir = "qlora_w4_" + str(lr).replace(".", "_")
    print(f"Model will be saved to: {save_dir}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        num_train_epochs=args.epochs,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none"
    )

    # Training with SFTTrainer (compatible with PEFT)
    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args
    )

    print("Starting training...")
    trainer.train()

    return trainer, save_dir


def save_model(base_model, trainer, tokenizer, save_dir):
    """"
    Save the trained model and tokenizer.
    Args:
        base_model (AutoModelForCausalLM): The base model used for training.
        trainer (SFTTrainer): The trained model trainer.
        tokenizer (AutoTokenizer): The tokenizer used for encoding.
        save_dir (str): Directory where the model will be saved.
    Returns:
        str: The directory where the merged model is saved.
    """
    # Save the LoRA adapters
    print("Saving LoRA adapters...")
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # Merge and save the full model
    print("Merging LoRA adapters with base model...")
    # Reload base model for merging
    merged_model = PeftModel.from_pretrained(base_model, save_dir)
    merged_model = merged_model.merge_and_unload()
    
    merged_save_dir = save_dir + "_merged"
    print(f"Saving merged model to: {merged_save_dir}")
    merged_model.save_pretrained(merged_save_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_save_dir)

    return merged_save_dir


def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Load quantized base model
    print("Loading base model...")
    base_model, tokenizer = load_base_model()

    # Prepare for QLoRA training
    model = prepare_QLoRA_training(base_model)

    # Load and prepare dataset
    train_dataset = prepare_dataset(args,tokenizer)

    # Training Model
    trainer, save_dir = train(args,model,tokenizer,train_dataset)

    # Save Model
    merged_save_dir = save_model(base_model,trainer,tokenizer,save_dir)

    print("Training completed successfully!")
    print(f"LoRA adapters saved to: {save_dir}")
    print(f"Merged model saved to: {merged_save_dir}")

if __name__ == "__main__":
    main()
    