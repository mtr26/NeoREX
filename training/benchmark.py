import argparse
import time
import os
import csv
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, TrainerCallback, AutoTokenizer
from datasets import load_dataset

import sys
# Add project root directory to path to resolve local model/training packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.neorex import NeoRexConfig, NeoRexForCausalLM
from model.rexmodel import REXConfig, REX

class TelemetryCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file
        self.start_time = None
        
        # Write header
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "loss", "vram_mb", "tokens_per_sec"])

    def on_step_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is None:
            return
            
        step_time = time.time() - self.start_time
        # Reset memory tracking for peak estimation
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        
        # Calculate tokens per sec
        batch_size = args.per_device_train_batch_size
        seq_len = 1024 # Hardcoded for benchmark
        tokens_per_sec = (batch_size * seq_len) / step_time
        
        loss = state.log_history[-1].get("loss", None) if state.log_history else None
        
        if loss is not None:
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([state.global_step, loss, vram_mb, tokens_per_sec])

class BenchmarkTrainer(Trainer):
    def __init__(self, optimizer_name="adam", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer_name = optimizer_name

    def create_optimizer(self):
        if self.optimizer_name == "muon":
            from training.trainer import Muon
            self.optimizer = Muon(self.model.parameters(), lr=self.args.learning_rate)
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return self.optimizer

def get_real_dataset(seq_len=1024):
    tokenizer_name = "mistralai/Mistral-7B-v0.1"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception:
        # Fallback to an un-gated equivalent if the user hasn't authenticated HF CLI
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"])
        
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=4)
    
    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # Drop the last chunk if it's smaller than seq_len
        total_length = (total_length // seq_len) * seq_len
        # Split by chunks of seq_len
        result = {
            k: [t[i : i + seq_len] for i in range(0, total_length, seq_len)]
            for k, t in concatenated_examples.items()
        }
        # Add labels
        result["labels"] = result["input_ids"].copy()
        return result
        
    lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=4)
    lm_dataset.set_format("torch")
    return lm_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["rex", "neorex"], required=True)
    parser.add_argument("--optimizer", type=str, choices=["adam", "muon"], required=True)
    parser.add_argument("--csv_out", type=str, required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    vocab_size = 32778
    if args.model == "rex":
        # Configure REX to ~510M parameters
        config = REXConfig(
            vocab_size=vocab_size,
            n_embd=1280,
            n_layers=20, # 20 layers for classic REX gives ~510M
            n_heads=20,
            n_kv_heads=4,
            max_len=1024,
            tie_word_embeddings=True
        )
        model = REX(config)
    else:
        # Configure NeoRex to ~514M parameters
        config = NeoRexConfig(
            vocab_size=vocab_size,
            d_model=1280,
            n_layers=24,
            n_heads=20,
            latent_dim=256,
            mlp_hidden=3456,
            sliding_window=1024,
            mtp_depth=3,
            tie_word_embeddings=True
        )
        model = NeoRexForCausalLM(config)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    if torch.cuda.is_available():
        model = model.to("cuda")
        
    try:
        model = torch.compile(model, mode="max-autotune")
    except Exception as e:
        print(f"Compilation failed, running eager: {e}")

    dataset = get_real_dataset(seq_len=1024)

    training_args = TrainingArguments(
        output_dir="./benchmark_output",
        per_device_train_batch_size=4,
        max_steps=args.steps,
        logging_steps=1,
        learning_rate=1e-3 if args.optimizer == "muon" else 3e-4,
        bf16=torch.cuda.is_bf16_supported(),
        dataloader_num_workers=2,
        report_to="none",
        remove_unused_columns=False,  # torch.compile wraps the model and hides the forward signature
    )

    trainer = BenchmarkTrainer(
        optimizer_name=args.optimizer,
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.add_callback(TelemetryCallback(args.csv_out))
    
    trainer.train()

if __name__ == "__main__":
    main()
