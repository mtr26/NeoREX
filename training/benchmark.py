import argparse
import time
import os
import csv
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset

import sys
# Add project root directory to path to resolve local model/training packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.neorex import NeoRexConfig, NeoRexForCausalLM
from model.rexmodel import REXConfig, REX

SEQ_LEN = 1024  # Benchmark sequence length

class BenchmarkTrainer(Trainer):
    """Trainer subclass that captures real per-step telemetry directly
    inside training_step, where the loss tensor is actually available.
    This replaces the broken TelemetryCallback that read from stale log_history.
    """
    def __init__(self, optimizer_name="adam", csv_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer_name = optimizer_name
        self.csv_path = csv_path

        # Write CSV header once
        if csv_path:
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["step", "loss", "vram_mb", "tokens_per_sec"])

    def create_optimizer(self):
        if self.optimizer_name == "muon":
            from training.trainer import Muon
            self.optimizer = Muon(self.model.parameters(), lr=self.args.learning_rate)
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Strip num_items_in_batch — newer HF Trainer passes it but our models don't accept it.
        inputs.pop("num_items_in_batch", None)
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to time each step and capture the real loss tensor value
        before it is detached and averaged by the parent class internals.
        """
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        loss = super().training_step(model, inputs, num_items_in_batch)

        step_time = time.perf_counter() - t0
        tokens_per_sec = (self.args.per_device_train_batch_size * SEQ_LEN) / step_time
        vram_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if torch.cuda.is_available() else 0.0
        )

        if self.csv_path:
            # loss is already scaled by grad_accum inside super(); with
            # gradient_accumulation_steps=1 (default) it equals the raw loss.
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    self.state.global_step,
                    round(loss.item() * self.args.gradient_accumulation_steps, 6),
                    round(vram_mb, 3),
                    round(tokens_per_sec, 2),
                ])
        return loss

def get_real_dataset(seq_len=1024):
    tokenizer_name = "mistralai/Mistral-7B-v0.1"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception:
        # Fallback to an un-gated equivalent if the user hasn't authenticated HF CLI
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    
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
            sliding_window=256,  # Must be < seq_len=1024 for SWA to actually restrict attention
            mtp_depth=3,
            tie_word_embeddings=True
        )
        model = NeoRexForCausalLM(config)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    if torch.cuda.is_available():
        model = model.to("cuda")
        
    try:
        #model = torch.compile(model, mode="max-autotune")
        pass
    except Exception as e:
        print(f"Compilation failed, running eager: {e}")

    dataset = get_real_dataset(seq_len=1024)

    training_args = TrainingArguments(
        output_dir="./benchmark_output",
        per_device_train_batch_size=4,
        max_steps=args.steps,
        logging_steps=100,
        save_strategy="no",
        learning_rate=1e-3 if args.optimizer == "muon" else 3e-4,
        bf16=torch.cuda.is_bf16_supported(),
        dataloader_num_workers=2,
        report_to="none",
        remove_unused_columns=False,  # torch.compile wraps the model and hides the forward signature,
        torch_compile=True,                      
        torch_compile_mode="default"
    )

    trainer = BenchmarkTrainer(
        optimizer_name=args.optimizer,
        csv_path=args.csv_out,
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
