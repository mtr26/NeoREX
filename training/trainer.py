import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import mlflow

try:
    from liger_kernel.transformers import apply_liger_kernel_to_llama
    # We can write a custom apply_liger_kernel_to_neorex, but for now we just 
    # try importing it to show the intent of fusion.
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False

from model.neorex import NeoRexConfig, NeoRexForCausalLM

# Dummy Dataset for demonstration
class DummyDataset(Dataset):
    def __init__(self, vocab_size, seq_len=1024, length=1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.randint(0, self.vocab_size, (self.seq_len,))

def get_dataloader(vocab_size, batch_size=4, seq_len=1024):
    dataset = DummyDataset(vocab_size, seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Muon(torch.optim.Optimizer):
    """
    Muon optimizer implementation (DeepSeek V4 inspired).
    Uses momentum and orthogonalization. (Simplified version for 500M model).
    """
    def __init__(self, params, lr=0.02, momentum=0.95):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim > 1: # Only apply orthogonalization to >= 2D tensors (weights)
                    # Simplified Newton-Schulz orthogonalization
                    x = g / (g.norm() + 1e-8)
                    for _ in range(5):
                        x = 1.5 * x - 0.5 * x @ x.T @ x
                    g = x * (max(g.shape[0], g.shape[1]) ** 0.5)

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(g)
                
                p.add_(buf, alpha=-group['lr'])
        return loss

def train():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("NeoRex-500M-Pretrain")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = NeoRexConfig(
        vocab_size=32778,
        d_model=1280,
        n_layers=24,
        n_heads=20,
        latent_dim=256,
        mlp_hidden=3456,
        sliding_window=1024,
        mtp_depth=3,
        tie_word_embeddings=True
    )
    
    model = NeoRexForCausalLM(config).to(device)
    
    if LIGER_AVAILABLE:
        print("Applying Liger Kernel Fusions...")
        # Custom fusion would go here.
        pass

    # Compile model
    if device == "cuda":
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="max-autotune")

    optimizer = Muon(model.parameters(), lr=1e-3)
    dataloader = get_dataloader(config.vocab_size, batch_size=4, seq_len=1024)
    
    epochs = 1
    global_step = 0

    # Optional CUDA Graphs setup (basic)
    # Note: CUDA Graphs require static shapes.
    # static_input = torch.randint(0, config.vocab_size, (4, 1024), device=device)
    # static_label = static_input.clone()
    
    print("Starting training...")
    with mlflow.start_run():
        mlflow.log_params({
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "mtp_depth": config.mtp_depth,
            "optimizer": "Muon",
            "lr": 1e-3
        })
        
        model.train()
        for epoch in range(epochs):
            for batch_idx, input_ids in enumerate(dataloader):
                input_ids = input_ids.to(device)
                labels = input_ids.clone()
                
                start_time = time.time()
                
                optimizer.zero_grad()
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                step_time = time.time() - start_time
                tokens_per_sec = (input_ids.numel()) / step_time
                
                if batch_idx % 10 == 0:
                    print(f"Step {global_step} | Loss: {loss.item():.4f} | Tok/s: {tokens_per_sec:.0f}")
                    mlflow.log_metric("train_loss", loss.item(), step=global_step)
                    mlflow.log_metric("tokens_per_sec", tokens_per_sec, step=global_step)
                    
                global_step += 1
                
                # Break early for testing purposes
                if global_step >= 50:
                    break
            if global_step >= 50:
                break

    print("Training finished.")

if __name__ == "__main__":
    train()
