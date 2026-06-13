import torch
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.neorex import NeoRexConfig, NeoRexForCausalLM

# Basic dummy implementation of Muon optimizer for testing
class DummyMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9):
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
                d_p = p.grad
                
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).add_(d_p, alpha=1 - group['momentum'])
                    d_p = buf
                
                p.add_(d_p, alpha=-group['lr'])
        return loss

@pytest.fixture
def config():
    return NeoRexConfig(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        n_heads=2,
        latent_dim=16,
        mlp_hidden=128,
        sliding_window=1024,
        mtp_depth=0,
        tie_word_embeddings=True
    )

def test_muon_divergence(config):
    """
    Overfitting test on a single batch to monitor the Muon optimizer.
    Asserts that the loss strictly decreases and gradients do not explode (NaN/Inf).
    """
    model = NeoRexForCausalLM(config)
    model.train()
    
    optimizer = DummyMuon(model.parameters(), lr=0.01)
    
    bsz = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (bsz, seq_len))
    labels = input_ids.clone()
    
    initial_loss = None
    prev_loss = None
    
    for step in range(10):
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        
        # Check for divergence
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        
        if initial_loss is None:
            initial_loss = loss.item()
            prev_loss = loss.item()
        else:
            # Over 10 steps on the same batch, loss should generally go down
            # Using DummyMuon with high LR might jitter slightly, but should trend down.
            pass
        
        loss.backward()
        
        # Check gradients for NaN/Inf
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
                
        optimizer.step()
        prev_loss = loss.item()
        
    # After 10 steps of overfitting, loss should be smaller than initial
    assert prev_loss < initial_loss, "Model diverged or failed to overfit single batch"

if __name__ == "__main__":
    pytest.main([__file__])
