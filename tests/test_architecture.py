import torch
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.neorex import NeoRexConfig, NeoRexForCausalLM

@pytest.fixture
def config():
    return NeoRexConfig(
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

@pytest.fixture
def model(config):
    model = NeoRexForCausalLM(config)
    model.eval()
    return model

def test_parameter_size(model):
    """
    Assert the model parameters fall strictly within the 480M-520M range.
    """
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f} M")
    
    assert 480_000_000 <= total_params <= 520_000_000, f"Parameter count {total_params} out of expected range"

def test_rope_caching(model):
    """
    Ensure the RoPE offset mechanism matches exactly between a full forward pass 
    and step-by-step KV cache generation.
    """
    device = next(model.parameters()).device
    # Generate random sequence
    seq_len = 10
    input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)
    
    # 1. Full forward pass
    with torch.no_grad():
        outputs_full = model(input_ids, use_cache=False)
        logits_full = outputs_full.logits
        
    # 2. Step-by-step with cache
    past_key_values = None
    logits_cached = []
    
    with torch.no_grad():
        for i in range(seq_len):
            curr_id = input_ids[:, i:i+1]
            outputs_cache = model(curr_id, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs_cache.past_key_values
            logits_cached.append(outputs_cache.logits)
            
    logits_cached = torch.cat(logits_cached, dim=1)
    
    # Compare logits. Should be exactly equal or very close (due to fp math)
    max_diff = torch.max(torch.abs(logits_full - logits_cached))
    assert max_diff < 1e-4, f"RoPE offset caching bug! Max diff: {max_diff}"

def test_interleaved_attention(model):
    """
    Verify local layers are SWA and global layers are MLA.
    """
    global_layers = 0
    local_layers = 0
    
    for i, layer in enumerate(model.model.layers):
        if i % 4 == 3:
            assert layer.is_global == True
            assert layer.attention.__class__.__name__ == "MultiHeadLatentAttention"
            global_layers += 1
        else:
            assert layer.is_global == False
            assert layer.attention.__class__.__name__ == "SlidingWindowAttention"
            local_layers += 1
            
    assert global_layers == 6
    assert local_layers == 18

def test_causality_preservation(model):
    """
    Adversarial test: Ensure that changing future tokens does not affect past logits/representations,
    both with and without user-provided attention masks.
    """
    device = next(model.parameters()).device
    seq_len = 12
    
    # Create two input sequences that are identical except for the very last token
    input1 = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)
    input2 = input1.clone()
    input2[0, -1] = (input1[0, -1] + 1) % model.config.vocab_size
    
    for use_mask in [False, True]:
        attention_mask = None
        if use_mask:
            attention_mask = torch.ones((1, seq_len), dtype=torch.bool, device=device)
            
        with torch.no_grad():
            outputs1 = model(input1, attention_mask=attention_mask, use_cache=False)
            outputs2 = model(input2, attention_mask=attention_mask, use_cache=False)
            
            # Compare logits for all tokens except the last one (since the last token is different)
            logits1 = outputs1.logits[:, :-1, :]
            logits2 = outputs2.logits[:, :-1, :]
            
            max_diff = torch.max(torch.abs(logits1 - logits2)).item()
            assert max_diff < 1e-5, f"Causality leak detected (use_mask={use_mask})! Max diff: {max_diff}"

def test_batched_masked_generation_equivalence(model):
    """
    Adversarial test: Verify step-by-step cached generation matches the full forward pass
    exactly on batched sequences with padding attention masks.
    """
    device = next(model.parameters()).device
    bsz = 2
    seq_len = 10
    prefill_len = 4
    
    input_ids = torch.randint(0, model.config.vocab_size, (bsz, seq_len), device=device)
    
    # Create padding mask where first two tokens of second batch sequence are masked
    attention_mask = torch.ones((bsz, seq_len), dtype=torch.bool, device=device)
    attention_mask[1, :2] = False
    
    # 1. Full forward pass
    with torch.no_grad():
        outputs_full = model(input_ids, attention_mask=attention_mask, use_cache=False)
        logits_full = outputs_full.logits
        
    # 2. Step-by-step cached generation
    with torch.no_grad():
        # Prefill
        prefill_ids = input_ids[:, :prefill_len]
        prefill_mask = attention_mask[:, :prefill_len]
        outputs_prefill = model(prefill_ids, attention_mask=prefill_mask, use_cache=True)
        past_key_values = outputs_prefill.past_key_values
        
        logits_cached = [outputs_prefill.logits]
        
        # Generation steps
        for i in range(prefill_len, seq_len):
            curr_id = input_ids[:, i:i+1]
            curr_mask = attention_mask[:, :i+1]
            outputs_step = model(curr_id, attention_mask=curr_mask, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs_step.past_key_values
            logits_cached.append(outputs_step.logits)
            
    logits_cached = torch.cat([logits_cached[0]] + logits_cached[1:], dim=1)
    
    max_diff = torch.max(torch.abs(logits_full - logits_cached)).item()
    assert max_diff < 1e-4, f"Batched cached generation divergence with mask! Max diff: {max_diff}"

def test_swa_window_boundary_caching(model):
    """
    Adversarial test: Generate a sequence longer than the sliding window and verify
    caching equivalence over SWA window boundaries.
    """
    # Create a small configuration with a small sliding window to test boundary crossings
    config = NeoRexConfig(
        vocab_size=1000,
        d_model=128,
        n_layers=4,
        n_heads=4,
        latent_dim=32,
        mlp_hidden=256,
        sliding_window=4, # small window
        mtp_depth=0,
        tie_word_embeddings=True
    )
    custom_model = NeoRexForCausalLM(config)
    custom_model.eval()
    
    device = next(custom_model.parameters()).device
    seq_len = 15
    prefill_len = 3
    
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device=device)
    
    # 1. Full forward pass
    with torch.no_grad():
        outputs_full = custom_model(input_ids, use_cache=False)
        logits_full = outputs_full.logits
        
    # 2. Step-by-step cached generation
    with torch.no_grad():
        # Prefill
        prefill_ids = input_ids[:, :prefill_len]
        outputs_prefill = custom_model(prefill_ids, use_cache=True)
        past_key_values = outputs_prefill.past_key_values
        logits_cached = [outputs_prefill.logits]
        
        # Generation steps
        for i in range(prefill_len, seq_len):
            curr_id = input_ids[:, i:i+1]
            outputs_step = custom_model(curr_id, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs_step.past_key_values
            logits_cached.append(outputs_step.logits)
            
    logits_cached = torch.cat([logits_cached[0]] + logits_cached[1:], dim=1)
    
    max_diff = torch.max(torch.abs(logits_full - logits_cached)).item()
    assert max_diff < 1e-4, f"SWA window boundary cached generation divergence! Max diff: {max_diff}"

if __name__ == "__main__":
    pytest.main([__file__])

