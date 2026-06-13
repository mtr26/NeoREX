import torch
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.neorex import NeoRexConfig, NeoRexForCausalLM

@pytest.fixture
def config():
    return NeoRexConfig(
        vocab_size=32768 + 10,
        d_model=256, # small for fast testing
        n_layers=2,
        n_heads=4,
        latent_dim=64,
        mlp_hidden=512,
        sliding_window=1024,
        mtp_depth=3,
        tie_word_embeddings=True
    )

def test_mtp_loss_trainer(config):
    """
    Test inside the trainer logic to ensure the Multi-Token Prediction auxiliary heads 
    correctly compute cross-entropy offset by n steps.
    """
    model = NeoRexForCausalLM(config)
    model.train()
    
    bsz = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (bsz, seq_len))
    labels = input_ids.clone() # predict next token = shift input
    
    # We should have loss from main head + 3 MTP heads
    outputs = model(input_ids, labels=labels)
    
    loss = outputs.loss
    logits = outputs.logits
    
    assert loss is not None
    assert not torch.isnan(loss)
    
    # Check that MTP loss is properly accumulated
    # Let's manually compute and compare
    hidden_states, _ = model.model(input_ids)
    main_logits = model.lm_head(hidden_states)
    
    shift_logits = main_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss()
    main_loss = loss_fct(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1))
    
    mtp_loss = 0.0
    for i, mtp_head in enumerate(model.mtp_heads):
        shift_amount = i + 2
        m_logits = mtp_head(hidden_states)[..., :-shift_amount, :].contiguous()
        m_labels = labels[..., shift_amount:].contiguous()
        mtp_loss += loss_fct(m_logits.view(-1, config.vocab_size), m_labels.view(-1))
        
    expected_loss = main_loss + mtp_loss * 0.5
    
    assert torch.allclose(loss, expected_loss), "MTP loss calculation is incorrect"

if __name__ == "__main__":
    pytest.main([__file__])
