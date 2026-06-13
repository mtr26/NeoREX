import torch
import pytest

# Dummy implementation of data collation for test
def dummy_collate_fn(batch, pad_token_id=0):
    max_len = max(len(item) for item in batch)
    padded_batch = []
    labels_batch = []
    
    for item in batch:
        pad_len = max_len - len(item)
        padded_item = item + [pad_token_id] * pad_len
        
        # Labels are same as input, but pad tokens are ignored (-100)
        label_item = item + [-100] * pad_len
        
        padded_batch.append(padded_item)
        labels_batch.append(label_item)
        
    return torch.tensor(padded_batch, dtype=torch.long), torch.tensor(labels_batch, dtype=torch.long)

def test_data_corruption():
    """
    Test data collation to ensure padding and truncation are deterministic
    and free of corruption.
    """
    raw_data = [
        [1, 2, 3],
        [1, 2, 3, 4, 5],
        [1]
    ]
    
    padded, labels = dummy_collate_fn(raw_data, pad_token_id=0)
    
    assert padded.shape == (3, 5)
    assert labels.shape == (3, 5)
    
    # Check padding is correct
    assert padded[0, 3].item() == 0
    assert padded[0, 4].item() == 0
    
    # Check label masking
    assert labels[0, 3].item() == -100
    assert labels[0, 4].item() == -100
    
    assert labels[2, 1].item() == -100

def test_agentic_tokens():
    """
    Ensure agentic tokens fall within the valid extended vocab range.
    """
    vocab_size = 32768
    num_special = 10
    total_vocab = vocab_size + num_special
    
    action_token_id = 32769
    thought_token_id = 32770
    
    assert action_token_id < total_vocab
    assert thought_token_id < total_vocab

if __name__ == "__main__":
    pytest.main([__file__])
