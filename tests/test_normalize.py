import torch

from utils.normalize import normalize_for_unit_mean
from utils.pooling import mean_pooling


def test_normalize_for_unit_mean():
    # last_hidden_states[i] is a token sequence; attention_mask[i] marks valid tokens
    # norms[i] = ||μ(normalize_for_unit_mean(X_i))||_2, expected to equal 1.0
    last_hidden_states = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )
    attention_mask = torch.tensor([[True, True], [True, False]])

    result = normalize_for_unit_mean(last_hidden_states, attention_mask)
    normalized_means = mean_pooling(result, attention_mask)
    norms = torch.linalg.norm(normalized_means, dim=-1)
    expected_norms = torch.tensor([1.0, 1.0])
    assert torch.allclose(norms, expected_norms, atol=1e-6)
