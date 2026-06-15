import torch

from utils.pooling import covariance_pooling, mean_pooling


def test_mean_pooling():
    # last_hidden_states[i] is a token sequence with padding; attention_mask[i] marks valid tokens
    # expected[i] = μ(X_i), computed analytically
    last_hidden_states = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )
    attention_mask = torch.tensor([[1, 1], [0, 1]])

    result = mean_pooling(last_hidden_states, attention_mask)
    expected = torch.tensor([[2.5, 3.5, 4.5], [10.0, 11.0, 12.0]])
    assert torch.allclose(result, expected)


def test_covariance_pooling():
    # last_hidden_states[i] is a token sequence with padding; attention_mask[i] marks valid tokens
    # expected[i] = Σ(X_i), computed analytically
    last_hidden_states = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [999.0, 999.0, 999.0]],
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 1, 1],
        ]
    )
    result = covariance_pooling(last_hidden_states, attention_mask)
    expected_b0 = torch.full((3, 3), 2.25, dtype=result.dtype)
    expected_b1 = torch.tensor(
        [
            [8.0 / 9.0, -4.0 / 9.0, -4.0 / 9.0],
            [-4.0 / 9.0, 8.0 / 9.0, -4.0 / 9.0],
            [-4.0 / 9.0, -4.0 / 9.0, 8.0 / 9.0],
        ],
        dtype=result.dtype,
    )
    expected = torch.stack([expected_b0, expected_b1], dim=0)
    assert result.shape == expected.shape
    assert torch.allclose(result, expected, atol=1e-6, rtol=0.0)
