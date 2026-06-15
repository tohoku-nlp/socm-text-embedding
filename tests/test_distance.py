import numpy as np
import torch

from utils.distance import (
    calc_batched_bures_wasserstein_squared_distance,
    calc_batched_euclidean_squared_distance,
)


def test_calc_batched_bures_wasserstein_squared_distance():
    # cov1[i] and cov2[i] are 2x2 covariance matrices
    # true_vals[i] = BW^2(cov1[i], cov2[i]), computed analytically
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    covs1 = torch.tensor(
        [
            [[2.0, 1.0], [1.0, 3.0]],
            [[3.0, 1.0], [1.0, 2.0]],
            [[4.0, 0.0], [0.0, 9.0]],
        ],
        device=device,
        dtype=dtype,
    )
    covs2 = torch.tensor(
        [
            [[5.0, 2.0], [2.0, 2.0]],
            [[2.0, 1.0], [1.0, 5.0]],
            [[1.0, 0.0], [0.0, 16.0]],
        ],
        device=device,
        dtype=dtype,
    )

    pred = calc_batched_bures_wasserstein_squared_distance(covs1, covs2).to(dtype)

    true_vals = torch.tensor(
        [
            np.sqrt(12 - 2 * np.sqrt(20 + 2 * np.sqrt(30))) ** 2,
            np.sqrt(12 - 2 * np.sqrt(15) - 2 * np.sqrt(3)) ** 2,
            (np.sqrt(2)) ** 2,
        ],
        device=device,
        dtype=dtype,
    )

    print("pred:", pred)
    print("true:", true_vals)

    assert torch.allclose(pred, true_vals, atol=1e-7, rtol=1e-7)


def test_calc_batched_euclidean_squared_distance():
    # xs[i] and ys[i] are 2D vectors
    # true_vals[i] = ||xs[i] - ys[i]||^2, computed analytically
    x0 = torch.tensor([0.0, 0.0])
    x1 = torch.tensor([2.0, 0.0])
    x2 = torch.tensor([0.0, 4.0])

    xs = torch.stack([x0, x0, x1])
    ys = torch.stack([x1, x2, x2])

    pred = calc_batched_euclidean_squared_distance(xs, ys)

    true_vals = torch.tensor([4.0, 16.0, 20.0])

    print("pred:", pred)
    print("true:", true_vals)

    assert torch.allclose(pred, true_vals, atol=1e-7, rtol=1e-7)
