def calc_socm(d_mu: float, d_sigma: float) -> float:
    """Compute the degree of Second-Order Collapse by Mean pooling (SOCM).

    SOCM(d_mu, d_sigma) := (1 - d_mu) * d_sigma

    Parameters
    ----------
    d_mu : float
        Scaled squared Euclidean distance between first-order statistics.
    d_sigma : float
        Scaled Bures-Wasserstein distance between second-order statistics.

    Returns
    -------
    float
        SOCM value in [0, 1]. Higher indicates more severe collapse.
    """
    return (1 - d_mu) * d_sigma
