import random

import numpy as np
import torch


def fix_seeds(seed: int = 0):
    """Fix random seeds.

    Parameters
    ----------
    seed : int, optional
        Random seed, by default 0
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
