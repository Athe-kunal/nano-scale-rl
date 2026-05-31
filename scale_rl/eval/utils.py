from math import comb

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)