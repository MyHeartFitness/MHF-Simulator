import numpy as np
from scipy.stats import truncnorm, beta as beta_dist

def _truncnorm_samples(mean, sd, low, high, size):
    # Convert bounds to std normal space for truncnorm
    a, b = (low - mean) / sd, (high - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size)

def sample_uniform(low, high, size):
    return np.random.uniform(low, high, size)

def sample_normal(low, high, mean, sd, size):
    return _truncnorm_samples(mean, sd, low, high, size)

def sample_lognormal(low, high, mean, sd, size):
    """
    mean/sd here are for the underlying normal (mu/sigma). We sample and truncate.
    """
    out = np.random.lognormal(mean=mean, sigma=sd, size=size)
    # Re-sample out-of-range values until all within [low, high]
    mask = (out < low) | (out > high)
    tries = 0
    while mask.any() and tries < 10:
        out[mask] = np.random.lognormal(mean=mean, sigma=sd, size=mask.sum())
        mask = (out < low) | (out > high)
        tries += 1
    # Final clamp (rare)
    return np.clip(out, low, high)

def sample_beta_scaled(low, high, alpha, beta, size):
    # Beta on [0,1], then scale to [low, high]
    raw = beta_dist.rvs(alpha, beta, size=size)
    return low + raw * (high - low)

def sample_binary(prob_true, size):
    return np.random.rand(size) < prob_true

def sample_categorical(proportions, size):
    """
    proportions: list of floats summing ~1.0
    returns indices 0..k-1
    """
    proportions = np.array(proportions, dtype=float)
    proportions = proportions / proportions.sum()
    return np.random.choice(len(proportions), p=proportions, size=size)
