"""augmentation.py — Audio sequence augmentations for Test-Time Augmentation (TTA).

This module implements sequence shifts and temporal flips used during TTA to
regularize predictions and reduce window boundary effects.
"""

import numpy as np
import torch


def apply_sequence_shift(
    tensor: torch.Tensor, shift: int, dims: int = 1
) -> torch.Tensor:
    """Applies a circular roll shift along the time sequence dimension.

    Summary:
        Augments sequences by shifting window steps circularly to combat boundary artifacts.

    Inputs:
        tensor (torch.Tensor): The input sequence tensor.
        shift (int): The shift size.
        dims (int): The dimension along which to shift. Default is 1 (time dimension).

    Outputs:
        torch.Tensor: The shifted tensor.

    Shapes:
        - Input: `(B, T, D)`
        - Output: `(B, T, D)`

    Side effects:
        None.

    Usage example:
        >>> x = torch.randn(10, 12, 1536)
        >>> x_shifted = apply_sequence_shift(x, 1)
    """
    if not shift:
        return tensor
    return torch.roll(tensor, shift, dims=dims)


def reverse_predictions_shift(
    preds: np.ndarray, shift: int, axis: int = 1
) -> np.ndarray:
    """Reverses the circular shift applied during test-time augmentation.

    Summary:
        Rolls predictions back to align with the original time axis.

    Inputs:
        preds (np.ndarray): Shifted predictions array.
        shift (int): The original shift amount.
        axis (int): The axis of shift. Default is 1.

    Outputs:
        np.ndarray: The unshifted predictions.

    Shapes:
        - Input: `(B, T, C)`
        - Output: `(B, T, C)`

    Side effects:
        None.

    Usage example:
        >>> p = np.random.rand(10, 12, 234)
        >>> p_orig = reverse_predictions_shift(p, 1)
    """
    if not shift:
        return preds
    return np.roll(preds, -shift, axis=axis)


def apply_temporal_flip(tensor: torch.Tensor, dims: int = 1) -> torch.Tensor:
    """Reverses a sequence along the time dimension.

    Summary:
        Augments sequences by flipping the time order (useful for bidirectional models).

    Inputs:
        tensor (torch.Tensor): Input sequence tensor.
        dims (int): Dimension to flip. Default is 1.

    Outputs:
        torch.Tensor: Flipped sequence tensor.

    Shapes:
        - Input: `(B, T, D)`
        - Output: `(B, T, D)`

    Side effects:
        None.

    Usage example:
        >>> x = torch.randn(10, 12, 1536)
        >>> x_flipped = apply_temporal_flip(x)
    """
    return tensor.flip(dims)


def reverse_flipped_predictions(preds: np.ndarray, axis: int = 1) -> np.ndarray:
    """Reverses predictions along the time axis to restore original order.

    Summary:
        Restores temporal sequence order after temporal flip prediction.

    Inputs:
        preds (np.ndarray): Flipped predictions array.
        axis (int): Time dimension axis index. Default is 1.

    Outputs:
        np.ndarray: Restored predictions array.

    Shapes:
        - Input: `(B, T, C)`
        - Output: `(B, T, C)`

    Side effects:
        None.

    Usage example:
        >>> p = np.random.rand(10, 12, 234)
        >>> p_orig = reverse_flipped_predictions(p)
    """
    return np.flip(preds, axis=axis).copy()
