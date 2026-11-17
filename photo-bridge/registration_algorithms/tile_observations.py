"""Tile-level displacement observations for the hierarchical registration."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _window(shape):
    height, width = shape
    wy = np.hanning(height).astype(np.float32)
    wx = np.hanning(width).astype(np.float32)
    return np.outer(wy, wx)


def score_shift(first: np.ndarray, second: np.ndarray, mask: np.ndarray,
                shift_x: float, shift_y: float) -> float:
    warp = np.array([[1.0, 0.0, shift_x],
                     [0.0, 1.0, shift_y]], np.float32)
    aligned = cv2.warpAffine(second, warp, (second.shape[1], second.shape[0]),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    valid = cv2.warpAffine(mask.astype(np.uint8), warp, (mask.shape[1], mask.shape[0]),
                           flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0) > 0
    if valid.sum() < 16:
        return float("nan")
    a = first[valid]
    b = aligned[valid]
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return float(np.sum(a * b) / denom) if denom > 1e-12 else float("nan")


def _peak_shifts(correlation: np.ndarray, top_k: int, min_separation: int):
    height, width = correlation.shape
    response = np.abs(correlation).astype(np.float32)
    # Ignore the periodic wrap border and the zero-shift DC region.
    margin = max(2, min_separation)
    response[:margin, :] = 0
    response[-margin:, :] = 0
    response[:, :margin] = 0
    response[:, -margin:] = 0
    peaks = []
    working = response.copy()
    for _ in range(top_k):
        index = int(np.argmax(working))
        value = float(working.flat[index])
        if value <= 0:
            break
        row, column = divmod(index, width)
        shift_y = row if row <= height // 2 else row - height
        shift_x = column if column <= width // 2 else column - width
        peaks.append((float(value), int(shift_x), int(shift_y)))
        rr = max(0, row - min_separation)
        cc = max(0, column - min_separation)
        working[rr:min(height, row + min_separation + 1),
                cc:min(width, column + min_separation + 1)] = 0
    return peaks


def tile_candidates(first: np.ndarray, second: np.ndarray,
                    valid_mask: np.ndarray, resolution_m: float,
                    top_k: int = 5, min_separation_px: int = 2):
    """Return top phase-correlation candidates for one tile.

    The returned shifts use the same convention as the ECC tile output:
    dx_m is east, dy_m is north.
    """
    if first.shape != second.shape or first.shape != valid_mask.shape:
        raise ValueError("tile arrays and mask must share a shape")
    window = _window(first.shape)
    first_window = first.astype(np.float32) * window
    second_window = second.astype(np.float32) * window
    spectrum = (np.fft.fft2(first_window)
                * np.conj(np.fft.fft2(second_window)))
    magnitude = np.abs(spectrum)
    spectrum = np.divide(spectrum, magnitude, out=np.zeros_like(spectrum),
                         where=magnitude > 1e-9)
    correlation = np.fft.ifft2(spectrum).real
    candidates = []
    seen = []
    for phase_score, shift_x, shift_y in _peak_shifts(
            correlation, top_k=top_k, min_separation=min_separation_px):
        # Phase peak sign is ambiguous from the FFT alone; score both signs and
        # keep the one that actually aligns the image content.
        for sign in (1.0, -1.0):
            sx = sign * shift_x
            sy = sign * shift_y
            if any(abs(sx - x) <= min_separation_px
                   and abs(sy - y) <= min_separation_px for x, y in seen):
                continue
            score = score_shift(first, second, valid_mask, sx, sy)
            if not np.isfinite(score):
                continue
            seen.append((sx, sy))
            candidates.append({
                "dx_m": float(-sx * resolution_m),
                "dy_m": float(sy * resolution_m),
                "score": float(score),
                "phase_score": float(phase_score),
                "shift_px": (float(sx), float(sy)),
            })
    candidates.sort(key=lambda row: row["score"], reverse=True)
    return candidates[:top_k]
