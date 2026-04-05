from __future__ import annotations

import numpy as np

from src.models.schemas import ToolCategory
from src.tools.registry import register_tool


@register_tool(
    name="compute_rsi",
    description="Compute Relative Strength Index for a price series",
    category=ToolCategory.INDICATOR,
    input_schema={
        "type": "object",
        "properties": {
            "prices": {
                "type": "array",
                "items": {"type": "number"},
                "description": "List of closing prices",
            },
            "period": {
                "type": "integer",
                "description": "RSI period (default 14)",
                "default": 14,
            },
        },
        "required": ["prices"],
    },
)
def compute_rsi(prices: list[float], period: int = 14) -> float | None:
    if len(prices) < period + 1:
        return None

    arr = np.array(prices, dtype=float)
    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


@register_tool(
    name="compute_sma",
    description="Compute Simple Moving Average",
    category=ToolCategory.INDICATOR,
    input_schema={
        "type": "object",
        "properties": {
            "prices": {"type": "array", "items": {"type": "number"}},
            "window": {"type": "integer", "description": "SMA window size"},
        },
        "required": ["prices", "window"],
    },
)
def compute_sma(prices: list[float], window: int) -> float | None:
    if len(prices) < window:
        return None
    return round(float(np.mean(prices[-window:])), 4)


@register_tool(
    name="compute_ema",
    description="Compute Exponential Moving Average",
    category=ToolCategory.INDICATOR,
    input_schema={
        "type": "object",
        "properties": {
            "prices": {"type": "array", "items": {"type": "number"}},
            "window": {"type": "integer", "description": "EMA window size"},
        },
        "required": ["prices", "window"],
    },
)
def compute_ema(prices: list[float], window: int) -> float | None:
    if len(prices) < window:
        return None
    arr = np.array(prices, dtype=float)
    multiplier = 2 / (window + 1)
    ema = arr[0]
    for price in arr[1:]:
        ema = (price - ema) * multiplier + ema
    return round(float(ema), 4)


@register_tool(
    name="compute_macd",
    description="Compute MACD (12, 26, 9) — returns macd_line, signal_line, histogram",
    category=ToolCategory.INDICATOR,
    input_schema={
        "type": "object",
        "properties": {
            "prices": {"type": "array", "items": {"type": "number"}},
        },
        "required": ["prices"],
    },
)
def compute_macd(
    prices: list[float],
) -> dict[str, float] | None:
    if len(prices) < 26:
        return None

    def _ema(data: np.ndarray, span: int) -> np.ndarray:
        result = np.empty_like(data)
        multiplier = 2 / (span + 1)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
        return result

    arr = np.array(prices, dtype=float)
    ema12 = _ema(arr, 12)
    ema26 = _ema(arr, 26)
    macd_line = ema12 - ema26

    signal_line = _ema(macd_line, 9)
    histogram = macd_line - signal_line

    return {
        "macd_line": round(float(macd_line[-1]), 4),
        "signal_line": round(float(signal_line[-1]), 4),
        "histogram": round(float(histogram[-1]), 4),
    }


@register_tool(
    name="compute_bollinger_bands",
    description="Compute Bollinger Bands (middle, upper, lower)",
    category=ToolCategory.INDICATOR,
    input_schema={
        "type": "object",
        "properties": {
            "prices": {"type": "array", "items": {"type": "number"}},
            "window": {"type": "integer", "default": 20},
            "num_std": {"type": "number", "default": 2.0},
        },
        "required": ["prices"],
    },
)
def compute_bollinger_bands(
    prices: list[float], window: int = 20, num_std: float = 2.0
) -> dict[str, float] | None:
    if len(prices) < window:
        return None

    recent = np.array(prices[-window:], dtype=float)
    middle = float(np.mean(recent))
    std = float(np.std(recent, ddof=1))

    return {
        "upper": round(middle + num_std * std, 4),
        "middle": round(middle, 4),
        "lower": round(middle - num_std * std, 4),
    }
