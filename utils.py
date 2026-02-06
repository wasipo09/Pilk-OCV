#!/usr/bin/env python3
"""Utility functions for Pilk-OCV options analysis.

Provides common utilities for API calls, caching, logging, and error handling.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

import requests
import ccxt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Simple in-memory cache
_CACHE = {}
CACHE_TTL = 300  # 5 minutes default TTL

T = TypeVar('T')


class APIError(Exception):
    """Base exception for API-related errors."""
    pass


class RateLimitError(APIError):
    """Raised when rate limit is hit."""
    pass


class DataValidationError(APIError):
    """Raised when API response data is invalid."""
    pass


def cache_get(key: str) -> Optional[Any]:
    """Get value from cache if not expired."""
    if key in _CACHE:
        value, timestamp = _CACHE[key]
        if time.time() - timestamp < CACHE_TTL:
            return value
        del _CACHE[key]
    return None


def cache_set(key: str, value: Any, ttl: Optional[int] = None) -> None:
    """Set value in cache with optional TTL."""
    _CACHE[key] = (value, time.time())


def clear_cache() -> None:
    """Clear all cached values."""
    global _CACHE
    _CACHE = {}


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (requests.exceptions.RequestException, APIError)
) -> Callable:
    """Decorator for retrying functions with exponential backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = base_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}: {e}")
                        raise

                    # Check if rate limited
                    if isinstance(e, requests.exceptions.HTTPError):
                        if hasattr(e.response, 'status_code') and e.response.status_code == 429:
                            logger.warning(f"Rate limited on {func.__name__}, attempt {attempt + 1}/{max_retries}")
                            # Use longer delay for rate limits
                            delay = min(delay * backoff_factor * 2, max_delay)
                        else:
                            logger.warning(f"Request failed on {func.__name__}, attempt {attempt + 1}/{max_retries}: {e}")
                            delay = min(delay * backoff_factor, max_delay)
                    else:
                        logger.warning(f"Error on {func.__name__}, attempt {attempt + 1}/{max_retries}: {e}")
                        delay = min(delay * backoff_factor, max_delay)

                    time.sleep(delay)

            raise last_exception  # type: ignore

        return wrapper
    return decorator


@retry_with_backoff(max_retries=3, base_delay=0.3)
def api_get(
    url: str,
    params: dict | None = None,
    timeout: int = 30,
    raise_on_error: bool = True
) -> dict:
    """Make GET request with error handling and retry logic.

    Args:
        url: The API endpoint URL
        params: Query parameters
        timeout: Request timeout in seconds
        raise_on_error: Whether to raise exception on HTTP errors

    Returns:
        JSON response as dict

    Raises:
        APIError: If the request fails and raise_on_error is True
    """
    try:
        response = requests.get(url, params=params, timeout=timeout)
        if raise_on_error:
            response.raise_for_status()

        data = response.json()

        # Validate response structure
        if not isinstance(data, dict):
            raise DataValidationError(f"Expected dict response, got {type(data).__name__}")

        # Check for API-level errors
        if 'error' in data:
            raise APIError(f"API returned error: {data['error']}")

        return data

    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout fetching {url}: {e}")
        raise APIError(f"Request timed out after {timeout}s") from e
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {url}: {e}")
        raise APIError(f"Request failed: {e}") from e
    except ValueError as e:
        logger.error(f"Invalid JSON response from {url}: {e}")
        raise DataValidationError(f"Invalid JSON response: {e}") from e


# CCXT exchange instance cache
_exchange_instances: dict = {}


def get_exchange(exchange_name: str) -> ccxt.Exchange:
    """Get or create cached CCXT exchange instance.

    Args:
        exchange_name: Name of exchange (e.g., 'binance')

    Returns:
        CCXT exchange instance
    """
    if exchange_name not in _exchange_instances:
        try:
            exchange_class = getattr(ccxt, exchange_name.lower())
            _exchange_instances[exchange_name] = exchange_class({
                'enableRateLimit': True,  # Let CCXT handle rate limiting
                'timeout': 30000,  # 30 second timeout
            })
            logger.debug(f"Created CCXT exchange instance for {exchange_name}")
        except AttributeError as e:
            raise APIError(f"Unknown exchange: {exchange_name}") from e
        except Exception as e:
            raise APIError(f"Failed to initialize {exchange_name}: {e}") from e

    return _exchange_instances[exchange_name]


@retry_with_backoff(max_retries=2, base_delay=0.2)
def fetch_ccxt_ticker(exchange_name: str, symbol: str) -> float:
    """Fetch ticker price using CCXT with retry logic.

    Args:
        exchange_name: Name of exchange
        symbol: Trading pair symbol (e.g., 'BTC/USDT')

    Returns:
        Last price as float

    Raises:
        APIError: If fetching fails
    """
    try:
        exchange = get_exchange(exchange_name)
        ticker = exchange.fetch_ticker(symbol)
        price = ticker.get('last')

        if price is None:
            raise DataValidationError(f"No 'last' price in ticker response for {symbol}")

        return float(price)

    except ccxt.NetworkError as e:
        raise APIError(f"Network error fetching {symbol}: {e}") from e
    except ccxt.ExchangeError as e:
        raise APIError(f"Exchange error fetching {symbol}: {e}") from e
    except (KeyError, ValueError) as e:
        raise DataValidationError(f"Invalid ticker response for {symbol}: {e}") from e


def validate_expiry(expiry: str) -> str:
    """Validate and normalize expiry string format.

    Args:
        expiry: Expiry string (e.g., '07FEB26' or '260207')

    Returns:
        Normalized expiry string

    Raises:
        ValueError: If expiry format is invalid
    """
    expiry = expiry.strip().upper()

    # Try to parse common formats
    for fmt in ('%d%b%y', '%y%m%d', '%d%b%Y'):
        try:
            datetime.strptime(expiry, fmt)
            return expiry
        except ValueError:
            continue

    raise ValueError(
        f"Invalid expiry format: '{expiry}'. "
        "Expected format like '07FEB26' or '260207'."
    )


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default on division by zero.

    Args:
        numerator: The dividend
        denominator: The divisor
        default: Value to return if denominator is zero

    Returns:
        Result of division or default value
    """
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default


def format_number(value: float, decimals: int = 2) -> str:
    """Format number for display with appropriate precision.

    Args:
        value: Number to format
        decimals: Number of decimal places

    Returns:
        Formatted string
    """
    try:
        return f"{value:.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"
