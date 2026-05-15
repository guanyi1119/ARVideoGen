import functools
import logging
import time

logger = logging.getLogger(__name__)


def retry_load_error(max_retries=3, delay=1):
    """Decorator to retry function calls on load errors."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Load error (attempt {attempt+1}/{max_retries}): {e}")
                    time.sleep(delay)
        return wrapper
    return decorator
