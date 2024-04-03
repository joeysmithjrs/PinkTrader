import time
from functools import wraps
import asyncio
import aiosqlite
from solana.exceptions import SolanaRpcException
from log import log_general

def handle_rate_limiting_solana_rpc(retry_attempts=5, retry_delay=10, doubling=True):
    def decorator(client_function):
        @wraps(client_function)
        def wrapper(*args, **kwargs):
            current_delay = retry_delay  # Initialize current delay to the initial retry delay
            for attempt in range(retry_attempts):
                try:
                    return client_function(*args, **kwargs)
                except SolanaRpcException as e:
                    if 'HTTPStatusError' in e.error_msg:
                        log_general.warning(f"Rate limit exceeded in {client_function.__name__}, attempt {attempt + 1} of {retry_attempts}. Retrying in {current_delay} seconds...")
                        time.sleep(current_delay)
                        if doubling:
                            current_delay *= 2  # Double the delay if doubling is True
                    else:
                        raise
            log_general.warning(f"All retry attempts failed for {client_function.__name__}.")
            return None
        return wrapper
    return decorator


def handle_rate_limiting_birdeye(retry_attempts=5, retry_delay=10, doubling=True):
    def decorator(client_function):
        @wraps(client_function)
        def wrapper(*args, **kwargs):
            current_delay = retry_delay
            for attempt in range(retry_attempts):
                response = client_function(*args, **kwargs)
                if response.status_code == 200:
                    return response.json()  # Assuming successful response returns JSON
                elif response.status_code == 429:
                    log_general.warning(f"Rate limit exceeded in {client_function.__name__}, attempt {attempt + 1} of {retry_attempts}. Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    if doubling:
                        current_delay *= 2
                else:
                    response.raise_for_status()  # Raises HTTPError for bad responses
            log_general.warning(f"All retry attempts failed for {client_function.__name__}.")
            return None
        return wrapper
    return decorator


def handle_sqlite_lock(retry_attempts=10, retry_delay=0.25, doubling=True):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = retry_delay
            for attempt in range(retry_attempts):
                try:
                    # Since the decorated function is async, we use await here
                    return await func(*args, **kwargs)
                except aiosqlite.OperationalError as e:
                    # Check if the exception is due to a lock
                    if 'locked' in str(e).lower():
                        log_general.warning(f"SQLite database is locked in {func.__name__}, attempt {attempt + 1} of {retry_attempts}. Retrying in {current_delay} seconds...")
                        await asyncio.sleep(current_delay)
                        if doubling:
                            current_delay *= 2
                    else:
                        # If the error is not a lock, re-raise it
                        raise
            # Optionally, you might want to raise an exception if all attempts fail
            log_general.warning(f"All retry attempts failed for {func.__name__}.")
            return None
        return wrapper
    return decorator

