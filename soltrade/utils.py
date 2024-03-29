import time
from functools import wraps

from solana.exceptions import SolanaRpcException
from log import log_general

def handle_rate_limiting_solana_rpc(retry_attempts=3, retry_delay=10):
    def decorator(client_function):
        @wraps(client_function)
        def wrapper(*args, **kwargs):
            for _ in range(retry_attempts):
                try:
                    return client_function(*args, **kwargs)
                except SolanaRpcException as e:
                    if 'HTTPStatusError' in e.error_msg:
                        log_general.warning(f"Rate limit exceeded in {client_function.__name__}, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        raise
            log_general.warning("Rate limit error persisting, skipping this iteration.")
            return None

        return wrapper

    return decorator


def handle_rate_limiting_birdeye(retry_attempts=3, retry_delay=10, doubling=False):
    def decorator(client_function):
        @wraps(client_function)
        def wrapper(*args, **kwargs):
            current_delay = retry_delay
            for attempt in range(retry_attempts):
                response = client_function(*args, **kwargs)
                if response.status_code == 200:
                    return response.json()  # Assuming successful response returns JSON
                elif response.status_code == 429:
                    log_general.warning(f"Rate limit exceeded in {client_function.__name__}, attempt {attempt+1} of {retry_attempts}. Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    if doubling:
                        current_delay *= 2
                else:
                    response.raise_for_status()  # Raises HTTPError for bad responses
            log_general.warning(f"All retry attempts failed for {client_function.__name__}.")
            return None
        return wrapper
    return decorator
