import time
from functools import wraps
import asyncio
import aiohttp
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
        async def wrapper(*args, **kwargs):
            current_delay = retry_delay
            for attempt in range(retry_attempts):
                try:
                    response = await client_function(*args, **kwargs)
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        log_general.warning(f"Rate limit exceeded in {client_function.__name__}, attempt {attempt + 1} of {retry_attempts}. Retrying in {current_delay} seconds...")
                        await asyncio.sleep(current_delay)
                        if doubling:
                            current_delay *= 2
                    else:
                        response.raise_for_status()
                except aiohttp.ClientError as e:
                    log_general.warning(f"HTTP client error in {client_function.__name__}: {e}")
                    return None
            log_general.warning(f"All retry attempts failed for {client_function.__name__}.")
            return None
        return wrapper
    return decorator

def handle_sqlite_lock(retry_attempts=10, retry_delay=0.25, doubling=True):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = retry_delay
            last_exception = None
            for attempt in range(retry_attempts):
                try:
                    return await func(*args, **kwargs)
                except aiosqlite.OperationalError as e:
                    last_exception = e
                    if 'locked' in str(e).lower():
                        log_general.warning(f"SQLite database is locked in {func.__name__}, attempt {attempt + 1} of {retry_attempts}. Retrying in {current_delay} seconds...")
                        await asyncio.sleep(current_delay)
                        if doubling:
                            current_delay *= 2
                    else:
                        raise
            log_general.warning(f"All retry attempts failed for {func.__name__}.")
            raise last_exception
        return wrapper
    return decorator

def handle_database_connection(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        self = args[0]  # Assumes the first argument to the function is `self`
        await self.open_database_connection()
        try:
            return await func(*args, **kwargs)
        finally:
            await self.close_database_connection()
    return wrapper

def handle_aiohttp_session(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        self = args[0]  # Assumes the first argument to the function is `self`
        await self.open_aiohttp_session()
        try:
            return await func(*args, **kwargs)
        finally:
            await self.close_aiohttp_session()
    return wrapper

def convert_or_default(value, target_type, default_value):
    try:
        return target_type(value)
    except (TypeError, ValueError):
        return default_value

def interval_to_seconds(interval):
    mapping = {
        '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
        '1H': 3600, '2H': 7200, '4H': 14400, '6H': 21600, '8H': 28800,
        '12H': 43200, '1D': 86400, '3D': 259200, '1W': 604800, '1M': 2592000
    }
    return mapping.get(interval, 0)

def format_universe_composition(market_cap_bins, universe):
    logging_statements = []
    for idx, bin_range in enumerate(market_cap_bins, start=1):
        bin_start, bin_end = bin_range
        token_count = len(universe[idx])
        logging_statements.append(f"${bin_start:,}-${bin_end:,}: {token_count} tokens")

    final_logging_statement = ", ".join(logging_statements)
    return f"Tradeable universe filtered with final market cap composition - {final_logging_statement}"