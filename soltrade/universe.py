import requests
import time
from config import config

def fetch_coins_by_market_cap(min_cap=1_000_000, max_cap=250_000_000, retries=5, wait=60):

    universe = {n : [] for n in range(1, 6)}
    retry_count, offset = 0, 0
    exhausted = False

    while offset <= 500:

        url = f"https://public-api.birdeye.so/public/tokenlist?sort_by=v24hUSD&sort_type=desc&offset={offset}&limit=50"

        headers = {
            "x-chain": "solana",
            "X-API-KEY": config().birdeye_api_key
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            coins = response.json()
            if not coins:
                break  
            for coin in coins['data']['tokens']:
                volume = float(coin.get('v24hUSD', 0) or 0)
                volume_change_pct = float(coin.get('v24hChangePercent', 0) or 0)
                market_cap = float(coin.get('mc', 0) or 0)
                liquidity = float(coin.get("liquidity", 0) or 0)
                name = str(coin.get('name', '_') or '_')
                address = str(coin.get('address', '_') or '_')
                symbol = str(coin.get('symbol', '_') or '_')

                preliminaries = volume != 0 and market_cap != 0 and liquidity != 0 and name != '_' and address != '_' and symbol != '_' and 'Wormhole' not in name

                if (min_cap < market_cap < max_cap) and preliminaries:
                    entry = {
                        'address': coin['address'],
                        'symbol': coin['symbol'],
                        'name': coin['name'],
                        'volume': volume,
                        'volume_change_pct': volume_change_pct,
                        'market_cap': market_cap,
                        'liquidity': liquidity,
                        'volume_pct_market_cap': volume / market_cap
                    }
                    # print(entry)
                    match market_cap:
                        case mc if 1_000_000 <= mc < 5_000_000:
                            universe[1].append(entry)
                        case mc if 5_000_000 <= mc < 25_000_000:
                            universe[2].append(entry)
                        case mc if 25_000_000 <= mc < 50_000_000:
                            universe[3].append(entry)
                        case mc if 50_000_000 <= mc < 100_000_000:
                            universe[4].append(entry)
                        case mc if 100_000_000 <= mc < 250_000_000:
                            universe[5].append(entry)
            offset += 50
            retry_count = 0  # Reset retry count after a successful request
        elif response.status_code == 429:
            print("Rate limit exceeded. Waiting before retrying...")
            time.sleep(wait)  # Wait for a specified time
            retry_count += 1
            if retry_count > retries:
                print("Maximum retries exceeded. Exiting...")
                break
        else:
            print(f"Failed to fetch data: HTTP {response.status_code}")
            break
    print(f"Initial universe filtering complete")
    print({key: len(val) for key, val in universe.items()})
    return universe

def filter_universe_by_age_and_safety(universe, min_hours_since_creation = 12, max_top_10_holders_pct = .30):
    pass

def filter_universe_by_volume_and_liquidity(universe, min_liquidity = 100_000, min_volume_pct_market_cap_quintile = 3, min_volume_change_pct_quintile = 2):
    def calculate_quintile_indices(length):
        return [int(length * i / 5) for i in range(6)]
    
    def filter_by_quintile(data, key, min_quintile):
        if len(data) < 5:  # If less than 5 items, exclude the whole tier
            return []
        sorted_data = sorted(data, key=lambda x: x[key])
        indices = calculate_quintile_indices(len(sorted_data))
        quintile_cutoff_index = indices[min_quintile - 1]
        return sorted_data[quintile_cutoff_index:]
    
    filtered_universe = []
    for _, coins in universe.items():
        liquidity_filtered_coins = [coin for coin in coins if coin['liquidity'] >= min_liquidity]

        volume_market_cap_filtered = filter_by_quintile(liquidity_filtered_coins, 'volume_pct_market_cap', min_volume_pct_market_cap_quintile)

        final_filtered = filter_by_quintile(volume_market_cap_filtered, 'volume_change_pct', min_volume_change_pct_quintile)
        
        filtered_universe.extend(final_filtered)
        
    return filtered_universe


if __name__ == '__main__':
    config_path = '.\config.ini'
    config(config_path) 
    universe = fetch_coins_by_market_cap()
    filtered_universe = filter_universe_by_volume_and_liquidity(universe)
    print(len(filtered_universe))
