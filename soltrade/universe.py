import requests
import time
import sqlite3
from datetime import datetime
from config import config

def fetch_coins_by_market_cap(min_cap=1_000_000, max_cap=250_000_000, retries=5, wait=60):

    universe = {n : [] for n in range(1, 6)}
    retry_count, offset = 0, 0
    conn = sqlite3.connect('../trading_algo.db')
    c = conn.cursor()

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
                now = datetime.now()
                # Check if the asset exists in the tradeable_assets table
                c.execute("SELECT token_address FROM tradeable_assets WHERE token_address = ?", (entry['address'],))
                exists = c.fetchone()

                if exists:
                    url = f"https://public-api.birdeye.so/defi/token_security?address={entry['address']}"
                    top10 = requests.get(url, headers=headers).json()['data'].get('Top10HoldersPercent')
                    c.execute('''INSERT INTO tradeable_asset_info (datetime, token_address, top10holderspct, volume, 
                                volume_change_pct, market_cap, liquidity, volume_pct_market_cap)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (now, entry['token_address'], top10, entry.get('volume'), 
                            entry.get('volume_change_pct'), entry.get('market_cap'), entry.get('liquidity'), entry.get('volume_pct_market_cap')))
                    
                preliminaries = volume != 0 and market_cap != 0 and liquidity != 0 and name != '_' and address != '_' and symbol != '_' and 'Wormhole' not in name
                
                if (min_cap < market_cap < max_cap) and preliminaries:
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
    conn.commit()
    conn.close()
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

def update_tradeable_assets():
    conn = sqlite3.connect('../trading_algo.db')
    c = conn.cursor()

    # Fetch the current universe of tradeable assets
    universe = fetch_coins_by_market_cap()
    filtered_universe = filter_universe_by_volume_and_liquidity(universe)
    print(len(filtered_universe))
    for coin in filtered_universe:
        print(coin)

    # First, update all assets to not currently tradeable before selectively enabling those that are.
    c.execute("UPDATE tradeable_assets SET currently_tradeable = FALSE")

    for asset in universe:
        now = datetime.now()

        # Check if the asset exists in the tradeable_assets table
        c.execute("SELECT token_address FROM tradeable_assets WHERE token_address = ?", (asset['token_address'],))
        exists = c.fetchone()

        if exists:
            # Update the currently_tradeable flag for existing assets
            c.execute("UPDATE tradeable_assets SET currently_tradeable = TRUE WHERE token_address = ?", (asset['token_address'],))
        else:
            # Insert new asset into tradeable_assets
            c.execute('''INSERT INTO tradeable_assets (token_address, name, symbol, creation_datetime, currently_tradeable)
                         VALUES (?, ?, ?, ?, ?)''', 
                      (asset['token_address'], asset['name'], asset['symbol'], asset['creation_datetime'], True))
            
            # Add first entry into tradeable_asset_info
            c.execute('''INSERT INTO tradeable_asset_info (datetime, token_address, top10holderspct, volume, 
                        volume_change_pct, market_cap, liquidity, volume_pct_market_cap)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', 
                    (now, asset['token_address'], asset.get('top10holderspct'), asset.get('volume'), 
                    asset.get('volume_change_pct'), asset.get('market_cap'), asset.get('liquidity'), asset.get('volume_pct_market_cap')))

    # Commit changes
    conn.commit()
    conn.close()


config_path = '.\config.ini'
config(config_path) 
update_tradeable_assets()