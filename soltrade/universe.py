import requests
import time
from config import config

def fetch_coins_by_volume_and_cap(min_cap=1_000_000, max_cap=25_000_000, retries=5, wait=60):

    coins_meeting_criteria = []
    retry_count, offset = 0, 0
    exhausted = False

    while True:

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
                volume = int(coin.get('v24hUSD', 0) or 0)
                market_cap = int(coin.get('mc', 0) or 0)
                liquidity = int(coin.get("liquidity", 0) or 0)
                enough_liquidity = liquidity > 100_000 # more complex logic later
                enough_volume = True if volume > market_cap / 5 else False

                if (volume != 0) and (volume < min_cap / 5):
                    exhausted = True
                    break

                if enough_volume and enough_liquidity and (min_cap < market_cap < max_cap) and ('Wormhole' not in coin['name']):
                    entry = {
                        'address': coin['address'],
                        'symbol': coin['symbol'],
                        'name': coin['name'],
                        'volume': volume,
                        'market_cap': market_cap,
                        'liquidity': liquidity
                    }
                    print(entry)
                    coins_meeting_criteria.append(entry)
            if exhausted:
                break
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
    print("Initial universe filtering complete")
    return coins_meeting_criteria


if __name__ == '__main__':
    config_path = '.\config.ini'
    config(config_path) 
    fetch_coins_by_volume_and_cap()