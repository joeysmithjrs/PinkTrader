import requests
import json
import time
import sqlite3
from datetime import datetime, timedelta
from log import log_general
from config import config
from utils import handle_rate_limiting_birdeye

class Universe:

    def __init__(self, universe_configs_path):
        self.path = universe_configs_path
        with open(self.path, 'r') as file:
            self.configs = json.load(file)
        self.universe_id = self.configs.get('universe_id')
        self.platform = self.configs.get('platform')
        self.token_list_sort_by = self.configs.get('token_list_sort_by')
        self.token_list_sort_type = self.configs.get('token_list_sort_type')
        self.page_limit = self.configs.get('token_list_page_limit')
        self.static_token_address_list = self.configs.get('static_token_address_list')
        self.intervals = self.configs.get('intervals')
        self.api_token_fetch_limit = self.configs.get('api_token_fetch_limit')
        self.market_cap_bins = self.configs.get('market_cap_bins')
        self.min_hours_since_creation = self.configs.get('min_hours_since_creation')
        self.max_top_10_holders_pct = self.configs.get('max_top_10_holders_pct')
        self.min_liquidity = self.configs.get('min_liquidity')
        self.min_volume_pct_market_cap_quintile = self.get('min_volume_pct_market_cap_quintile')
        self.min_volume_change_pct_quintile = self.get('min_volume_change_pct_quintile')
        self.headers = {
            "x-chain": self.platform,
            "X-API-KEY": config().birdeye_api_key
        }
        self.conn, self.cur = None, None
        self.init_db_column()

    def __del__(self):
        self.close_database_connection()

    def open_database_connection(self):
        self.conn = sqlite3.connect(self.configs.database_path)
        self.cur = self.conn.cursor()
        log_general.info("SQLite Database connection opened")
    
    def close_database_connection(self):
        self.conn.commit()
        self.conn.close()
        log_general.info("SQLite Database connection closed")

    def init_df_column(self):
        self.open_database_connection()
        column_name = self.configs.get('universe_id')
        if column_name:
            self.cur.execute(f"PRAGMA table_info(tradeable_assets)")
            columns = [info[1] for info in self.cur.fetchall()]
            if column_name not in columns:
                self.cur.execute(f"ALTER TABLE tradeable_assets ADD COLUMN {column_name} BOOLEAN DEFAULT FALSE")
                self.close_database_connection()
        else:
            log_general.error(f"Critical error: parameter ('universe_id') not found in {self.path}")
            self.close_database_connection()
            exit()

    def update_tradeable_assets_info(self, entry):
        now = datetime.now()
        top10 = self.fetch_token_security_info(entry)
        self.cur.execute('''INSERT INTO tradeable_asset_info (datetime, token_address, top10holderspct, volume, 
                    volume_change_pct, market_cap, liquidity, volume_pct_market_cap)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (now, entry.get('token_address'), top10, entry.get('volume'), 
                    entry.get('volume_change_pct'), entry.get('market_cap'), entry.get('liquidity'), entry.get('volume_pct_market_cap')))
        self.conn.commit()
        log_general.info(f"Token info updated for token_address: {entry.get('token_address')} for universe_id: {self.configs.get('universe_id')}")
    
    @handle_rate_limiting_birdeye()
    def fetch_token_security_info(self, entry):
        url = f"https://public-api.birdeye.so/defi/token_security?address={entry.get('address')}"
        return requests.get(url, headers=self.headers)
    
    @handle_rate_limiting_birdeye()
    def fetch_token_list_page(self, offset):
        url = f"https://public-api.birdeye.so/public/tokenlist?sort_by={self.configs.get("token_list_sort_by")}&sort_type={self.configs.get("token_list_sort_type")}&offset={offset}&limit={self.configs.get("token_list_page_limit")}"
        return requests.get(url, headers=self.headers)
    
    def fill_entry(self, coin):
        volume = float(coin.get('v24hUSD', 0) or 0)
        volume_change_pct = float(coin.get('v24hChangePercent', 0) or 0)
        market_cap = float(coin.get('mc', 0) or 0)
        liquidity = float(coin.get("liquidity", 0) or 0)
        name = str(coin.get('name', '_') or '_')
        address = str(coin.get('address', '_') or '_')
        symbol = str(coin.get('symbol', '_') or '_')

        entry = {'address': coin['address'], 'symbol': coin['symbol'],
                'name': coin['name'], 'volume': volume,
                'volume_change_pct': volume_change_pct, 'market_cap': market_cap,
                'liquidity': liquidity, 'volume_pct_market_cap': volume / market_cap}
        preliminaries = volume != 0 and market_cap != 0 and liquidity != 0 and name != '_' and address != '_' and symbol != '_' and 'Wormhole' not in name

        return entry, preliminaries
    
    def token_exists_in_database(self, token_address):
        self.cur.execute("SELECT token_address FROM tradeable_assets WHERE token_address = ?", (token_address))
        return self.curr.fetchone()

    def fetch_coins_by_market_cap(self):

        self.open_database_connection()
        universe = {n : [] for n in range(1, len(self.market_cap_bins)+1)}
        min_mc, max_mc = self.market_cap_bins[0][0], self.market_cap_bins[-1][1]
        offset = 0
        while offset <= self.api_token_fetch_limit:

            coins = self.fetch_token_list_page(offset)
            if not coins:
                break  
            for coin in coins['data']['tokens']:
                entry, preliminaries = self.fill_entry(coin)
                market_cap = entry.get('market_cap')
                if self.token_exists_in_database(entry.get('address')):
                    self.update_token_info()
                                    
                if (min_mc < market_cap < max_mc) and preliminaries:
                    for idx, (lower_bound, upper_bound) in enumerate(self.market_cap_bins, start=1):
                        if lower_bound <= market_cap < upper_bound:
                            universe[idx].append(entry)
                            break
            offset += 50
        log_general.info(f"Queried {self.api_token_fetch_limit + 50} tokens from birdeye with sort_by: {self.token_list_sort_by} and sort_type: {self.token_list_sort_type} for universe_id: {self.universe_id}")
        self.close_database_connection()
        return universe

    def filter_universe_by_age_and_safety(self, universe):
        pass

    def filter_universe_by_volume_and_liquidity(self, universe):
        def calculate_quintile_indices(length):
            return [int(length * i / 5) for i in range(len(self.market_cap_bins)+1)]
        
        def filter_by_quintile(data, key, min_quintile):
            if len(data) < 5:  # If less than 5 items, exclude the whole tier
                return []
            sorted_data = sorted(data, key=lambda x: x[key])
            indices = calculate_quintile_indices(len(sorted_data))
            quintile_cutoff_index = indices[min_quintile - 1]
            return sorted_data[quintile_cutoff_index:]
        
        filtered_universe = []
        for _, coins in universe.items():
            liquidity_filtered_coins = [coin for coin in coins if coin['liquidity'] >= self.min_liquidity]

            volume_market_cap_filtered = filter_by_quintile(liquidity_filtered_coins, 'volume_pct_market_cap', self.min_volume_pct_market_cap_quintile)

            final_filtered = filter_by_quintile(volume_market_cap_filtered, 'volume_change_pct', self.min_volume_change_pct_quintile)
            
            filtered_universe.extend(final_filtered)
            
        return filtered_universe

    def update_tradeable_assets(self):
        
        universe = self.fetch_coins_by_market_cap()
        filtered_universe = self.filter_universe_by_volume_and_liquidity(universe)
        filtered_universe = self.filter_universe_by_age_and_safety(universe)

        log_general.info("Tradeable universe of composition ")

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

# update using utils @handle_rate_limiting, potentially make async
def fetch_new_ohlcv_data(token_address, interval, unix_time_start, unix_time_end, retries=5, wait_time=60):
    url = f"https://public-api.birdeye.so/defi/ohlcv?address={token_address}&type={interval}&time_from={unix_time_start}&time_to={unix_time_end}"
    try:
        response = requests.get(url)
        if response.status_code == 200 and response.json().get("success"):
            return response.json().get("data", {}).get("items", [])
        else:
            # If response code is not 200 or success flag is False, raise an exception to trigger a retry
            raise Exception(f"Failed to fetch data: {response.status_code} {response.text}")
    except Exception as e:
        if retries > 0:
            print(f"Error fetching data, retrying in {wait_time} seconds... Error: {e}")
            time.sleep(wait_time)
            return fetch_new_ohlcv_data(token_address, interval, unix_time_start, unix_time_end, retries-1, wait_time*2)
        else:
            print(f"Failed to fetch data after retries. Error: {e}")
            return []

def update_tradeable_asset_prices(interval='1H'):
    conn = sqlite3.connect(config.database_path)
    c = conn.cursor()

    c.execute("SELECT token_address FROM tradeable_assets WHERE currently_tradeable = TRUE")
    token_addresses = [row[0] for row in c.fetchall()]

    now = datetime.now()
    unix_time_end = int(time.mktime(now.timetuple()))

    for token_address in token_addresses:
        c.execute('''SELECT datetime FROM tradeable_asset_prices WHERE token_address = ? AND interval = ? ORDER BY datetime DESC LIMIT 1''', 
                  (token_address, interval))
        result = c.fetchone()

        if result:
            last_update = datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S')
            unix_time_start = int(time.mktime(last_update.timetuple()))
        else:
            c.execute("SELECT creation_datetime FROM tradeable_assets WHERE token_address = ?", (token_address,))
            creation_datetime = datetime.strptime(c.fetchone()[0], '%Y-%m-%d %H:%M:%S')
            if now - creation_datetime < timedelta(weeks=1):
                unix_time_start = int(time.mktime(creation_datetime.timetuple()))
            else:
                unix_time_start = int(time.mktime((now - timedelta(weeks=1)).timetuple()))

        new_ohlcv_data = fetch_new_ohlcv_data(token_address, interval, unix_time_start, unix_time_end)

        for data_point in new_ohlcv_data:
            c.execute('''INSERT INTO tradeable_asset_prices (token_address, datetime, open, high, low, close, volume, interval)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                      (token_address, datetime.fromtimestamp(data_point["unixTime"]).strftime('%Y-%m-%d %H:%M:%S'), 
                       data_point["o"], data_point["h"], data_point["l"], data_point["c"], data_point["v"], interval))

    conn.commit()
    conn.close()

update_tradeable_assets()
