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

    def set_currently_tradeable_to_false(self):
        self.cur.execute(f"UPDATE tradeable_assets SET {self.universe_id} = FALSE")
        self.conn.commit()

    def set_currently_tradeable_to_true(self, token_address):
        self.cur.execute(f"UPDATE tradeable_assets SET {self.universe_id} = TRUE WHERE token_address = {token_address}")
        self.conn.commit()

    def get_all_currently_tradeable_assets(self):
        self.cur.execute(f"SELECT token_address FROM tradeable_assets WHERE {self.universe_id} = TRUE")
        return [row[0] for row in self.cur.fetchall()]

    @handle_rate_limiting_birdeye()
    def fetch_token_security_info(self, token_address):
        url = f"https://public-api.birdeye.so/defi/token_security?address={token_address}"
        return requests.get(url, headers=self.headers)
    
    @handle_rate_limiting_birdeye()
    def fetch_token_list_page(self, offset):
        url = f"https://public-api.birdeye.so/public/tokenlist?sort_by={self.configs.get("token_list_sort_by")}&sort_type={self.configs.get("token_list_sort_type")}&offset={offset}&limit={self.configs.get("token_list_page_limit")}"
        return requests.get(url, headers=self.headers)
    
    @handle_rate_limiting_birdeye()
    def fetch_new_ohlcv_data(self, token_address, interval, unix_time_start, unix_time_end):
        url = f"https://public-api.birdeye.so/defi/ohlcv?address={token_address}&type={interval}&time_from={unix_time_start}&time_to={unix_time_end}"
        return requests.get(url, headers=self.headers)
    
    def token_exists_in_database(self, token_address):
        self.cur.execute("SELECT token_address FROM tradeable_assets WHERE token_address = ?", (token_address))
        return self.curr.fetchone()
    
    def insert_into_tradeable_assets_info(self, entry):
        now = time.time()
        self.cur.execute('''INSERT INTO tradeable_asset_info (unixtime, token_address, top_10_holders_pct, volume, 
                    volume_change_pct, market_cap, liquidity, volume_pct_market_cap)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (now, entry.get('token_address'), entry.get('top_10_holders_pct'), entry.get('volume'), 
                    entry.get('volume_change_pct'), entry.get('market_cap'), entry.get('liquidity'), entry.get('volume_pct_market_cap')))
        self.conn.commit()
        log_general.info(f"token_address: {entry.get('token_address')} information updated in tradeable_assets_info for universe_id: {self.configs.get('universe_id')}")

    def insert_into_tradaeble_assets(self, entry):
        self.cur.execute('''INSERT INTO tradeable_assets (token_address, name, symbol, platform, creation_unixtime, ?)
                  VALUES (?, ?, ?, ?, ?)''', 
                  (self.universe_id, entry['token_address'], entry['name'], entry['symbol'], self.platform, entry['creation_time'], True))
        self.conn.commit()
        log_general.info(f"token_address: {entry.get('token_address')} added to tradeable_assets and set true for universe_id: {self.configs.get('universe_id')}")

    def insert_into_tradeable_asset_prices(self, token_address, entries, interval):
        for data_point in entries:
            self.cur.execute('''INSERT INTO tradeable_asset_prices (token_address, unixtime, open, high, low, close, volume, interval)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (token_address, data_point["unixTime"],data_point["o"], data_point["h"], data_point["l"], data_point["c"], data_point["v"], interval))
        self.conn.commit()
        log_general.info(f"{len(entries)} OHCLV data points added to tradeable_asset_prices for token_address: {token_address} interval: {interval}")

    def fill_entry(self, coin):
        volume = float(coin.get('v24hUSD', 0) or 0)
        volume_change_pct = float(coin.get('v24hChangePercent', 0) or 0)
        market_cap = float(coin.get('mc', 0) or 0)
        liquidity = float(coin.get("liquidity", 0) or 0)
        name = str(coin.get('name', '_') or '_')
        token_address = str(coin.get('address', '_') or '_')
        symbol = str(coin.get('symbol', '_') or '_')
        security_info = self.fetch_token_security_info(token_address)
        top10holderspct = float(security_info['data'].get('top_10_holders_pct'))
        creation_time = int(security_info['data'].get('creationTime'))

        entry = {'token_address': token_address, 'symbol': symbol, 'name': name, 'volume': volume,
                'volume_change_pct': volume_change_pct, 'market_cap': market_cap,
                'liquidity': liquidity, 'volume_pct_market_cap': volume / market_cap,
                'top_10_holders_pct': top10holderspct, 'creation_time': creation_time}
        preliminaries = volume != 0 and market_cap != 0 and liquidity != 0 and name != '_' and token_address != '_' and symbol != '_' and 'Wormhole' not in name

        return entry, preliminaries

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
                if self.token_exists_in_database(entry.get('token_address')):
                    self.insert_into_tradeable_assets_info(entry)
                if (min_mc < market_cap < max_mc) and preliminaries:
                    for idx, (lower_bound, upper_bound) in enumerate(self.market_cap_bins, start=1):
                        if lower_bound <= market_cap < upper_bound:
                            universe[idx].append(entry)
                            break
            offset += 50
        log_general.info(f"Queried {self.api_token_fetch_limit + 50} tokens from birdeye with sort_by: {self.token_list_sort_by} and sort_type: {self.token_list_sort_type} for universe_id: {self.universe_id}")
        self.close_database_connection()
        return universe

    def filter_universe_by_age_and_security(self, universe):
        min_creation_time = time.time() - (self.min_hours_since_creation * 3600)
        filtered_universe = {n: [] for n in range(1, len(self.market_cap_bins) + 1)}
        
        total_initial_count = sum(len(universe[key]) for key in universe)
        total_filtered_count = 0
        
        for key in universe:
            filtered_universe[key] = [
                asset for asset in universe[key]
                if asset['top_10_holders_pct'] <= self.max_top_10_holders_pct 
                and asset['creation_time'] <= min_creation_time
            ]
            total_filtered_count += len(filtered_universe[key])
        
        total_filtered_out_percentage = round((1 - float(total_filtered_count / total_initial_count)) * 100, 2)
        log_general.info(f"{total_filtered_out_percentage}% of initial universe filtered out by age and security for universe_id: {self.universe_id}")
        
        return filtered_universe
    
    def filter_universe_by_volume_and_liquidity(self, universe):
        def calculate_quintile_indices(length):
            return [int(length * i / 5) for i in range(len(self.market_cap_bins)+1)]

        def filter_by_quintile(data, key, min_quintile):
            if len(data) == 0:
                return []
            if 0 < len(data) < 5:
                return [max(data, key=lambda x: x[key])]

            sorted_data = sorted(data, key=lambda x: x[key])
            indices = calculate_quintile_indices(len(sorted_data))
            quintile_cutoff_index = indices[min_quintile - 1]
            return sorted_data[quintile_cutoff_index:]


        total_initial_count = sum(len(coins) for _, coins in universe.items())
        filtered_universe = []

        for _, coins in universe.items():
            liquidity_filtered_coins = [coin for coin in coins if coin['liquidity'] >= self.min_liquidity]
            volume_market_cap_filtered = filter_by_quintile(liquidity_filtered_coins, 'volume_pct_market_cap', self.min_volume_pct_market_cap_quintile)
            final_filtered = filter_by_quintile(volume_market_cap_filtered, 'volume_change_pct', self.min_volume_change_pct_quintile)
            filtered_universe.extend(final_filtered)

        total_filtered_count = len(filtered_universe)
        total_filtered_out_percentage = round((1 - float(total_filtered_count / total_initial_count)) * 100, 2)

        log_general.info(f"{total_filtered_out_percentage}% of initial universe further filtered out by volume and liquidity for universe_id: {self.universe_id}")
        return filtered_universe

    
    def update_tradeable_assets(self):
        
        log_general.info(f"Beginning universe selection process for universe_id: {self.universe_id}")
        universe = self.fetch_coins_by_market_cap()
        security_filtered_universe = self.filter_universe_by_age_and_security(universe)
        final_filtered_universe = self.filter_universe_by_volume_and_liquidity(security_filtered_universe)
        log_general.info("Tradeable universe of composition ")

        self.open_database_connection()
        self.set_currently_tradeable_to_false()

        for asset in final_filtered_universe:
            token_address = asset.get('token_address')
            if self.token_exists_in_database(token_address):
                self.set_currently_tradeable_to_true(token_address)
            else:
                self.insert_into_tradaeble_assets(asset)
                self.insert_into_tradeable_assets_info(asset)

        self.close_database_connection()

    def update_tradeable_asset_prices(self):
        def interval_to_seconds(interval):
            mapping = {
                '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
                '1H': 3600, '2H': 7200, '4H': 14400, '6H': 21600, '8H': 28800,
                '12H': 43200, '1D': 86400, '3D': 259200, '1W': 604800, '1M': 2592000
            }
            return mapping.get(interval, 0)

        self.open_database_connection()
        unix_time_end = int(time.time())
        token_addresses = self.get_all_currently_tradeable_assets()

        for token_address in token_addresses:
            for interval in self.intervals:
                self.cur.execute('''SELECT unixtime FROM tradeable_asset_prices WHERE token_address = ? AND interval = ? ORDER BY unixtime DESC LIMIT 1''', 
                        (token_address, interval))
                result = self.cur.fetchone()

                if result:
                    last_update_unix = result[0]
                else:
                    self.cur.execute("SELECT creation_unixtime FROM tradeable_assets WHERE token_address = ?", (token_address,))
                    creation_unixtime_result = self.cur.fetchone()
                    creation_unixtime = creation_unixtime_result[0] if creation_unixtime_result else unix_time_end
                    if unix_time_end - creation_unixtime < interval_to_seconds('1W'):
                        last_update_unix = creation_unixtime
                    else:
                        last_update_unix = unix_time_end - interval_to_seconds('1W')

                interval_seconds = interval_to_seconds(interval)
                if unix_time_end - last_update_unix >= interval_seconds:
                    new_ohlcv_data = self.fetch_new_ohlcv_data(token_address, interval, last_update_unix, unix_time_end).get("data", {}).get("items", [])
                    if new_ohlcv_data:
                        self.insert_into_tradeable_asset_prices(token_address, new_ohlcv_data, interval)

        self.close_database_connection()
