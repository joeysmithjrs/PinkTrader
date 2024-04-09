import asyncio
import aiosqlite
import json
from log import log_general, log_transaction
from utils import handle_sqlite_lock
from wallet import find_balance
from config import config
from transactions import perform_swap
from indicators import init_indicator

class Strategy:
    async def __init__(self, configs):
        self.strategy_id = configs['strategy_id']
        self.universe_id = configs['universe_id']
        self.check_exit_rule_minutes = configs.get('check_exit_rule_minutes')
        self.minimum_periods = self.configs.get('minimum_periods')
        self.base_token_address = self.configs.get('base_token_address')
        self.risk_management = self.configs.get('risk_management')
        self.indicator_dict = self.configs.get('indicators')
        self.token_list = self.current_holdings = self.conn = self.cur = self.indicators, self.indicator_cols = None
        await self.init_indicators()

    async def init_indicators(self):
        interval_indicators, indicator_cols = {}, []
        for interval, indicators in self.indicator_dict.items():
            indicator_instances = []
            for name, args_list in indicators.items():
                for args in args_list:
                    indi = init_indicator(name, *args)
                    indicator_cols.extend(indi.cols)
                    indicator_instances.append(indi)
            interval_indicators[interval] = indicator_instances
        self.indicators = interval_indicators
        indicator_cols = list(set(indicator_cols))
        await self.init_db_columns(indicator_cols)

    @handle_sqlite_lock()
    async def init_db_columns(self, names):
        try:
            self.open_database_connection()
            for name in names:
                await self.cur.execute(f"PRAGMA table_info(tradable_asset_prices_indicators)")
                columns = [info[1] for info in await self.cur.fetchall()]
                if name not in columns:
                    await self.cur.execute(f"ALTER TABLE tradable_asset_prices_indicators ADD COLUMN {name} REAL")
                    log_general.info(f"Column name: {name} added to tradable_asset_prices_indicators")
        finally:
            self.close_database_connection()

    async def pre_next(self):
        try:
            await self.open_database_connection()
            self.update_indicators()
            self.token_list = await self.query_tradeable_assets()
            self.current_holdings = await self.query_portfolio_tokens()
            self.update_buy_size_limit()

        finally: 
            await self.close_database_connection()
    
    async def post_next(self):
        try:
            self.open_database_connection()
            self.insert_into_indicator_database()
        finally:
            self.close_database_connection()

    async def update_indicators(self):
        for token in self.token_list:
            if token not in self.indicator_values:
                self.indicator_values[token] = {}
            
            for indicator_name, settings in self.indicator_dict.items():
                if indicator_name not in self.indicator_values[token]:
                    self.indicator_values[token][indicator_name] = {}
                
                for setting in settings:
                    interval, *args = setting  # First item is interval, the rest are args
                    
                    if interval not in self.indicator_values[token][indicator_name]:
                        self.indicator_values[token][indicator_name][interval] = []
                    
                    historical_values = self.indicator_values[token][indicator_name][interval]
                    
                    # Calculate the new indicator value
                    new_value = await indicator_map[indicator_name](token, interval, self.db_connection, *args, historical_values=historical_values)
                    
                    # Append the new value, maintaining the list's length up to self.minimum_periods
                    historical_values.append(new_value)
                    if len(historical_values) > self.minimum_periods:
                        historical_values.pop(0)

    async def insert_into_indicator_database():
        pass

    async def next(self):
        raise NotImplementedError("Subclasses should implement this method")

    async def run(self):
        await self.pre_next()
        await self.next()

    async def open_database_connection(self):
        self.conn = await aiosqlite.connect(config().database_path)
        self.cur = await self.conn.cursor()
        log_general.info("SQLite Database connection opened")
    
    async def close_database_connection(self):
        await self.conn.commit()
        await self.conn.close()
        log_general.info("SQLite Database connection closed")

    async def get_prices_for_all_current_holdings(self):
        # fetch multiprice from jupiter
        pass

    async def update_trailing_stop_loss_take_profit(self):
        pass

    async def check_stop_loss_take_profit(self):
        pass

    async def buy(self, token_address):
        log_transaction.info(f"Buy signal detected for token_address: {token_address} strategy_id: {self.strategy_id}")
        size = self.position_sizer()
        if size <= 0 or size > self.buy_size_limit:
            log_transaction.info("Position size out of limits; trade not executed.")
            return
        await asyncio.run(perform_swap("BUY", size, token_address))

    async def sell(self, token_address, pct_position=1.0):
        if pct_position > 1:
            pct_position = 1
        size = await self.get_balance(token_address) * pct_position
        log_transaction.info(f"Sell signal detected for token_address: {token_address} strategy_id: {self.strategy_id}")
        await asyncio.run(perform_swap("SELL", size, token_address))

    async def update_buy_size_limit(self):
        # need to edit for total portfolio value with a max usdc pct
        self.buy_size_limit = await find_balance(self.base_token_address) * self.risk_management['portfolio_allocation_pct'] * self.risk_management['buy_size_limit_pct']

    async def query_portfolio_tokens(self):
        tokens = []
        await self.cur.execute("SELECT DISTINCT token_address FROM portfolio_composition_by_strategy WHERE strategyID=?", (self.strategyID,))
        rows = await self.cur.fetchall()
        tokens = [row[0] for row in rows]
        return tokens

    async def query_tradeable_assets(self):
        tokens = []
        await self.cur.execute("SELECT DISTINCT token_address FROM tradeable_assets WHERE ?=TRUE", (self.strategyID,))
        rows = await self.cur.fetchall()
        tokens = [row[0] for row in rows]
        return tokens

    async def fetch_ohlcv_data(self, token_address, interval):
        await self.open_database_connection()
        data = []
        await self.cur.execute("SELECT unixtime, open, high, low, close, volume FROM tradable_asset_prices_indicators WHERE token_address=? AND interval=? ORDER BY datetime DESC LIMIT ?", (token_address, interval, self.minimum_periods))
        data = await self.cur.fetchall()
        await self.close_database_connection()
        return data

    async def get_balance(self, token_address):
        await self.open_database_connection()
        await self.cur.execute("SELECT token_balance FROM portfolio_composition_by_strategy WHERE token_address=? AND strategyID=?", (token_address, self.strategyID))
        balance = await self.cur.fetchone()
        await self.close_database_connection()
        return balance[0] if balance else 0