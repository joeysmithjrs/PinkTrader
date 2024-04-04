import asyncio
import aiosqlite
import json
from log import log_general, log_transaction
from utils import handle_sqlite_lock
from wallet import find_balance
from config import config
from transactions import perform_swap

class Strategy:
    def __init__(self, strategy_configs_path):
        self.path = strategy_configs_path
        with open(self.path, 'r') as file:
            self.configs = json.load(file)
        self.strategy_id = self.configs.get('strategy_id')
        self.universe_id = self.configs.get('universe_id')
        self.check_exit_rule_minutes = self.configs.get('check_exit_rule_minutes')
        self.minimum_periods = self.configs.get('minimum_periods')
        self.base_token_address = self.configs.get('base_token_address')
        self.params = self.configs.get('paramaters')
        self.risk_management = self.configs.get('risk_management')
        self.token_list, self.current_holdings = None, None
        self.conn, self.cur = None, None

    async def pre_next(self):
        self.token_list = await self.query_tradeable_assets()
        self.current_holdings = await self.query_portfolio_tokens()
        self.update_buy_size_limit()

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
        # fetch multiprice from birdeye 
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
        self.buy_size_limit = await find_balance(self.base_token_address) * 0.15

    async def query_portfolio_tokens(self):
        await self.open_database_connection()
        tokens = []
        await self.cur.execute("SELECT DISTINCT token_address FROM portfolio_composition_by_strategy WHERE strategyID=?", (self.strategyID,))
        rows = await self.cur.fetchall()
        tokens = [row[0] for row in rows]
        await self.close_database_connection()
        return tokens

    async def query_tradeable_assets(self):
        await self.open_database_connection()
        tokens = []
        await self.cur.execute("SELECT DISTINCT token_address FROM portfolio_composition_by_strategy WHERE strategyID=?", (self.strategyID,))
        rows = await self.cur.fetchall()
        tokens = [row[0] for row in rows]
        await self.close_database_connection()
        return tokens

    async def fetch_ohlcv_data(self, token_address, interval):
        await self.open_database_connection()
        data = []
        await self.cur.execute("SELECT datetime, open, high, low, close, volume FROM tradeable_asset_prices WHERE token_address=? AND interval=? ORDER BY datetime DESC LIMIT ?", (token_address, interval, self.minimum_periods))
        data = await self.cur.fetchall()
        await self.close_database_connection()
        return data

    async def get_balance(self, token_address):
        await self.open_database_connection()
        await self.cur.execute("SELECT token_balance FROM portfolio_composition_by_strategy WHERE token_address=? AND strategyID=?", (token_address, self.strategyID))
        balance = await self.cur.fetchone()
        await self.close_database_connection()
        return balance[0] if balance else 0