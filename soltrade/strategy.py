import asyncio
import aiosqlite
from log import log_general, log_transaction
from transactions import perform_swap

class Strategy:
    def __init__(self, db_path='trading_algo.db'):
        self.strategyID = 'unique_strategy_id'  # Set unique identifier
        self.priceUpdateMinutes = 60  # Desired frequency
        self.tokenList = []  # Initialize empty, fill dynamically
        self.minimum_periods = 5  # Set to minimum data points needed
        self.intervals = ['1H', '1D']  # Desired intervals
        self.buy_size_limit = 0.0  # Updated dynamically
        self.db_path = db_path

    async def next(self):
        log_general.debug("Analyzing the market; no trade has been executed.")
        await self.update_buy_size_limit()
        self.tokenList = await self.query_portfolio_tokens()

        for token_address in self.tokenList:
            for interval in self.intervals:
                data = await self.fetch_ohlcv_data(token_address, interval)
                # Placeholder for analysis; implement logic here

    async def buy(self, token_address):
        log_transaction.info("Buy signal detected.")
        size = self.position_sizer()
        if size <= 0 or size > self.buy_size_limit:
            log_transaction.info("Position size out of limits; trade not executed.")
            return
        await asyncio.run(perform_swap("BUY", size, token_address))

    async def sell(self, token_address, pct_position=1.0):
        if pct_position > 1:
            pct_position = 1
        size = await self.get_balance(token_address) * pct_position
        log_transaction.info("Sell signal detected.")
        await asyncio.run(perform_swap("SELL", size, token_address))

    async def update_buy_size_limit(self):
        self.buy_size_limit = await get_portfolio_balance_usdc() * 0.15

    async def get_balance(self, token_address):
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute("SELECT token_balance FROM portfolio_composition_by_strategy WHERE token_address=? AND strategyID=?", (token_address, self.strategyID))
            balance = await cur.fetchone()
        return balance[0] if balance else 0

    async def query_portfolio_tokens(self):
        tokens = []
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute("SELECT DISTINCT token_address FROM portfolio_composition_by_strategy WHERE strategyID=?", (self.strategyID,))
            rows = await cur.fetchall()
            tokens = [row[0] for row in rows]
        return tokens

    async def fetch_ohlcv_data(self, token_address, interval):
        data = []
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute("SELECT datetime, open, high, low, close, volume FROM tradeable_asset_prices WHERE token_address=? AND interval=? ORDER BY datetime DESC LIMIT ?", (token_address, interval, self.minimum_periods))
            data = await cur.fetchall()
        return data
