import asyncio
import aiosqlite
import pandas as pd
from log import log_general, log_transaction
from utils import handle_sqlite_lock, handle_database_connection
from wallet import find_balance
from config import config
from transactions import perform_swap
from indicators import init_indicator

class OHCLV:
    pass

class Strategy:
    def __init__(self, configs):
        self.strategy_id = configs.get('strategy_id')
        self.universe_id = configs.get('universe_id')
        self.lookback_period = configs.get('lookback_period', 10)
        self.check_exit_rule_minutes = configs.get('check_exit_rule_minutes', None)
        self.base_token_address = configs.get('base_token_address', None)
        self.risk_management = configs.get('risk_management', None)
        self.indicator_dict = configs.get('indicators', None)

        self.simulation_or_backtest = None
        self.token_list = None
        self.current_holdings = None
        self.conn = None
        self.cur = None
        self.indicators = None
        self.indicator_cols = None
        self.indis = None
        self.ohclv = None

    @classmethod
    async def create(cls, configs):
        instance = cls(configs)
        instance.simulation_or_backtest = config().simulation or config().backtest
        assert set(instance.lookback_period.keys()) == set(instance.indicators.keys()), "The keys of lookback_period and indicators must match."
        assert instance.strategy_id is not None and isinstance(instance.strategy_id, str), "strategy_id must be a non-empty string"
        assert instance.universe_id is not None and isinstance(instance.universe_id, str), "universe_id must be a non-empty string"
        await instance.init_indicators()
        return instance

    async def init_indicators(self):
        interval_indicators, indicator_cols = {}, []
        for interval, indicators in self.indicator_dict.items():
            indicator_instances = []
            for name, args_list in indicators.items():
                for args in args_list:
                    indi = init_indicator(name, *args)
                    self.lookback_period[interval] = max(self.lookback_period[interval], indi.length)
                    indicator_cols.extend(indi.cols)
                    indicator_instances.append(indi)
            interval_indicators[interval] = indicator_instances
        self.indicators = interval_indicators
        indicator_cols = list(set(indicator_cols))
        await self.init_db_columns(indicator_cols)

    @handle_sqlite_lock()
    @handle_database_connection()
    async def init_db_columns(self, names):
        await self.cur.execute(f"PRAGMA table_info(tradable_asset_indicators)")
        columns = [info[1] for info in await self.cur.fetchall()]
        for name in names:
            if name not in columns:
                await self.cur.execute(f"ALTER TABLE tradable_asset_indicators ADD COLUMN {name} REAL")
                log_general.info(f"Column name: {name} added to tradable_asset_indicators")

    @handle_database_connection()
    async def pre_next(self):
        self.token_list = await self.query_tradeable_assets()
        self.current_holdings = await self.query_portfolio_tokens()
        self.update_ohclv_and_indicators()
        self.update_buy_size_limit()

    @handle_database_connection()
    async def post_next(self):
        self.insert_into_indicator_database()

    async def update_ohclv_and_indicators(self):
        self.indis = { token: { interval: { indi.id: None for indi in self.indicators[interval] } for interval in self.indicators } for token in self.token_list }
        new_indicator_database_entries = { interval: { indi.id: pd.DataFrame(columns=['unixtime', 'token_address', 'interval'].extend(indi.cols)) for indi in self.indicators[interval] } for interval in self.indicators } 
        for token in self.token_list:
            for interval, indis in self.indicators.items():
                data = self.fetch_joined_ohlcv_and_indicators_data(token, interval, self.lookback_period[interval])
                columns = ['open', 'high', 'low', 'close', 'volume'] 
                self.ohclv[token][interval] = data[columns]
                for indi in indis:
                    fetched_indi_cols = data[list(indi.cols)]
                    if fetched_indi_cols.notna().all():
                        self.indis[token][interval][indi.id] = fetched_indi_cols
                        break
                    new_calcs = indi.calculate(data[columns + list(indi.cols)])
                    diff_mask = (fetched_indi_cols != new_calcs)
                    rows_with_diff = diff_mask.any(axis=1)
                    df_diff_rows = new_calcs.loc[rows_with_diff]
                    df_final_diff = df_diff_rows.loc[:, indi.cols + ['unixtime']]
                    df_final_diff['token_address'] = token
                    df_final_diff['interval'] = interval
                    new_indicator_database_entries[interval][indi.id] = pd.concat([new_indicator_database_entries[interval][indi.id], df_final_diff], ignore_index=True)
                    self.indis[token][interval][indi.id] = new_calcs[['unixtime'] + list(indi.cols)]

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

    @handle_database_connection()
    async def fetch_last_ohlcv_data(self, token_address, interval, length):
        columns = ['unixtime', 'open', 'high', 'low', 'close', 'volume']
        query = f"""SELECT {','.join(columns)} 
                    FROM tradable_asset_prices 
                    WHERE token_address = ? 
                    AND interval = ? 
                    ORDER BY datetime 
                    DESC LIMIT ?"""
        await self.cur.execute(query, (token_address, interval, length))
        data = await self.cur.fetchall()
        return pd.DataFrame(data, columns=columns)
    
    @handle_database_connection()
    async def fetch_ohlcv_data_range(self, token_address, interval, unixtimestart, unixtimeend):
        columns = ['unixtime', 'open', 'high', 'low', 'close', 'volume']
        query = f"SELECT {','.join(columns)} FROM tradable_asset_prices FROM tradable_asset_prices WHERE token_address = ? AND interval = ? AND unixtime >= ? AND unixtime <= ? ORDER BY unixtime ASC"
        await self.cur.execute(query, (token_address, interval, unixtimestart, unixtimeend))
        data = await self.cur.fetchall()
        return pd.DataFrame(data, columns=columns)
    
    @handle_database_connection()
    async def fetch_indicators_data_range(self, token_address, interval, unixtimestart, unixtimeend):
        columns = ', '.join(['unixtime'] + self.indicator_cols)
        query = f"SELECT {columns} FROM tradable_asset_indicators  # Assuming this is the correct table name WHERE token_address = ? AND interval = ? AND unixtime >= ? AND unixtime <= ? ORDER BY unixtime ASC"
        await self.cur.execute(query, (token_address, interval, unixtimestart, unixtimeend))
        data = await self.cur.fetchall()
        return pd.DataFrame(data, columns=columns)

    @handle_database_connection()
    async def fetch_joined_ohlcv_and_indicators_data(self, token_address, interval, length, unixtimestart, unixtimeend):
        ohlcv_columns = ['unixtime', 'open', 'high', 'low', 'close', 'volume']
        indicator_columns = [f'a.{col}' for col in self.indicator_cols]
        subquery_ohlcv = f"""SELECT {', '.join(ohlcv_columns)}
                             FROM tradable_asset_prices
                             WHERE token_address = ? AND interval = ?
                             ORDER BY datetime 
                             DESC LIMIT ? """
        main_query = f"""WITH last_ohlcv AS ({subquery_ohlcv})
                         SELECT last_ohlcv.*, {', '.join(indicator_columns)}
                         FROM last_ohlcv
                         LEFT JOIN tradable_asset_indicators a ON last_ohlcv.unixtime = a.unixtime
                         AND last_ohlcv.token_address = a.token_address
                         AND last_ohlcv.interval = a.interval
                         ORDER BY last_ohlcv.unixtime ASC
                         """
        await self.cur.execute(main_query, (token_address, interval, length))
        data = await self.cur.fetchall()
        columns = ohlcv_columns + self.indicator_cols
        df = pd.DataFrame(data, columns=columns)
        df.set_index('unixtime', inplace=True)
        return df

    @handle_database_connection()
    async def get_balance(self, token_address):
        await self.cur.execute("SELECT token_balance FROM portfolio_composition_by_strategy WHERE token_address=? AND strategyID=?", (token_address, self.strategyID))
        balance = await self.cur.fetchone()
        return balance[0] if balance else 0