import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('trading_algo.db')
c = conn.cursor()

# Create table for tradeable assets
c.execute('''CREATE TABLE IF NOT EXISTS tradeable_assets (
    token_address TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    creation_datetime DATETIME NOT NULL,
    currently_tradeable BOOLEAN NOT NULL
)''')

# Create table for tradeable asset info
c.execute('''CREATE TABLE IF NOT EXISTS tradeable_asset_info (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    datetime DATETIME NOT NULL,
    token_address TEXT NOT NULL,
    top10holderspct REAL,
    volume REAL,
    volume_change_pct REAL,
    market_cap REAL,
    liquidity REAL,
    volume_pct_market_cap REAL,
    FOREIGN KEY (token_address) REFERENCES tradeable_assets (token_address)
    ON DELETE CASCADE
)''')

# Create table for tradeable asset prices
c.execute('''CREATE TABLE IF NOT EXISTS tradeable_asset_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token_address TEXT NOT NULL,
    datetime DATETIME NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    interval TEXT,
    FOREIGN KEY (token_address) REFERENCES tradeable_assets (token_address)
    ON DELETE CASCADE
)''')

# Create table for algorithmic trades
c.execute('''CREATE TABLE IF NOT EXISTS algorithmic_trades (
    transactionID TEXT PRIMARY KEY,
    datetime DATETIME NOT NULL,
    token_address TEXT NOT NULL,
    strategyID TEXT NOT NULL,
    buy_sell TEXT NOT NULL,
    price REAL,
    fees REAL,
    FOREIGN KEY (token_address) REFERENCES tradeable_assets (token_address)
    ON DELETE CASCADE
)''')

# Create table for portfolio balances
c.execute('''CREATE TABLE IF NOT EXISTS portfolio_balances (
    datetime DATETIME NOT NULL,
    wallet_address TEXT NOT NULL,
    usdc_balance REAL,
    solana_balance REAL,
    solana_balance_usd REAL,
    amt_spl_tokens INTEGER,
    spl_token_balance_usd REAL,
    total_portfolio_balance_usd REAL
)''')

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Database initialized successfully.")
