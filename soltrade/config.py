import os
import base58
import configparser  # Import the configparser module

from solders.pubkey import Pubkey
from solders.keypair import Keypair
from solana.rpc.api import Client
from log import log_general


class Config:
    def __init__(self, path):
        self.path = path
        self.database_path = None
        self.simulation_database_path = None
        self.birdeye_api_key = None
        self.private_key = None
        self.custom_rpc_https = None
        self.price_update_seconds = None
        self.trading_interval_minutes = None
        self.slippage = None  # BPS
        self.computeUnitPriceMicroLamports = None
        self.simulation = False
        self.backtest = False
        self.load_config()

    def load_config(self):
        if not os.path.exists(self.path):
            log_general.error(
                "Soltrade was unable to detect the config file. Are you sure config.ini has not been renamed or removed?")
            exit(1)

        config = configparser.ConfigParser()
        config.read(self.path)
        
        try:
            self.database_path = config.get('PATHS', 'DATABASE_PATH')
            self.simulation_database_path = config.get('PATHS', 'SIMULATION_DATABASE_PATH')
            self.birdeye_api_key = config.get('DATA', 'BIRDEYE_API_KEY')
            self.private_key = config.get('KEYS', 'SOLANA_PRIVATE_KEY')
            self.custom_rpc_https = config.get('RPC', 'DEFAULT_RPC')
            self.price_update_seconds = config.getint('SETTINGS', 'PRICE_UPDATE_SECONDS')
            self.trading_interval_minutes = config.getint('SETTINGS', 'TRADING_INTERVAL_MINUTES')
            self.slippage = config.getint('SETTINGS', 'SLIPPAGE_TOLERANCE_BPS')
            self.computeUnitPriceMicroLamports = config.getint('SETTINGS', 'COMPUTE_PRICE_MICRO_LAMPORTS') 
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            log_general.error(f"Missing configuration or section: {e}")
            exit(1)

    @property
    def keypair(self):
        try:
            return Keypair.from_bytes(base58.b58decode(self.private_key))
        except Exception as e:
            log_general.error(f"Error decoding private key: {e}")
            exit(1)

    @property
    def public_address(self):
        return self.keypair.pubkey()

    @property
    def client(self):
        rpc_url = self.custom_rpc_https
        return Client(rpc_url)
    
    @property
    def decimals(self):
        response = self.client.get_account_info_json_parsed(Pubkey.from_string(config().other_mint)).to_json()
        json_response = json.loads(response)
        value = 10**json_response["result"]["value"]["data"]["parsed"]["info"]["decimals"]
        return value


_config_instance = None


def config(path=None):
    global _config_instance
    if _config_instance is None and path is not None:
        _config_instance = Config(path)
    return _config_instance
