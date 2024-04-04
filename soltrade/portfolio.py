import asyncio
import json
import os
import signal
from log import log_general
from concurrent.futures import ProcessPoolExecutor
from universe import Universe
from strategy import Strategy

class Portfolio:
    def __init__(self, strategies_config_path, universes_config_path):
        self.strategies_config_path = strategies_config_path
        self.universes_config_path = universes_config_path
        self.universes = {}  # key: universe_id, value: Universe object
        self.strategies = []  # List of Strategy objects
        self.loop = asyncio.get_event_loop()
        # Register signal handlers for graceful shutdown
        for signame in ('SIGINT', 'SIGTERM'):
            self.loop.add_signal_handler(getattr(signal, signame), lambda: asyncio.ensure_future(self.shutdown()))

        self.load_configs()

    def load_configs(self):
        # Load Universe configurations
        for filename in os.listdir(self.universes_config_path):
            with open(os.path.join(self.universes_config_path, filename), 'r') as f:
                config = json.load(f)
                self.universes[config['universe_id']] = Universe(config)
                
        # Load Strategy configurations
        for filename in os.listdir(self.strategies_config_path):
            with open(os.path.join(self.strategies_config_path, filename), 'r') as f:
                config = json.load(f)
                self.strategies.append(Strategy(config))

    async def shutdown(self):
        log_general.info("Shutdown initiated. Cleaning up...")
        
        # Cancel all tasks to stop them gracefully
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        [task.cancel() for task in tasks]

        log_general.info(f"Cancelling {len(tasks)} outstanding tasks")
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Any additional cleanup can be performed here

        log_general.info("Cleanup complete. Exiting.")
        self.loop.stop()

    async def schedule_method(self, method, interval_minutes):
        """Schedule an asynchronous method to be called at regular intervals."""
        while True:
            await asyncio.sleep(interval_minutes * 60)  # Convert minutes to seconds
            asyncio.create_task(method())

    async def run_strategy_methods(self, strategy):
        """Schedule strategy-specific methods based on their timing configurations."""
        universe_config = self.universes[strategy.universe_id]
        await asyncio.gather(
            self.schedule_method(strategy.run, universe_config['ohclv_update_minutes']),
            self.schedule_method(strategy.check_exit_rule, strategy.check_exit_rule_minutes)
        )
    
    async def run_universe_methods(self, universe):
        """Schedule universe-specific methods based on their timing configurations."""
        await asyncio.gather(
            self.schedule_method(universe.update_tradeable_assets, universe.tradeable_assets_update_minutes),
            self.schedule_method(universe.fetch_new_ohlc_data, universe.ohclv_update_minutes)
        )

    async def main(self):

        tasks = []
        for _, universe in self.universes.items():
            tasks.append(asyncio.create_task(self.run_universe_methods(universe)))
        
        for strategy in self.strategies:
            tasks.append(asyncio.create_task(self.run_strategy_methods(strategy)))

        # Wait on all scheduled tasks indefinitely
        # This will keep the program running as long as any of these tasks are still running
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    portfolio_path = "/path/to/strategies"
    universes_path = "/path/to/universes"
    portfolio = Portfolio(portfolio_path, universes_path)
    try:
        asyncio.run(portfolio.main())
    except asyncio.CancelledError:
        # Expected exception when tasks are cancelled during shutdown
        pass
    finally:
        log_general.info("Application exited.")