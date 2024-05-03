# Python Algorithmic Trading Framework for Solana Ecosystem

## Overview
This open-source Python framework is designed for mid-frequency algorithmic trading within the Solana ecosystem. It caters especially to trading low market cap crypto assets, enabling both seasoned and novice software engineers to deploy systematic trading strategies effectively. The framework features a SQLite data management system and supports custom strategy and indicator development, making it highly customizable and robust. Built with `asyncio`, it allows for efficient asynchronous execution of trading operations.

## Features
- **Asynchronous Execution**: Leverages Python's `asyncio` library for non-blocking, concurrent code execution, enhancing performance and scalability.
- **Data Management**: Utilizes SQLite for local data storage, ensuring fast access and reliable management of trading data.
- **Custom Strategy and Indicator Development**: Users can create custom strategies and indicators by extending the base classes provided.
- **Universe Filtering**: Filter assets based on market capitalization, liquidity, and token creation time via a straightforward configuration.
- **Multiple Trading Modes**: Supports backtesting, live simulation, and live trading. Note that backtesting requires prior data from either live trading or simulation to populate the SQLite database.
- **Risk Management Tools**: Includes built-in functionalities like take profit, stop loss, trailing stop loss, and trailing take profit to manage trading risks effectively.
- **Multi-Asset and Multi-Timescale**: Enables trading strategies that span multiple assets and timescales.

## API Keys Requirement
To utilize this framework, you will need:
- A premium API key from **Birdeye.so**
- An API key from **Jupiter.ag**

## Upcoming Features
- **Documentation**: Comprehensive documentation is in progress and will be available soon to assist users in setting up and customizing their trading systems.
- **Expanded Blockchain Support**: Future updates will include support for other blockchain networks such as Ethereum Mainnet and Base, among others.

## Getting Started
To get started with this trading framework, clone this repository and ensure you have the required API keys. Detailed setup instructions will be provided in the forthcoming documentation.

## Contribution
Contributions are welcome! If you have ideas for improvements or have found a bug, please feel free to submit an issue or pull request.

## Disclaimer
Trading cryptocurrencies involves significant risk and can result in the loss of your invested capital. You should not invest more than you can afford to lose and should ensure that you fully understand the risks involved. This software is provided 'as-is', and you use it at your own risk.

---

For more information and updates, keep an eye on this repository!
