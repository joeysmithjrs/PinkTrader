import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from collections import OrderedDict

class MarketPosition:
    def __init__(self, txid, token_address, entry_price, position_size, dual_pct_adj=False):
        self.token_address : str = token_address
        self.avg_price : float = entry_price
        self.current_price : float = entry_price
        self.position_size : float = position_size
        self.dual_pct_adj : bool = dual_pct_adj
        self.entries : OrderedDict = {txid: (entry_price, position_size)}
        self.exits : OrderedDict # same format as self.entries
        self.stop_loss: list
        self.take_profit : list
        self.trailing_stop_loss : list
        self.trailing_take_profit : list
        self.true_trailing_take_profit : list

    def advise(self, current_price):
        # accepts argument current_price 
        # iterates through all stop_loss and take profit dicts
        # checks for exit rules and returns 0 if no action needs to be taken 
        # returns a tuple of ({exit_type}, price, pct_exit)
        # exit type is just the str conversion of class variables stop_loss, take_profit etc.

    def confrim(self, exit):
        # takes in argument exit = ({exit_type}, price, pct_exit) signalling a successful execution of advise()
        # if pct_exit = 1, then we delete this class instance, logging is handled by our transactions class
        # if pct_exit < 1, then we delete the corresponding exit condition
        # adjust_exit_sizes(exit)
        
    def adjust_exit_sizes(self, exit):
        # if dual_pct_adj is true:
        # if we just exited a take profit with pct_exit .50 and we have a hanging stop_loss with pct_exit .50, then we must change the pct_exit of the stop_loss to 1
        # this type of readjustment must be made across all exit conditions both take profit and stop loss. the lists will be in order of exit prices, so as we iterate through the list if the sum of the pct_exits equals 1 we, will delete the rest of the list items
        # if dual pct_adj is false:
        # we will do the same exact thing, however if we exit a take_profit we only readjust the trailing_take_profit and true_trailing_take_profit
        # if we exit a stop_loss, we will only readjust the trailing_stop_loss

    def add_entry(txid, entry_price, position_size):
        self.entries[txid] = (entry_price, position_size)
        self.avg_price = # weighted average of price/size tuples in self.entries
        self.position_size # sum of size value in entries value tuple minus sum of size value in exits value tuple

    def remove_exit_condition(self, exit_type, idx):
        # exit type can be 'stop_loss', 'take_profit', 'trailing_stop_loss', 'trailing_take_profit', 'true_trailing_take_profit'
        # remove exit condition by index using getattr on exit_type
        # do error handling on index and exit type

    def get_exit_condition(self, exit_type, idx):
        # same type of thing as remove_exit_condition except we just return the item at index instead of delete
    
    def add_stop_loss(self, pct_position = 1, pct_exit = 1):
        price = self.current_price - self.current_price * pct_exit
        entry = (price, pct_position)
        # check if there are entries in self.stop_loss with a higher price that have pct_exit summing to 1. If so, do not add entry
        # if the sum of all of the entries' pct_exit including the new one is greater than 1, use log_general.warning() and readjust all pct exits to sum to 1 (divide each value by the sum)
        self.stop_loss.append(entry)
        # sort self.stop_loss list by price value descending 

    def add_take_profit(self, pct_position = 1, pct_exit = 1):
        price = self.current_price + self.current_price * pct_exit
        entry = (price, pct_position)
        # check if there are entries in self.take_profit with a lower price that have pct_exit summing to 1. If so, use log_general.warning() and do not add entry
        # if the sum of all of the entries' pct_exit including the new one is greater than 1, use log_general.warning() and readjust all pct exits to sum to 1 (divide each value by the sum)
        self.take_profit.append(entry)
        # sort self.stop_loss list by price value descending 

    def add_trailing_stop_loss(self, pct_trail, pct_exit = 1):
        entry = (self.current_price - (self.current_price * pct_trail), pct_trail, pct_exit)
        # check if there are entries in self.trailing_stop_loss with lower pct_trail that have pct_exit summing to 1. If so, do not add entry
        # if the sum of all of the entries' pct_exit including the new one is greater than one, use log_general.warning() and readjust all pct exits to sum to 1 (divide each value by the sum)
        self.trailing_stop_loss.append(entry)
        # sort self.trailing_stop_loss by pct_trail ascending

    def add_trailing_take_profit(self, profit_target_pct, pct_trail, pct_exit = 1):
        trigger_price = self.current_price + (self.current_price * profit_target_pct)
        entry = (None, trigger_price, pct_trail, pct_exit)
        self.trailing_take_profit.append(entry)

    def add_true_trailing_take_profit(self, pct_trail, pct_exit = 1):
        entry = (self.current_price + (self.current_price * pct_trail), pct_trail, pct_exit)
        # check if there are entries in self.true_trailing_take_profit with lower pct_trail that have pct_exit summing to 1. If so, do not add entry
        # if the sum of all of the entries' pct_exit including the new one is greater than one, use log_general.warning() and readjust all pct exits to sum to 1 (divide each value by the sum)
        self.trailing_stop_loss.append(entry)
        # sort self.trailing_stop_loss by pct_trail ascending



class PositionContainer:
    def __init__(self):
        self.active_holdings : dict

class Stream:
    __slots__ = ['data']

    def __init__(self, data):
        self.data : np.array = self.arrange(data)

    def __iter__(self):
        return iter(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

    def arrange(self, data):
        if isinstance(data, pd.Series):
            return data.values
        else:
            raise ValueError("Input must be a pandas Series")
        
    def mean(self, length=1, lag=0) -> float:
        if lag + length > len(self.data):
            length = len(self.data) - lag 
        segment = self.data[-1 - lag - length: -1 - lag if lag > 0 else None]
        return np.mean(segment)
    
    def max(self, length=1, lag=0) -> float:
        if lag + length > len(self.data):
            length = len(self.data) - lag
        segment = self.data[-1 - lag - length: -1 - lag if lag > 0 else None]
        return np.max(segment)

    def min(self, length=1, lag=0) -> float:
        if lag + length > len(self.data):
            length = len(self.data) - lag
        segment = self.data[-1 - lag - length: -1 - lag if lag > 0 else None]
        return np.min(segment)

    def std_dev(self, length=1, lag=0) -> float:
        if lag + length > len(self.data):
            length = len(self.data) - lag
        segment = self.data[-1 - lag - length: -1 - lag if lag > 0 else None]
        return np.std(segment)

    def covariance(self, other, length=1, lag=0) -> float:
        if isinstance(other, Stream):
            available_length = min(len(self.data) - lag, len(other.data) - lag)
            length = min(length, available_length)
            if length > 0:
                x = self.data[-1 - lag - length: -1 - lag if lag > 0 else None]
                y = other.data[-1 - lag - length: -1 - lag if lag > 0 else None]
                return np.cov(x, y)[0, 1]
        return 0

    def crossed_above(self, other, lag=0) -> bool:
        if isinstance(other, Stream):
            if lag < len(self.data) - 1 and lag < len(other.data) - 1:
                return self.data[-1 - lag] > other.data[-1 - lag] and self.data[-2 - lag] <= other.data[-2 - lag]
        elif isinstance(other, (int, float)):
            if lag < len(self.data) - 1:
                return self.data[-1 - lag] > other and self.data[-2 - lag] <= other
        return False

    def crossed_below(self, other, lag=0) -> bool:
        if isinstance(other, Stream):
            if lag < len(self.data) - 1 and lag < len(other.data) - 1:
                return self.data[-1 - lag] < other.data[-1 - lag] and self.data[-2 - lag] >= other.data[-2 - lag]
        elif isinstance(other, (int, float)):
            if lag < len(self.data) - 1:
                return self.data[-1 - lag] < other and self.data[-2 - lag] >= other
        return False
    
    def has_crossed_above(self, other, lookback=0) -> bool:
        for lag in range(lookback + 1):
            if self.crossed_above(other, lag):
                return True
        return False

    def has_crossed_below(self, other, lookback=0) -> bool:
        for lag in range(lookback + 1):
            if self.crossed_below(other, lag):
                return True
        return False
    
    def is_above(self, value, lag=0) -> bool:
        return self.data[-1 - lag] > value if lag < len(self.data) else False

    def is_below(self, value, lag=0) -> bool:
        return self.data[-1 - lag] < value if lag < len(self.data) else False

    def is_rising(self, length=1, lag=0) -> bool:
        return self.slope(lag, length) > 0

    def is_falling(self, length=1, lag=0) -> bool:
        return self.slope(lag, length) < 0
    
    def average_gain(self, length=1, lag=0):
        if lag + length > len(self.data):
            length = len(self.data) - lag
        segment = self.data[-1 - lag - length: -1 - lag if lag > 0 else None]
        changes = np.diff(segment)
        previous_values = segment[:-1]
        valid_indices = previous_values != 0 
        valid_changes = changes[valid_indices]
        valid_previous_values = previous_values[valid_indices]
        percentage_changes = valid_changes / valid_previous_values * 100
        gains = np.where(percentage_changes > 0, percentage_changes, 0)
        return np.mean(gains) if len(gains) > 0 else 0 

    def average_loss(self, length=1, lag=0):
        if lag + length > len(self.data):
            length = len(self.data) - lag
        segment = self.data[-1 - lag - length: -1 - lag if lag > 0 else None]
        changes = np.diff(segment)
        previous_values = segment[:-1]
        valid_indices = previous_values != 0 
        valid_changes = changes[valid_indices]
        valid_previous_values = previous_values[valid_indices]
        percentage_changes = valid_changes / valid_previous_values * 100
        losses = np.where(percentage_changes < 0, -percentage_changes, 0)
        return np.mean(losses) if len(losses) > 0 else 0 
    
    def slope(self, length=1, lag=0) -> float:
        if lag + length >= len(self.data):
            return None
        x = np.arange(length + 1).reshape(-1, 1)
        y = self.data[-1 - lag - length: -1 - lag if lag > 0 else None].reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        return float(model.coef_[0][0]) if model else 0.0

    def predict(self, future_steps=1, length=1, lag=0) -> float | None:
        model = self._fit_model(lag, length)
        if model:
            highest_x = length 
            predicted_x = np.array([[highest_x + future_steps]])
            predicted_y = model.predict(predicted_x)
            return float(predicted_y[0][0])
        return None

    def _fit_model(self, length=1, lag=0):
        if lag + length >= len(self.data):
            return None
        x = np.arange(length + 1).reshape(-1, 1)
        y = self.data[-1 - lag - length: -1 - lag if lag > 0 else None].reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        return model

class StreamContainer:
    __slots__ = ['unixtime', 'streams']
    
    def __init__(self, data, alias_list=None):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")

        self.unixtime = data.index
        self.streams = {}

        if alias_list is None:
            alias_list = data.columns

        if len(alias_list) != len(data.columns):
            raise ValueError("Alias list must match the number of columns in the data")

        for alias, column in zip(alias_list, data.columns):
            self.streams[alias] = Stream(data[column])

    def __getattr__(self, name):
        if name in self.streams:
            return self.streams[name]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getitem__(self, key):
        return self.streams[key]

    def __repr__(self):
        return f"<StreamContainer with streams: {list(self.streams.keys())}>"
