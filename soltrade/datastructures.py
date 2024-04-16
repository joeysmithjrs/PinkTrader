import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class Stream:
    def __init__(self, data):
        self.data = self.arrange(data)

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

    def is_rising(self, lag=0, length=1) -> bool:
        return self.slope(lag, length) > 0

    def is_falling(self, lag=0, length=1) -> bool:
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
    
    def slope(self, lag=0, length=1) -> float:
        if lag + length >= len(self.data):
            return None
        x = np.arange(length + 1).reshape(-1, 1)
        y = self.data[-1 - lag - length: -1 - lag if lag > 0 else None].reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        return float(model.coef_[0][0]) if model else 0.0

    def predict(self, future_steps=1, lag=0, length=1) -> float | None:
        model = self._fit_model(lag, length)
        if model:
            highest_x = length 
            predicted_x = np.array([[highest_x + future_steps]])
            predicted_y = model.predict(predicted_x)
            return float(predicted_y[0][0])
        return None

    def _fit_model(self, lag=0, length=1):
        if lag + length >= len(self.data):
            return None
        x = np.arange(length + 1).reshape(-1, 1)
        y = self.data[-1 - lag - length: -1 - lag if lag > 0 else None].reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        return model

class StreamContainer:
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
