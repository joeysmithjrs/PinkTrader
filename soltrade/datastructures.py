from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

class Stream:
    def __init__(self, data):
        self.data = self.arrange(data)

    def __iter__(self):
        return iter(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

    def arrange(self, data):
        if isinstance(data, pd.Series):
            return data.tolist()
        else:
            raise ValueError("Input must be a pandas Series")
        
    def std_dev(self, length=1, lag=0):
        if lag + length > len(self.data):
            return None  # Not enough data
        segment = self.data[-1 - lag - length: -1 - lag if lag > 0 else None]
        return np.std(segment, ddof=1)  # Using population standard deviation (ddof=0 for sample)

    def covariance(self, other, length=1, lag=0):
        if isinstance(other, Stream) and lag + length <= len(self.data) and lag + length <= len(other.data):
            x = self.data[-1 - lag - length: -1 - lag if lag > 0 else None]
            y = other.data[-1 - lag - length: -1 - lag if lag > 0 else None]
            return np.cov(x, y)[0][1]  # Returns the covariance between x and y
        return None

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
    
    def has_crossed_above(self, lookback, other) -> bool:
        for lag in range(lookback + 1):
            if self.crossed_above(other, lag):
                return True
        return False

    def has_crossed_below(self, lookback, other) -> bool:
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

    def slope(self, lag, length):
        if lag + length >= len(self.data):
            return None
        x = np.arange(length + 1).reshape(-1, 1)
        y = np.array(self.data[-1 - lag - length: -1 - lag if lag > 0 else None]).reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        return model.coef_[0][0] if model else 0

    def predict(self, future_steps=1, lag=0, length=1):
        model = self._fit_model(lag, length)
        if model:
            highest_x = length 
            predicted_x = np.array([[highest_x + future_steps]])
            predicted_y = model.predict(predicted_x)
            return predicted_y[0][0]
        return None

    def _fit_model(self, lag, length):
        if lag + length >= len(self.data):
            return None
        x = np.arange(length + 1).reshape(-1, 1)
        y = np.array(self.data[-1 - lag - length: -1 - lag if lag > 0 else None]).reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        return model


