import numpy as np
import pandas as pd
from datastructures import Stream, StreamContainer
from abc import ABC, abstractmethod
from log import log_general

class Indicator(ABC):
    input_arguments: int = 1
    output_streams: int = 1
    n_self_referential: int = 1
    category: str = None  # 'price_bound', 'zero_mean', 'non_zero_mean_bounded', 'non_zero_mean_unbounded'
    bounds: tuple = None  # (min_value, max_value)

    def __init__(self, *args):
        if len(args) != self.input_arguments:
            raise ValueError(f"{self.__class__.__name__} expects exactly {self.input_arguments} argument(s), got {len(args)}")
        self.args = args
        self.length = args[0]
        self.id = self.indicator_id()
        self.cols = self.column_names()
        self.prevs = None
        assert len(self.cols) == len(set(self.cols)), "Column names must be unique"
        if not hasattr(self, 'stream_aliases'):
            self.stream_aliases = self.default_stream_aliases()
        assert len(self.stream_aliases) == len(set(self.stream_aliases)), "Stream aliases must be unique"
        self.alias_to_cols = dict(zip(self.stream_aliases, self.cols))

    def next(self, ohclv: StreamContainer, prevs: StreamContainer):
        data_length = len(next(iter(prevs.streams.values())).data)
        first_valid_idx = self.n_self_referential
        all_nan_init = False 
        self.prevs = prevs

        for stream in prevs.streams.values():
            if np.isnan(stream.data[:first_valid_idx]).all():
                all_nan_init = True
                default_value = self.determine_default_value(ohclv, stream)
                stream.data[:first_valid_idx] = default_value

        start_idx = 0 if all_nan_init else first_valid_idx

        for idx in range(first_valid_idx, data_length):
            try:
                calculated_values = self.calculate(ohclv, idx)
                for alias, value in zip(self.stream_aliases, calculated_values):
                    if self.bounds:
                        value = max(min(value, self.bounds[1]), self.bounds[0])
                    prevs.streams[alias].data[idx] = value
            except Exception as e:
                self.handle_calculation_error(e, idx)

        return prevs, (start_idx, data_length - 1)

    def determine_default_value(self, ohclv, stream):
        if self.category in [None, 'zero_mean', 'non_zero_mean_unbounded']:
            return 0
        elif self.category == 'price_bound':
            return ohclv.close[0]
        elif self.category == 'non_zero_mean_bounded':
            if not self.bounds:
                raise ValueError("Bounds must be set for non_zero_mean_bounded category")
            return sum(self.bounds) / 2
        else:
            raise ValueError("Unknown category")

    def handle_calculation_error(self, error, idx):
        log_general.error(f"Error calculating indicator {self.id} at index {idx}: {str(error)}")
        raise

    @abstractmethod
    def calculate(self, ohclv, prevs, idx):
        """ Implement the calculation logic for the indicator in subclasses """
        pass

    def indicator_id(self) -> str:
        base_name = self.__class__.__name__.upper()
        return f"{base_name}_{'_'.join(map(str, self.args))}"

    def column_names(self) -> tuple:
        return tuple(f"{self.id}_STREAM_{i+1}" for i in range(self.output_streams))

    def default_stream_aliases(self):
        if self.output_streams == 1:
            return ['out']
        else:
            return [f'out{i+1}' for i in range(self.output_streams)]
        
class EMA(Indicator):
    category = 'price_bound'

    def calculate(self, ohclv, idx):
        multiplier = 2 / (self.length + 1)
        out = ohclv.close[idx] * multiplier + self.prevs.out[idx-1] * (1-multiplier)
        return (out,)

class ATR(Indicator):
    category = 'non_zero_mean_unbounded'
    
    def calculate(self, ohclv, idx):
        high_low = ohclv.high[idx] - ohclv.low[idx]
        high_close = abs(ohclv.high[idx] - ohclv.close[idx-1])
        low_close = abs(ohclv.low[idx] - ohclv.close[idx-1])
        tr = max(high_low, high_close, low_close)
        out = (self.prevs.out[idx-1] * (self.length - 1) + tr) / self.length
        return (out,)

class BOLLINGER(Indicator):
    input_arguments = 2
    output_streams = 3
    stream_aliases = ['upper', 'middle', 'lower']
    category = 'price_bound'

    def __init__(self, *args):
        super().__init__(*args)
        self.std_devs = self.args[1]
        self.ema_indicator = EMA(self.length)
    
    def calculate(self, ohclv, idx):
        lag = len(ohclv.close.data) - 1 - idx
        middle = ohclv.close.mean(length=self.length, lag=lag)
        std = ohclv.close.std_dev(length=self.length, lag=lag)
        upper = middle + std * self.std_devs
        lower = middle - std * self.std_devs
        return (upper, middle, lower)
    
class RSI(Indicator):
    category = 'non_zero_mean_bounded'
    bounds = (0, 100)

    def calculate(self, ohclv, idx):
        lag = len(ohclv.close.data) - 1 - idx
        last_avg_gain = ohclv.close.average_gain(length=self.length-1, lag=lag+1)
        last_avg_loss = ohclv.close.average_loss(length=self.length-1, lag=lag+1)
        current_gain = ohclv.close.average_gain()
        current_loss = ohclv.close.average_loss()
        num = (last_avg_gain * (self.length - 1)) + current_gain
        denom = (last_avg_loss * (self.length - 1)) + current_loss
        div = num / denom if denom != 0 else 0
        out = 100 - (100 / (1 + div))
        return (out,)
    
def init_indicator(name, *args) -> Indicator:
    match name:
        case "EMA":
            return EMA(*args)
        case "ATR":
            return ATR(*args)
        case "BOLLINGER":
            return BOLLINGER(*args)
        case "RSI":
            return RSI(*args)
        case _:
            raise NotImplementedError(f"{name} has not been implemented correctly.")
    