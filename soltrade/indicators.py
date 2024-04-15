import numpy as np
import pandas as pd
from datastructures import Stream, StreamContainer
from abc import ABC, abstractmethod

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
        assert len(self.cols) == len(set(self.cols)), "Column names must be unique"
        if not hasattr(self, 'stream_aliases'):
            self.stream_aliases = self.default_stream_aliases()
        assert len(self.stream_aliases) == len(set(self.stream_aliases)), "Stream aliases must be unique"
        self.alias_to_cols = dict(zip(self.stream_aliases, self.cols))

    def next(self, ohclv: StreamContainer, prevs: StreamContainer):
        data_length = len(next(iter(prevs.streams.values())).data)
        first_valid_idx = self.n_self_referential
        all_nan_init = False  # Flag to check if any stream was initialized due to all NaNs

        for stream in prevs.streams.values():
            if np.isnan(stream.data[:first_valid_idx]).all():
                all_nan_init = True  # Set flag if initializing any stream
                if self.category in [None, 'zero_mean', 'non_zero_mean_unbounded']:
                    default_value = 0
                elif self.category == 'price_bound':
                    default_value = ohclv.close[0]  # Assuming there is a 'close' stream in ohclv
                elif self.category == 'non_zero_mean_bounded':
                    if not self.bounds:
                        raise ValueError("Bounds must be set for non_zero_mean_bounded category")
                    default_value = sum(self.bounds) / 2
                else:
                    raise ValueError("Unknown category")

                stream.data[:first_valid_idx] = default_value

        start_idx = 0 if all_nan_init else first_valid_idx

        for idx in range(first_valid_idx, data_length):
            calculated_values = self.calculate(ohclv, prevs, idx)
            for alias, value in zip(self.stream_aliases, calculated_values):
                if self.bounds:
                    value = max(min(value, self.bounds[1]), self.bounds[0])
                prevs.streams[alias].data[idx] = value

        return prevs, (start_idx, data_length - 1)

    @abstractmethod
    def calculate(self):
        """ Implement the calculation logic for the indicator in subclasses """
        pass

    def indicator_id(self) -> str:
        base_name = self.__class__.__name__.upper()
        return f"{base_name}_{'_'.join(map(str, self.args))}"

    def column_names(self) -> tuple:
        return tuple(f"{self.id}_STREAM_{i+1}" for i in range(self.output_streams))

    def default_stream_aliases(self):
        # Generate default stream aliases based on the number of output data points
        if self.output_streams == 1:
            return ['out']
        else:
            return [f'out{i+1}' for i in range(self.output_streams)]
        
class EMA(Indicator):
    category = 'price_bound'

    def calculate(self, ohclv, prevs, idx):
        multiplier = 2 / (self.length + 1)
        new_val = ohclv.close[idx] * multiplier + prevs.out[idx-1] * (1-multiplier)
        return (new_val,)

class ATR(Indicator):
    category = 'non_zero_mean_unbounded'
    
    def calculate(self, df):
        # Placeholder for the ATR calculation logic
        return df

class BOLLINGER(Indicator):
    input_arguments = 2
    output_streams = 2
    stream_aliases = ['upper', 'lower']
    category = 'price_bound'

    def __init__(self, *args):
        super().__init__(*args)
        self.std_devs = self.args[1]
    
    def calculate(self, df):
        # Placeholder for the Bollinger Bands calculation logic
        return df

def init_indicator(name, *args) -> Indicator:
    match name:
        case "EMA":
            return EMA(*args)
        case "ATR":
            return ATR(*args)
        case "BOLLINGER":
            return BOLLINGER(*args)
        case _:
            raise NotImplementedError(f"{name} has not been implemented correctly.")
    