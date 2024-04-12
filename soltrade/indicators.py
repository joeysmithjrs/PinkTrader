import numpy as np
import pandas as pd

class Indicator:
    input_arguments = None 

    def __init__(self, *args):
        # Validate the number of input arguments if input_arguments is specified
        if self.input_arguments is not None and len(args) != self.input_arguments:
            raise ValueError(f"{self.__class__.__name__} expects exactly {self.input_arguments} argument(s), got {len(args)}")
        self.args = args
        self.length = args[0] # First argument defaults to length
        self.output_datapoints = getattr(self, 'output_datapoints', 1)  # Default to 1 if not specified
        self.id = self.indicator_id()
        self.cols = self.column_names()

    def calculate(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def indicator_id(self):
        base_name = self.__class__.__name__.upper()
        return f"{base_name}_{'_'.join(map(str, self.args))}"
    
    def column_names(self):
        return tuple(f"{self.id}_STREAM_{i+1}" for i in range(self.output_datapoints))

class EMA(Indicator):
    input_arguments = 1
    output_datapoints = 1

    def __init__(self, *args):
        super().__init__(*args)
    
    def calculate(self, df):
        # Placeholder for the EMA calculation logic
        return df

class ATR(Indicator):
    input_arguments = 1
    output_datapoints = 1

    def __init__(self, *args):
        super().__init__(*args)
    
    def calculate(self, df):
        # Placeholder for the ATR calculation logic
        return df

class BOLLINGER(Indicator):
    input_arguments = 2
    output_datapoints = 2

    def __init__(self, *args):
        super().__init__(*args)
        self.std_devs = self.args[1]  # The standard deviations parameter
    
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
    