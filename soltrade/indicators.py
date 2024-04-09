import pandas as pd

class Indicator:
    input_arguments = None 

    def __init__(self, *args):
        # Validate the number of input arguments if input_arguments is specified
        if self.input_arguments is not None and len(args) != self.input_arguments:
            raise ValueError(f"{self.__class__.__name__} expects exactly {self.input_arguments} argument(s), got {len(args)}")
        self.args = args
        self.output_datapoints = getattr(self, 'output_datapoints', 1)  # Default to 1 if not specified
        self.cols = self.column_names()

    def calculate(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def column_names(self):
        base_name = self.__class__.__name__.upper()
        if self.output_datapoints == 1:
            return (f"{base_name}_{'_'.join(map(str, self.args))}",)
        else:
            return tuple(f"{base_name}_{'_'.join(map(str, self.args))}_{i+1}" for i in range(self.output_datapoints))

class EMA(Indicator):
    input_arguments = 1
    output_datapoints = 1

    def __init__(self, *args):
        super().__init__(*args)
        self.length = self.args[0]  # The length parameter
    
    def calculate(self, df):
        # Placeholder for the EMA calculation logic
        return df

class ATR(Indicator):
    input_arguments = 1
    output_datapoints = 1

    def __init__(self, *args):
        super().__init__(*args)
        self.length = self.args[0]  # The length parameter
    
    def calculate(self, df):
        # Placeholder for the ATR calculation logic
        return df

class BOLLINGER(Indicator):
    input_arguments = 2
    output_datapoints = 2

    def __init__(self, *args):
        super().__init__(*args)
        self.length = self.args[0]  # The length parameter
        self.std_devs = self.args[1]  # The standard deviations parameter
    
    def calculate(self, df):
        # Placeholder for the Bollinger Bands calculation logic
        return df

    
    
def init_indicator(name, *args):
    match name:
        case "EMA":
            return EMA(*args)
        case "ATR":
            return ATR(*args)
        case "BOLLINGER":
            return BOLLINGER(*args)
        case _:
            raise NotImplementedError(f"{name} has not been implemented correctly.")
    
# Calculates EMA using DataFrame
def calculate_ema(dataframe: pd.DataFrame, length: int):
    ema = dataframe['close'].ewm(span=length, adjust=False).mean()
    return ema.iat[-1]


# Calculates BB using SMA indicator and DataFrame
def calculate_bbands(dataframe: pd.DataFrame, length: int, std_devs : float):
    sma = dataframe['close'].rolling(length).mean()
    std = dataframe['close'].rolling(length).std() * std_devs 
    upper_bband = sma + std 
    lower_bband = sma - std
    return upper_bband, lower_bband


# Calculates RSI using custom EMA indicator and DataFrame
def calculate_rsi(dataframe: pd.DataFrame, length: int):
    delta = dataframe['close'].diff()
    up = delta.clip(lower=0)
    down = delta.clip(upper=0).abs()
    upper_ema = up.ewm(com=length - 1, adjust=False, min_periods=length).mean()
    lower_ema = down.ewm(com=length - 1, adjust=False, min_periods=length).mean()
    rsi = upper_ema / lower_ema
    rsi = 100 - (100 / (1 + rsi))
    return rsi.iat[-1]
