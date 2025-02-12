import numpy as np
import pandas as pd
import sys

def cal_change(df: pd.DataFrame, array_indicators: list):
    """
        

    Parameters
    ----------
    df : TYPE dataframe
    DESCRIPTION.
    array_indicators : TYPE [[result_column, value_column], 
                             [result_column, value_column]] 
                	   e.g. [['maCH', 'ma'], ['openCH', 'open']]
    DESCRIPTION. Actual value diff 

    Returns
    -------
    df : TYPE dataframe
    DESCRIPTION.

    """
        
    for result_column, value_column in array_indicators:
        df[result_column] = df[value_column].diff()
        
    df = df.dropna(axis=0)
    df = df.reset_index(drop=True)
            
    return df
    
#=========================================================================#

def cal_future_change(df: pd.DataFrame, num_roll: int, array_indicators: list):
    """
    

    Parameters
    ----------
    df : TYPE dataframe
        DESCRIPTION.
    num_roll : TYPE number of future data that are wanted to view
        DESCRIPTION.
    array_indicators : TYPE [[result_column, value_column], 
                             [result_column, value_column]] 
                	   e.g. [['maCH', 'ma'], ['openCH', 'open']]
        DESCRIPTION. Actual value diff

    Returns
    -------
    df : TYPE dataframe
    DESCRIPTION.

    """
    env_view_offset = 1
    
    for result_column, value_column in array_indicators:
        df[result_column] = df[value_column].shift(- num_roll)
        df[result_column] = df[result_column] - df[value_column]
        df[result_column] = df[result_column].shift(env_view_offset)
            
    return df

#=========================================================================#

def cal_diff(df: pd.DataFrame, array_indicators: list[list]):
    """
    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.
    array_indicators : list[list]
        DESCRIPTION. For example - data_M1 = cal_diff(data_M1, [['MA&Close_diff', 'close', 'MA_20'], 
                                                                ['diff_bb', 'upper_bb', 'lower_bb']])

    Returns
    -------
    df : TYPE - pd.DataFrame
        DESCRIPTION.

    """
    for result, firstCol, secondCol in array_indicators:
        df[result] = df[firstCol] - df[secondCol]
    
    return df

#=========================================================================#

def MA_SMA(df: pd.DataFrame, period: int=20, apply_to: str='close'):
    
    if apply_to == 'close' or apply_to == 'high' or apply_to == 'low' or apply_to == 'open':
        column_name = f'MA_{period}_{apply_to}'
        df[column_name] = df[apply_to].rolling(period).mean()
    else:
        print(f'Error: MA_SMA indicator can be applied to column "close", "high", "low", "open" only. Receive {apply_to}')
        sys.exit(1)
        
    return df

#=========================================================================#

def Bollinger_Bands(df, period=20, deviations=2.0):
    df['middle_band'] = df['close'].rolling(window=period).mean()
    
    data = df.copy()
    
    # Calculate rolling standard deviation
    data['rolling_std'] = data['close'].rolling(window=period).std()
    
    # Calculate the upper and lower Bollinger Bands
    df['upper_band'] = data['middle_band'] + (deviations * data['rolling_std'])
    df['lower_band'] = data['middle_band'] - (deviations * data['rolling_std'])
        
    return df

#=========================================================================#

def MACD(df, fast_EMA=12, slow_EMA=26, MACD_SMA=9):
    data = df.copy()
        
    data['fast_ema'] = data['close'].ewm(span=fast_EMA, adjust=False).mean()
    data['slow_ema'] = data['close'].ewm(span=slow_EMA, adjust=False).mean()
    data['macd'] = data['fast_ema'] - data['slow_ema']
    
    data['signal'] = data['macd'].ewm(span=MACD_SMA, adjust=False).mean()
    
    #data['histogram'] = data['macd'] - data['signal']
        
    df['macd'] = data['macd'].iloc[slow_EMA - 1: ]
    df['macd_signal'] = data['signal'].iloc[slow_EMA - 1 + MACD_SMA - 1: ]
        
    return df

#=========================================================================#

def ATR(df, period=14):
    data = df.copy()
    
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = np.abs(data['high'] - data['close'].shift(1))
    data['tr3'] = np.abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)

    # Calculate the ATR using a simple moving average of the True Range
    data['atr'] = data['true_range'].rolling(window=period).mean()
    
    df[f'atr_{period}'] = data['atr']
    
    return df

#=========================================================================#