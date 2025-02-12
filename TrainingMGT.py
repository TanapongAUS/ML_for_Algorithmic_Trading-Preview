import json

import MetaTrader5 as metaTrader
from MT5 import MT5
from Init import Init
from Utils import Utils

from Indicators import ATR, MA_SMA
from DataPrep import DataPrep

from Models import Mod_LSTM_Keras

import time
import sys

#=============================================================================#
# Global configuration
#=============================================================================#
IS_CONNECT_MT5 = False
IS_SPLIT_DATA = False
SPLITTING_RATIO = 0.5
IS_SAVE_NUMPY_DATA = False
IS_CREATE_NEW_MODEL = True
IS_CLONE_MODEL = False
ORIGINAL_MODEL_NAME = "model-class_1"



#=============================================================================#
# Read environment setup configuration
#=============================================================================#
# | Read JSON file |
# ==================
with open('TradeConfig.json', 'r') as json_file:
    account_config = json.load(json_file)
    
with open('TrainConfig.json', 'r') as json_file:
    train_config = json.load(json_file)

start_time_data = "2024-12-05 00:00:00" # train_config['time_start_train']
   
    
   
#=============================================================================#
# MT5 login, Connect to Metatrader5
#=============================================================================#
if IS_CONNECT_MT5:
    
    MT5.initialize(account_config['MT5_path'], account_config['login'], account_config['password'], account_config['server'], MT=metaTrader)
    time.sleep(3)
    
    # Replace the value in the JSON keys
    account_config["symbol_digits"] = MT5.get_digits(account_config['symbol'], MT=metaTrader)
    account_config["point"] = MT5.get_minimal_point(account_config['symbol'], MT=metaTrader)
    account_config["minimal_lot"] = MT5.get_minimal_lot(account_config['symbol'], MT=metaTrader)
    account_config["trade_contract_size"] = MT5.get_trade_contract_size(account_config['symbol'], MT=metaTrader)
    account_config["swap_long"] = MT5.get_swap_long(account_config['symbol'], MT=metaTrader)
    account_config["swap_short"] = MT5.get_swap_short(account_config['symbol'], MT=metaTrader)
    
    isValueCorrect = False
    trade_tick_value_profit = 0
    trade_tick_value_loss = 0
    num_retry = 10
    
    for num in range(num_retry):
        trade_tick_value_profit = MT5.get_value_profit_point(account_config['symbol'], MT=metaTrader)
        trade_tick_value_loss = MT5.get_value_loss_point(account_config['symbol'], MT=metaTrader)
        time.sleep(1)
        
        if trade_tick_value_profit != 0 and trade_tick_value_loss != 0:
            break
    
    if trade_tick_value_profit != 0 and trade_tick_value_loss != 0:
        account_config["trade_tick_value_profit"] = trade_tick_value_profit
        account_config["trade_tick_value_loss"] = trade_tick_value_loss
    else:
        print("System is unable to save symbol info in JSON file correctly!")
        sys.exit(1)
    
    # Write the updated data back to the JSON file
    with open('TradeConfig.json', 'w') as file:
        json.dump(account_config, file, indent=4)
        
    # Reopen and read JSON file
    with open('TradeConfig.json', 'r') as json_file:
        account_config = json.load(json_file)
        
    with open('TrainConfig.json', 'r') as json_file:
        train_config = json.load(json_file)
    
    #=============================================================================#
    # Collecting data
    #=============================================================================#
    Init.init()
    
    if Init.check_data() == False:
        Utils.collect_data_first(MT=metaTrader)
    else: 
        Utils.collect_data_cont(MT=metaTrader)
    
    # shutdown connection to the MetaTrader 5 terminal
    MT5.shutdown(MT=metaTrader)



#=============================================================================#
# Pulling Data From CSV Files
print("\r==> Retrieving data... 🔍", end='\r', flush=True)
#=============================================================================#
"""
Parameters
----------
time : TYPE datetime - e.g. "2024-07-01 00:00:00"
    DESCRIPTION. Specify start date and time to pull data.
"""
data_M1 = DataPrep.pull_data_csv("TIMEFRAME_M1", True, start_time_data)
data_M1 = data_M1.reset_index(drop=True)

data_H1 = DataPrep.pull_data_csv("TIMEFRAME_H1", False, start_time_data)
data_H1 = data_H1.reset_index(drop=True)

data_H4 = DataPrep.pull_data_csv("TIMEFRAME_H4", False, start_time_data)
data_H4 = data_H4.reset_index(drop=True)

print("\r==> Retrieving data is complete! ✅")



#=============================================================================#
# Apply indicators
print("\r==> Calculating indicators... 💻", end='\r', flush=True)
#=============================================================================#
# Time signal - Check time for trading
data_M1 = tradeTime(data_M1, account_config['Monday_start_time'], 
                    account_config['weekday_start_time'], 
                    account_config['weekday_stop_time'], 
                    account_config['Friday_stop_time'], 
                    account_config['Friday_closeAll_time'])

# Lag indicators >>> Example Indicators
data_M1 = MA_SMA(ATR(data_M1, period=14), period=50)
data_H1 = MA_SMA(ATR(data_H1, period=14),  period=50)
data_H4 = MA_SMA(ATR(data_H4, period=14), period=50)

# Lead indicators >>> This part is not available for preview
#data_M1 = superTrend(data_M1, periods=3, multiplier=2.0)
#data_H1 = fibonacci(MP(superTrend(data_H1, periods=3, multiplier=2.0), lookback=168, upTrend=True, downTrend=True), lookback_period=24, validation_period=12)
#data_H4 = superTrend(data_H4, periods=3, multiplier=2.0)

print("\r==> Calculating indicators is complete! ✅")

#=============================================================================#
# Apply strategy
#=============================================================================#
# This part is not available for preview



#=============================================================================#
# Calculate change & drop n/a
#=============================================================================#
# This part is not available for preview



#=============================================================================#
# Train & test split
#=============================================================================#
# This part is not available for preview



#=============================================================================#
# Normalise data by STD, and create one-hot labels
#=============================================================================#
# This part is not available for preview



#=============================================================================#
# Combine data from every timeframe and Arrange dataset
#=============================================================================#
# This part is not available for preview



#=============================================================================================#
# Initiate model
#=============================================================================================#
if IS_CREATE_NEW_MODEL:
    print("\r==> Creating learning model... 📈", end='\r', flush=True)
    # Record as .txt file
    Utils.save_txt("KPI-model_predict", 0, 0, 0)
    
    # Create model
    tradeMod = Mod_LSTM_Keras(array_timestep = array_timestep,
                              array_feature = array_feature, 
                              lstm_array_units = [[512,256], [128,64,32,train_config['action_array'][0]], [128,64,32,train_config['action_array'][1]]],
                              dropout_rate=0.2, regularizer_rate=0.0001, 
                              kernel_weight_max_norm=5.0, rnn_weight_max_norm=3.0, bias_max_norm=1.0)
        
    tradeMod = tradeMod.create_model()
    
    if IS_CLONE_MODEL:
        
        if ORIGINAL_MODEL_NAME == None:
            print("Error: Unsuccessful cloning model, the original model is not existed.")
            sys.exit(1)
        
        original_mod = Utils.load_model(ORIGINAL_MODEL_NAME)
        res = Utils.clone_weight_and_bias(original_mod=original_mod, new_mod=tradeMod)
        
        if res == False:
            print("Error: Unsuccessful creating and saving new model.")
            sys.exit(1)
        
    else:
        # Save model
        Utils.save_model(tradeMod, "model-class_1")
        
    print("\r==> Creating learning model is complete! ✅")
    