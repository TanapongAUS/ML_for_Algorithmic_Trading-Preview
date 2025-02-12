import json
import pandas as pd

from datetime import datetime
from MT5 import MT5

"""
Created on Sat Aug 24 10:42:00 2024

@author: tanap
"""
class Utils:
    
    def collect_data_first(MT=None):
        with open('TradeConfig.json', 'r') as json_file:
            account_config = json.load(json_file)
            
        with open('TrainConfig.json', 'r') as json_file:
            train_config = json.load(json_file)
            
        data_list = [["TIMEFRAME_M1","files/data/data_M1.csv"], ["TIMEFRAME_M5","files/data/data_M5.csv"],
                     ["TIMEFRAME_M15","files/data/data_M15.csv"], ["TIMEFRAME_M30","files/data/data_M30.csv"],
                     ["TIMEFRAME_H1","files/data/data_H1.csv"], ["TIMEFRAME_H4","files/data/data_H4.csv"],
                     ["TIMEFRAME_D1","files/data/data_D1.csv"]]
        
        current_data = MT5.get_hist_data_fromPos(account_config['symbol'], "TIMEFRAME_M1", MT=MT)
        time_end = current_data['time'][0]
        #time_end = time_end.replace(hour=0, minute=0, second=0, microsecond=0)
        time_end = time_end.strftime("%Y-%m-%d %H:%M:%S")
        
        for timeframe, path in data_list:
            df = MT5.get_hist_data_dateRange(account_config['symbol'], timeframe, train_config['start_collect_data'], time_end, MT=MT)
            df.to_csv(path, index=False)
        
        # Checking for correctness of current data by comparing the lastest datapoint of every timeframe with current data_M1
        #Current M1 time
        current_time = pd.read_csv("files/data/data_M1.csv")
        current_time = current_time.iloc[-1, 0]
        #current_time = pd.to_datetime(current_time)
        current_time = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
        del data_list[0]
        
        for timeframe, path in data_list:
            data = pd.read_csv(path)
            data = data[:-1]
            data.to_csv(path, index=False)
            
        print("Data is collected successfully!")
    
    #=========================================================================#