"""
MetaTrader 5 Functions

@author: Tanapong Dechsakul
"""

import datetime as dt
import pandas as pd
import pytz

class MT5:
#    print("MetaTrader5 package author: ",mt5.__author__)
#    print("MetaTrader5 package version: ",mt5.__version__)
    
#    with open('TradeConfig.json', 'r') as json_file:
#        config = json.load(json_file)
    
#    MT5_path = config['MT5_path']
#    login = config['login']
#    password = config['password']
#    server = config['server']
    
    #========================================================
    # Connection
    #========================================================
    
    def initialize (MT5_path, login, password, server, MT=None):
        """
        

        Parameters
        ----------
        path : TYPE
            DESCRIPTION.
        login : TYPE
            DESCRIPTION.
        password : TYPE
            DESCRIPTION.
        server : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        
        # establish MetaTrader 5 connection to a specified trading account
        if not MT.initialize(path=MT5_path, login=login, password=password, server=server, timeout=60000):
            print("initialize() failed, error code =",MT.last_error())
            #quit()
         
        print("MetaTrader is connected successfully! ")
        # Display MetaTrader info
        print("MetaTrader package author: ",MT.__author__)
        # Display MetaTrader version
        print("MetaTrader package version: ",MT.__version__)
        # Display data on connection status, server name and trading account
        #print("MetaTrader terminal info: ",MT.terminal_info())
    
    def connect (login, password, server, MT=None):
        authorized = MT.login(login=login, password=password, server=server, timeout=60000)
        
        if authorized:
            print("connected to account #{}".format(login))
        else:
            print("failed to connect at account #{}, error code: {}".format(login, MT.last_error()))
            
    def shutdown (MT=None):
        # shutdown connection to the MetaTrader 5 terminal
        MT.shutdown()
        print("Shutdown connection to MetaTrader successfully")

    #========================================================
    # Symbol Info
    #========================================================
    def current_price(symbol, MT=None):
        return MT.symbol_info_tick(symbol)
    
    def get_minimal_point(symbol, MT=None):
        return MT.symbol_info(symbol).point
    
    def get_minimal_lot(symbol, MT=None):
        return MT.symbol_info(symbol).volume_min
    
    def get_digits(symbol, MT=None):
        return MT.symbol_info(symbol).digits
    
    def get_trade_contract_size(symbol, MT=None):
        return MT.symbol_info(symbol).trade_contract_size
    
    def get_value_profit_point(symbol, MT=None):
        return MT.symbol_info(symbol).trade_tick_value_profit
    
    def get_value_loss_point(symbol, MT=None):
        return MT.symbol_info(symbol).trade_tick_value_loss
    
    def get_swap_long(symbol, MT=None):
        return MT.symbol_info(symbol).swap_long
    
    def get_swap_short(symbol, MT=None):
        return MT.symbol_info(symbol).swap_short
    
    #while True:
    #    symbols = mt5.symbols_get("NAS100.a")
    #    
    #    for symbol in symbols:
    #        print("symbol: {}, bid price: {}, ask price: {}".format(symbol.name, symbol.bid, symbol.ask))
    
    #while True:
    #    symbol = mt5.symbol_info("NAS100.a")
    #    print("symbol: {}, bid price: {}, ask price: {}".format(symbol.name, symbol.bid, symbol.ask))
    #    time.sleep(1)
    
    #while True:
    #    tick = mt5.symbol_info_tick("NAS100.a")
    #    print("time: {}, bid price: {}, ask price: {}".format(tick.time, tick.bid, tick.ask))
    #    time.sleep(1)
    
    #========================================================
    # Account Information
    #========================================================
    
    def get_balance(MT=None):
        return MT.account_info().balance
    
    def get_equity(MT=None):
        return MT.account_info().equity
    
    def get_margin(MT=None):
        return MT.account_info().margin
    
    #========================================================
    # Getting Price Data
    #========================================================
    
    # Check timezone
    # pytz.all_timezones
    
    # Extract historical data from date
    def get_hist_data_fromDate(symbol, timeframe, time_start=None, num_bar=1, MT=None):
        """
        Parameters
        ----------
        symbol : TYPE Str - e.g. "EURUSD", "NAS100.a"
        timeframe : TYPE str - e.g. "TIMEFRAME_M1", "TIMEFRAME_M5"
        time_start : TYPE str - e.g. "YYYY-MM-DD HH:MM:SS"
        num_bar : TYPE int - e.g. 100, 2000
            DESCRIPTION: Number of bar that is required to be collected.
    
        Returns
        -------
        hist_data_df : TYPE pd dataframe
        """
        
        current_tz = pytz.timezone("Australia/South")
        required_tz = pytz.timezone("Etc/UTC")
        
        if time_start == None:
            time_start = current_tz.localize(dt.datetime.now()).replace(tzinfo=required_tz)
        else:
            time_start = dt.datetime.strptime(time_start, "%Y-%m-%d %H:%M:%S").replace(tzinfo=required_tz)
        
        hist_data = MT.copy_rates_from(symbol, getattr(MT, timeframe), time_start, num_bar)   
        hist_data_df = pd.DataFrame(hist_data) 
        hist_data_df.time = pd.to_datetime(hist_data_df.time, unit="s")
        
        return hist_data_df
    
    # Extract historical data from date
    def get_hist_data_fromPos(symbol, timeframe, bar_start=0, num_bar=1, MT=None):
        """
        Parameters
        ----------
        symbol : TYPE Str - e.g. "EURUSD", "NAS100.a"
        timeframe : TYPE str - e.g. "TIMEFRAME_M1", "TIMEFRAME_M5"
        bar_start : TYPE int - e.g. 100, 2000
            DESCRIPTION. The default is 0.
        num_bar : TYPE int - e.g. 100, 2000
            DESCRIPTION. Number of bar that is required to be collected, the default is 1.
    
        Returns
        -------
        hist_data_df : TYPE pd dataframe
        """
        
        hist_data = MT.copy_rates_from_pos(symbol, getattr(MT, timeframe), bar_start, num_bar)   
        hist_data_df = pd.DataFrame(hist_data) 
        hist_data_df.time = pd.to_datetime(hist_data_df.time, unit="s")
        
        return hist_data_df
    
    # Extract historical data for the specified period of time
    def get_hist_data_dateRange(symbol, timeframe, time_start=None, time_end=None, MT=None):
        """
        Parameters
        ----------
        symbol : TYPE Str - e.g. "EURUSD", "NAS100.a"
        timeframe : TYPE str - e.g. "TIMEFRAME_M1", "TIMEFRAME_M5"
        time_start : TYPE str - e.g. "YYYY-MM-DD HH:MM:SS"
        time_end : TYPE str - e.g. "YYYY-MM-DD HH:MM:SS"
        
        Returns
        -------
        hist_data_df : TYPE pd dataframe
        """
        
        current_tz = pytz.timezone("Australia/South")
        required_tz = pytz.timezone("Etc/UTC")
        
        if time_start == None:
            time_start = current_tz.localize(dt.datetime.now()).replace(tzinfo=required_tz)
        else:
            time_start = dt.datetime.strptime(time_start, "%Y-%m-%d %H:%M:%S").replace(tzinfo=required_tz)
            
        if time_end == None:
            time_end = current_tz.localize(dt.datetime.now()).replace(tzinfo=required_tz)
        else:
            time_end = dt.datetime.strptime(time_end, "%Y-%m-%d %H:%M:%S").replace(tzinfo=required_tz)
        
        hist_data = MT.copy_rates_range(symbol, getattr(MT, timeframe), time_start, time_end)   
        hist_data_df = pd.DataFrame(hist_data) 
        hist_data_df.time = pd.to_datetime(hist_data_df.time, unit="s")
        
        return hist_data_df
    
    #========================================================
    