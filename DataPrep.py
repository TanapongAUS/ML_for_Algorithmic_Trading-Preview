import sys
import time
from datetime import datetime
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
"""
MetaTrader 5 Functions

@author: Tanapong Dechsakul
"""
class DataPrep:
    
    def pull_data_csv(timeframe, main_tf=False, time_start=None, time_end=None):
        """
        

        Parameters
        ----------
        timeframe : TYPE str - e.g. "TIMEFRAME_M1", "TIMEFRAME_H4"
            DESCRIPTION.
        main_tf : TYPE bool - True, if the dataframe is the main timeframe, 
                    else False
            DESCRIPTION. The default is False. If False, the data points of 
            that timeframe must - offset points, so when we calculate the
            indicators, the data points will leave n/a value which could 
            reach the date of the main timeframe cause error when conbining
            data on every timeframe
        time_start : TYPE, optional of str format "%Y-%m-%d %H:%M:%S" 
                    - e.g. "YYYY-MM-DD HH:MM:SS"
            DESCRIPTION. The default is None.
        time_end : TYPE, optional of str format "%Y-%m-%d %H:%M:%S" 
                    - e.g. "YYYY-MM-DD HH:MM:SS"
            DESCRIPTION. The default is None.

        Returns
        -------
        data: After finishing pulling data by specifying the datetime
            DESCRIPTION.

        """
        if timeframe == "TIMEFRAME_M1":
            data = pd.read_csv("files/data/data_M1.csv")
            with open('TradeConfig.json', 'r') as json_file:
                trade_config = json.load(json_file)
                lookback = trade_config['lookback_M1']
        elif timeframe == "TIMEFRAME_M5":
            data = pd.read_csv("files/data/data_M5.csv")
            with open('TradeConfig.json', 'r') as json_file:
                trade_config = json.load(json_file)
                lookback = trade_config['lookback_M5']
        elif timeframe == "TIMEFRAME_M15":
            data = pd.read_csv("files/data/data_M15.csv")
            with open('TradeConfig.json', 'r') as json_file:
                trade_config = json.load(json_file)
                lookback = trade_config['lookback_M15']
        elif timeframe == "TIMEFRAME_M30":
            data = pd.read_csv("files/data/data_M30.csv")
            with open('TradeConfig.json', 'r') as json_file:
                trade_config = json.load(json_file)
                lookback = trade_config['lookback_M30']
        elif timeframe == "TIMEFRAME_H1":
            data = pd.read_csv("files/data/data_H1.csv")
            with open('TradeConfig.json', 'r') as json_file:
                trade_config = json.load(json_file)
                lookback = trade_config['lookback_H1']
        elif timeframe == "TIMEFRAME_H4":
            data = pd.read_csv("files/data/data_H4.csv")
            with open('TradeConfig.json', 'r') as json_file:
                trade_config = json.load(json_file)
                lookback = trade_config['lookback_H4']
        elif timeframe == "TIMEFRAME_D1":
            data = pd.read_csv("files/data/data_D1.csv")
            with open('TradeConfig.json', 'r') as json_file:
                trade_config = json.load(json_file)
                lookback = trade_config['lookback_D1']
        else:
            print("Error: wrong parameter 'timeframe'")
            sys.exit(1)
        
        # Initiate offset value
        offset_value_type_I = 336
        offset_value_type_II = 180
        offset_value_type_III = 60
        
        # Convert data to datetime
        if timeframe == "TIMEFRAME_D1":
            data['time'] = data['time'] + " 00:00:00"
            
        data['time'] = pd.to_datetime(data['time'], format="%Y-%m-%d %H:%M:%S")

        # Calculate for start point
        start_point = 0
        end_point = len(data)
        if time_start != None:
            time_start = pd.to_datetime(time_start)
            time_start = time_start.strftime("%Y-%m-%d %H:%M:%S")
            time_start = datetime.strptime(time_start, "%Y-%m-%d %H:%M:%S")
            
            left = 0 
            right = 0
            for idx in range (len(data)):
                if left==left:
                    data_start_time = data.iloc[left, 0]
                    compared_time = data_start_time.strftime("%Y-%m-%d %H:%M:%S")
                    compared_time = datetime.strptime(compared_time, "%Y-%m-%d %H:%M:%S")
                    
                    #time_start = pd.to_datetime(time_start)
                    #time_start = time_start.strftime("%Y-%m-%d %H:%M:%S")
                    #time_start = datetime.strptime(time_start, "%Y-%m-%d %H:%M:%S")
                
                    if compared_time >= time_start:
                        if left - lookback >= 0:
                            start_point = left - lookback
                            
                            if main_tf == False:
                                if timeframe == "TIMEFRAME_D1":
                                    start_point = start_point - offset_value_type_III
                                elif timeframe == "TIMEFRAME_H4":
                                    start_point = start_point - offset_value_type_II
                                else:
                                    start_point = start_point - offset_value_type_I
                                    
                                if start_point < 0:
                                    print("Error: the range of data is incorrect, there are not enough data for the start point of " + timeframe)
                                    sys.exit(1)
                        break
                left += 1
                
                right -= 1
                if right==right:
                    data_start_time = data.iloc[right, 0]
                    compared_time = data_start_time.strftime("%Y-%m-%d %H:%M:%S")
                    compared_time = datetime.strptime(compared_time, "%Y-%m-%d %H:%M:%S")
                    
                    #time_start = pd.to_datetime(time_start)
                    #time_start = time_start.strftime("%Y-%m-%d %H:%M:%S")
                    #time_start = datetime.strptime(time_start, "%Y-%m-%d %H:%M:%S")
                
                    if compared_time <= time_start:
                        if (len(data) + right) - lookback >= 0:
                            start_point = (len(data) + right) - lookback
                            
                            if main_tf == False:
                                if timeframe == "TIMEFRAME_D1":
                                    start_point = start_point - offset_value_type_III
                                elif timeframe == "TIMEFRAME_H4":
                                    start_point = start_point - offset_value_type_II
                                else:
                                    start_point = start_point - offset_value_type_I
                                    
                                if start_point < 0:
                                    print("Error: the range of data is incorrect, there are not enough data for the start point of " + timeframe)
                                    sys.exit(1)
                        break
                    
        
        if time_end != None:
            for idx in range (start_point, len(data)):
                data_end_time = data.iloc[idx, 0]
                compared_time = data_end_time.strftime("%Y-%m-%d %H:%M:%S")
                compared_time = datetime.strptime(compared_time, "%Y-%m-%d %H:%M:%S")
                
                time_end = pd.to_datetime(time_end)
                time_end = time_end.strftime("%Y-%m-%d %H:%M:%S")
                time_end = datetime.strptime(time_end, "%Y-%m-%d %H:%M:%S")
                
                
                if compared_time >= time_end:
                    end_point = idx
                    break
        
        data = data[start_point: end_point]
        
        return data
            
        
    
    #=========================================================================#
 
    def arrange_data(main_df: pd.DataFrame, main_lookback: int, array_sub_df: list[list]=None, array_lookback: list[list]=None):
        """
        

        Parameters
        ----------
        main_df : TYPE
            DESCRIPTION.
        main_lookback : TYPE
            DESCRIPTION.
        array_sub_df : TYPE array, optional
            DESCRIPTION. The default is None.
        array_lookback : TYPE array, optional
            DESCRIPTION. The default is None.
        function_type : TYPE "train_test" or "predict", optional
            DESCRIPTION. The default is "train_test".

        Returns
        -------
        df : TYPE
            DESCRIPTION.

        """
        
        if array_sub_df is None or array_lookback is None:
            # | Create Main numpy array to collect all data | 
            # ===============================================
            N = len(main_df) - main_lookback + 1
            T = main_lookback
            D = main_df.shape[1] - 1 # The context of - 1 is to delete 'time' column
            data = np.zeros((N, T, D)).astype(np.float32)
            
            marker = 0
            for idx in range (main_lookback - 1, len(main_df)):
                temp_data = main_df.iloc[idx - main_lookback + 1 : idx + 1, 1 : ]
                data[marker, :, :] = temp_data
                
                marker += 1
            
            return data
        
        # | Create Main numpy array to collect all data | 
        # ===============================================
        N = len(main_df) - main_lookback + 1
        T = main_lookback
        D = main_df.shape[1] - 1 # The context of - 1 is to delete 'time' column
        data = np.zeros((N, T, D)).astype(np.float32)
            
        data_array = []
        data_array.append(data)
            
        for num in range (len(array_sub_df)):
            T = array_lookback[num]
            D = array_sub_df[num].shape[1] - 1 # The context of - 1 is to delete 'time' column
            data = np.zeros((N, T, D)).astype(np.float32)
            data_array.append(data)
            
        # Creating array of marker to store idx of each different dataframes
        marker = []
        for idx in range (len(array_sub_df) + 1):
            marker.append(0)
            
        # Start Combining
        for idx in range (main_lookback - 1, len(main_df)):
            temp_data = main_df.iloc[idx - main_lookback + 1 : idx + 1, 1 : ]
            data_array[0][marker[0], :, :] = temp_data
                
            # Get 'time' data from main_df
            main_time = main_df.iloc[idx, 0]
            main_time = main_time.strftime("%Y-%m-%d %H:%M:%S")
            main_time = datetime.strptime(main_time, "%Y-%m-%d %H:%M:%S")
               
            # Access to each data array
            for num_sub_df in range (len(array_sub_df)):
                for idx_sub_df in range (marker[num_sub_df + 1], len(array_sub_df[num_sub_df])): # the context of marker[num_sub_df + 1] because marker has 5 elements including main marker which the data has been collected earlier. 
                    # | Get 'time' data from sub_df | 
                    # ===============================
                    # Time of previous bar
                    prev_bar = idx_sub_df - 1
                    if idx_sub_df == 0:
                        prev_time = array_sub_df[num_sub_df].iloc[0, 0]
                        prev_time = prev_time.strftime("%Y-%m-%d %H:%M:%S")
                        prev_time = datetime.strptime(prev_time, "%Y-%m-%d %H:%M:%S")
                    else:
                        prev_time = array_sub_df[num_sub_df].iloc[prev_bar, 0]
                        prev_time = prev_time.strftime("%Y-%m-%d %H:%M:%S")
                        prev_time = datetime.strptime(prev_time, "%Y-%m-%d %H:%M:%S")
                        
                    # Time from current bar
                    cur_bar = idx_sub_df
                    if idx_sub_df == len(array_sub_df[num_sub_df]) - 1:
                        cur_time = array_sub_df[num_sub_df].iloc[len(array_sub_df[num_sub_df]) - 1, 0]
                        cur_time = cur_time.strftime("%Y-%m-%d %H:%M:%S")
                        cur_time = datetime.strptime(cur_time, "%Y-%m-%d %H:%M:%S")
                    else:
                        cur_time = array_sub_df[num_sub_df].iloc[idx_sub_df, 0]
                        cur_time = cur_time.strftime("%Y-%m-%d %H:%M:%S")
                        cur_time = datetime.strptime(cur_time, "%Y-%m-%d %H:%M:%S")
                
                    # Compare time
                    if main_time > prev_time and main_time < cur_time:
                        temp_data = array_sub_df[num_sub_df].iloc[prev_bar - array_lookback[num_sub_df]: prev_bar, 1:]
                        
                        data_array[num_sub_df + 1][marker[0], :, :] = temp_data
                        marker[num_sub_df + 1] = idx_sub_df
                        break
                    elif main_time == cur_time:
                        temp_data = array_sub_df[num_sub_df].iloc[cur_bar - array_lookback[num_sub_df]: cur_bar, 1:]
                            
                        data_array[num_sub_df + 1][marker[0], :, :] = temp_data
                        marker[num_sub_df + 1] = idx_sub_df
                        break
                    elif idx_sub_df == len(array_sub_df[num_sub_df]) - 1 and main_time > cur_time:
                        extended_time = cur_time + (cur_time - prev_time)
                        if main_time < extended_time:
                            temp_data = array_sub_df[num_sub_df].iloc[cur_bar - array_lookback[num_sub_df]: cur_bar, 1:]
                            data_array[num_sub_df + 1][marker[0], :, :] = temp_data
                        else:
                            temp_data = array_sub_df[num_sub_df].iloc[cur_bar + 1 - array_lookback[num_sub_df]: cur_bar + 1, 1:]
                            data_array[num_sub_df + 1][marker[0], :, :] = temp_data
                        break
                
            # Move the main idx
            marker[0] += 1
            
        return data_array

    #=========================================================================#
    
    def transform_data(df, model_type="Keras"):
        if model_type == "Keras":
            dataset = df.iloc[ : , 1: ].to_numpy()
            return dataset
        
        elif model_type == "Pytorch":
            # Convert data into Torch tensor and put the dataset into Dataloader
            dataset = df.iloc[ : , 1: ].to_numpy()
            dataset = torch.tensor(dataset).to(dtype=torch.float32)
            dataset = DataLoader(dataset, batch_size=1, shuffle=False)
            return dataset
        
        else:
            print("Error: wrong parameter 'model_type'")
            sys.exit(1)
    
    #=========================================================================#
    
    def frame(dataset, numState, model_type="Keras"):
        """
        

        Parameters
        ----------
        dataset : TYPE numpy array or Pytorch dataLoader
            DESCRIPTION.
        numState : TYPE int
            DESCRIPTION.
        model_type : TYPE "Keras" or "Pytorch", optional
            DESCRIPTION. The default is "Keras".

        Returns
        -------
        data : TYPE
            DESCRIPTION.

        """
        if model_type == "Keras":
            
            data = []
            for idx in range (len(dataset)):
                temp_data = dataset[idx][numState]
                temp_data = np.expand_dims(temp_data, axis=0)
                data.append(temp_data)
            
            shape_data = []
            for num_shape in range(len(data)):
                temp_shape = data[num_shape].shape[1]
                shape_data.append(temp_shape)
            
            padded_data = []
            for num in range (len(data)):
                temp_padded_data = pad_sequences(data[num], maxlen=max(shape_data), padding='post')
                padded_data.append(temp_padded_data)

            return padded_data
        
        elif model_type == "Pytorch":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            for batch_idx, batch in enumerate(dataset):
                if batch_idx in [numState]:  
                    data = batch[ : , : ].to(device)
                    break
            
            return data
        
        else:
            print("Error: wrong parameter 'model_type'")
            sys.exit(1)
    
    #=========================================================================#
    
    # Custom Data Generator
    def data_generator(X: list, Y: dict, num_datapoint: int, sleep_time: float = 0.5):
        def generator():
            while True:  # Infinite generator loop for training
                for idx in range(num_datapoint):
                    data = []
                    for dataset in range(len(X)):
                        x = X[dataset][idx: idx + 1, :, :]  # Extract one data point
                        data.append(x)
                    
                    # Ensure data is converted to numpy arrays
                    # data = [np.array(d) for d in data]
    
                    target = {}
                    keys = list(Y.keys())
                    for key in keys:
                        target[key] = np.array(Y[key][idx: idx + 1])  # Convert targets to numpy arrays
                    
                    yield tuple(data), target  # Yield data and target
                    time.sleep(sleep_time)  # Introduce delay
    
        # Define the output signature for the generator
        output_signature = (
            tuple(tf.TensorSpec(shape=(1, None, None), dtype=tf.float32) for _ in range(len(X))),  # Inputs
            {key: tf.TensorSpec(shape=(1, None), dtype=tf.float32) for key in Y.keys()}  # Targets
        )
    
        return tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    
    #=========================================================================#
    
    def pad_3d_sequences(data: np.array, max_timesteps: int):
        """
        Pad a 3D array of sequences to the maximum timesteps.
        
        Args:
            data: List or array of 3D sequences, shape (num_samples, timesteps, features).
            max_timesteps: The desired maximum length of timesteps.
        
        Returns:
            Padded 3D array with shape (num_samples, max_timesteps, features).
        """
        num_samples = len(data)
        num_features = data[0].shape[1]
        padded_data = np.zeros((num_samples, max_timesteps, num_features))
        
        for i, sample in enumerate(data):
            timesteps = sample.shape[0]
            padded_data[i, :timesteps, :] = sample  # Copy original data into the padded array
        
        return padded_data
    
    #=========================================================================#