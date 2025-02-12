import sys
import os

"""
Created on Fri Jul 26 14:38:31 2024

@author: tanap
"""
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, concatenate, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm

class Mod_LSTM_Keras:
    def __init__(self, array_timestep: list, array_feature: list, lstm_array_units: list, dropout_rate: float=0.2, regularizer_rate: float=None, kernel_weight_max_norm: float=None, rnn_weight_max_norm: float=None, bias_max_norm: float=None):     
        """
        

        Parameters
        ----------
        array_timestep : TYPE
            DESCRIPTION.
        array_feature : TYPE
            DESCRIPTION.
        lstm_array_units : TYPE
            DESCRIPTION - first array = price layer
                        - second array = trend layer
                        - third array = action layer
                        - fourth array = sl layer
        dropout_rate : TYPE, optional
            DESCRIPTION. The default is 0.2.

        Returns
        -------
        None.

        """
        
        if len(lstm_array_units) != 3:
            print("Error: the lstm_array_units must have three types, but got lstm_array_units = {}".format(len(lstm_array_units)))
            sys.exit(1)
            
        if len(lstm_array_units[0]) < 2:
            print("Error: the lstm_array_units for the price layer (main timeframe) cannot contain less than two layers, but got lstm_array_units = {}".format(len(lstm_array_units[0])))
            sys.exit(1)
            
        if len(lstm_array_units[1]) < 2:
            print("Error: the lstm_array_units for the trend layer cannot contain less than two layers, but got lstm_array_units = {}".format(len(lstm_array_units[1])))
            sys.exit(1)
        
        if len(lstm_array_units[2]) < 2:
            print("Error: the lstm_array_units for the projectile layer cannot contain less than two layers, but got lstm_array_units = {}".format(len(lstm_array_units[2])))
            sys.exit(1)
        
        # Initialize model parameters  
        self.array_timestep = array_timestep
        self.array_feature = array_feature
        self.lstm_array_units = lstm_array_units
        
        # Determining layer parameters
        self.dropout_rate = dropout_rate
        
        if regularizer_rate != None:
            self.regularizer_rate = l2(regularizer_rate)
        else:
            self.regularizer_rate = regularizer_rate
        
        if kernel_weight_max_norm != None:
            self.kernel_weight_max_norm = MaxNorm(max_value=kernel_weight_max_norm)
        else:
            self.kernel_weight_max_norm = kernel_weight_max_norm
            
        if rnn_weight_max_norm != None:
            self.rnn_weight_max_norm = MaxNorm(max_value=rnn_weight_max_norm)
        else:
            self.rnn_weight_max_norm = rnn_weight_max_norm
        
        if bias_max_norm != None:
            self.bias_max_norm = MaxNorm(max_value=bias_max_norm)
        else:
            self.bias_max_norm = bias_max_norm
        
        self.model = self.initiate_model()

    def initiate_model(self):
        # | Price Layer - Main timeframe |
        # ================================
        # Define the input layer
        input_layer_main = Input(shape=(max(self.array_timestep), self.array_feature[0]), name='input_main')
        input_layer_sub_1 = Input(shape=(max(self.array_timestep), self.array_feature[1]), name='input_sub1')
        # input_layer_sub_2 = Input(shape=(max(self.array_timestep), self.array_feature[2]), name='input_sub2')
        # Concatenate all outputs from the Individual Price Layer
        concat_input = concatenate([input_layer_main, input_layer_sub_1], name='concat_input_layer')
        
        # LSTM 
        lstm_layer = LSTM(units=self.lstm_array_units[0][0], 
                          kernel_regularizer=self.regularizer_rate, recurrent_regularizer=self.regularizer_rate, bias_regularizer=self.regularizer_rate,
                          kernel_constraint=self.kernel_weight_max_norm, recurrent_constraint=self.rnn_weight_max_norm, bias_constraint=self.bias_max_norm,
                          dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate,
                          return_sequences=True)(concat_input)
        
        if len(self.lstm_array_units[0]) > 2:
            for num_layer in range (1, len(self.lstm_array_units[0]) - 1):
                lstm_layer = LSTM(units=self.lstm_array_units[0][num_layer],
                                  kernel_regularizer=self.regularizer_rate, recurrent_regularizer=self.regularizer_rate, bias_regularizer=self.regularizer_rate,
                                  kernel_constraint=self.kernel_weight_max_norm, recurrent_constraint=self.rnn_weight_max_norm, bias_constraint=self.bias_max_norm,
                                  dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate,
                                  return_sequences=True)(lstm_layer)
        
        # Last layer before output layer
        price_output = LSTM(self.lstm_array_units[0][-1], 
                            kernel_regularizer=self.regularizer_rate, recurrent_regularizer=self.regularizer_rate, bias_regularizer=self.regularizer_rate,
                            kernel_constraint=self.kernel_weight_max_norm, recurrent_constraint=self.rnn_weight_max_norm, bias_constraint=self.bias_max_norm,
                            dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate,
                            return_sequences=False)(lstm_layer)
        
        # Output layer
        # price_output = Dense(self.lstm_array_units[0][-1], activation='linear', name='price_output_main')(lstm_layer)
        
        # | Trend Layer | - Predict up trend as classification
        # ====================================================
        up_trend_layer = Dense(self.lstm_array_units[1][0], activation='relu',
                               kernel_regularizer=self.regularizer_rate, bias_regularizer=self.regularizer_rate, activity_regularizer=self.regularizer_rate,
                               kernel_constraint=self.kernel_weight_max_norm, bias_constraint=self.bias_max_norm)(price_output)
        
        up_trend_layer = Dropout(self.dropout_rate)(up_trend_layer)
        
        if len(self.lstm_array_units[1]) > 2:
            for num_layer in range (1, len(self.lstm_array_units[1]) - 1):
                up_trend_layer = Dense(self.lstm_array_units[1][num_layer], activation='relu',
                                       kernel_regularizer=self.regularizer_rate, bias_regularizer=self.regularizer_rate, activity_regularizer=self.regularizer_rate,
                                       kernel_constraint=self.kernel_weight_max_norm, bias_constraint=self.bias_max_norm)(up_trend_layer)
                
                up_trend_layer = Dropout(self.dropout_rate)(up_trend_layer)
            
        up_trend_output = Dense(self.lstm_array_units[1][-1], activation='softmax', name='up_trend')(up_trend_layer)
        
        # | Trend Layer | - Predict down trend as classification
        # ======================================================
        down_trend_layer = Dense(self.lstm_array_units[1][0], activation='relu',
                                 kernel_regularizer=self.regularizer_rate, bias_regularizer=self.regularizer_rate, activity_regularizer=self.regularizer_rate,
                                 kernel_constraint=self.kernel_weight_max_norm, bias_constraint=self.bias_max_norm)(price_output)
        
        down_trend_layer = Dropout(self.dropout_rate)(down_trend_layer)
        
        if len(self.lstm_array_units[1]) > 2:
            for num_layer in range (1, len(self.lstm_array_units[1]) - 1):
                down_trend_layer = Dense(self.lstm_array_units[1][num_layer], activation='relu',
                                         kernel_regularizer=self.regularizer_rate, bias_regularizer=self.regularizer_rate, activity_regularizer=self.regularizer_rate,
                                         kernel_constraint=self.kernel_weight_max_norm, bias_constraint=self.bias_max_norm)(down_trend_layer)
                
                down_trend_layer = Dropout(self.dropout_rate)(down_trend_layer)
        
        down_trend_output = Dense(self.lstm_array_units[1][-1], activation='softmax', name='down_trend')(down_trend_layer)
        
        # | Projectile Layer |
        # ====================
        projectile_layer = Dense(self.lstm_array_units[2][0], activation='relu',
                                 kernel_regularizer=self.regularizer_rate, bias_regularizer=self.regularizer_rate, activity_regularizer=self.regularizer_rate,
                                 kernel_constraint=self.kernel_weight_max_norm, bias_constraint=self.bias_max_norm)(price_output)
        
        projectile_layer = Dropout(self.dropout_rate)(projectile_layer)
        
        if len(self.lstm_array_units[2]) > 2:
            for num_layer in range (1, len(self.lstm_array_units[2]) - 1):
                projectile_layer = Dense(self.lstm_array_units[2][num_layer], activation='relu',
                                         kernel_regularizer=self.regularizer_rate, bias_regularizer=self.regularizer_rate, activity_regularizer=self.regularizer_rate,
                                         kernel_constraint=self.kernel_weight_max_norm, bias_constraint=self.bias_max_norm)(projectile_layer)
                
                projectile_layer = Dropout(self.dropout_rate)(projectile_layer)
        
        projectile_output = Dense(self.lstm_array_units[2][-1], activation='softmax', name='projectile')(projectile_layer)
        
        
        
        return Model(inputs = [input_layer_main, input_layer_sub_1], 
                     outputs = [up_trend_output, down_trend_output, projectile_output])
    
        # =====================================================================
    
    def create_model(self):
        return self.model
    
    def save_model(model, model_name):
        model_dir = "models"
        # If the directory is not existed, creat one
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # Specify the path where you want to save the model
        model_path = os.path.join(model_dir, f'{model_name}.keras')
        # Save the model
        model.save(model_path)
    
    def load_model(model_name):
        model_path = os.path.join('models', f'{model_name}.keras')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found at {model_path}")
            
        return load_model(model_path)
        
    #=========================================================================#