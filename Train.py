import os
import json
import re
import numpy as np
import pandas as pd
import time as t
from datetime import datetime, time
from DataPrep import DataPrep
from Backtest import backtest
from Callback import Callback
from Utils import Utils

#=============================================================================#
# Global configuration
#=============================================================================#
IS_TRAIN_MODEL = False
IS_EVALUATE_MODEL = True
EVALUATION_EPISODE = 20
IS_RANDOM_LEARNING_RATE = False
IS_DATA_GENERATOR = False
BATCH_SIZE = 24



#=============================================================================#
# Read environment setup configuration
#=============================================================================#
# | Read JSON file |
# ==================
with open('TrainConfig.json', 'r') as json_file:
    train_config = json.load(json_file)

num_episodes = train_config['num_episodes']
training_downtime_str = train_config['training_downtime']
training_downtime_end_str = train_config['training_downtime_end']
downtime_duration = train_config['downtime_duration_secs']
sleep_time = train_config['sleep_time']



#=============================================================================#
# Load model
#=============================================================================#
if os.path.exists("models/predict_model.keras"):
    
    tradeMod = Utils.load_model("predict_model")
    
    # Testing data directory
    save_dir = "files/env_data"
    
    # Find the highest existing index for the given base filename
    existing_files = [
        f for f in os.listdir(save_dir) 
        if f.startswith("test_main_padded") and f.endswith(".npz")
    ]

    # Extract existing indices
    existing_indices = []
    for file in existing_files:
        match = re.search(r"test_main_padded_(\d+)\.npz", file)
        if match:
            existing_indices.append(int(match.group(1)))
    
    # Sort indexs
    existing_indices = sorted(existing_indices)
    
    # Load input data
    test_main_padded = np.load(f"files/env_data/test_main_padded_{existing_indices[-1]}.npz")['data']
    test_sub_padded = np.load(f"files/env_data/test_sub_padded_{existing_indices[-1]}.npz")['data']
    padded_data = [test_main_padded, test_sub_padded]
    # Load CSV data
    env_test_main_dataset = pd.read_csv(f"files/env_data/env_test_main_dataset_{existing_indices[-1]}.csv")
    env_test_sub_dataset = pd.read_csv(f"files/env_data/env_test_sub_dataset{existing_indices[-1]}.csv")
    
    predicted_winRate, predicted_reward, predicted_totalTrade = backtest([env_test_main_dataset, env_test_sub_dataset], padded_data, tradeMod, "test", verbose=2)          
    
    # Record as .txt file
    Utils.save_txt("KPI-model_predict", predicted_winRate, predicted_reward, predicted_totalTrade)
    
    # Print out
    print("==> predict_model is loaded!")
    
elif os.path.exists("models/model-class_2.keras"):
    tradeMod = Utils.load_model("model-class_2")
    print("==> model-class_2 is loaded!")

else:
    tradeMod = Utils.load_model("model-class_1")
    print("==> model-class_1 is loaded!")



#=============================================================================#
# Training
from tensorflow.keras.optimizers import Adam
#=============================================================================#
# Setup optimizer
optimizer = Adam(learning_rate = 0.0001, global_clipnorm = 1.0)

if IS_RANDOM_LEARNING_RATE:
    # Function to generate a random learning rate
    def random_learning_rate(min_lr=0.0001, max_lr=0.001):
        return np.random.uniform(min_lr, max_lr)

# Comply model with optimizer
tradeMod.compile(optimizer = optimizer, 
                 loss = {'up_trend': 'categorical_crossentropy',
                         'down_trend': 'categorical_crossentropy',
                         'projectile': 'categorical_crossentropy'
                         },
                 metrics = {'up_trend': ['accuracy'],
                            'down_trend': ['accuracy'],
                            'projectile': ['accuracy']
                            }
                 )

for episode in range(num_episodes):
    
    if IS_TRAIN_MODEL:
        
        print("\n================ Train Mode ================")
        print("==> Training in progress...")
        print(f"==> Episode: {episode+1}/{num_episodes}")
        # Training data directory
        save_dir = "files/numpy_data"
    
        # Find the highest existing index for the given base filename
        existing_files = [
            f for f in os.listdir(save_dir) 
            if f.startswith("input_main_padded") and f.endswith(".npz")
        ]
    
        # Extract existing indices
        existing_indices = []
        for file in existing_files:
            match = re.search(r"input_main_padded_(\d+)\.npz", file)
            if match:
                existing_indices.append(int(match.group(1)))
        
        for indice in existing_indices:
            # Load input data
            input_main_padded = np.load(f"files/numpy_data/input_main_padded_{indice}.npz")['data']
            input_sub_padded = np.load(f"files/numpy_data/input_sub_padded_{indice}.npz")['data']
            padded_data = [input_main_padded, input_sub_padded]
        
            # Load target data
            train_target = np.load(f"files/numpy_data/train_target_{indice}.npz", allow_pickle=True)
            # Convert to a dictionary
            train_target = {key: train_target[key] for key in train_target.keys()}
            
            # Train model
            if IS_DATA_GENERATOR:
                # Generate data
                data_gen = DataPrep.data_generator(padded_data, train_target, len(padded_data[0]), sleep_time=0.5)
                # Train data
                history = tradeMod.fit(data_gen, steps_per_epoch=len(padded_data[0]), epochs=1, batch_size=BATCH_SIZE, verbose=0)
                
            else:
                # Train data
                history = tradeMod.fit(padded_data, train_target, epochs=1, batch_size=BATCH_SIZE, verbose=0)
            
            # Save model
            Utils.save_model(tradeMod, "model-class_1")
            
            # Extract metrics from the History object
            loss = history.history['loss'][-1]
            down_trend_accuracy = history.history['down_trend_accuracy'][-1]
            down_trend_loss = history.history['down_trend_loss'][-1]
            projectile_accuracy = history.history['projectile_accuracy'][-1]
            projectile_loss = history.history['projectile_loss'][-1]
            up_trend_accuracy = history.history['up_trend_accuracy'][-1]
            up_trend_loss = history.history['up_trend_loss'][-1]
            
            # Get the current time
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%d/%m/%Y %H:%M:%S")
            # Print the file number, formatted date and time
            print(f"\nFile: {indice}/{len(existing_indices)}")
            print("Time: ", formatted_datetime)
            
            # Print the metrics
            print("=== | Loss | ===")
            print(f"Up Trend Loss: {up_trend_loss} \nDown Trend Loss: {down_trend_loss} \nProjectile Loss: {projectile_loss}")
            print(f"==> Total Loss: {loss}")
            
            print("=== | Accuracy | ===")
            print(f"Up Trend Accuracy: {up_trend_accuracy} \nDown Trend Accuracy: {down_trend_accuracy} \nProjectile Accuracy: {projectile_accuracy}")
            
            print("===========================================")
            
            # Extract the time component (HH:MM:SS)
            current_time = current_datetime.time()
            # Convert the string of training_downtime to a time object
            training_downtime = datetime.strptime(training_downtime_str, "%H:%M:%S").time()
            training_downtime_end = datetime.strptime(training_downtime_end_str, "%H:%M:%S").time()
            
            # Sleep
            if current_time >= training_downtime and current_time < training_downtime_end:
                t.sleep(downtime_duration)
            else:
                t.sleep(sleep_time)
    
    #=================================================
    # Evaluation process
    #=================================================
    
    if (IS_EVALUATE_MODEL == True and episode + 1 >= EVALUATION_EPISODE) or (IS_EVALUATE_MODEL == True and IS_TRAIN_MODEL == False):
        
        print("\n================ Evaluation Mode ================")
        
        # Testing data directory
        save_dir = "files/env_data"
        
        # Find the highest existing index for the given base filename
        existing_files = [
            f for f in os.listdir(save_dir) 
            if f.startswith("test_main_padded") and f.endswith(".npz")
        ]

        # Extract existing indices
        existing_indices = []
        for file in existing_files:
            match = re.search(r"test_main_padded_(\d+)\.npz", file)
            if match:
                existing_indices.append(int(match.group(1)))
        
        # Sort indexs
        existing_indices = sorted(existing_indices)
        
        # Evaluate with previous dataset |
        #=================================
        print("==> Evaluating model with training dataset - Acceptance condition_1 ...")
        # Load input data
        test_main_padded = np.load(f"files/env_data/test_main_padded_{existing_indices[-2]}.npz")['data']
        test_sub_padded = np.load(f"files/env_data/test_sub_padded_{existing_indices[-2]}.npz")['data']
        padded_data = [test_main_padded, test_sub_padded]
    
        # Load target data
        test_target = np.load(f"files/env_data/test_target_{existing_indices[-2]}.npz", allow_pickle=True)
        # Convert to a dictionary
        test_target = {key: test_target[key] for key in test_target.keys()}
        
        # Evaluate
        info = tradeMod.evaluate(padded_data, test_target, batch_size=BATCH_SIZE, verbose=0)
        # Extract metrics from the History object
        loss = info[0]
        up_trend_loss = info[1]
        down_trend_loss = info[2]
        projectile_loss = info[3]
        down_trend_accuracy = info[4]
        projectile_accuracy = info[5]
        up_trend_accuracy = info[6]
        
        # Get the current time
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%d/%m/%Y %H:%M:%S")
        # Print the formatted date and time
        print("\nTime: ", formatted_datetime)
        
        # Print the metrics
        print("=== | Loss | ===")
        print(f"Up Trend Loss: {up_trend_loss} \nDown Trend Loss: {down_trend_loss} \nProjectile Loss: {projectile_loss}")
        print(f"==> Total Loss: {loss}")
        
        print("=== | Accuracy | ===")
        print(f"Up Trend Accuracy: {up_trend_accuracy} \nDown Trend Accuracy: {down_trend_accuracy} \nProjectile Accuracy: {projectile_accuracy}")
        
        print("===========================================")
        
        # Backtest with previous dataset |
        #=================================
        print("==> Backtesting model with training dataset - Acceptance condition_1 ...")
        # Load CSV data
        env_test_main_dataset = pd.read_csv(f"files/env_data/env_test_main_dataset_{existing_indices[-2]}.csv")
        env_test_sub_dataset = pd.read_csv(f"files/env_data/env_test_sub_dataset_{existing_indices[-2]}.csv")
        
        # Backtest
        train_winRate, train_reward, train_totalTrade = backtest([env_test_main_dataset, env_test_sub_dataset], padded_data, tradeMod, "train", verbose=2)
        
        # Check for Acceptance Condition - 1 |
        #=====================================
        """
        result_I = Callback.filter_I(train_winRate, train_reward)
        
        if result_I != True:
            continue
        
        else:
            # If pass the acceptance-1, save the model for a higher class
            Utils.save_model(tradeMod, "model-class_2")
            # Save as txt file
            Utils.save_txt("KPI-model_class_2", train_winRate, train_reward, train_totalTrade)
            print("The model-class_2 is saved! ^-^")
        """
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        # Evaluate with current dataset |
        #================================
        print("==> Evaluating model with testing dataset - Acceptance condition_2 ...")
        # Load input data
        test_main_padded = np.load(f"files/env_data/test_main_padded_{existing_indices[-1]}.npz")['data']
        test_sub_padded = np.load(f"files/env_data/test_sub_padded_{existing_indices[-1]}.npz")['data']
        padded_data = [test_main_padded, test_sub_padded]
    
        # Load target data
        test_target = np.load(f"files/env_data/test_target_{existing_indices[-1]}.npz", allow_pickle=True)
        # Convert to a dictionary
        test_target = {key: test_target[key] for key in test_target.keys()}
        
        # Evaluate
        info = tradeMod.evaluate(padded_data, test_target, batch_size=BATCH_SIZE, verbose=0)
        # Extract metrics from the History object
        loss = info[0]
        up_trend_loss = info[1]
        down_trend_loss = info[2]
        projectile_loss = info[3]
        down_trend_accuracy = info[4]
        projectile_accuracy = info[5]
        up_trend_accuracy = info[6]
        
        # Get the current time
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%d/%m/%Y %H:%M:%S")
        # Print the formatted date and time
        print("\nTime: ", formatted_datetime)
        
        # Print the metrics
        print("=== | Loss | ===")
        print(f"Up Trend Loss: {up_trend_loss} \nDown Trend Loss: {down_trend_loss} \nProjectile Loss: {projectile_loss}")
        print(f"==> Total Loss: {loss}")
        
        print("=== | Accuracy | ===")
        print(f"Up Trend Accuracy: {up_trend_accuracy} \nDown Trend Accuracy: {down_trend_accuracy} \nProjectile Accuracy: {projectile_accuracy}")
        
        print("===========================================")
        
        # Backtest with current dataset |
        #================================
        print("==> Backtesting model with testing dataset - Acceptance condition_2 ...")
        # Load CSV data
        env_test_main_dataset = pd.read_csv(f"files/env_data/env_test_main_dataset_{existing_indices[-1]}.csv")
        env_test_sub_dataset = pd.read_csv(f"files/env_data/env_test_sub_dataset_{existing_indices[-1]}.csv")
        
        # Backtest
        eval_winRate, eval_reward, eval_totalTrade = backtest([env_test_main_dataset, env_test_sub_dataset], padded_data, tradeMod, "test", verbose=2)
        
        # Check for Acceptance Condition - 2 |
        #=====================================
        """
        # Load saving predict_model
        predict_winRate, predict_reward, predict_totalTrade = Utils.load_txt("KPI-model_predict")
        
        result_II = Callback.filter_II(eval_winRate, predict_winRate, eval_reward, predict_reward, eval_totalTrade, predict_totalTrade)
        
        if result_II != True:
            continue
        else:
            Utils.save_model(tradeMod, "predict_model")
            Utils.save_txt("KPI-model_predict", eval_winRate, eval_reward, eval_totalTrade)
            print("The optimal predicting model is saved! ^-^")
        """
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        # Hault the system if the model training is unrequired |
        #=======================================================
        if IS_TRAIN_MODEL == False:
            break
        