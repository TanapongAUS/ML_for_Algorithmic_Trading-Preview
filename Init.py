import os
import csv

"""
Created on Fri Jul 26 14:38:31 2024

@author: tanap
"""
class Init:
    def init():
        if not os.path.exists("models"):
            os.makedirs("models")
            
        if not os.path.exists("files"):
            os.makedirs("files")
            
        if not os.path.exists("files/data"):
            os.makedirs("files/data")
            
        if not os.path.exists("files/std_value/training"):
            os.makedirs("files/std_value/training")
            
        if not os.path.exists("files/std_value/env"):
            os.makedirs("files/std_value/env")
        
        if not os.path.exists("files/training_log"):
            os.makedirs("files/training_log")
            
        if not os.path.exists("files/trading_Log"):
            os.makedirs("files/trading_Log")
            
        if not os.path.exists("files/trading_Log/Trading_Log.csv"):
            with open("files/trading_Log/Trading_Log.csv", 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                header = ["Date/Time", "Orders"]
                csv_writer.writerow(header)
        
        if not os.path.exists("files/result"):
            os.makedirs("files/result")
        
        if not os.path.exists("files/analysis"):
            os.makedirs("files/analysis")
        
        print("Program files are initiated successfully!")
    
    #+++==================================================+++
    
    def check_data():
        if not os.path.exists("files/data/data_M1.csv"):
            return False
        
        return True