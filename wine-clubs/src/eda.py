import pandas as pd 
import numpy as np

'''
import file, throw into dataframe
'''

def get_dataframe(filename):
    filename = "data/Order-Data-Sample.csv"
    return pd.read_csv(filename)

def print_stats(data):
    print(data.info())
    print(data.describe())
    for col in data.columns:
        if type(data[col].values[0])==str:
            print(col,data[col].unique())

def eliminate_nans(data):
    '''
    This function takes in a dataframe with nan values...
    '''
    pass

def get_intimacy_score(data):
    '''
    This function takes in a dataframe and creates a feature for total bottles poured that day and applies feature to all customers for that day.
    '''
    pass

def get_dow(data):
    data["CalendarDate"] = np.array([x[:10] for x in data["OrderCompletedDate"]])
    data = data.drop(columns=["OrderCompletedDate","OrderCompletedDate.1"])
    return data

    








