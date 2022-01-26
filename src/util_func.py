import numpy as np
from sklearn.metrics import mean_squared_error

def sMAPE(predicted_values, targets):  
    smape_values = np.column_stack((predicted_values.reshape(-1,1), targets.reshape(-1,1)))
    smape = np.array([abs(a-b)/(abs(a)+abs(b)) for (a,b) in smape_values]).mean()*100
    return smape

def RMSE(predicted_values, targets):
    return np.sqrt(mean_squared_error(targets, predicted_values))

def MAE(predicted_values, targets):
    return abs(predicted_values-targets).mean()