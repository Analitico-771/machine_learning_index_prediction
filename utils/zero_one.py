import pandas as pd

def zero_one(inputDF):
    inputDF['SPY Returns'] = 0.0
    inputDF.loc[(inputDF['SPY'] >= 0), 'SPY Returns'] = 1
    inputDF.loc[(inputDF['SPY'] < 0), 'SPY Returns'] = 0
    inputDF.drop(columns='SPY', inplace=True)
    inputDF['SPY Returns'] = inputDF['SPY Returns'].shift(-1)
    inputDF['SPY Open'] = inputDF['SPY Open'].shift(-1)
    inputDF['SPY Close'] = inputDF['SPY Close'].shift(-1)
    inputDF.rename(columns={'SPY Returns':'SPY'}, inplace=True)
    inputDF.dropna(inplace=True)
    return inputDF