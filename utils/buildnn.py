import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, Model

def buildnn(SectorDF, layers):
    number_input_features = len(X.iloc[0])
    number_output_neurons = 1
    nn = Sequential()
    nn.add(Dense(units=(number_input_features + 1) // 2 , input_dim=number_input_features, activation="relu"))
    for layer in layers:
        nn.add(Dense(units=layer, activation="relu"))
    nn.add(Dense(units=number_output_neurons, activation="sigmoid"))
    return nn