import keras.models
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics import accuracy_score
from IPython.display import clear_output
import sys
tf.get_logger().setLevel("ERROR")


def build_model():
    dnn_model = Sequential([
        Dense(units=64, activation='relu'),
        Dense(units=64, activation='relu'),
        Dense(units=1)
    ])
    dnn_model.compile(loss='mean_absolute_error',
                      optimizer=tf.keras.optimizers.Adam(0.00001))
    return dnn_model


url = 'Real estate.csv'
column_names = ['num', 'transaction_date', 'house_date', 'dist_to_MRT',
                'num_convenience', 'lat', 'long', 'house_price']
x = pd.read_csv(url, names=column_names)
x = x.drop(0)

for col in column_names:
    x[col] = pd.to_numeric(x[col])

x.pop('num')
y = x.pop("house_price")

x_train = x.sample(frac=0.8, random_state=0)
x_test = x.drop(x_train.index)
y_train = y.sample(frac=0.8, random_state=0)
y_test = y.drop(y_train.index)

wrong_load_input = True
wrong_save_input = True
load_input = input("Would you like to load a model? Y/n ")
while wrong_load_input:
    if load_input == "Y" or load_input == "y":
        load = input("Type the name of the model you would like to load: ")
        model = keras.models.load_model(load + ".keras")
        wrong_load_input = False
        wrong_save_input = False

    elif load_input == "N" or load_input == "n":
        EPOCHS = 700
        model = build_model()
        model.fit(x_train, y_train, epochs=EPOCHS, batch_size=60, shuffle=True, verbose=1)
        wrong_load_input = False
    else:
        load_input = input("Would you like to load a model? Y/n ")

y_pred = model.predict(x_test).flatten()
error = y_pred - y_test
print(error.mean())
print(error.median())

while wrong_save_input:
    save_input = input("Would you like to save this model? Y/n ")
    if save_input == "Y" or save_input == "y":
        model_name = input("Input model name: ")
        model.save(model_name + ".keras")
        wrong_save_input = False
    elif save_input == "N" or save_input == "n":
        sys.exit()
    else:
        save_input = input("Would you like to save this model? Y/n ")
